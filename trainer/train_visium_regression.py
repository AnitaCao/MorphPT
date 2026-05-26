#!/usr/bin/env python3
"""
train_visium_regression.py
──────────────────────────
Train VisiumRegressor for gene expression prediction from H&E patches.

Data modes (--data_mode):
  memmap   fast pre-cached dataset (run build_memmap.py first)
  raw      PIL on-the-fly (slower, no preprocessing needed)

Fuse modes (--fuse):
  identity  single scale: --scales 10.0x  or  --scales 2.5x
  gate      dual scale:   --scales 2.5x,10.0x

Backbone modes:
  --freeze_backbone 1  frozen (all 6 jobs in train_jobs.sh)
  --freeze_backbone 0  end-to-end (backbone goes into optimizer via get_param_groups)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from data.visium_dataset import VisiumHDPredictionDataset
from data.visium_dataset_raw import VisiumHDPredictionDatasetRaw
from models.visium_regression import VisiumRegressor
from trainer.trainer_base import setup_ddp, cleanup_ddp, is_main, log

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── Parser ─────────────────────────────────────────────────────────────────
def get_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data mode
    p.add_argument("--data_mode",   choices=["memmap", "raw"], default="memmap")
    # Memmap args
    p.add_argument("--cache_dir",   type=str,
                   default="/hpc/group/jilab/rz179/MorphPT_MOE/cache_visium")
    # Raw args
    p.add_argument("--root_dir",    type=str,
                   default="/hpc/group/jilab/boxuan/visiumHD/human_crc")
    p.add_argument("--morphpt_data_dir", type=str,
                   default="/hpc/group/jilab/hz/MorphPT/data/visiumHD/human_crc")
    p.add_argument("--img_variant", type=str, default="raw",
                   choices=["raw", "mask_target", "mask_context"])
    # Split params (raw mode — must match build_memmap.py settings)
    p.add_argument("--grid_size",   type=int,  default=5)
    p.add_argument("--test_tiles",  type=int,  default=4)
    p.add_argument("--val_tiles",   type=int,  default=3)
    p.add_argument("--min_cells",   type=int,  default=300)
    p.add_argument("--buffer_zone", action="store_true", default=False)
    p.add_argument("--split_seed",  type=int,  default=42)
    p.add_argument("--split_type", type=str, default="spatial",
               choices=["spatial", "random"])
    # Shared data args
    p.add_argument("--scales",      type=str,  default="10.0x")
    p.add_argument("--fuse",        type=str,  default="identity",
                   choices=["identity", "gate"])
    p.add_argument("--img_size",    type=int,  default=224)
    # Model args
    p.add_argument("--model",       type=str,
                   default="vit_base_patch16_dinov3.lvd1689m")
    p.add_argument("--ckpt_path",   type=str,  default=None)
    p.add_argument("--head_type",   type=str,  default="linear",
                   choices=["linear", "mlp"])
    p.add_argument("--freeze_backbone", type=int, default=1)
    p.add_argument("--lora_blocks", type=str,  default="0")
    p.add_argument("--lora_rank",   type=int,  default=8)
    p.add_argument("--lora_alpha",  type=int,  default=16)
    p.add_argument("--lora_dropout",type=float,default=0.05)
    p.add_argument("--lora_targets",type=str,  default="qkv")
    p.add_argument("--unfreeze_lora",type=int, default=0)
    p.add_argument("--gate_dropout",type=float,default=0.1)
    # Training args
    p.add_argument("--batch_size",  type=int,  default=256)
    p.add_argument("--epochs",      type=int,  default=30)
    p.add_argument("--lr",          type=float,default=3e-4)
    p.add_argument("--weight_decay",type=float,default=0.01)
    p.add_argument("--gate_wd",     type=float,default=0.1)
    p.add_argument("--lora_lr_scale",type=float,default=0.1)
    p.add_argument("--workers",     type=int,  default=8)
    p.add_argument("--eval_every",  type=int,  default=1)
    p.add_argument("--warmup_epochs",type=int, default=3,
                   help="Linear LR warmup epochs (0 = no warmup).")
    # Early stopping
    p.add_argument("--patience",    type=int,  default=8,
                   help="Stop if val_pearson does not improve for this many "
                        "evals. 0 = disabled.")
    p.add_argument("--min_epochs",  type=int,  default=10,
                   help="Never stop before this epoch (let LR schedule warm up).")
    # Loss
    p.add_argument("--loss_type",   type=str,  default="mse",
                   choices=["mse", "mixed"],
                   help="'mixed' adds (1 - pearson) to MSE.")
    p.add_argument("--pearson_weight", type=float, default=0.1,
                   help="Weight for (1 - pearson) term in mixed loss.")
    # Output
    p.add_argument("--out_dir",     type=str,  required=True)
    p.add_argument("--wandb_project",type=str, default="visium_regression")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--target_gene_idx", type=int, default=None, help="Target a specific gene idx")
    p.add_argument("--target_gene_name", type=str, default=None, help="Target gene name for wandb metadata")
    return p


# ── Metrics ────────────────────────────────────────────────────────────────
def pearson_per_gene(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N, G) → (G,) per-gene Pearson r."""
    pm  = pred   - pred.mean(dim=0)
    tm  = target - target.mean(dim=0)
    num = (pm * tm).sum(dim=0)
    den = (pm.pow(2).sum(dim=0) * tm.pow(2).sum(dim=0)).clamp(min=1e-8).sqrt()
    return num / den


def mixed_loss(pred: torch.Tensor, target: torch.Tensor,
               pearson_weight: float = 0.1) -> torch.Tensor:
    """
    MSE + pearson_weight * (1 - mean_pearson_r).

    The Pearson term directly optimizes the evaluation metric.
    Clamp r to [0, 1] because mini-batch Pearson can go negative
    from noise, and we don't want (1 - r) > 1 to dominate the loss.
    """
    mse = F.mse_loss(pred, target)
    r   = pearson_per_gene(pred, target)           # (G,)
    r   = r.clamp(min=0.0, max=1.0).mean()         # scalar
    return mse + pearson_weight * (1.0 - r)


# ── Eval ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, device, amp_dtype, target_gene_idx=None):
    model.eval()
    loss_sum, n_elem = 0.0, 0
    all_pred, all_y  = [], []
    for imgs, expr, _ in loader:
        if target_gene_idx is not None:
            expr = expr[:, target_gene_idx].unsqueeze(1)
        imgs = imgs.to(device, non_blocking=True)
        expr = expr.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=torch.cuda.is_available()):
            pred = model(imgs)
            loss = F.mse_loss(pred, expr, reduction="sum")
        loss_sum += float(loss.item())
        n_elem   += expr.numel()
        all_pred.append(pred.float().cpu())
        all_y.append(expr.float().cpu())
    return torch.cat(all_pred), torch.cat(all_y), loss_sum, n_elem


def gather_eval(pred, y, loss_sum, n_elem, device, is_ddp, rank):
    if is_ddp:
        def _gather(t):
            t = t.to(device)
            sizes = [torch.zeros(1, dtype=torch.long, device=device)
                     for _ in range(dist.get_world_size())]
            dist.all_gather(sizes, torch.tensor([t.shape[0]], device=device))
            max_sz = max(s.item() for s in sizes)
            padded = torch.zeros(max_sz, t.shape[1], device=device, dtype=t.dtype)
            padded[:t.shape[0]] = t
            bufs = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
            dist.all_gather(bufs, padded)
            return torch.cat([b[:s.item()] for b, s in zip(bufs, sizes)]).cpu()
        pred = _gather(pred)
        y    = _gather(y)
        t = torch.tensor([loss_sum, float(n_elem)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum, n_elem = t[0].item(), int(t[1].item())

    if rank == 0:
        # Note: MSE is in standardized (z-score) space — label as zMSE
        zmse = loss_sum / max(1, n_elem)
        corr = pearson_per_gene(pred, y)
        return zmse, corr.mean().item(), corr
    return None, None, None


# ── Dataset builder ────────────────────────────────────────────────────────
def build_datasets(args):
    scales = [s.strip() for s in args.scales.split(",")]

    if args.data_mode == "memmap":
        def _ds(split):
            return VisiumHDPredictionDataset(
                cache_dir = args.cache_dir,
                split     = split,
                scales    = scales,
                fuse      = args.fuse,
                augment   = (split == "train"),
                split_type = args.split_type,
            )
    else:
        def _ds(split):
            return VisiumHDPredictionDatasetRaw(
                root_dir         = args.root_dir,
                morphpt_data_dir = args.morphpt_data_dir,
                split            = split,
                scales           = scales,
                fuse             = args.fuse,
                img_variant      = args.img_variant,
                img_size         = args.img_size,
                augment          = (split == "train"),
                grid_size        = args.grid_size,
                test_tiles       = args.test_tiles,
                val_tiles        = args.val_tiles,
                min_cells        = args.min_cells,
                buffer_zone      = args.buffer_zone,
                seed             = args.split_seed,
            )

    return _ds("train"), _ds("val"), _ds("test")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args  = get_parser().parse_args()
    rank, local_rank, world_size, is_ddp = setup_ddp()
    device    = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                             else "cpu")
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    use_wandb = False
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
        if HAS_WANDB and args.wandb_project:
            wandb.init(project=args.wandb_project, config=vars(args),
                       name=out_dir.name, dir=str(out_dir))
            use_wandb = True

    # ── Datasets ──────────────────────────────────────────────────────────
    scales = [s.strip() for s in args.scales.split(",")]
    log(rank, f"Data mode={args.data_mode}  scales={scales}  fuse={args.fuse}")
    train_ds, val_ds, test_ds = build_datasets(args)
    num_genes = train_ds.gene_mean.shape[0]
    out_dim = 1 if args.target_gene_idx is not None else num_genes
    log(rank, f"  Train={len(train_ds):,}  Val={len(val_ds):,}  "
              f"Test={len(test_ds):,}  Genes={num_genes:,}  OutDim={out_dim}")

    train_sampler = DistributedSampler(train_ds, shuffle=True,
                                       seed=args.seed) if is_ddp else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    nw = args.workers
    pf = 4 if nw > 0 else None
    pw = nw > 0

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=pw, prefetch_factor=pf)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              sampler=val_sampler, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=pw, prefetch_factor=pf)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True,
                              persistent_workers=pw, prefetch_factor=pf)

    # ── Model ─────────────────────────────────────────────────────────────
    log(rank, "Building model...")
    model = VisiumRegressor(
        model_name      = args.model,
        img_size        = args.img_size,
        out_dim         = out_dim,
        pretrained      = True,
        fuse            = args.fuse,
        freeze_backbone = bool(args.freeze_backbone),
        lora_blocks     = args.lora_blocks,
        lora_rank       = args.lora_rank,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        lora_targets    = args.lora_targets,
        unfreeze_lora   = bool(args.unfreeze_lora),
        ckpt_path       = args.ckpt_path,
        head_type       = args.head_type,
        gate_dropout    = args.gate_dropout,
    ).to(device)

    if is_ddp:
        model     = DDP(model, device_ids=[local_rank],
                        find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    # ── Optimizer — use get_param_groups so ALL trainable params are covered
    # This is correct for both frozen and end-to-end modes.
    log(rank, "Building optimizer param groups:")
    param_groups = raw_model.get_param_groups(
        lr           = args.lr,
        weight_decay = args.weight_decay,
        gate_wd      = args.gate_wd,
        lora_lr_scale= args.lora_lr_scale,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # LR schedule: optional linear warmup → cosine decay
    warmup_ep = min(args.warmup_epochs, args.epochs - 1)
    if warmup_ep > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_ep,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_ep, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_ep],
        )
        log(rank, f"LR schedule: {warmup_ep}-epoch warmup → cosine")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )
        log(rank, "LR schedule: cosine (no warmup)")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    best_corr    = -float("inf")
    best_path    = out_dir / "best.pth"
    log_rows     = []
    no_improve   = 0          # consecutive evals without improvement
    stop_signal  = torch.zeros(1, device=device)  # broadcast flag for DDP

    log(rank, f"\n{'ep':>4} {'tr_zMSE':>9} {'va_zMSE':>9} "
              f"{'va_corr':>8} {'lr':>10} {'secs':>6}")
    log(rank, "─" * 55)

    # ── Training loop ──────────────────────────────────────────────────────
    for ep in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        model.train()
        tr_loss, tr_n = 0.0, 0
        t0 = time.time()

        for imgs, expr, _ in train_loader:
            if args.target_gene_idx is not None:
                expr = expr[:, args.target_gene_idx].unsqueeze(1)
            imgs = imgs.to(device, non_blocking=True)
            expr = expr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=torch.cuda.is_available()):
                pred = model(imgs)
                if args.loss_type == "mixed":
                    loss = mixed_loss(pred, expr, args.pearson_weight)
                else:
                    loss = F.mse_loss(pred, expr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * expr.size(0)
            tr_n    += expr.size(0)

        scheduler.step()

        # Aggregate train loss across GPUs
        t_agg = torch.tensor([tr_loss, float(tr_n)], device=device)
        if is_ddp:
            dist.all_reduce(t_agg, op=dist.ReduceOp.SUM)
        tr_zmse = (t_agg[0] / t_agg[1].clamp(min=1)).item()

        # ── Validation ─────────────────────────────────────────────────────
        if ep % args.eval_every == 0 or ep == args.epochs:
            pred_v, y_v, vl, vn = eval_epoch(raw_model, val_loader,
                                              device, amp_dtype, args.target_gene_idx)
            va_zmse, va_corr, per_gene_corr = gather_eval(
                pred_v, y_v, vl, vn, device, is_ddp, rank
            )

            if is_main(rank):
                dt     = time.time() - t0
                cur_lr = scheduler.get_last_lr()[0]
                log(rank, f"{ep:4d} {tr_zmse:9.4f} {va_zmse:9.4f} "
                           f"{va_corr:8.4f} {cur_lr:10.2e} {dt:6.1f}")

                row = {"epoch": ep, "tr_zMSE": round(tr_zmse, 5),
                       "va_zMSE": round(va_zmse, 5),
                       "va_mean_pearson": round(va_corr, 5)}
                log_rows.append(row)

                if use_wandb:
                    wandb.log({**row, "lr": cur_lr}, step=ep)
                    # Log actual gate weights from a small val batch
                    if args.fuse == "gate":
                        imgs_sample, _, _ = next(iter(val_loader))
                        imgs_sample = imgs_sample[:64].to(device)
                        w = raw_model.get_gate_weights(imgs_sample).cpu()
                        wandb.log({
                            f"gate_w_{scales[0]}_mean": w[:, 0].mean().item(),
                            f"gate_w_{scales[1]}_mean": w[:, 1].mean().item(),
                        }, step=ep)

                ckpt = {
                    "epoch":          ep,
                    "state_dict":     raw_model.state_dict(),
                    "va_zmse":        va_zmse,
                    "va_mean_pearson":va_corr,
                    "args":           vars(args),
                }
                if va_corr > best_corr:
                    best_corr  = va_corr
                    no_improve = 0
                    torch.save(ckpt, best_path)
                    torch.save(per_gene_corr,
                               out_dir / "per_gene_pearson_best.pt")
                    log(rank, f"  ↑ New best  val_pearson={best_corr:.4f}")
                else:
                    no_improve += 1
                    if args.patience > 0:
                        log(rank, f"  no_improve={no_improve}/{args.patience}")

                if ep == args.epochs:
                    torch.save(ckpt, out_dir / "last.pth")
                    torch.save(per_gene_corr,
                               out_dir / "per_gene_pearson_last.pt")

                # ── Early stopping check (rank 0 sets signal) ───────────────
                if (args.patience > 0
                        and ep >= args.min_epochs
                        and no_improve >= args.patience):
                    log(rank, f"  Early stop at epoch {ep} "
                              f"(patience={args.patience}, "
                              f"min_epochs={args.min_epochs})")
                    torch.save(ckpt, out_dir / "last.pth")
                    torch.save(per_gene_corr,
                               out_dir / "per_gene_pearson_last.pt")
                    stop_signal.fill_(1.0)

            # Broadcast stop signal so all DDP ranks exit together
            if is_ddp:
                dist.broadcast(stop_signal, src=0)
            if stop_signal.item() > 0:
                break

    # ── Final test eval (rank 0 only) ──────────────────────────────────────
    cleanup_ddp(is_ddp)
    if not is_main(rank):
        return

    (out_dir / "train_log.json").write_text(json.dumps(log_rows, indent=2))

    log(0, f"\n{'='*55}")
    log(0, "Final test evaluation — best checkpoint")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["state_dict"])

    pred_t, y_t, tl, tn = eval_epoch(raw_model, test_loader, device, amp_dtype, args.target_gene_idx)
    test_zmse  = tl / max(1, tn)
    test_corr  = pearson_per_gene(pred_t, y_t)
    mean_r     = test_corr.mean().item()
    median_r   = test_corr.median().item()

    log(0, f"Test zMSE (standardized)  : {test_zmse:.4f}")
    log(0, f"Test mean   Pearson r      : {mean_r:.4f}")
    log(0, f"Test median Pearson r      : {median_r:.4f}")
    log(0, f"Genes r > 0.1             : {(test_corr > 0.1).sum():,} / {len(test_corr)}")
    log(0, f"Genes r > 0.2             : {(test_corr > 0.2).sum():,} / {len(test_corr)}")
    log(0, f"Genes r > 0.3             : {(test_corr > 0.3).sum():,} / {len(test_corr)}")

    test_results = {
        "note_mse":          "zMSE — MSE in per-gene standardized (z-score) space",
        "test_zMSE":         float(test_zmse),
        "test_mean_pearson": float(mean_r),
        "test_median_pearson": float(median_r),
        "best_val_pearson":  float(best_corr),
        "args":              vars(args),
    }
    (out_dir / "test_results.json").write_text(
        json.dumps(test_results, indent=2)
    )
    torch.save(test_corr, out_dir / "per_gene_pearson_test.pt")
    log(0, f"\nSaved → {out_dir}/test_results.json")

    if use_wandb:
        wandb.summary["test_mean_pearson"]   = mean_r
        wandb.summary["test_median_pearson"] = median_r
        wandb.summary["test_zMSE"]           = test_zmse
        wandb.finish()


if __name__ == "__main__":
    main()