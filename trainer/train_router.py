#!/usr/bin/env python3
"""
Router training for CellPT MoE — DDP
=====================================

4 fusion modes A/B test:
  --fuse avg     mean(emb_a, emb_b) → Linear(d, 7)
  --fuse concat  cat(emb_a, emb_b)  → Linear(2d, 7)
  --fuse gate    MLP gate → weighted avg → Linear(d, 7)
  --fuse late    avg(head_a(emb_a), head_b(emb_b))  [separate heads]

Launch (single node, multi-GPU):
  torchrun --nproc_per_node=4 train_router.py --fuse avg ...

Launch (single GPU, no DDP):
  python train_router.py --fuse avg ...
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import timm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.lora import apply_lora_to_timm_vit
from data.dataset import CellParquetMultiView, AugCfg, MemmapDataset

# Enable TF32 for faster matmul on Ampere+ GPUs (H100/H200)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trainer_base import (
    setup_ddp, cleanup_ddp, is_main, log,
    compute_class_weights, macro_f1, per_class_metrics,
    confusion_matrix_np, eval_epoch_local, gather_eval,
    get_base_parser
)

from models.model import MultiViewClassifier


# (Model definition moved to models/model.py)


# ─────────────────────────────────────────────
# Router-Specific Metrics & Training
# ─────────────────────────────────────────────


def per_fine_routing_accuracy(pred_coarse, ds_val, class_to_idx,
                               fine_col="label",
                               coarse_col="coarse_label",
                               ii_all=None,
                               fine_to_coarse_dict=None):
    """For each fine-grained cell type, compute what % are routed to the
    correct coarse class.  Returns list of dicts sorted by accuracy.
    Works with both CellParquetMultiView (has .df) and MemmapDataset
    (has ._fine_labels / ._coarse_labels)."""
    # ── Get fine/coarse label arrays ─────────────────────────────────────
    ds = ds_val.dataset if hasattr(ds_val, 'dataset') else ds_val
    if isinstance(ds, MemmapDataset):
        if ds._fine_labels is None or ds._coarse_labels is None:
            return None
        full_fine_labels   = ds._fine_labels
        full_coarse_labels = ds._coarse_labels
    else:
        df = ds.df
        if fine_col not in df.columns:
            return None
        full_fine_labels   = df[fine_col].values
        full_coarse_labels = df[coarse_col].values

    if ii_all is not None:
        # ii_all from evaluation already corresponds to absolute index
        # since __getitem__ in the raw dataset returns absolute indices.
        # But if the user overrides ds._y or otherwise has subset wrapper,
        # Subset returns the inner sample, so the batch gets the absolute idx.
        idx_np = ii_all.cpu().numpy()
        fine_labels = full_fine_labels[idx_np]
        coarse_labels = full_coarse_labels[idx_np]
    else:
        # if not provided, just filter assuming subset sequential if applicable
        if hasattr(ds_val, 'indices'):
            idx_np = ds_val.indices
            fine_labels = full_fine_labels[idx_np]
            coarse_labels = full_coarse_labels[idx_np]
        else:
            fine_labels = full_fine_labels
            coarse_labels = full_coarse_labels
        
    pred_np = pred_coarse.cpu().numpy()

    # Build fine → correct coarse_idx mapping
    fine_to_coarse_idx = {}
    if fine_to_coarse_dict is not None:
        for fl, cl in fine_to_coarse_dict.items():
            if cl in class_to_idx:
                fine_to_coarse_idx[fl] = class_to_idx[cl]
    else:
        seen = set()
        for fl, cl in zip(fine_labels, coarse_labels):
            if (fl, cl) not in seen:
                seen.add((fl, cl))
                if cl in class_to_idx:
                    fine_to_coarse_idx[fl] = class_to_idx[cl]

    fine_conf_mat = {}

    rows = []
    for fine_name in sorted(fine_to_coarse_idx.keys()):
        mask = fine_labels == fine_name
        n = int(mask.sum())
        if n == 0:
            continue
        correct_idx = fine_to_coarse_idx[fine_name]
        coarse_name = [k for k, v in class_to_idx.items() if v == correct_idx][0]
        correct = int((pred_np[mask] == correct_idx).sum())
        acc = correct / n
        rows.append({
            "fine_class": fine_name,
            "coarse_class": coarse_name,
            "n": n,
            "correct": correct,
            "acc": round(acc, 4),
        })

        # Calculate exactly how this fine class's cells were distributed
        fine_preds = pred_np[mask]
        dist = {cname: 0 for cname in class_to_idx.keys()}
        for cidx in fine_preds:
            cname_pred = [k for k, v in class_to_idx.items() if v == cidx][0]
            dist[cname_pred] += 1
        fine_conf_mat[fine_name] = dist
        
    return sorted(rows, key=lambda x: x["acc"]), fine_conf_mat


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, opt, scaler, device, amp_dtype,
                    class_weight, cons_w=0.0, epoch=1):
    model.train()
    loss_sum = 0.0
    cons_sum = 0.0
    n = 0

    # Ramp cons_w from 0 over epochs 3-7
    ramp = min(1.0, max(0, epoch - 2) / 5.0)
    eff_cons_w = cons_w * ramp

    for x, y, _tissue in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=torch.cuda.is_available()):
            logits, _emb, aux = model(x)
            if aux.get("is_log_prob"):
                loss_ce = F.nll_loss(logits, y, weight=class_weight)
            else:
                loss_ce = F.cross_entropy(logits, y, weight=class_weight)

            loss_cons = torch.tensor(0.0, device=device)
            # Log feature similarity (cosine) if we have embeddings
            if "emb_views" in aux:
                P = aux["emb_views"]                             # [B, 2, d]
                Pn = F.normalize(P, dim=-1)
                cos = (Pn[:, 0] * Pn[:, 1]).sum(dim=1)           # [B]
                # We no longer use cos for loss, it's just for logging/monitoring 
                # (eval loop tracks cv_cos_sum but train loop tracks loss_cons)
            
            # Compute KL Divergence logit consistency if z_list available
            if eff_cons_w > 0 and "z_list" in aux:
                z0, z1 = aux["z_list"]
                Tcons = getattr(model.module if hasattr(model, 'module') else model, 'cons_T', 3.0) 
                # Softmax temp scaling
                p0 = F.log_softmax(z0 / Tcons, dim=1)
                q0 = F.softmax(z1 / Tcons, dim=1)
                p1 = F.log_softmax(z1 / Tcons, dim=1)
                q1 = F.softmax(z0 / Tcons, dim=1)
                
                kl_loss = F.kl_div(p0, q0, reduction="batchmean") + F.kl_div(p1, q1, reduction="batchmean")
                loss_cons = kl_loss * (Tcons * Tcons)

            loss = loss_ce + eff_cons_w * loss_cons

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        loss_sum += float(loss_ce.item()) * y.size(0)
        cons_sum += float(loss_cons.item()) * y.size(0)
        n += y.size(0)

    return loss_sum / max(1, n), cons_sum / max(1, n)


# (Evaluation helpers moved to trainer_base.py)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = get_base_parser()
    
    # Override defaults specific to router
    ap.set_defaults(
        train_split="router_shards",
        wandb_project="cellpt-router"
    )
    ap.add_argument("--fine_to_coarse", type=str, default=None,
                    help="Path to fine_to_coarse.json to avoid implicit label mapping bugs")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="Fraction of training data to use for validation")
    
    args = ap.parse_args()

    # ── DDP setup ──
    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # Barrier so all ranks wait for dir creation
    if is_ddp:
        dist.barrier()

    # ── Class map ──
    with open(args.class_map) as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    fine_to_coarse_dict = None
    if getattr(args, "fine_to_coarse", None) and Path(args.fine_to_coarse).exists():
        with open(args.fine_to_coarse, "r") as f:
            fine_to_coarse_dict = json.load(f)
        log(rank, f"Loaded fine_to_coarse mapping from {args.fine_to_coarse}")

    log(rank, f"Classes ({num_classes}): {class_to_idx}")
    log(rank, f"Fuse: {args.fuse}  |  World size: {world_size}")

    # ── W&B ──
    use_wandb = HAS_WANDB and args.wandb_project and is_main(rank)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"router_{args.fuse}",
            config=vars(args),
            dir=str(out_dir),
        )
        log(rank, f"W&B run: {wandb.run.url}")

    # ── Datasets ──
    if args.cache_dir:
        # Fast path: memmap tensors, zero PIL decode at training time
        log(rank, f"Using MemmapDataset from {args.cache_dir}")
        ds_train_full = MemmapDataset(
            cache_dir=args.cache_dir,
            split_name=args.train_split,
            view="both",
            aug=AugCfg(enable=True),
            return_meta=True,
        )
        ds_val_full = MemmapDataset(
            cache_dir=args.cache_dir,
            split_name=args.train_split,
            view="both",
            aug=AugCfg(enable=False),
            return_meta=True,
        )
        ds_test = None
    else:
        # Fallback: parquet + PIL decode (original behaviour)
        if not args.train_dir:
            raise ValueError("Must provide --train_dir if not using --cache_dir")
        log(rank, f"Using CellParquetMultiView (parquet mode)")
        ds_train_full = CellParquetMultiView(
            parquet_path=args.train_dir,
            class_to_idx=class_to_idx,
            label_col=args.label_col,
            size=args.img_size,
            aug=AugCfg(enable=True),
            return_meta=True,
        )
        ds_val_full = CellParquetMultiView(
            parquet_path=args.train_dir,
            class_to_idx=class_to_idx,
            label_col=args.label_col,
            size=args.img_size,
            aug=AugCfg(enable=False),
            return_meta=True,
        )
        test_dir = args.test_dir if args.test_dir else args.val_dir
        if test_dir:
            ds_test = CellParquetMultiView(
                parquet_path=test_dir,
                class_to_idx=class_to_idx,
                label_col=args.label_col,
                size=args.img_size,
                aug=AugCfg(enable=False),
                return_meta=True,
            )
        else:
            ds_test = None

    dataset_len = len(ds_train_full)
    indices = torch.randperm(dataset_len, generator=torch.Generator().manual_seed(args.seed)).numpy().astype(np.int32)
    val_len = int(dataset_len * args.val_frac)
    val_indices = indices[:val_len]
    train_indices = indices[val_len:]

    ds_train = torch.utils.data.Subset(ds_train_full, train_indices)
    ds_val = torch.utils.data.Subset(ds_val_full, val_indices)
    if ds_test is None:
        ds_test = ds_val

    log(rank, f"Train: {len(ds_train):,}  Val: {len(ds_val):,}  "
              f"Test: {len(ds_test):,}")

    # ── Class weights ──
    cw = compute_class_weights(ds_train_full, args.label_col, num_classes, class_to_idx)
    if args.no_class_weight:
        cw = torch.ones(num_classes)
        log(rank, "Class weights: DISABLED (uniform)")
    else:
        log(rank, f"Class weights: {dict(zip(class_to_idx.keys(), cw.numpy().round(3)))}")

    # ── Data loaders with DDP sampler ──
    train_sampler = DistributedSampler(ds_train, shuffle=True, seed=args.seed) if is_ddp else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4, 
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    eff_bs = args.batch_size * world_size
    log(rank, f"Effective batch size: {eff_bs} ({args.batch_size} × {world_size} GPUs)")

    # ── AMP ──
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # ── Model ──
    model = MultiViewClassifier(
        model_name=args.model,
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained=True,
        fuse=args.fuse,
        use_cosine_head=args.use_cosine_head,
        cons_T=args.cons_T,
        verbose=is_main(rank),
    ).to(device)

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # LoRA
    lora_params = []
    if args.use_lora:
        lora_params = apply_lora_to_timm_vit(
            model.backbone,
            last_n_blocks=args.lora_blocks,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
            verbose=is_main(rank),
        )
        model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(rank, f"Params: {n_train:,} trainable / {n_total:,} total "
              f"({100 * n_train / n_total:.2f}%)")

    # Wrap in DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    # Unwrap for param access
    raw_model = model.module if is_ddp else model

    # ── Optimizer ──
    # Split params: ndim>1 gets weight decay, ndim==1 (bias, LN) does not
    # LoRA params handled separately with different lr
    lora_set = set(id(p) for p in lora_params)

    decay_params = []
    no_decay_params = []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad or id(p) in lora_set:
            continue
        if p.ndim == 1:  # bias, LayerNorm weight/bias
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "lr": args.lr_head, "weight_decay": 0.0},
    ]
    if lora_params:
        param_groups.append(
            {"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.0}
        )

    opt = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    cw_device = cw.to(device)

    # ── Resume from periodic checkpoint ──
    start_epoch = 1
    best = -1.0
    log_rows = []
    best_path = out_dir / "best.pt"

    if args.resume and Path(args.resume).exists():
        log(rank, f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt.get("best_ckpt_score", ckpt.get("best_macro_f1", -1.0))
        # Restore log_rows if available
        log_path = out_dir / "train_log.json"
        if log_path.exists():
            log_rows = json.loads(log_path.read_text())
        log(rank, f"  Resumed at epoch {start_epoch}, best_score={best:.4f}")

    # ── Training loop ──
    log(rank, f"\n{'epoch':>5} {'tr_loss':>8} {'tr_cons':>8} {'va_loss':>8} {'va_acc':>7} "
              f"{'va_mF1':>7} {'route':>7} {'cv_cos':>7} {'lr':>10} {'secs':>5}")
    log(rank, "─" * 95)

    # val_loader (with DistributedSampler) is used by ALL ranks for distributed eval

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Set epoch for sampler (ensures different shuffle each epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss, tr_cons = train_one_epoch(
            model, train_loader, opt, scaler, device, amp_dtype, cw_device,
            cons_w=args.cons_weight, epoch=epoch,
        )

        # ---- Validation ----
        if epoch == args.epochs or (epoch % args.eval_every == 0):
            pred_l, yy_l, ii_l, l_sum, n_l, g_sum, g_n, cvcl, cvnl = eval_epoch_local(
                raw_model, val_loader, device, amp_dtype, cw_device
            )
            va_loss, va_acc, mf1, pred_all, yy_all, ii_all, avg_gate, cv_cos = gather_eval(
                pred_l, yy_l, ii_l, l_sum, n_l, g_sum, g_n, cvcl, cvnl,
                num_classes, device, is_ddp, rank
            )

            # Score logic
            if is_main(rank):
                score = args.ckpt_score_w * mf1 + (1 - args.ckpt_score_w) * va_acc

                secs = time.time() - t0
                m_row = {
                    "epoch": epoch, "tr_loss": round(tr_loss, 4),
                    "tr_cons": round(tr_cons, 4), "va_loss": round(va_loss, 4),
                    "va_acc": round(va_acc, 4), "va_mF1": round(mf1, 4),
                    "score": round(score, 4), "secs": round(secs, 1)
                }
                if cv_cos is not None:
                    m_row["cv_cos"] = round(cv_cos, 4)

                g_str = ""
                if avg_gate is not None:
                    g_str = f"{avg_gate[0]:.2f}/{avg_gate[1]:.2f}"
                    m_row["gate_2p5x"], m_row["gate_10x"] = avg_gate

                # Fine routing accuracy
                fra, _ = per_fine_routing_accuracy(
                    pred_all, ds_val, class_to_idx,
                    fine_col=args.fine_col if hasattr(args, "fine_col") else "label",
                    coarse_col=args.label_col,
                    ii_all=ii_all,
                    fine_to_coarse_dict=fine_to_coarse_dict
                )
                if fra and len(fra) > 0:
                    bot_3 = " ".join([f"{r['fine_class']}:{r['acc']:.2f}" for r in fra[:3]])
                    # Add worst fine acc to score? (Optional)
                    # score = 0.5*mf1 + 0.5*fra[0]['acc']
                else:
                    bot_3 = ""

                log_rows.append(m_row)
                cur_lr = scheduler.get_last_lr()[0]
                log(rank, f"{epoch:5d} {tr_loss:8.4f} {tr_cons:8.4f} {va_loss:8.4f} "
                          f"{va_acc:7.4f} {mf1:7.4f} {g_str:>7} "
                          f"{cv_cos if cv_cos else 0.0:7.4f} {cur_lr:10.2e} {secs:5.1f}")
                if bot_3:
                    log(rank, f"      Bot3 fine acc: {bot_3}")

                # ── Per-coarse class metrics ──
                ep_rows = per_class_metrics(pred_all, yy_all, num_classes, idx_to_class)
                log(rank, f"      {'Class':<22} {'N':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
                log(rank, f"      {'─' * 52}")
                for r in sorted(ep_rows, key=lambda x: -x["f1"]):
                    log(rank, f"      {r['class']:<22} {r['n']:>8,} "
                              f"{r['prec']:>7.3f} {r['rec']:>7.3f} {r['f1']:>7.3f}")
                if use_wandb:
                    ep_tbl = wandb.Table(columns=["class", "n", "prec", "rec", "f1"])
                    for r in ep_rows:
                        ep_tbl.add_data(r["class"], r["n"], r["prec"], r["rec"], r["f1"])
                    wandb.log({"per_class_epoch": ep_tbl}, step=epoch)

                # Save best
                if score > best:
                    best = score
                    torch.save({
                        "model": raw_model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "best_ckpt_score": best,
                        "val_acc": va_acc,
                        "macro_f1": mf1,
                        "args": vars(args)
                    }, best_path)
                    log(rank, f"      [New best score: {best:.4f}]")

                # WandB
                if use_wandb:
                    wandb.log(m_row, step=epoch)

        # ── Periodic checkpoint every 5 epochs (outside do_eval) ──
        if is_main(rank) and (epoch % 5 == 0):
            ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_ckpt_score": best,
                "args": vars(args),
                "class_to_idx": class_to_idx,
            }, ckpt_path)
            print(f"  → periodic {ckpt_path}", flush=True)

        scheduler.step()

    # ── Tear down DDP before final eval ──────────────────────────────────────
    # Non-rank-0 processes exit here; rank 0 runs final eval as a single
    # process, no need for NCCL after this point.
    cleanup_ddp(is_ddp)
    if not is_main(rank):
        return

    # ── Final per-class report (rank 0 only) ─────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"Best checkpoint score: {best:.4f}")
    print(f"{'═' * 60}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])

    test_label = "test" if (ds_test is not ds_val) else "val"
    print(f"Final eval on {test_label} set ({len(ds_test):,} cells)")

    test_loader_full = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    pred, yy, ii, _ls, _n, _gs, _gn, _cc, _cn = eval_epoch_local(
        raw_model, test_loader_full, device, amp_dtype, cw_device,
    )

    # ── Per-coarse class metrics ──
    rows = per_class_metrics(pred, yy, num_classes, idx_to_class)
    print(f"\n{'Class':<22} {'N':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("─" * 54)
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(f"{r['class']:<22} {r['n']:>8,} {r['prec']:>7.3f} "
              f"{r['rec']:>7.3f} {r['f1']:>7.3f}")

    # ── Confusion matrix ──
    cm = confusion_matrix_np(pred, yy, num_classes)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    header = f"{'':>20}" + "".join(f"{c[:8]:>10}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>20}" + "".join(f"{cm[i, j]:>10,}" for j in range(num_classes))
        print(row_str)

    cm_json = {"class_names": class_names,
               "matrix": cm.tolist()}

    # ── Per-fine routing accuracy ──
    fine_rows, fine_cm = per_fine_routing_accuracy(
        pred, ds_test, class_to_idx,
        fine_col="label", coarse_col=args.label_col,
        ii_all=ii,
        fine_to_coarse_dict=fine_to_coarse_dict
    )

    if fine_rows:
        print(f"\n{'Fine Class':<28} {'Coarse':>20} {'N':>8} {'Acc':>7}")
        print("─" * 67)
        for r in fine_rows:
            print(f"{r['fine_class']:<28} {r['coarse_class']:>20} "
                  f"{r['n']:>8,} {r['acc']:>7.3f}")
        avg_fine_acc = np.mean([r["acc"] for r in fine_rows])
        print(f"\nMean per-fine routing accuracy: {avg_fine_acc:.4f}")

        print(f"\nFine-to-Coarse Confusion Matrix (rows=true fine, cols=pred coarse):")
        header = f"{'':>28}" + "".join(f"{c[:8]:>10}" for c in class_names)
        print(header)
        for fine_name in sorted(fine_cm.keys()):
            row_str = f"{fine_name:>28}" + "".join(f"{fine_cm[fine_name][cname]:>10,}" for cname in class_names)
            print(row_str)

    # ── Save files ──
    (out_dir / "train_log.json").write_text(json.dumps(log_rows, indent=2))
    (out_dir / "per_class.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm_json, indent=2))
    if fine_rows:
        (out_dir / "per_fine.json").write_text(json.dumps(fine_rows, indent=2))
        (out_dir / "fine_confusion_matrix.json").write_text(json.dumps(fine_cm, indent=2))

    # ── W&B logging ──
    if use_wandb:
        wandb.summary["best_ckpt_score"] = best
        wandb.summary["best_macro_f1"] = ckpt.get("best_macro_f1", best)
        wandb.summary["best_route_acc"] = ckpt.get("best_route_acc", 0.0)

        # Per-coarse table
        tbl = wandb.Table(columns=["class", "n", "prec", "rec", "f1"])
        for r in rows:
            tbl.add_data(r["class"], r["n"], r["prec"], r["rec"], r["f1"])
        wandb.log({"per_class": tbl})

        # Confusion matrix heatmap
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=yy.numpy(),
            preds=pred.numpy(),
            class_names=class_names,
        )})

        # Per-fine table
        if fine_rows:
            fine_tbl = wandb.Table(
                columns=["fine_class", "coarse_class", "n", "correct", "acc"])
            for r in fine_rows:
                fine_tbl.add_data(
                    r["fine_class"], r["coarse_class"],
                    r["n"], r["correct"], r["acc"])
            wandb.log({"per_fine_routing": fine_tbl})
            wandb.summary["mean_fine_routing_acc"] = avg_fine_acc

            # Create detailed heatmap array for wandb
            # Rows are fine classes (alphabetical), Cols are coarse targets
            fine_names_sorted = sorted(fine_cm.keys())
            heatmap_data = []
            for fine_name in fine_names_sorted:
                heatmap_data.append([fine_cm[fine_name][c] for c in class_names])
            wandb.log({
                "fine_confusion_matrix": wandb.plots.HeatMap(
                    x_labels=class_names,
                    y_labels=fine_names_sorted,
                    matrix_values=heatmap_data,
                    show_text=False
                )
            })

        wandb.finish()

    print(f"\nDone. Checkpoint: {best_path}")



if __name__ == "__main__":
    main()