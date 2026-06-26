#!/usr/bin/env python3
"""
Expert training for CellPT MoE — DDP
======================================
Fine-grained classifier for a single coarse group (e.g. Cancer, Lymphoid).

Two initialization strategies:
  Strategy B (default): --router_ckpt best.pt
    Inherits backbone + LoRA + gate from router checkpoint.
    Head re-initialized for N fine classes.
    LoRA lr kept low to preserve domain adaptation.

  Strategy A (ablation): --router_ckpt best.pt --no_inherit_lora
    Inherits backbone + gate only. Fresh LoRA from scratch.

  No checkpoint: omit --router_ckpt
    Everything from frozen DINOv2 pretrained weights.

Launch (multi-GPU):
  torchrun --nproc_per_node=2 trainer/train_expert.py \
    --router_ckpt experiments/router_best/best.pt \
    --train_dir prepared/splits_v3_seed1337/expert_Cancer/shards_capped \
    --class_map prepared/splits_v3_seed1337/expert_Cancer/class_to_idx.json \
    --expert_name Cancer \
    --label_col label \
    --fuse gate \
    --val_frac 0.15 \
    --out_dir experiments/expert_Cancer

Launch (single GPU):
  python trainer/train_expert.py ...
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
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import timm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.lora import apply_lora_to_timm_vit
from models.model import MultiViewClassifier
from data.dataset import CellParquetMultiView, AugCfg, MemmapDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trainer_base import (
    setup_ddp, cleanup_ddp, is_main, log,
    compute_class_weights, macro_f1, per_class_metrics,
    confusion_matrix_np, eval_epoch_local, gather_eval,
    get_base_parser
)


# ─────────────────────────────────────────────
# Checkpoint Loading
# ─────────────────────────────────────────────

def load_router_checkpoint(model, ckpt_path, inherit_lora, device, rank=0):
    """Load router checkpoint into expert model.

    inherit_lora=True  (Strategy B): transfer backbone + LoRA + gate
    inherit_lora=False (Strategy A): transfer backbone + gate only, fresh LoRA
    Always skips: classification head (dimension mismatch)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    router_state = ckpt["model"]

    filtered = {}
    skipped = []
    for k, v in router_state.items():
        # Strip DDP module prefix if present
        k_stripped = k.replace("module.", "")
        
        # Always skip head (expert has different num_classes)
        # MultiViewClassifier could have 'head.' and 'head_b.'
        if k_stripped.startswith("head.") or k_stripped.startswith("head_b.") or "cosine_scale" in k_stripped:
            skipped.append(k_stripped)
            continue
        # Optionally skip LoRA (Strategy A)
        if not inherit_lora and "lora_" in k_stripped:
            skipped.append(k_stripped)
            continue
        filtered[k_stripped] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    strategy = "B (inherit LoRA + gate)" if inherit_lora else "A (fresh LoRA, inherit gate)"
    log(rank, f"Router checkpoint loaded — Strategy {strategy}")
    log(rank, f"  From: {ckpt_path}")
    log(rank, f"  Transferred: {len(filtered)} params")
    log(rank, f"  Skipped: {len(skipped)} keys ({', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})")
    log(rank, f"  Missing (new head): {missing}")
    if unexpected:
        log(rank, f"  Unexpected: {unexpected}")

    router_info = {
        "epoch": ckpt.get("epoch", "?"),
        "macro_f1": ckpt.get("macro_f1", "?"),
        "best_ckpt_score": ckpt.get("best_ckpt_score", "?"),
    }
    log(rank, f"  Router trained {router_info['epoch']} epochs, "
              f"score={router_info['best_ckpt_score']}")

    return router_info


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, opt, scaler, device, amp_dtype,
                    class_weight, epoch=1):
    model.train()
    loss_sum = 0.0
    n = 0

    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=torch.cuda.is_available()):
            logits, _emb, aux = model(x)
            if aux.get("is_log_prob"):
                loss = F.nll_loss(logits, y, weight=class_weight)
            else:
                loss = F.cross_entropy(logits, y, weight=class_weight)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        loss_sum += float(loss.item()) * y.size(0)
        n += y.size(0)

    return loss_sum / max(1, n)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = get_base_parser()

    # Expert-specific defaults
    ap.set_defaults(
        train_split="expert_shards",
        wandb_project="cellpt-expert",
        fuse="gate",
        lr_lora=3e-5,      # lower than router to preserve domain adaptation
        lr_head=3e-4,
        epochs=30,
    )
    ap.add_argument("--router_ckpt", type=str, default=None,
                    help="Path to router best.pt. Omit for training from scratch.")
    ap.add_argument("--no_inherit_lora", action="store_true",
                    help="Strategy A: do NOT inherit router LoRA, fresh LoRA from scratch.")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="Fraction of training data for validation")
    ap.add_argument("--expert_name", type=str, default="",
                    help="Expert group name for logging (e.g. Cancer, Lymphoid)")

    args = ap.parse_args()

    # ── DDP setup ──
    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Override architecture args from router ckpt ──
    if args.router_ckpt and Path(args.router_ckpt).exists():
        # Load lightly on CPU just to read args
        ckpt_meta = torch.load(args.router_ckpt, map_location="cpu", weights_only=False)
        r_args = ckpt_meta.get("args", {})
        if r_args:
            log(rank, f"Auto-syncing architecture args from {args.router_ckpt}")
            for k in ["model", "fuse", "use_lora", "lora_blocks", "lora_targets", "lora_rank", "lora_alpha", "lora_dropout"]:
                if k in r_args:
                    old_val = getattr(args, k, None)
                    new_val = r_args[k]
                    if old_val != new_val:
                        log(rank, f"  Overriding --{k}: {old_val} -> {new_val}")
                    setattr(args, k, new_val)
        del ckpt_meta
        if is_ddp:
            dist.barrier()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    if is_ddp:
        dist.barrier()

    # ── Class map ──
    with open(args.class_map) as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    expert_name = args.expert_name or f"expert_{num_classes}cls"
    log(rank, f"Expert: {expert_name}")
    log(rank, f"Classes ({num_classes}): {class_to_idx}")
    log(rank, f"Fuse: {args.fuse}  |  World size: {world_size}")

    # ── W&B ──
    use_wandb = HAS_WANDB and args.wandb_project and is_main(rank)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{expert_name}_{args.fuse}",
            config=vars(args),
            dir=str(out_dir),
        )
        log(rank, f"W&B run: {wandb.run.url}")

    # ── Datasets ──
    if args.cache_dir:
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
    else:
        if not args.train_dir:
            raise ValueError("Must provide --train_dir or --cache_dir")
        log(rank, f"Using CellParquetMultiView from {args.train_dir}")
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

    # ── Train/Val split ──
    dataset_len = len(ds_train_full)
    indices = torch.randperm(dataset_len,
                             generator=torch.Generator().manual_seed(args.seed)
                             ).numpy().astype(np.int32)
    val_len = int(dataset_len * args.val_frac)
    val_indices = indices[:val_len]
    train_indices = indices[val_len:]

    ds_train = Subset(ds_train_full, train_indices)
    ds_val = Subset(ds_val_full, val_indices)

    log(rank, f"Train: {len(ds_train):,}  Val: {len(ds_val):,}")

    # ── Class weights ──
    cw = compute_class_weights(ds_train_full, args.label_col, num_classes, class_to_idx)
    if args.no_class_weight:
        cw = torch.ones(num_classes)
        log(rank, "Class weights: DISABLED (uniform)")
    else:
        log(rank, f"Class weights: {dict(zip(class_to_idx.keys(), cw.numpy().round(3)))}")

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

    # Apply LoRA (must match router config for Strategy B compatibility)
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

    # ── Load router checkpoint ──
    router_info = None
    inherit_lora = not args.no_inherit_lora
    if args.router_ckpt and Path(args.router_ckpt).exists():
        router_info = load_router_checkpoint(
            model, args.router_ckpt, inherit_lora, device, rank
        )
    else:
        log(rank, "No router checkpoint — training from scratch.")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(rank, f"Params: {n_train:,} trainable / {n_total:,} total "
              f"({100 * n_train / n_total:.2f}%)")

    # Wrap in DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    raw_model = model.module if is_ddp else model

    # ── Optimizer ──
    lora_set = set(id(p) for p in lora_params)

    decay_params = []
    no_decay_params = []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad or id(p) in lora_set:
            continue
        if p.ndim == 1:
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

    # ── Data loaders ──
    train_sampler = DistributedSampler(ds_train, shuffle=True, seed=args.seed) if is_ddp else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    eff_bs = args.batch_size * world_size
    log(rank, f"Effective batch size: {eff_bs} ({args.batch_size} × {world_size} GPUs)")

    # ── AMP ──
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

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
        best = ckpt.get("best_ckpt_score", -1.0)
        log_path = out_dir / "train_log.json"
        if log_path.exists():
            log_rows = json.loads(log_path.read_text())
        log(rank, f"  Resumed at epoch {start_epoch}, best_score={best:.4f}")

    # ── Training loop ──
    log(rank, f"\n{'epoch':>5} {'tr_loss':>8} {'va_loss':>8} {'va_acc':>7} "
              f"{'va_mF1':>7} {'gate':>7} {'lr':>10} {'secs':>5}")
    log(rank, "─" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss = train_one_epoch(
            model, train_loader, opt, scaler, device, amp_dtype,
            cw_device, epoch=epoch,
        )

        # ── Validation ──
        if epoch == args.epochs or (epoch % args.eval_every == 0):
            pred_l, yy_l, ii_l, l_sum, n_l, g_sum, g_n, cvcl, cvnl = eval_epoch_local(
                raw_model, val_loader, device, amp_dtype, cw_device
            )
            va_loss, va_acc, mf1, pred_all, yy_all, ii_all, avg_gate, cv_cos = gather_eval(
                pred_l, yy_l, ii_l, l_sum, n_l, g_sum, g_n, cvcl, cvnl,
                num_classes, device, is_ddp, rank
            )

            if is_main(rank):
                score = args.ckpt_score_w * mf1 + (1 - args.ckpt_score_w) * va_acc

                secs = time.time() - t0
                m_row = {
                    "epoch": epoch, "tr_loss": round(tr_loss, 4),
                    "va_loss": round(va_loss, 4),
                    "va_acc": round(va_acc, 4), "va_mF1": round(mf1, 4),
                    "score": round(score, 4), "secs": round(secs, 1),
                }

                g_str = ""
                if avg_gate is not None:
                    g_str = f"{avg_gate[0]:.2f}/{avg_gate[1]:.2f}"
                    m_row["gate_2p5x"], m_row["gate_10x"] = avg_gate

                log_rows.append(m_row)
                cur_lr = scheduler.get_last_lr()[0]
                log(rank, f"{epoch:5d} {tr_loss:8.4f} {va_loss:8.4f} "
                          f"{va_acc:7.4f} {mf1:7.4f} {g_str:>7} "
                          f"{cur_lr:10.2e} {secs:5.1f}")

                # Per-class metrics
                ep_rows = per_class_metrics(pred_all, yy_all, num_classes, idx_to_class)
                log(rank, f"      {'Class':<28} {'N':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
                log(rank, f"      {'─' * 58}")
                for r in sorted(ep_rows, key=lambda x: -x["f1"]):
                    log(rank, f"      {r['class']:<28} {r['n']:>8,} "
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
                        "args": vars(args),
                        "class_to_idx": class_to_idx,
                        "router_ckpt": args.router_ckpt,
                    }, best_path)
                    log(rank, f"      [New best score: {best:.4f}]")

                if use_wandb:
                    wandb.log(m_row, step=epoch)

        # Periodic checkpoint
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

    # ── Tear down DDP, final eval on rank 0 ──
    cleanup_ddp(is_ddp)
    if not is_main(rank):
        return

    # ── Final report ──
    print(f"\n{'═' * 60}")
    print(f"Expert: {expert_name}  |  Best score: {best:.4f}")
    print(f"{'═' * 60}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])

    print(f"Final eval on val set ({len(ds_val):,} cells)")

    val_loader_full = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    pred, yy, ii, _ls, _n, _gs, _gn, _cc, _cn = eval_epoch_local(
        raw_model, val_loader_full, device, amp_dtype, cw_device,
    )

    # Per-class
    rows = per_class_metrics(pred, yy, num_classes, idx_to_class)
    print(f"\n{'Class':<28} {'N':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("─" * 60)
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(f"{r['class']:<28} {r['n']:>8,} {r['prec']:>7.3f} "
              f"{r['rec']:>7.3f} {r['f1']:>7.3f}")

    # Confusion matrix
    cm = confusion_matrix_np(pred, yy, num_classes)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    header = f"{'':>28}" + "".join(f"{c[:10]:>12}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>28}" + "".join(f"{cm[i, j]:>12,}" for j in range(num_classes))
        print(row_str)

    cm_json = {"class_names": class_names, "matrix": cm.tolist()}

    # Save
    (out_dir / "train_log.json").write_text(json.dumps(log_rows, indent=2))
    (out_dir / "per_class.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm_json, indent=2))

    if use_wandb:
        wandb.summary["best_ckpt_score"] = best
        wandb.summary["best_macro_f1"] = ckpt.get("macro_f1", best)

        tbl = wandb.Table(columns=["class", "n", "prec", "rec", "f1"])
        for r in rows:
            tbl.add_data(r["class"], r["n"], r["prec"], r["rec"], r["f1"])
        wandb.log({"per_class": tbl})

        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=yy.numpy(),
            preds=pred.numpy(),
            class_names=class_names,
        )})

        wandb.finish()

    print(f"\nDone. Checkpoint: {best_path}")


if __name__ == "__main__":
    main()