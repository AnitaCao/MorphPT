#!/usr/bin/env python3
"""
Single-View Baseline Training
=============================

Trains a single-view model (ResNet, Swin, DINOv2/v3) on either the 2.5x or 10x view
for ablation studies against the Multi-view MoE architecture.

Usage:
  torchrun --nproc_per_node=4 trainer/train_single_view.py \
    --train_dir prepared/splits/router_shards \
    --class_map prepared/splits/fine_to_coarse.json \
    --view 10x \
    --model resnet50 \
    --out_dir experiments/ablation_resnet50_10x
"""

import argparse
import json
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

from models.lora import apply_lora_to_timm_vit, apply_lora_to_timm_swin, apply_lora_to_timm_convnext
from data.dataset import CellParquetSingleView, AugCfg, MemmapDataset

from trainer_base import (
    setup_ddp, cleanup_ddp, is_main, log,
    compute_class_weights, macro_f1, per_class_metrics,
    confusion_matrix_np, get_base_parser
)

# Enable TF32 for faster matmul on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SingleViewClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, img_size: int, pretrained: bool = True):
        super().__init__()
        
        create_kwargs = dict(pretrained=pretrained, num_classes=num_classes)
        try:
            self.backbone = timm.create_model(model_name, **create_kwargs, img_size=img_size)
        except TypeError:
            self.backbone = timm.create_model(model_name, **create_kwargs)
            
    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        return logits


@torch.no_grad()
def eval_epoch_single(model, loader, device, amp_dtype, cw_device):
    model.eval()
    loss_sum = 0.0
    n = 0
    all_pred = []
    all_y = []

    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for batch in loader:
            if len(batch) == 3 and isinstance(batch[2], dict) and "index" in batch[2]:
                x, y, _meta = batch
            else:
                x, y, _tissue = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=cw_device)

            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            all_pred.append(logits.argmax(1).detach().cpu())
            all_y.append(y.detach().cpu())

    pred = torch.cat(all_pred) if all_pred else torch.empty(0, dtype=torch.long)
    yy = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)
    return pred, yy, loss_sum, n


def gather_eval_single(pred, yy, loss_sum, n_local, num_classes, device, is_ddp, rank):
    if is_ddp:
        def _gather_tensor(t):
            t = t.to(device)
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
            dist.all_gather(sizes, torch.tensor([t.shape[0]], device=device))
            max_sz = max(s.item() for s in sizes)
            padded = torch.zeros(max_sz, dtype=t.dtype, device=device)
            padded[:t.shape[0]] = t
            bufs = [torch.zeros(max_sz, dtype=t.dtype, device=device) for _ in range(dist.get_world_size())]
            dist.all_gather(bufs, padded)
            return torch.cat([b[:s.item()] for b, s in zip(bufs, sizes)]).cpu()

        pred_all = _gather_tensor(pred)
        yy_all = _gather_tensor(yy)

        t_loss = torch.tensor([loss_sum], device=device)
        t_n = torch.tensor([n_local], device=device, dtype=torch.float)
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_n, op=dist.ReduceOp.SUM)
        loss_sum_g = t_loss.item()
        n_g = int(t_n.item())
    else:
        pred_all = pred
        yy_all = yy
        loss_sum_g = loss_sum
        n_g = n_local

    if rank == 0:
        acc = (pred_all == yy_all).float().mean().item() if yy_all.numel() else 0.0
        mf1 = macro_f1(pred_all, yy_all, num_classes)
        va_loss = loss_sum_g / max(1, n_g)
        return va_loss, acc, mf1, pred_all, yy_all
    return None, None, None, None, None


def main():
    ap = get_base_parser()
    
    # Disable required flag for --fuse since it's unused in single view
    for action in ap._actions:
        if action.dest == "fuse":
            action.required = False
            
    ap.set_defaults(
        train_split="router_shards",
        lr_head=3e-4, lr_lora=1e-4, epochs=30, fuse="none" # Unused, just to satisfy base parser
    )
    ap.add_argument("--view", type=str, required=True, choices=["2p5x", "10x"],
                    help="Which view to train on: '2p5x' or '10x'")
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone weights and only train the head")

    args = ap.parse_args()

    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    if is_ddp: dist.barrier()

    with open(args.class_map) as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    log(rank, f"Classes ({num_classes}): {class_to_idx}")
    log(rank, f"Model: {args.model} | View: {args.view} | World size: {world_size}")

    # ── W&B ──
    use_wandb = HAS_WANDB and args.wandb_project and is_main(rank)
    if use_wandb:
        run_name = f"sv_{args.model}_{args.view}"
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=run_name, config=vars(args), dir=str(out_dir),
        )

    # ── Datasets ──
    view_map_memmap = {"2p5x": "a", "10x": "b"}
    
    if args.cache_dir:
        log(rank, f"Using MemmapDataset from {args.cache_dir} for {args.view}")
        ds_train_full = MemmapDataset(
            cache_dir=args.cache_dir, split_name=args.train_split,
            view=view_map_memmap[args.view], aug=AugCfg(enable=True),
            class_to_idx=class_to_idx, label_col=args.label_col
        )
        ds_val_full = MemmapDataset(
            cache_dir=args.cache_dir, split_name=args.train_split,
            view=view_map_memmap[args.view], aug=AugCfg(enable=False),
            class_to_idx=class_to_idx, label_col=args.label_col
        )
    else:
        if not args.train_dir: raise ValueError("Must provide --train_dir or --cache_dir")
        log(rank, f"Using CellParquetSingleView from {args.train_dir} for {args.view}")
        ds_train_full = CellParquetSingleView(
            parquet_path=args.train_dir, class_to_idx=class_to_idx,
            img_col=f"img_path_{args.view}", label_col=args.label_col,
            size=args.img_size, aug=AugCfg(enable=True)
        )
        ds_val_full = CellParquetSingleView(
            parquet_path=args.train_dir, class_to_idx=class_to_idx,
            img_col=f"img_path_{args.view}", label_col=args.label_col,
            size=args.img_size, aug=AugCfg(enable=False)
        )

    # ── Splits ──
    dataset_len = len(ds_train_full)
    indices = torch.randperm(dataset_len, generator=torch.Generator().manual_seed(args.seed)).numpy()
    val_len = int(dataset_len * 0.05)
    
    ds_train = Subset(ds_train_full, indices[val_len:])
    ds_val = Subset(ds_val_full, indices[:val_len])
    log(rank, f"Train: {len(ds_train):,}  Val: {len(ds_val):,}")

    def print_distribution(ds, name):
        if is_main(rank):
            counts = torch.zeros(num_classes)
            if isinstance(ds, MemmapDataset):
                y = ds._y
                for idx in range(num_classes):
                    counts[idx] = int((y == idx).sum())
            elif isinstance(ds, Subset):
                # When ds is a Subset of a MemmapDataset
                if isinstance(ds.dataset, MemmapDataset):
                    y = ds.dataset._y[ds.indices]
                    for idx in range(num_classes):
                        counts[idx] = int((y == idx).sum())
                else:
                    # When ds is a Subset of a Parquet Dataset
                    labels = ds.dataset._labels[ds.indices]
                    for label, idx in class_to_idx.items():
                        counts[idx] = (labels == label).sum()
            else:
                for label, idx in class_to_idx.items():
                    counts[idx] = (ds.df[args.label_col] == label).sum()
            
            log(rank, "-" * 50)
            log(rank, f"Dataset Label Distribution for '{name}' (col='{args.label_col}'):")
            for idx in range(num_classes):
                cname = idx_to_class[idx]
                log(rank, f"  {cname:<25}: {int(counts[idx]):,}")
            log(rank, "-" * 50)

    # Print data distribution for verification
    print_distribution(ds_train_full, "Train Full Dataset (Before Split)")
    print_distribution(ds_train, "Train Split")
    print_distribution(ds_val, "Val Split")

    cw = compute_class_weights(ds_train_full, args.label_col, num_classes, class_to_idx)
    if args.no_class_weight: cw = torch.ones(num_classes)
    cw_device = cw.to(device)

    # ── Model ──
    model = SingleViewClassifier(args.model, num_classes, args.img_size)

    # Freeze strategy
    if args.freeze_backbone:
        for name, p in model.backbone.named_parameters():
            if "head" not in name and "fc" not in name:
                p.requires_grad = False
        log(rank, "Frozen backbone. Only training head.")

    model.to(device)

    # Apply LoRA if requested
    lora_params = []
    if args.use_lora and ("vit" in args.model or "swin" in args.model or "convnext" in args.model) and not args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        # Only training the classifier head by default alongside LoRA
        if hasattr(model.backbone, 'head'):
            for p in model.backbone.head.parameters():
                p.requires_grad = True

        if "swin" in args.model:
            lora_params = apply_lora_to_timm_swin(
                model.backbone, last_n_blocks=args.lora_blocks, r=args.lora_rank, 
                alpha=args.lora_alpha, dropout=args.lora_dropout, 
                targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
                verbose=is_main(rank),
            )
        elif "convnext" in args.model:
            lora_params = apply_lora_to_timm_convnext(
                model.backbone, last_n_blocks=args.lora_blocks, r=args.lora_rank, 
                alpha=args.lora_alpha, dropout=args.lora_dropout, 
                targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
                verbose=is_main(rank),
            )
        else:
            lora_params = apply_lora_to_timm_vit(
                model.backbone, last_n_blocks=args.lora_blocks, r=args.lora_rank, 
                alpha=args.lora_alpha, dropout=args.lora_dropout, 
                targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
                verbose=is_main(rank),
            )
        model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(rank, f"Params: {n_train:,} trainable / {n_total:,} total ({100 * n_train / n_total:.2f}%)")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if is_ddp else model

    # ── Optimizer ──
    lora_set = set(id(p) for p in lora_params)
    decay_params, no_decay_params = [], []

    for name, p in raw_model.named_parameters():
        if not p.requires_grad or id(p) in lora_set: continue
        if p.ndim == 1: no_decay_params.append(p)
        else: decay_params.append(p)

    param_groups = [
        {"params": decay_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "lr": args.lr_head, "weight_decay": 0.0},
    ]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.0})

    opt = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    train_sampler = DistributedSampler(ds_train, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.workers, pin_memory=True
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # ── Resume from periodic checkpoint ──
    start_epoch = 1
    best = -1.0
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
        log(rank, f"  Resumed at epoch {start_epoch}, best_score={best:.4f}")

    log(rank, f"\n{'epoch':>5} {'tr_loss':>8} {'va_loss':>8} {'va_acc':>7} {'va_mF1':>7} {'lr':>10} {'secs':>5}")
    log(rank, "─" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if train_sampler: train_sampler.set_epoch(epoch)
        
        # Train Step
        model.train()
        loss_sum = 0.0
        n_tr = 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=cw_device)
                
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item()) * y.size(0)
            n_tr += y.size(0)
            
        tr_loss = loss_sum / max(1, n_tr)

        # Eval Step
        if epoch == args.epochs or (epoch % args.eval_every == 0):
            pred_l, yy_l, l_sum, n_l = eval_epoch_single(raw_model, val_loader, device, amp_dtype, cw_device)
            va_loss, va_acc, mf1, _p, _y = gather_eval_single(pred_l, yy_l, l_sum, n_l, num_classes, device, is_ddp, rank)

            if is_main(rank):
                score = args.ckpt_score_w * mf1 + (1 - args.ckpt_score_w) * va_acc
                secs = time.time() - t0
                cur_lr = scheduler.get_last_lr()[0]
                log(rank, f"{epoch:5d} {tr_loss:8.4f} {va_loss:8.4f} {va_acc:7.4f} {mf1:7.4f} {cur_lr:10.2e} {secs:5.1f}")

                if score > best:
                    best = score
                    torch.save({
                        "model": raw_model.state_dict(),
                        "epoch": epoch, "best_ckpt_score": best,
                        "args": vars(args), "class_to_idx": class_to_idx,
                    }, best_path)
                    log(rank, f"      [New best score: {best:.4f}  (Score = {args.ckpt_score_w:.2f}*mF1 + {(1-args.ckpt_score_w):.2f}*Acc)]")

                # Print per-class metrics every 5 epochs
                if epoch % 5 == 0:
                    ep_rows = per_class_metrics(_p, _y, num_classes, idx_to_class)
                    log(rank, f"\n      {'Class':<25} {'N':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
                    log(rank, f"      {'─' * 55}")
                    for r in sorted(ep_rows, key=lambda x: -x["f1"]):
                        log(rank, f"      {r['class']:<25} {r['n']:>8,} "
                                  f"{r['prec']:>7.3f} {r['rec']:>7.3f} {r['f1']:>7.3f}")
                    log(rank, "")

                if use_wandb:
                    wandb.log({"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss, 
                               "va_acc": va_acc, "va_mF1": mf1, "score": score}, step=epoch)

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
    print(f"Single View: {args.model} ({args.view})")
    print(f"Best score: {best:.4f}  (Formula: {args.ckpt_score_w:.2f}*mF1 + {(1-args.ckpt_score_w):.2f}*Acc)")
    print(f"{'═' * 60}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])

    if args.test_dir and Path(args.test_dir).exists():
        print(f"Final eval on TEST set: {args.test_dir}")
        ds_test = CellParquetSingleView(
            parquet_path=args.test_dir, class_to_idx=class_to_idx,
            img_col=f"img_path_{args.view}", label_col=args.label_col,
            size=args.img_size, aug=AugCfg(enable=False)
        )
        print_distribution(ds_test, "Test Split (Final Eval)")
        val_loader_full = DataLoader(
            ds_test, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    else:
        print(f"Final eval on VAL set ({len(ds_val):,} cells)")
        print_distribution(ds_val, "Val Split (Final Eval)")
        val_loader_full = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    
    pred, yy, _l_sum, _n = eval_epoch_single(
        raw_model, val_loader_full, device, amp_dtype, cw_device
    )

    # Per-class metrics
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

    log(rank, f"\nDone. Best Model Saved: {best_path} (Score = {args.ckpt_score_w:.2f}*mF1 + {(1-args.ckpt_score_w):.2f}*Acc)")

if __name__ == "__main__":
    main()
