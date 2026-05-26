#!/usr/bin/env python3
"""
Evaluation Script for CellPT MoE Router
=======================================

Loads a trained checkpoint and evaluates it on the test/val set.

Launch (single node, multi-GPU):
  torchrun --nproc_per_node=4 eval.py --fuse avg --resume /path/to/best.pt ...

Launch (single GPU, no DDP):
  python eval.py --fuse avg --resume /path/to/best.pt ...
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

from models.lora import apply_lora_to_timm_vit
from data.dataset import CellParquetMultiView, AugCfg, MemmapDataset

# Enable TF32 for faster matmul on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from trainer_base import (
    setup_ddp, cleanup_ddp, is_main, log,
    compute_class_weights, per_class_metrics,
    confusion_matrix_np, eval_epoch_local, gather_eval,
    get_base_parser
)
from models.model import MultiViewClassifier
from train_router import per_fine_routing_accuracy


# ─────────────────────────────────────────────
# Main Evaluation Target
# ─────────────────────────────────────────────

def main():
    ap = get_base_parser()
    
    # Require --resume for eval
    ap.add_argument("--eval_output", type=str, default="", 
                    help="Optional JSON file path to dump evaluation metrics.")
    
    # Override defaults for eval script context
    ap.set_defaults(
        train_split="router_shards",
        val_split="val_balanced",
        test_split="test_shards",
        wandb_project="" # Disable wandb by default for eval
    )
    
    args = ap.parse_args()
    
    if not args.resume or not Path(args.resume).exists():
        raise ValueError(f"--resume must be provided and valid for eval.py, got '{args.resume}'")

    # ── DDP setup ──
    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.resume).parent
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    if is_ddp:
        dist.barrier()

    # ── Class map ──
    with open(args.class_map) as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    log(rank, f"Classes ({num_classes}): {class_to_idx}")
    log(rank, f"Fuse: {args.fuse}  |  World size: {world_size}")
    log(rank, f"Evaluating checkpoint: {args.resume}")

    # ── Datasets ──
    if args.cache_dir:
        log(rank, f"Using MemmapDataset from {args.cache_dir}")
        ds_test_split = args.test_split if args.test_split else args.val_split
        ds_test = MemmapDataset(
            cache_dir=args.cache_dir,
            split_name=ds_test_split,
            view="both",
            return_meta=True,
        )
    else:
        log(rank, f"Using CellParquetMultiView (parquet mode)")
        test_dir = args.test_dir if args.test_dir else args.val_dir
        if not test_dir:
            raise ValueError("Must provide --test_dir, --val_dir, or --cache_dir for evaluation.")
            
        ds_test = CellParquetMultiView(
            parquet_path=test_dir,
            class_to_idx=class_to_idx,
            label_col=args.label_col,
            size=args.img_size,
            aug=AugCfg(enable=False),
            return_meta=True,
        )

    log(rank, f"Evaluation Set: {len(ds_test):,} cells")

    # ── Data loader with DDP sampler ──
    test_sampler = DistributedSampler(ds_test, shuffle=False) if is_ddp else None
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    # ── AMP ──
    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)

    # ── Dummy class weights (to satisfy loss func, though we only care about metrics during eval) ──
    cw_device = torch.ones(num_classes, device=device)

    # ── Load checkpoint early to get structure args ──
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    
    # Override structural/LoRA args from the checkpoint if available
    # so the user doesn't have to specify them manually
    if "args" in ckpt:
        ckpt_args = ckpt["args"]
        for key in ["fuse", "use_cosine_head", "cons_T", 
                    "use_lora", "lora_blocks", "lora_rank", 
                    "lora_alpha", "lora_dropout", "lora_targets"]:
            if key in ckpt_args:
                setattr(args, key, ckpt_args[key])
        if is_main(rank):
            log(rank, "Restored model structure config (fuse/LoRA) from checkpoint args.")

    # ── Model ──
    model = MultiViewClassifier(
        model_name=args.model,
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained=False, # We load weights shortly
        fuse=args.fuse,
        use_cosine_head=args.use_cosine_head,
        cons_T=args.cons_T,
        verbose=is_main(rank),
    ).to(device)

    # LoRA
    if args.use_lora:
        apply_lora_to_timm_vit(
            model.backbone,
            last_n_blocks=args.lora_blocks,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
            verbose=is_main(rank),
        )
        model.to(device)

    # Wrap in DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if is_ddp else model

    # ── Load weights ──
    # Backward compatibility with checkpoints without strict structures
    if "model" in ckpt:
        raw_model.load_state_dict(ckpt["model"])
        if is_main(rank): log(rank, f"Loaded weights from epoch {ckpt.get('epoch', 'unknown')} with score {ckpt.get('best_ckpt_score', 'unknown')}")
    else:
        raw_model.load_state_dict(ckpt)
        if is_main(rank): log(rank, "Loaded weights (raw state dict format).")
        
    log(rank, "Starting evaluation phase...")
    t0 = time.time()

    # ── Evaluate ──
    pred_l, yy_l, ii_l, l_sum, n_l, g_sum, g_n, cvcl, cvnl = eval_epoch_local(
        raw_model, test_loader, device, amp_dtype, cw_device
    )
    
    va_loss, va_acc, mf1, pred_all, yy_all, ii_all, avg_gate, cv_cos = gather_eval(
        pred_l, yy_l, ii_l, l_sum, n_l, g_sum, g_n, cvcl, cvnl,
        num_classes, device, is_ddp, rank
    )

    # ── Stop non-main ranks ──
    cleanup_ddp(is_ddp)
    if not is_main(rank):
        return

    # ── Main rank reporting ──
    secs = time.time() - t0
    
    print(f"\n{'═' * 60}")
    print(f"Evaluation finished in {secs:.1f}s")
    print(f"Global Loss: {va_loss:.4f}  |  Global Acc: {va_acc:.4f}  |  Macro F1: {mf1:.4f}")
    if cv_cos is not None:
        print(f"Cross-View Cosine Sim: {cv_cos:.4f}")
    if avg_gate is not None:
        print(f"Gate Weights (2.5x / 10x): {avg_gate[0]:.2f} / {avg_gate[1]:.2f}")
    print(f"{'═' * 60}")

    # ── Per-coarse class metrics ──
    rows = per_class_metrics(pred_all, yy_all, num_classes, idx_to_class)
    print(f"\n{'Class':<22} {'N':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("─" * 54)
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(f"{r['class']:<22} {r['n']:>8,} {r['prec']:>7.3f} "
              f"{r['rec']:>7.3f} {r['f1']:>7.3f}")

    # ── Confusion matrix ──
    cm = confusion_matrix_np(pred_all, yy_all, num_classes)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    header = f"{'':>20}" + "".join(f"{c[:8]:>10}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>20}" + "".join(f"{cm[i, j]:>10,}" for j in range(num_classes))
        print(row_str)

    # ── Per-fine routing accuracy ──
    fine_rows = per_fine_routing_accuracy(
        pred_all, ds_test, class_to_idx,
        fine_col=args.fine_col if hasattr(args, "fine_col") else "label",
        coarse_col=args.label_col,
        ii_all=ii_all
    )

    if fine_rows:
        print(f"\n{'Fine Class':<28} {'Coarse':>20} {'N':>8} {'Acc':>7}")
        print("─" * 67)
        for r in fine_rows:
            print(f"{r['fine_class']:<28} {r['coarse_class']:>20} "
                  f"{r['n']:>8,} {r['acc']:>7.3f}")
        avg_fine_acc = np.mean([r["acc"] for r in fine_rows])
        print(f"\nMean per-fine routing accuracy: {avg_fine_acc:.4f}")

    # Optionally log to JSON
    if args.eval_output:
        out_eval_path = Path(args.eval_output)
        out_eval_path.parent.mkdir(parents=True, exist_ok=True)
        dump = {
            "loss": va_loss,
            "accuracy": va_acc,
            "macro_f1": mf1,
            "per_class": rows,
            "confusion_matrix": {"class_names": class_names, "matrix": cm.tolist()}
        }
        if fine_rows:
            dump["per_fine_routing"] = fine_rows
            dump["mean_fine_routing_acc"] = avg_fine_acc
            
        out_eval_path.write_text(json.dumps(dump, indent=2))
        print(f"\nSaved evaluation metrics to {out_eval_path}")

if __name__ == "__main__":
    main()
