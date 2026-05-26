#!/usr/bin/env python3
"""
Extract embeddings from a trained router checkpoint.

Outputs a parquet with per-cell: label, tissue, coordinates,
and embeddings (2.5x, 10x, fused) for downstream analysis
(heatmaps, clustering, UMAP, etc.)

Usage (memmap):
    python scripts/extract_embeddings.py \
        --checkpoint experiments/router_gate_all_even_r16/best.pt \
        --cache_dir cache_224 --split val_balanced \
        --class_map prepared/splits_v3_seed1337/coarse_to_id.json \
        --out prepared/embeddings_gate_all_even_r16.parquet

Usage (parquet):
    python scripts/extract_embeddings.py \
        --checkpoint experiments/router_gate_all_even_r16/best.pt \
        --parquet prepared/splits_v3_seed1337/val_balanced.parquet \
        --class_map prepared/splits_v3_seed1337/coarse_to_id.json \
        --label_col coarse_label \
        --out prepared/embeddings_gate_all_even_r16.parquet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import MultiViewClassifier
from models.lora import apply_lora_to_timm_vit
from data.dataset import CellParquetMultiView, MemmapDataset, AugCfg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(ckpt_path, class_to_idx):
    """Rebuild MultiViewClassifier from checkpoint args, load weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    num_classes = len(class_to_idx)

    model = MultiViewClassifier(
        model_name=args.get("model", "vit_base_patch16_dinov3.lvd1689m"),
        num_classes=num_classes,
        img_size=args.get("img_size", 224),
        pretrained=False,
        fuse=args["fuse"],
        use_cosine_head=args.get("use_cosine_head", False),
        cons_T=args.get("cons_T", 3.0),
        verbose=True,
    )

    for p in model.backbone.parameters():
        p.requires_grad = False

    if args.get("use_lora", 1):
        apply_lora_to_timm_vit(
            model.backbone,
            last_n_blocks=args.get("lora_blocks", "4"),
            r=args.get("lora_rank", 8),
            alpha=args.get("lora_alpha", 16),
            dropout=0.0,
            targets=tuple(
                t.strip() for t in
                args.get("lora_targets", "qkv").split(",") if t.strip()
            ),
            verbose=True,
        )
        model.to(DEVICE)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"Loaded: {ckpt_path}")
    print(f"  fuse={args['fuse']}  epoch={ckpt.get('epoch', '?')}  "
          f"score={ckpt.get('best_ckpt_score', '?')}")
    return model, args


@torch.no_grad()
def extract_embeddings(model, loader, idx_to_class, fine_to_coarse=None):
    """Run inference, collect per-view and fused embeddings + gate weights."""
    model = model.to(DEVICE).eval()
    amp_dtype = (torch.bfloat16
                 if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)

    emb2_all, emb10_all, embf_all = [], [], []
    pred_all, y_all, gate_all = [], [], []

    for x, y, meta in tqdm(loader, desc="Extracting"):
        x = x.to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=(DEVICE == "cuda")):
            logits, emb, aux = model(x)

        pred_all.append(logits.argmax(1).cpu())
        y_all.append(y)

        # Per-view embeddings
        if "emb_views" in aux:
            P = aux["emb_views"].float()   # [B, 2, D]
            e2 = F.normalize(P[:, 0], dim=1)
            e10 = F.normalize(P[:, 1], dim=1)
        else:
            e2 = e10 = F.normalize(emb.float(), dim=1)

        ef = F.normalize(emb.float(), dim=1)
        emb2_all.append(e2.cpu().numpy().astype(np.float32))
        emb10_all.append(e10.cpu().numpy().astype(np.float32))
        embf_all.append(ef.cpu().numpy().astype(np.float32))

        if "gate_weights" in aux:
            gate_all.append(aux["gate_weights"].cpu().numpy().astype(np.float32))

    # Assemble
    preds = torch.cat(pred_all).numpy()
    labels = torch.cat(y_all).numpy()

    out = pd.DataFrame({
        "label_id": labels.astype(int),
        "pred_id": preds.astype(int),
        "label": [idx_to_class[int(i)] for i in labels],
        "pred": [idx_to_class[int(i)] for i in preds],
    })

    if fine_to_coarse:
        out["coarse_label"] = out["label"].map(fine_to_coarse).fillna("unknown")

    out["emb_2p5x"] = list(np.concatenate(emb2_all))
    out["emb_10x"] = list(np.concatenate(emb10_all))
    out["emb_fused"] = list(np.concatenate(embf_all))

    if gate_all:
        gates = np.concatenate(gate_all)
        out["gate_2p5x"] = gates[:, 0]
        out["gate_10x"] = gates[:, 1]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--class_map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    # Data source
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--split", default="val_balanced")
    ap.add_argument("--parquet", default=None)
    # Optional
    ap.add_argument("--fine_to_coarse", default=None,
                    help="Path to fine_to_coarse.json")
    ap.add_argument("--label_col", default=None,
                    help="Dataset label column. Defaults to checkpoint args label_col, then coarse_label.")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Subsample for quick tests (0 = all)")
    args = ap.parse_args()

    with open(args.class_map) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    fine_to_coarse = None
    if args.fine_to_coarse and Path(args.fine_to_coarse).exists():
        with open(args.fine_to_coarse) as f:
            fine_to_coarse = json.load(f)

    model, train_args = build_model(args.checkpoint, class_to_idx)
    label_col = args.label_col or train_args.get("label_col", "coarse_label")

    # Dataset
    if args.cache_dir:
        print(f"MemmapDataset: {args.cache_dir}/{args.split}")
        ds = MemmapDataset(
            cache_dir=args.cache_dir, split_name=args.split,
            view="both", return_meta=True,
        )
    elif args.parquet:
        print(f"CellParquetMultiView: {args.parquet}")
        ds = CellParquetMultiView(
            parquet_path=args.parquet, class_to_idx=class_to_idx,
            label_col=label_col,
            size=train_args.get("img_size", 224),
            aug=AugCfg(enable=False), return_meta=True,
        )
    else:
        raise ValueError("Provide --cache_dir or --parquet")

    if 0 < args.max_samples < len(ds):
        from torch.utils.data import Subset
        idx = np.random.default_rng(42).choice(len(ds), args.max_samples, replace=False)
        ds = Subset(ds, idx.tolist())
        print(f"Subsampled to {len(ds):,}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Total: {len(ds):,} cells\n")

    df = extract_embeddings(model, loader, idx_to_class, fine_to_coarse)

    # ── Attach source metadata in the same row order ─────────────────────
    base_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    if hasattr(base_ds, "df"):
        meta_df = base_ds.df.reset_index(drop=True)
        if isinstance(ds, torch.utils.data.Subset):
            meta_df = meta_df.iloc[ds.indices].reset_index(drop=True)
        keep_cols = [
            "cell_id", "img_path_10x", "img_path_2p5x", "label", "coarse_label",
            "tissue", "x_centroid", "y_centroid", "patch_id", "coarse_id",
        ]
        meta_df = meta_df[[c for c in keep_cols if c in meta_df.columns]].copy()
        meta_df = meta_df.rename(columns={"label": "fine_label"})
        dup_cols = [c for c in meta_df.columns if c in df.columns]
        df = pd.concat([meta_df, df.drop(columns=dup_cols)], axis=1)

    # ── Attach fine/coarse string labels from memmap metadata ────────────
    # MemmapDataset stores _fine_labels and _coarse_labels arrays;
    # these give us the 23 fine class names needed for clustering analysis.
    if hasattr(base_ds, "_fine_labels") and base_ds._fine_labels is not None:
        if isinstance(ds, torch.utils.data.Subset):
            df["fine_label"] = [str(base_ds._fine_labels[i]) for i in ds.indices]
        else:
            df["fine_label"] = [str(base_ds._fine_labels[i]) for i in range(len(ds))]
        print(f"  Fine labels: {df['fine_label'].nunique()} classes")

    if hasattr(base_ds, "_coarse_labels") and base_ds._coarse_labels is not None:
        if isinstance(ds, torch.utils.data.Subset):
            df["coarse_label_str"] = [str(base_ds._coarse_labels[i]) for i in ds.indices]
        else:
            df["coarse_label_str"] = [str(base_ds._coarse_labels[i]) for i in range(len(ds))]

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path), index=False)

    dim = len(df["emb_fused"].iloc[0]) if len(df) > 0 else "?"
    print(f"\nSaved: {out_path} ({len(df):,} cells, {dim}d embeddings)")


if __name__ == "__main__":
    main()
