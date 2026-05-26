#!/usr/bin/env python3
"""
Extract frozen DINOv2 embeddings from test shards for dendrogram / UMAP analysis.

For each cell saves:
  emb_2p5x  : 2.5x view embedding  (768-d, L2-normalised)
  emb_10x   : 10x view embedding   (768-d, L2-normalised)
  emb_fused : mean of both views   (768-d, L2-normalised)

The backbone is completely frozen (pretrained weights, no fine-tuning).
Use these embeddings to validate coarse-group design WITHOUT circular reasoning.

Usage:
    python scripts/extract_embeddings_frozen.py \
        --test_shards  prepared/splits_v2_seed1337_nobreast/test_shards \
        --out          results/embeddings/frozen_dinov2_embeddings.parquet \
        --max_per_class 2000 \
        --batch_size 256 \
        --workers 12
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from data.dataset import CellParquetMultiView, AugCfg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fine-to-coarse mapping — verified against
#   prepared/splits_v2_seed1337_nobreast/fine_class_to_idx.json
#   prepared/splits_v2_seed1337_nobreast/expert_groups.json
FINE_TO_COARSE = {
    # Cancer expert
    "Colon cancer cells":    "Cancer",
    "Liver cancer cells":    "Cancer",
    "Lung cancer cells":     "Cancer",
    "Ovary cancer cells":    "Cancer",
    "Pancreas cancer cells": "Cancer",
    "Skin cancer cells":     "Cancer",
    # Lymphoid expert
    "B cells":               "Lymphoid",
    "NK cells":              "Lymphoid",
    "T cells":               "Lymphoid",
    # Neuroglial expert
    "Astrocytes":            "Neuroglial",
    "Microglia":             "Neuroglial",
    "Neurons":               "Neuroglial",
    "Oligodendrocytes":      "Neuroglial",
    # Tissue_Vascular expert
    "Endothelial cells":     "Tissue_Vascular",
    "Epithelial cells":      "Tissue_Vascular",
    "Fibroblasts":           "Tissue_Vascular",
    "Myeloid cells":         "Tissue_Vascular",
    "Pericytes":             "Tissue_Vascular",
    "Smooth muscle cells":   "Tissue_Vascular",
    # Singletons (no expert)
    "Stromal cells":              "Stromal",
    "Stem and progenitor cells":  "Stem_Progenitor",
}

ALL_FINE_CLASSES = sorted(FINE_TO_COARSE.keys())   # 21 classes
FINE_TO_IDX      = {c: i for i, c in enumerate(ALL_FINE_CLASSES)}


# ── feature extraction (unchanged from original) ──────────────────────────────

@torch.no_grad()
def forward_features(backbone, x):
    feats = backbone.forward_features(x)
    if feats.ndim == 3:
        emb = feats[:, 0]             # CLS token  (ViT)
    elif feats.ndim == 4:
        emb = feats.mean(dim=(2, 3))  # GAP        (CNN)
    else:
        emb = feats
    return emb


def load_backbone_checkpoint(backbone, checkpoint_path: str):
    """Load common SimCLR/backbone checkpoint formats into a timm backbone."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("state_dict", "model", "backbone", "encoder"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break

    if not isinstance(state, dict):
        raise RuntimeError(f"Could not find a state dict in {checkpoint_path}")

    prefixes = (
        "module.", "model.", "backbone.", "encoder.", "encoder_q.",
        "net.", "visual.",
    )
    cleaned = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        kk = k
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
                    changed = True
        cleaned[kk] = v

    incompatible = backbone.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  missing keys: {len(incompatible.missing_keys)}")
    print(f"  unexpected keys: {len(incompatible.unexpected_keys)}")


@torch.no_grad()
def extract_from_parquet(
    parquet_path: Path,
    backbone,
    class_to_idx: dict[str, int],
    label_col: str,
    fine_to_coarse: dict[str, str],
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Run frozen backbone on one tissue shard, return embeddings + metadata."""

    ds = CellParquetMultiView(
        parquet_path = str(parquet_path),
        class_to_idx = class_to_idx,
        label_col    = label_col,
        size         = img_size,
        aug          = AugCfg(enable=False),
        return_meta  = True,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    use_cuda  = (DEVICE == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    emb2_list, emb10_list, embf_list = [], [], []

    for x, _y, _meta in loader:
        x = x.to(DEVICE, non_blocking=True)          # (B, 2, C, H, W)

        B, S, C, H, W = x.shape
        xs = x.reshape(B * S, C, H, W)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
            emb_all = forward_features(backbone, xs)  # (B*S, D)

        P   = emb_all.reshape(B, S, -1).float()       # (B, 2, D)
        e2  = F.normalize(P[:, 0], dim=1)             # 2.5x
        e10 = F.normalize(P[:, 1], dim=1)             # 10x
        ef  = F.normalize(0.5 * (e2 + e10), dim=1)   # fused

        emb2_list.append(e2.cpu().numpy().astype(np.float32))
        emb10_list.append(e10.cpu().numpy().astype(np.float32))
        embf_list.append(ef.cpu().numpy().astype(np.float32))

    # Read metadata directly from parquet (avoids depending on DataLoader meta fields)
    meta_cols = [
        "cell_id", "img_path_10x", "img_path_2p5x", "label", "coarse_label",
        "tissue", "x_centroid", "y_centroid", "patch_id", "coarse_id",
    ]
    available_cols = pd.read_parquet(parquet_path).columns
    meta_cols = [c for c in meta_cols if c in available_cols]
    df_meta = pd.read_parquet(
        parquet_path,
        columns=meta_cols,
    ).reset_index(drop=True)

    if "coarse_label" not in df_meta.columns and "label" in df_meta.columns:
        df_meta["coarse_label"] = df_meta["label"].map(fine_to_coarse)
    df_meta["target_label"] = df_meta[label_col]
    df_meta["label_id"] = df_meta[label_col].map(class_to_idx).astype("Int32")
    df_meta["emb_2p5x"]    = list(np.concatenate(emb2_list,  axis=0))
    df_meta["emb_10x"]     = list(np.concatenate(emb10_list, axis=0))
    df_meta["emb_fused"]   = list(np.concatenate(embf_list,  axis=0))

    return df_meta


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_shards",   default=None,
                    help="prepared/splits_v2_seed1337_nobreast/test_shards/")
    ap.add_argument("--parquet",       default=None,
                    help="Single parquet to extract, e.g. a 500-per-class core subset.")
    ap.add_argument("--out",           required=True,
                    help="Output parquet path")
    ap.add_argument("--model",         default="vit_base_patch16_dinov3.lvd1689m",
                    help="timm model name (frozen backbone)")
    ap.add_argument("--checkpoint",    default=None,
                    help="Optional backbone checkpoint, useful for a frozen SimCLR encoder.")
    ap.add_argument("--class_map",     default=None,
                    help="JSON class map for label_col. Defaults to built-in fine class map.")
    ap.add_argument("--fine_to_coarse", default=None,
                    help="Optional fine_to_coarse JSON. Defaults to built-in mapping.")
    ap.add_argument("--label_col",     default="label",
                    help="Label column used only for dataset indexing and label_id.")
    ap.add_argument("--img_size",      type=int, default=224)
    ap.add_argument("--max_per_class", type=int, default=2000,
                    help="Max rows per label_col after extraction. 0 = no cap.")
    ap.add_argument("--batch_size",    type=int, default=256)
    ap.add_argument("--workers",       type=int, default=12)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    rng      = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if (args.test_shards is None) == (args.parquet is None):
        raise ValueError("Provide exactly one of --test_shards or --parquet")

    if args.class_map:
        with open(args.class_map) as f:
            class_to_idx = json.load(f)
    else:
        class_to_idx = FINE_TO_IDX

    fine_to_coarse = FINE_TO_COARSE
    if args.fine_to_coarse:
        with open(args.fine_to_coarse) as f:
            fine_to_coarse = json.load(f)

    # Load backbone ONCE before the shard loop
    print(f"Loading backbone: {args.model}  device: {DEVICE}")
    create_kwargs = dict(pretrained=(args.checkpoint is None), num_classes=0)
    try:
        backbone = timm.create_model(args.model, **create_kwargs, img_size=args.img_size)
    except TypeError:
        backbone = timm.create_model(args.model, **create_kwargs)
    backbone = backbone.to(DEVICE)
    if args.checkpoint:
        load_backbone_checkpoint(backbone, args.checkpoint)
    backbone.eval()

    if args.parquet:
        shards = [Path(args.parquet)]
    else:
        shards = sorted(Path(args.test_shards).glob("*.parquet"))
    print(f"Found {len(shards)} parquet file(s)\n")

    all_dfs = []
    for shard in tqdm(shards, desc="shards"):
        df = extract_from_parquet(
            shard, backbone, class_to_idx, args.label_col, fine_to_coarse,
            args.img_size, args.batch_size, args.workers,
        )
        all_dfs.append(df)
        tqdm.write(f"  {shard.stem}: {len(df):,} cells")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Per-class cap (keeps class balance for dendrogram / UMAP)
    if args.max_per_class > 0:
        parts = []
        for label, grp in df_all.groupby(args.label_col):
            if len(grp) > args.max_per_class:
                idx = rng.choice(grp.index, size=args.max_per_class, replace=False)
                parts.append(grp.loc[idx])
            else:
                parts.append(grp)
        df_all = pd.concat(parts, ignore_index=True)

    df_all.to_parquet(out_path, index=False)

    # Save label map alongside
    label_map_path = out_path.with_suffix(".label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # Summary
    print(f"\nClasses: {df_all[args.label_col].nunique()} | "
          f"Tissues: {df_all['tissue'].nunique()} | "
          f"Cells: {len(df_all):,}")
    print()
    for label, n in df_all[args.label_col].value_counts().sort_index().items():
        coarse = fine_to_coarse.get(label, df_all.loc[df_all[args.label_col] == label, "coarse_label"].iloc[0]
                                    if "coarse_label" in df_all.columns else "?")
        print(f"  [{coarse:<20}]  {label:<35}  {n:>6,}")
    print(f"\nSaved      → {out_path}")
    print(f"Label map  → {label_map_path}")


if __name__ == "__main__":
    main()
