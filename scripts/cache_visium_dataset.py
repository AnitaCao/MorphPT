#!/usr/bin/env python3
"""
cache_visium_dataset.py  (v3 — decoupled caching layer)
───────────────────────────────────────────────────────
Pure data preparation layer for Visium HD patches.
Aligns cells across spatial views, computes spatial grid tile_ids,
applies preliminary global coverage filtering, and writes heavy disk memmaps.

Does NOT assign train/val/test splits or compute expression normalization stats.
Use `generate_split_layout.py` downstream to define experimental split layouts.

Outputs:
  meta.csv          canonical cell order: cell_id, x_centroid, y_centroid, mmap_idx, tile_id
  expr.npy          (N, G) float32, globally filtered genes, aligned to meta.csv
  gene_list.txt     G kept gene names
  cache_stats.json  base caching configuration summary
  build_args.json   all args for reproducibility
  images_{scale}.npy  (N, img_size, img_size, 3) uint8 per scale
"""

import argparse
import json
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

VISIUM_ROOT = "/hpc/group/jilab/boxuan/visiumHD"
MORPH_ROOT  = "/hpc/group/jilab/hz/MorphPT/data/visiumHD"


# ── Args ───────────────────────────────────────────────────────────────────
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",          type=str, required=True,
                   choices=["human_crc", "human_lungcancer", "human_pancreas",
                            "mouse_brain", "mouse_embryo", "mouse_intestine", "mouse_kidney"])
    p.add_argument("--out_dir",          type=str, required=True)
    p.add_argument("--visium_root",      type=str, default=VISIUM_ROOT)
    p.add_argument("--morph_root",       type=str, default=MORPH_ROOT)
    p.add_argument("--scales",           type=str, default="2.5x,10.0x")
    p.add_argument("--img_variant",      type=str, default="raw",
                   choices=["raw", "mask_target", "mask_context"])
    p.add_argument("--img_size",         type=int, default=224)
    p.add_argument("--grid_size",        type=int, default=5,
                   help="NxN spatial grid to assign tile_ids.")
    p.add_argument("--min_coverage",     type=float, default=0.10,
                   help="Min fraction of cells expressing gene to keep it globally.")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--workers",          type=int, default=16)
    return p


# ── Spatial grid mapping ───────────────────────────────────────────────────
def compute_tile_ids(df, grid_size):
    x_min, x_max = df["x_centroid"].min(), df["x_centroid"].max()
    y_min, y_max = df["y_centroid"].min(), df["y_centroid"].max()
    x_step = (x_max - x_min) / grid_size * 1.0001
    y_step = (y_max - y_min) / grid_size * 1.0001

    df = df.copy()
    df["x_bin"]   = np.floor((df["x_centroid"] - x_min) / x_step).astype(int).clip(0, grid_size - 1)
    df["y_bin"]   = np.floor((df["y_centroid"] - y_min) / y_step).astype(int).clip(0, grid_size - 1)
    df["tile_id"] = df["y_bin"] * grid_size + df["x_bin"]

    tile_counts = df["tile_id"].value_counts()
    logging.info(f"Grid {grid_size}×{grid_size} mapped. Occupied tiles: {len(tile_counts)}")
    for t_id, count in tile_counts.sort_index().items():
        logging.info(f"  Tile {t_id:<3}: {count:>7,} cells")

    # Drop temporary bins to keep meta.csv pure and canonical
    df = df.drop(columns=["x_bin", "y_bin"])
    return df, tile_counts.to_dict()


# ── Image loader ───────────────────────────────────────────────────────────
def _load_image(task):
    mmap_idx, img_path, img_size = task
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
            return mmap_idx, np.array(img, dtype=np.uint8), None
    except Exception as e:
        return mmap_idx, None, str(e)


def build_image_memmap(canonical_df, root_dir, dataset, out_dir,
                       scale, img_variant, img_size, workers):
    img_col    = f"{img_variant}_img_path"
    meta_path  = root_dir / f"meta/{scale}/{dataset}.csv"
    scale_meta = pd.read_csv(meta_path)[["cell_id", img_col]]
    scale_meta["cell_id"] = scale_meta["cell_id"].astype(str).str.strip()

    merged = canonical_df[["cell_id", "mmap_idx"]].merge(
        scale_meta, on="cell_id", how="left"
    )
    missing = merged[img_col].isna().sum()
    if missing > 0:
        logging.warning(f"  [{scale}] {missing} cells missing image → gray placeholder")

    mmap_path = out_dir / f"images_{scale}.npy"
    shape     = (len(merged), img_size, img_size, 3)
    size_gb   = np.prod(shape) / 1e9
    logging.info(f"  [{scale}] Allocating {shape}  ({size_gb:.1f} GB)")

    images_mmap = np.lib.format.open_memmap(
        str(mmap_path), mode="w+", dtype=np.uint8, shape=shape
    )

    tasks = []
    for _, row in merged.iterrows():
        idx = int(row["mmap_idx"])
        if pd.isna(row[img_col]):
            images_mmap[idx] = 128
        else:
            tasks.append((idx, str(root_dir / row[img_col]), img_size))

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_load_image, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks),
                           desc=f"  [{scale}]", dynamic_ncols=True):
            idx, arr, err = future.result()
            if err is None:
                images_mmap[idx] = arr
            else:
                logging.warning(f"  [{scale}] Failed idx={idx}: {err}")
                images_mmap[idx] = 128
                failed.append({"mmap_idx": idx, "error": err})

    images_mmap.flush()
    del images_mmap
    gc.collect()

    logging.info(f"  [{scale}] Done. Written={len(tasks)-len(failed):,}  "
                 f"Failed={len(failed)}")
    if failed:
        (out_dir / f"failed_{scale}.json").write_text(json.dumps(failed, indent=2))


# ── Expression: load, align, filter ───────────────────────────────────────
def build_expression(canonical_df, root_dir, out_dir, min_coverage):
    logging.info("Loading expression matrix...")
    X_sparse   = sio.mmread(root_dir / "expr/expr.mtx").tocsr()
    cells_list = (root_dir / "expr/cells.txt").read_text().splitlines()
    genes_list = (root_dir / "expr/genes.txt").read_text().splitlines()

    # Ensure (cells × genes)
    if X_sparse.shape[0] == len(genes_list) and X_sparse.shape[1] == len(cells_list):
        X_sparse = X_sparse.T.tocsr()

    logging.info(f"  Raw: {X_sparse.shape[0]:,} cells × {X_sparse.shape[1]:,} genes")

    cell_to_expr = {str(c).strip(): i for i, c in enumerate(cells_list)}

    # Align to canonical order
    logging.info("  Aligning to canonical cell order...")
    ordered_idx = [cell_to_expr.get(str(cid).strip()) for cid in canonical_df["cell_id"]]
    missing     = sum(1 for x in ordered_idx if x is None)
    if missing > 0:
        logging.warning(f"  {missing} canonical cells not found in expr")

    # Build dense aligned matrix
    X_aligned = np.zeros((len(canonical_df), X_sparse.shape[1]), dtype=np.float32)
    valid     = [(i, j) for i, j in enumerate(ordered_idx) if j is not None]
    if valid:
        dst, src = zip(*valid)
        X_aligned[list(dst)] = np.array(X_sparse[list(src)].todense(), dtype=np.float32)

    del X_sparse
    gc.collect()

    # Filter genes globally by coverage across all valid cells
    coverage   = (X_aligned > 0).mean(axis=0)
    keep_mask  = coverage >= min_coverage
    n_kept     = int(keep_mask.sum())

    logging.info(f"  Global gene filter (≥{min_coverage:.0%} across {len(canonical_df):,} cells):")
    logging.info(f"    Before : {X_aligned.shape[1]:,} genes")
    logging.info(f"    Kept   : {n_kept:,} genes")
    logging.info(f"    Dropped: {X_aligned.shape[1] - n_kept:,} genes")

    X_filtered = X_aligned[:, keep_mask]
    kept_genes = [g for g, k in zip(genes_list, keep_mask) if k]

    del X_aligned
    gc.collect()

    # Save raw unlogged counts directly
    np.save(str(out_dir / "expr.npy"), X_filtered)
    logging.info(f"  Saved expr.npy  shape={X_filtered.shape}")

    (out_dir / "gene_list.txt").write_text("\n".join(kept_genes))
    logging.info(f"  Saved gene_list.txt  ({n_kept} genes)")

    del X_filtered
    gc.collect()

    return n_kept


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args    = get_parser().parse_args()
    scales  = [s.strip() for s in args.scales.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset   = args.dataset
    root_dir  = Path(args.visium_root) / dataset
    morph_dir = Path(args.morph_root)  / dataset

    logging.info("=" * 60)
    logging.info(f"Dataset      : {dataset}")
    logging.info(f"Out dir      : {out_dir}")
    logging.info(f"Scales       : {scales}")
    logging.info(f"Grid Size    : {args.grid_size}×{args.grid_size}")
    logging.info(f"Min coverage : ≥{args.min_coverage:.0%}")
    logging.info("=" * 60)

    (out_dir / "build_args.json").write_text(json.dumps(vars(args), indent=2))

    # ── Step 1: Canonical cell list ────────────────────────────────────────
    logging.info("\n[Step 1] Canonical cell list (intersection across scales + spatial)...")
    spatial   = pd.read_csv(morph_dir / "spatial.csv")
    valid_ids = set(spatial["cell_id"].astype(str).str.strip())
    logging.info(f"  Spatial CSV     : {len(valid_ids):,} cells")

    for scale in scales:
        meta_path    = root_dir / f"meta/{scale}/{dataset}.csv"
        ids_in_scale = set(pd.read_csv(meta_path)["cell_id"].astype(str).str.strip())
        before       = len(valid_ids)
        valid_ids   &= ids_in_scale
        logging.info(f"  After ∩ {scale:<6} : {len(valid_ids):,}  "
                     f"(dropped {before - len(valid_ids):,})")

    canonical_ids = sorted(valid_ids)
    logging.info(f"  Canonical total : {len(canonical_ids):,} (sorted by cell_id)")

    df = pd.DataFrame({"cell_id": canonical_ids})
    df = df.merge(
        spatial[["cell_id", "x_centroid", "y_centroid"]].assign(
            cell_id=lambda d: d["cell_id"].astype(str).str.strip()
        ),
        on="cell_id", how="left"
    )
    df = df.reset_index(drop=True)
    df["mmap_idx"] = np.arange(len(df))

    # ── Step 2: Spatial grid tile mapping ──────────────────────────────────
    logging.info(f"\n[Step 2] Spatial grid mapping...")
    df, tile_counts = compute_tile_ids(df, grid_size=args.grid_size)

    # ── Step 3: Expression ─────────────────────────────────────────────────
    logging.info(f"\n[Step 3] Expression cache...")
    n_genes = build_expression(df, root_dir, out_dir, args.min_coverage)

    # ── Step 4: Image memmaps ──────────────────────────────────────────────
    for scale in scales:
        logging.info(f"\n[Step 4] Image memmap: {scale}")
        build_image_memmap(
            canonical_df = df,
            root_dir     = root_dir,
            dataset      = dataset,
            out_dir      = out_dir,
            scale        = scale,
            img_variant  = args.img_variant,
            img_size     = args.img_size,
            workers      = args.workers,
        )

    # ── Step 5: Metadata ───────────────────────────────────────────────────
    logging.info(f"\n[Step 5] Saving base metadata...")
    df.to_csv(out_dir / "meta.csv", index=False)
    logging.info(f"  Saved meta.csv  ({len(df):,} rows)")

    cache_stats = {
        "dataset":      dataset,
        "n_cells":      len(df),
        "n_genes":      n_genes,
        "scales":       scales,
        "grid_size":    args.grid_size,
        "min_coverage": args.min_coverage,
        "seed":         args.seed,
        "tile_counts":  {int(k): int(v) for k, v in tile_counts.items()}
    }
    (out_dir / "cache_stats.json").write_text(json.dumps(cache_stats, indent=2))

    # ── Summary ────────────────────────────────────────────────────────────
    N = len(df)
    logging.info("\n" + "=" * 60)
    logging.info("Base Cache complete.")
    for scale in scales:
        gb = N * args.img_size * args.img_size * 3 / 1e9
        logging.info(f"  images_{scale}.npy : ({N:,}, {args.img_size}, "
                     f"{args.img_size}, 3)  ~{gb:.1f} GB")
    logging.info(f"  expr.npy          : ({N:,}, {n_genes})")
    logging.info(f"  gene_list.txt     : {n_genes} genes")
    logging.info(f"  out_dir           : {out_dir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()