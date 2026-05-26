#!/usr/bin/env python3
"""
CellPT MoE: Build Balanced Benchmark Test Set & Extract Showcase Slides
=======================================================================
1. 从 test_shards 中按细类(Fine Class)且按切片比例(per-slide)均匀抽样，生成核心 Benchmark。
2. 提取 2-3 个 Showcase Slide 的全部 5 个 test patches，用于空间可视化。

Usage:
  python scripts/build_core_test_split.py \
    --test_shards_dir prepared/splits_v3_seed1337/test_shards \
    --out_dir prepared/core_test_split \
    --cells_per_class 1500 \
    --showcase_slides Xenium_V1_FFPE_Human_Breast_IDC Xenium_Preview_Human_Lung_Cancer \
    --seed 42
"""

import argparse
import json
import os
from pathlib import Path
from glob import glob
from collections import defaultdict

import pandas as pd
import numpy as np


def build_fine_to_coarse() -> dict:
    groups = {
        "Cancer": [
            "Breast cancer cells", "Ovary cancer cells", "Colon cancer cells",
            "Skin cancer cells", "Lung cancer cells", "Pancreas cancer cells",
            "Liver cancer cells",
        ],
        "Neuroglial": ["Microglia", "Oligodendrocytes", "Astrocytes", "Neurons"],
        "Stromal": ["Stromal cells"],
        "Stem_Progenitor": ["Stem and progenitor cells"],
        "Lymphoid": ["NK cells", "B cells", "T cells"],
        "Tissue_Structural": [
            "Adipocytes", "Epithelial cells", "Pericytes", "Fibroblasts",
        ],
        "Vascular": ["Endothelial cells", "Myeloid cells", "Smooth muscle cells"],
    }
    m = {}
    for coarse, fines in groups.items():
        for fine in fines:
            m[fine] = coarse
    return m


def stratified_sample(group, n_target, rng):
    """Sample n_target from group, stratified by slide.
    Every slide contributes at least 1 sample.
    Uses largest-remainder method for exact target count."""
    if n_target >= len(group):
        return group

    slide_groups = list(group.groupby("tissue", sort=False))
    slide_counts = np.array([len(sg) for _, sg in slide_groups])

    # Proportional allocation, min 1 per slide
    fracs = slide_counts / slide_counts.sum() * n_target
    allocs = np.maximum(1, np.floor(fracs)).astype(int)
    allocs = np.minimum(allocs, slide_counts)

    # Distribute remainder by largest-remainder
    remainder = n_target - allocs.sum()
    if remainder > 0:
        headroom = slide_counts - allocs
        residuals = np.where(headroom > 0, fracs - allocs, -999)
        for idx in np.argsort(-residuals):
            if remainder <= 0:
                break
            give = min(int(remainder), int(headroom[idx]))
            allocs[idx] += give
            remainder -= give
    elif remainder < 0:
        over = -remainder
        for idx in np.argsort(-allocs):
            if over <= 0:
                break
            trim = min(int(over), int(allocs[idx] - 1))
            allocs[idx] -= trim
            over -= trim

    parts = []
    for (tissue, sg), n_take in zip(slide_groups, allocs):
        n_take = int(n_take)
        if n_take <= 0:
            continue
        if n_take >= len(sg):
            parts.append(sg)
        else:
            idx = rng.choice(sg.index.to_numpy(), size=n_take, replace=False)
            parts.append(sg.loc[idx])

    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_shards_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cells_per_class", type=int, default=1500)
    ap.add_argument("--showcase_slides", nargs="*", default=[
        "Xenium_V1_FFPE_Human_Breast_IDC",
        "Xenium_Preview_Human_Lung_Cancer",
    ])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ftc = build_fine_to_coarse()
    c2id = {n: i for i, n in enumerate(sorted(set(ftc.values())))}
    expected_fines = set(ftc.keys())

    # =================================================================
    # Load all test shards
    # =================================================================
    test_files = sorted(glob(os.path.join(args.test_shards_dir, "*.parquet")))
    if not test_files:
        raise RuntimeError(f"No parquet files in {args.test_shards_dir}")

    print(f"Loading {len(test_files)} test shards...")
    df_all = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
    print(f"Total cells in test pool: {len(df_all):,}")
    print(f"Slides: {df_all['tissue'].nunique()}")
    print(f"Fine classes found: {df_all['label'].nunique()}\n")

    missing = expected_fines - set(df_all["label"].unique())
    if missing:
        print(f"WARNING: Missing fine classes in test_shards: {missing}\n")

    # =================================================================
    # Task 1: Balanced Core Benchmark
    # =================================================================
    print(f"{'═' * 80}")
    print(f"TASK 1: Building Balanced Core Benchmark")
    print(f"  Target: {args.cells_per_class} cells/class, per-slide stratified")
    print(f"{'═' * 80}\n")

    parts = []
    stats = []
    for label in sorted(expected_fines):
        group = df_all[df_all["label"] == label]
        n_raw = len(group)

        if n_raw == 0:
            stats.append({"fine": label, "raw": 0, "sampled": 0,
                          "slides": 0, "kept": 0})
            continue

        n_target = min(args.cells_per_class, n_raw)
        if n_target >= n_raw:
            sampled = group
        else:
            sampled = stratified_sample(group, n_target, rng)

        n_slides = group["tissue"].nunique()
        n_kept = sampled["tissue"].nunique()
        parts.append(sampled)
        stats.append({"fine": label, "raw": n_raw, "sampled": len(sampled),
                      "slides": n_slides, "kept": n_kept})

    df_bench = pd.concat(parts, ignore_index=True)
    df_bench = df_bench.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    bench_path = out / "core_benchmark_test.parquet"
    df_bench.to_parquet(bench_path, index=False)

    # Report
    print(f"  {'Fine Class':<30} {'Raw':>8} {'Sampled':>8} {'Slides':>7} {'Kept':>6}")
    print(f"  {'─' * 62}")

    coarse_counts = defaultdict(int)
    for s in stats:
        flag = " keep" if s["sampled"] == s["raw"] and s["raw"] > 0 else ""
        print(f"  {s['fine']:<30} {s['raw']:>8,} {s['sampled']:>8,} "
              f"{s['slides']:>7} {s['kept']:>6}{flag}")
        if s["fine"] in ftc:
            coarse_counts[ftc[s["fine"]]] += s["sampled"]

    print(f"  {'─' * 62}")
    print(f"  {'TOTAL':<30} {len(df_all):>8,} {len(df_bench):>8,}\n")

    print(f"  Coarse distribution:")
    for c in sorted(coarse_counts):
        print(f"    {c:<22} {coarse_counts[c]:>8,}")
    if coarse_counts:
        vals = list(coarse_counts.values())
        print(f"    Max/min: {max(vals):,} / {min(vals):,} = "
              f"{max(vals) / max(min(vals), 1):.1f}x")

    print(f"\n  Saved: {bench_path} ({len(df_bench):,} cells)")

    # =================================================================
    # Task 2: Extract Showcase Slides (5 test patches each)
    # =================================================================
    print(f"\n{'═' * 80}")
    print(f"TASK 2: Extracting Showcase Slides (existing test patches)")
    print(f"{'═' * 80}\n")

    for tissue in args.showcase_slides:
        df_slide = df_all[df_all["tissue"] == tissue].copy()

        if len(df_slide) == 0:
            print(f"  [{tissue}] Not found in test_shards, skipping.\n")
            continue

        # Ensure coarse columns exist
        if "coarse_label" not in df_slide.columns:
            df_slide["coarse_label"] = df_slide["label"].map(ftc)
        if "coarse_id" not in df_slide.columns:
            df_slide["coarse_id"] = df_slide["coarse_label"].map(c2id).astype(np.int16)

        out_path = out / f"showcase_{tissue}.parquet"
        df_slide.to_parquet(out_path, index=False)

        # Report
        n_patches = df_slide["patch_id"].nunique() if "patch_id" in df_slide.columns else "?"
        n_fines = df_slide["label"].nunique()
        n_coarse = df_slide["coarse_label"].nunique()

        print(f"  [{tissue}]")
        print(f"    Cells: {len(df_slide):,}  |  Patches: {n_patches}  |  "
              f"Fine types: {n_fines}  |  Coarse groups: {n_coarse}")

        # Spatial extent
        x_range = df_slide["x_centroid"].max() - df_slide["x_centroid"].min()
        y_range = df_slide["y_centroid"].max() - df_slide["y_centroid"].min()
        print(f"    Spatial extent: {x_range:.0f} x {y_range:.0f} μm")

        # Per-patch summary
        if "patch_id" in df_slide.columns:
            print(f"    {'Patch':>7} {'Cells':>8} {'Fine types':>11} "
                  f"{'X range':>12} {'Y range':>12}")
            print(f"    {'─' * 54}")
            for pid, pg in df_slide.groupby("patch_id", sort=True):
                xr = f"{pg['x_centroid'].min():.0f}-{pg['x_centroid'].max():.0f}"
                yr = f"{pg['y_centroid'].min():.0f}-{pg['y_centroid'].max():.0f}"
                print(f"    {pid:>7} {len(pg):>8,} {pg['label'].nunique():>11} "
                      f"{xr:>12} {yr:>12}")

        # Fine class breakdown
        print(f"\n    {'Fine Class':<30} {'Count':>7}")
        print(f"    {'─' * 39}")
        for lbl, cnt in df_slide["label"].value_counts().items():
            print(f"    {lbl:<30} {cnt:>7,}")

        print(f"\n    Saved: {out_path}\n")

    # =================================================================
    # Save metadata
    # =================================================================
    meta = {
        "cells_per_class": args.cells_per_class,
        "seed": args.seed,
        "benchmark_cells": len(df_bench),
        "benchmark_fines": len([s for s in stats if s["sampled"] > 0]),
        "showcase_slides": args.showcase_slides,
        "fine_to_coarse": ftc,
        "coarse_to_id": c2id,
    }
    meta_path = out / "benchmark_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()