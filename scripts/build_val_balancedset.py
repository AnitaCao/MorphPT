#!/usr/bin/env python3
"""
Build a balanced validation subset from test_shards.
Each coarse class capped at --cap (default 30K), proportional per-slide sampling.

Usage:
  python build_val_balanced.py \
    --test_dir splits_v3_seed1337/test_shards \
    --out_dir  splits_v3_seed1337/val_balanced \
    --label_col coarse_label \
    --cap 30000 \
    --seed 1337
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="Directory of test shards")
    ap.add_argument("--out_dir", required=True, help="Output directory for balanced val")
    ap.add_argument("--label_col", default="coarse_label")
    ap.add_argument("--cap", type=int, default=30_000,
                    help="Max samples per coarse class")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load all test shards
    test_dir = Path(args.test_dir)
    shards = sorted(test_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"No .parquet files in {test_dir}")

    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    print(f"Loaded {len(df):,} cells from {len(shards)} shards")

    label_col = args.label_col
    cap = args.cap

    # Show raw distribution
    raw_counts = df[label_col].value_counts()
    print(f"\nRaw test distribution:")
    for cls, cnt in raw_counts.items():
        print(f"  {cls:<22} {cnt:>10,}")

    # Per-class proportional sampling
    sampled_parts = []
    for cls in sorted(raw_counts.index):
        cls_df = df[df[label_col] == cls]
        n_raw = len(cls_df)

        if n_raw <= cap:
            # Keep all
            sampled_parts.append(cls_df)
            print(f"  {cls:<22} keep all {n_raw:,}")
        else:
            # Proportional per-slide sampling
            rate = cap / n_raw
            slide_parts = []
            if "slide_id" in cls_df.columns:
                for sid, sdf in cls_df.groupby("slide_id"):
                    n_take = max(1, int(round(len(sdf) * rate)))
                    idx = rng.choice(len(sdf), size=n_take, replace=False)
                    slide_parts.append(sdf.iloc[idx])
                sampled = pd.concat(slide_parts, ignore_index=True)
            else:
                idx = rng.choice(n_raw, size=cap, replace=False)
                sampled = cls_df.iloc[idx]
            sampled_parts.append(sampled)
            print(f"  {cls:<22} {n_raw:,} → {len(sampled):,}")

    val_df = pd.concat(sampled_parts, ignore_index=True)

    # Shuffle
    val_df = val_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Show balanced distribution
    print(f"\nBalanced val: {len(val_df):,} cells")
    bal_counts = val_df[label_col].value_counts()
    for cls, cnt in bal_counts.items():
        print(f"  {cls:<22} {cnt:>10,}")

    # Save as single parquet
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "val_balanced.parquet"
    val_df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()