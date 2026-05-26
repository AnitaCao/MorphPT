#!/usr/bin/env python3
"""Create a balanced parquet subset for representation analysis."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--label_col", default="coarse_label")
    ap.add_argument("--max_per_class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    if args.label_col not in df.columns:
        raise RuntimeError(
            f"Missing label_col={args.label_col!r}. Columns: {list(df.columns)}"
        )

    parts = []
    input_counts = df[args.label_col].value_counts().sort_index()
    for label, grp in df.groupby(args.label_col, sort=True):
        n = min(len(grp), args.max_per_class)
        parts.append(grp.sample(n=n, random_state=args.seed, replace=False))

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sampled_counts = out[args.label_col].value_counts().sort_index()
    expected_total = int(input_counts.clip(upper=args.max_per_class).sum())
    full_cap_total = int(len(input_counts) * args.max_per_class)

    if len(out) != expected_total:
        raise RuntimeError(
            f"Sample count mismatch: got {len(out):,}, expected {expected_total:,}"
        )

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"Input : {args.in_parquet} ({len(df):,} rows)")
    print(f"Output: {out_path} ({len(out):,} rows)")
    print(f"Classes: {len(sampled_counts)}")
    print(f"Expected total with available cells: {expected_total:,}")
    print(f"Full-cap total ({len(input_counts)} classes x {args.max_per_class}): {full_cap_total:,}")
    if expected_total != full_cap_total:
        short = input_counts[input_counts < args.max_per_class]
        print("Classes below requested cap:")
        print(short.to_string())
    print(sampled_counts.to_string())


if __name__ == "__main__":
    main()
