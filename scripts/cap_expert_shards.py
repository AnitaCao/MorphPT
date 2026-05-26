#!/usr/bin/env python3
"""
Cap overrepresented fine classes in expert shards and write to new directory.

Usage:
  # Cancer: cap Breast and Lung to 20K
  python scripts/cap_expert_shards.py \
    --shards_dir prepared/splits_v3_seed1337/expert_Cancer/shards \
    --out_dir prepared/splits_v3_seed1337/expert_Cancer/shards_capped \
    --caps "Breast cancer cells:20000" "Lung cancer cells:20000"

  # Tissue_Structural: cap Epi and Fib to 60K
  python scripts/cap_expert_shards.py \
    --shards_dir prepared/splits_v3_seed1337/expert_Tissue_Structural/shards \
    --out_dir prepared/splits_v3_seed1337/expert_Tissue_Structural/shards_capped \
    --caps "Epithelial cells:60000" "Fibroblasts:60000"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_caps(cap_args):
    """Parse 'ClassName:N' strings into dict."""
    caps = {}
    for c in cap_args:
        name, val = c.rsplit(":", 1)
        caps[name.strip()] = int(val)
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--caps", nargs="+", required=True,
                    help='Per-class caps as "ClassName:N" pairs')
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    CAPS = parse_caps(args.caps)
    print(f"Caps: {CAPS}\n")

    # Load all shards
    shard_files = sorted(shards_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    print(f"Loaded {len(df):,} cells from {len(shard_files)} shards\n")

    # Apply caps
    print("Applying caps:")
    for label, cap in CAPS.items():
        mask = df["label"] == label
        n = mask.sum()
        if n > cap:
            drop = rng.choice(df.index[mask].to_numpy(), size=n - cap, replace=False)
            df = df.drop(drop).reset_index(drop=True)
            print(f"  {label:<30} {n:>8,} → {cap:>8,}")
        else:
            print(f"  {label:<30} {n:>8,}  (no cap needed)")

    print(f"\nAfter caps: {len(df):,} cells")

    # Write per-tissue shards
    print(f"\nWriting to {out_dir}/")
    for tissue, g in df.groupby("tissue"):
        g.to_parquet(out_dir / f"{tissue}.parquet", index=False)
        print(f"  {tissue:<50} {len(g):>8,}")

    # Final distribution
    print(f"\nFinal distribution:")
    for label, cnt in df["label"].value_counts().sort_values(ascending=False).items():
        print(f"  {label:<30} {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    print(f"  {'TOTAL':<30} {len(df):>8,}")


if __name__ == "__main__":
    main()