#!/usr/bin/env python3
"""
generate_split_layout.py
────────────────────────
Generates custom train/val/test split layouts for a prebuilt Visium HD cache.
Implements the "One Cache, Many Splits" decoupled architecture.

Reads base metadata (`meta.csv`) from the root cache directory, applies custom
tile-based holdout configurations (via random sampling or explicit lists),
computes leakage-free expression normalization statistics exclusively on the
training subset, and outputs a highly compact layout directory containing:
  splits.csv        lightweight mapping: mmap_idx, split
  expr_stats.npz    training-set normalization parameters (gene_mean, gene_std)
  split_stats.json  serialized exact tile lists and layout summary

Usage:
  python scripts/generate_split_layout.py --cache_dir cache_crc --layout_name default
  python scripts/generate_split_layout.py --cache_dir cache_crc --layout_name test_tiles_3_14 \
      --explicit_test_tiles 3,14 --explicit_val_tiles 0,2
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_parser():
    p = argparse.ArgumentParser(description="Generate a decoupled split layout for a Visium HD cache.")
    p.add_argument("--cache_dir",           type=str, required=True,
                   help="Path to the prebuilt root cache directory.")
    p.add_argument("--layout_name",         type=str, default="default",
                   help="Name of the layout folder to create inside splits/.")
    p.add_argument("--grid_size",           type=int, default=5,
                   help="Spatial grid size (used for buffer computation if cache_stats.json is missing).")
    p.add_argument("--min_cells",           type=int, default=100,
                   help="Min cells per tile to be eligible for random holdouts.")
    p.add_argument("--test_tiles",          type=int, default=4,
                   help="Number of random test tiles to sample.")
    p.add_argument("--val_tiles",           type=int, default=3,
                   help="Number of random validation tiles to sample.")
    p.add_argument("--explicit_test_tiles", type=str, default=None,
                   help="Comma-separated list of exact integer tile IDs to force as Test.")
    p.add_argument("--explicit_val_tiles",  type=str, default=None,
                   help="Comma-separated list of exact integer tile IDs to force as Val.")
    p.add_argument("--buffer_zone",         action="store_true", default=False,
                   help="Exclude adjacent tiles around test tiles to prevent spatial leakage.")
    p.add_argument("--seed",                type=int, default=42,
                   help="Random seed for tile holdout sampling.")
    return p


def main():
    args = get_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Root cache directory not found: {cache_dir}")

    layout_dir = cache_dir / "splits" / args.layout_name
    layout_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 65)
    logging.info(f"Generating Split Layout : '{args.layout_name}'")
    logging.info(f"Root Cache Dir          : {cache_dir}")
    logging.info(f"Output Layout Dir       : {layout_dir}")
    logging.info("=" * 65)

    # ── 1. Load root cache configuration and metadata ──
    meta_path = cache_dir / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Base metadata not found: {meta_path}")

    logging.info(f"Loading canonical metadata from {meta_path}...")
    df = pd.read_csv(meta_path)
    
    # Try reading cache_stats.json to inherit grid_size automatically
    stats_path = cache_dir / "cache_stats.json"
    grid_size = args.grid_size
    if stats_path.exists():
        try:
            cache_stats = json.loads(stats_path.read_text())
            grid_size = cache_stats.get("grid_size", args.grid_size)
            logging.info(f"Inherited grid_size={grid_size} from base cache configuration.")
        except Exception as e:
            logging.warning(f"Could not parse cache_stats.json: {e}")

    # ── 2. Determine Tile Sets ──
    tile_counts = df["tile_id"].value_counts()
    all_tiles   = sorted(tile_counts.index.values.tolist())
    
    test_set, val_set, buffer_set = set(), set(), set()
    eligible_tiles, ineligible_tiles = [], []

    # If explicit tiles are provided, bypass random selection entirely
    if args.explicit_test_tiles is not None or args.explicit_val_tiles is not None:
        logging.info("Using explicit user-defined tile configurations (bypassing random seed logic).")
        if args.explicit_test_tiles:
            test_set = set(int(t.strip()) for t in args.explicit_test_tiles.split(",") if t.strip())
        if args.explicit_val_tiles:
            val_set = set(int(t.strip()) for t in args.explicit_val_tiles.split(",") if t.strip())
            
        # Verify valid presence
        for t in test_set.union(val_set):
            if t not in all_tiles:
                logging.warning(f"Explicitly specified tile {t} has no mapped cells in meta.csv.")
                
        ineligible_tiles = [t for t in all_tiles if tile_counts[t] < args.min_cells]
    else:
        # Perform random sampling over eligible tiles
        eligible_tiles = tile_counts[tile_counts >= args.min_cells].index.values.tolist()
        ineligible_tiles = [t for t in all_tiles if t not in eligible_tiles]
        
        logging.info(f"Total mapped tiles : {len(all_tiles)}")
        logging.info(f"Eligible tiles     (≥{args.min_cells} cells): {len(eligible_tiles)}")
        logging.info(f"Ineligible tiles   (<{args.min_cells} cells): {len(ineligible_tiles)} → forced train")
        
        needed = args.test_tiles + args.val_tiles
        if len(eligible_tiles) < needed:
            raise ValueError(
                f"Only {len(eligible_tiles)} eligible tiles available, but requested {needed} "
                f"({args.test_tiles} test + {args.val_tiles} val). "
                f"Lower --min_cells or decrease requested holdout quantities."
            )
            
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(eligible_tiles, size=needed, replace=False).tolist()
        test_set = set(chosen[:args.test_tiles])
        val_set  = set(chosen[args.test_tiles:])

    # Optional buffer zone calculation
    if args.buffer_zone:
        logging.info("Computing single-tile adjacency buffer zone around Test tiles...")
        for tid in test_set:
            tx, ty = tid % grid_size, tid // grid_size
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ntx, nty = tx + dx, ty + dy
                    if 0 <= ntx < grid_size and 0 <= nty < grid_size:
                        nb = nty * grid_size + ntx
                        if nb in all_tiles and nb not in test_set and nb not in val_set:
                            buffer_set.add(nb)
        logging.info(f"Buffer tiles assigned to 'excluded': {len(buffer_set)}")

    train_set = set(all_tiles) - test_set - val_set - buffer_set

    # Map assignments to each cell
    def _assign_split(tid):
        if tid in test_set:   return "test"
        if tid in val_set:    return "val"
        if tid in buffer_set: return "excluded"
        return "train"

    df["split"] = df["tile_id"].map(_assign_split)
    
    # ── 3. Save compact splits.csv mapping ──
    splits_df = df[["mmap_idx", "split"]].copy()
    splits_csv_path = layout_dir / "splits.csv"
    splits_df.to_csv(splits_csv_path, index=False)
    logging.info(f"Saved highly lightweight layout splits mapping → {splits_csv_path} ({splits_csv_path.stat().st_size / 1e6:.2f} MB)")

    # ── 4. Compute strict leakage-free normalization parameters ──
    expr_path = cache_dir / "expr.npy"
    if not expr_path.exists():
        raise FileNotFoundError(f"Shared expression cache not found: {expr_path}")

    logging.info(f"Opening zero-copy shared expression cache: {expr_path} ...")
    expr_mmap = np.load(str(expr_path), mmap_mode="r")
    
    train_mask = df["split"].values == "train"
    n_train_cells = int(train_mask.sum())
    logging.info(f"Computing expression mean & std strictly over {n_train_cells:,} training cells...")
    
    if n_train_cells == 0:
        raise ValueError("Zero cells assigned to training split! Check your tile holdout logic.")

    # Load training rows into memory for fast vectorized stats computation
    X_train = expr_mmap[train_mask].astype(np.float32)
    gene_mean = X_train.mean(axis=0).astype(np.float32)
    gene_std  = np.clip(X_train.std(axis=0).astype(np.float32), 1e-5, None)
    
    stats_npz_path = layout_dir / "expr_stats.npz"
    np.savez(str(stats_npz_path), gene_mean=gene_mean, gene_std=gene_std)
    logging.info(f"Saved layout-specific training normalization stats → {stats_npz_path}")

    # ── 5. Serialize total reproducibility state descriptor ──
    counts = df["split"].value_counts().to_dict()
    
    split_stats = {
        "layout_name":      args.layout_name,
        "grid_size":        grid_size,
        "seed":             args.seed if args.explicit_test_tiles is None else None,
        "min_cells":        args.min_cells,
        "buffer_zone":      args.buffer_zone,
        "split_counts":     {k: int(v) for k, v in counts.items()},
        # Exact tile list serialization guarantees transparent restorability
        "train_tiles":      sorted(list(train_set)),
        "val_tiles":        sorted(list(val_set)),
        "test_tiles":       sorted(list(test_set)),
        "buffer_tiles":     sorted(list(buffer_set)),
        "ineligible_tiles": sorted(ineligible_tiles),
    }
    
    json_path = layout_dir / "split_stats.json"
    json_path.write_text(json.dumps(split_stats, indent=2))
    logging.info(f"Serialized complete tile layout JSON descriptor → {json_path}")

    logging.info("\n" + "=" * 65)
    logging.info(f"Split Layout '{args.layout_name}' generated successfully.")
    for s in ["train", "val", "test", "excluded"]:
        n = counts.get(s, 0)
        logging.info(f"  {s:<10}: {n:>7,} cells  ({n / len(df) * 100:.1f}%)")
    logging.info(f"  Test Tiles : {sorted(list(test_set))}")
    logging.info(f"  Val Tiles  : {sorted(list(val_set))}")
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
