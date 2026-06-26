#!/usr/bin/env python3
"""
select_top_genes_fixed.py
─────────────────────
Select top N genes by coverage from a dataset cache for a predefined set of tiles.

The original script selects genes across the whole dataset. This version adds a
`--tiles` argument that defaults to a hard‑coded list per dataset:

- test:      [3]
- crc:       [3, 17, 23]
- lung:      [7, 11, 18]
- pancreas:  1‑25 (25 tiles)

Only the specified tiles are used when computing coverage and variance.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(os.environ.get("MORPHPT_ROOT", Path(__file__).resolve().parents[1]))  # repo root; override with MORPHPT_ROOT to point at your data/cache

CACHE_DIRS = {
    'crc':      PROJECT / 'cache_crc',
    'lung':     PROJECT / 'cache_lung',
    'pancreas': PROJECT / 'cache_pancreas',
    'test':     PROJECT / 'cache_test',  # assuming a test cache exists
}

# Pre‑defined tile selections per dataset
DEFAULT_TILES = {
    'test':      [3],
    'crc':       [3, 17, 23],
    'lung':      [7, 11, 18],
    'pancreas':  list(range(1, 26)),  # 25 tiles
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='crc',
                   choices=['crc', 'lung', 'pancreas', 'test'])
    p.add_argument('--topn',    type=int, default=500,
                   help='Number of genes to select after filtering')
    p.add_argument('--min_coverage', type=float, default=0.1,
                   help='Minimum coverage threshold (default: 0.1).')
    p.add_argument('--method', type=str, default='variance',
                   choices=['variance', 'coverage'],
                   help='Metric to sort genes by after coverage filtering')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory (default: cache_dir/per_gene)')
    p.add_argument('--split',   type=str, default='spatial',
                   choices=['random', 'spatial', 'all'],
                   help='Which split to use for metrics calculation. "all" uses entire dataset.')
    p.add_argument('--seed',    type=int, default=42,
                   help='Random seed identifier to include in the output filename.')
    p.add_argument('--tiles',   type=int, nargs='*', default=None,
                   help='Explicit list of tile IDs to use (overrides defaults)')
    return p.parse_args()


def main():
    args = get_args()
    cache = CACHE_DIRS[args.dataset]
    out_dir = Path(args.out_dir) if args.out_dir else cache / 'per_gene_fixed'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine test tiles (provided) and train tiles (all others)
    test_tiles = args.tiles if args.tiles is not None else DEFAULT_TILES.get(args.dataset, [])
    if not test_tiles:
        raise ValueError(f'No test tile list defined for dataset {args.dataset}')
    print(f'Dataset   : {args.dataset}')
    print(f'Test tiles: {test_tiles}')

    # Load gene list (same for all tiles)
    genes = (cache / 'gene_list.txt').read_text().splitlines()
    n_genes = len(genes)
    print(f'Total genes: {n_genes}')

    # Load metadata to determine all tile ids
    meta_file = ('meta_random_split.csv' if args.split == 'random'
                 else 'meta.csv')
    meta = pd.read_csv(cache / meta_file)

    # Identify all available tiles for this dataset
    all_tiles = set(meta['tile_id'].unique())
    train_tiles = list(all_tiles - set(test_tiles))
    if not train_tiles:
        raise ValueError('No train tiles available after excluding test tiles.')
    print(f'Train tiles (used for coverage/variance): {train_tiles}')

    # Determine cell indices for training based on selected train tiles
    tile_mask = meta['tile_id'].isin(train_tiles)
    if args.split != 'all':
        tile_mask &= meta['split'] == 'train'
    train_idx = meta.loc[tile_mask, 'mmap_idx'].values
    split_label = f"tiles_{'_'.join(map(str, train_tiles))}"
    if args.split != 'all':
        split_label = f"train_{split_label}"
    else:
        split_label = f"all_{split_label}"

    # Load expression matrix – assumed to contain all cells from all tiles.
    # We will restrict the index list to cells belonging to the selected tiles.
    # Duplicate tile selection logic removed; train_idx and split_label are already set based on train_tiles above.

    # Sort indices for sequential memmap reads
    train_idx = np.sort(train_idx)
    print(f'Computing coverage & variance on {len(train_idx):,} cells (train tiles {train_tiles})...')

    expr = np.load(str(cache / 'expr.npy'), mmap_mode='r')

    # Batch compute stats to avoid OOM
    batch_size = 5000
    sum_expressed = np.zeros(n_genes, dtype=np.float64)
    sum_val = np.zeros(n_genes, dtype=np.float64)
    sum_sq_val = np.zeros(n_genes, dtype=np.float64)

    for i in range(0, len(train_idx), batch_size):
        batch = expr[train_idx[i:i+batch_size]]
        sum_expressed += (batch > 0).sum(axis=0)
        sum_val += batch.sum(axis=0)
        sum_sq_val += (batch ** 2).sum(axis=0)

    n = len(train_idx)
    coverage = sum_expressed / n
    mean = sum_val / n
    variance = (sum_sq_val / n) - (mean ** 2)
    variance = np.maximum(variance, 0)  # precision safety

    # 1. Filter by coverage
    mask = coverage >= args.min_coverage
    pass_indices = np.where(mask)[0]
    print(f'Genes passing {args.min_coverage*100:.0f}% coverage: {len(pass_indices)} / {n_genes}')

    # 2. Sort by chosen method
    metric_values = variance if args.method == 'variance' else coverage
    filtered_metrics = metric_values[pass_indices]
    sort_in_filtered = np.argsort(filtered_metrics)[::-1]
    selected_in_filtered = sort_in_filtered[:args.topn]
    top_idx = pass_indices[selected_in_filtered]

    out_csv_name = f'top{len(top_idx)}_{args.method}_mincov{args.min_coverage}_{split_label}_seed{args.seed}.csv'
    df = pd.DataFrame({
        'gene_idx':   top_idx,
        'gene_name':  [genes[i] for i in top_idx],
        'variance':   variance[top_idx],
        'coverage':   coverage[top_idx],
        'rank':       np.arange(1, len(top_idx) + 1),
    })

    out_csv = out_dir / out_csv_name
    df.to_csv(out_csv, index=False)

    print('\nTop genes selected:')
    print(df.head(10)[['rank', 'gene_name', 'variance', 'coverage']].to_string(index=False))
    print(f'\nSaved → {out_csv}')
    print(f'Total jobs to run: {len(df)}')


if __name__ == '__main__':
    main()
