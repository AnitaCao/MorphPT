#!/usr/bin/env python3
"""
select_top_genes.py
───────────────────
Select top N genes by coverage from a dataset cache.
Saves a gene index file for use in per-gene training.

Usage:
  python select_top_genes.py --dataset crc --topn 500
  python select_top_genes.py --dataset crc --min_coverage 0.3
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='crc',
                   help='Dataset identifier (e.g., crc, lung, mouse_brain).')
    p.add_argument('--split_layout', type=str, default='default',
                   help='Custom layout folder name under splits/ (default: "default").')
    p.add_argument('--topn',    type=int, default=500,
                   help='Number of genes to select after filtering')
    p.add_argument('--min_coverage', type=float, default=0.1,
                   help='Minimum coverage threshold (default: 0.1).')
    p.add_argument('--method', type=str, default='variance',
                   choices=['variance', 'coverage'],
                   help='Metric to sort genes by after coverage filtering')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory (default: layout folder or cache_dir/per_gene)')
    p.add_argument('--split',   type=str, default='spatial',
                   choices=['random', 'spatial', 'all'],
                   help='Which split to use for metrics calculation. "all" uses entire dataset.')
    p.add_argument('--seed',    type=int, default=42,
                   help='Random seed identifier to include in the output filename.')
    return p.parse_args()


def main():
    args    = get_args()
    cache   = PROJECT / f"cache_{args.dataset}"
    if not cache.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache}")

    layout_dir = cache / "splits" / args.split_layout
    out_dir = Path(args.out_dir) if args.out_dir else (layout_dir if layout_dir.exists() else cache / 'per_gene')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gene list
    genes = (cache / 'gene_list.txt').read_text().splitlines()
    n_genes = len(genes)
    print(f'Dataset    : {args.dataset}')
    print(f'Layout     : {args.split_layout}')
    print(f'Total genes: {n_genes}')

    # Load expression and compute metrics
    meta_file = ('meta_random_split.csv' if args.split == 'random' else 'meta.csv')
    meta      = pd.read_csv(cache / meta_file)
    
    if layout_dir.exists():
        print(f"Joining layout splits mapping from {layout_dir / 'splits.csv'}")
        splits = pd.read_csv(layout_dir / "splits.csv")
        meta = meta.merge(splits, on="mmap_idx", how="inner")
    elif "split" not in meta.columns:
        meta["split"] = "train"
    
    if args.split == 'all':
        train_idx = meta['mmap_idx'].values
        split_label = 'allcells'
    else:
        train_idx = meta[meta['split'] == 'train']['mmap_idx'].values
        split_label = f'train_{args.split_layout}'

    # OPTIMIZATION: Sort indices for making the memmap disk read sequential!
    train_idx = np.sort(train_idx)

    print(f'Computing coverage & variance on {len(train_idx):,} cells (Split: {args.split})...')
    expr = np.load(str(cache / 'expr.npy'), mmap_mode='r')

    # Batch compute stats to avoid OOM
    batch_size    = 5000
    sum_expressed = np.zeros(n_genes, dtype=np.float64)
    sum_val       = np.zeros(n_genes, dtype=np.float64)
    sum_sq_val    = np.zeros(n_genes, dtype=np.float64)

    for i in range(0, len(train_idx), batch_size):
        batch = expr[train_idx[i:i+batch_size]]
        sum_expressed += (batch > 0).sum(axis=0)
        sum_val       += batch.sum(axis=0)
        sum_sq_val    += (batch**2).sum(axis=0)

    n = len(train_idx)
    coverage = sum_expressed / n
    mean     = sum_val / n
    variance = (sum_sq_val / n) - (mean**2)
    variance = np.maximum(variance, 0) # Precision safety

    # 1. Filter by coverage
    mask = coverage >= args.min_coverage
    pass_indices = np.where(mask)[0]
    
    print(f'Genes passing {args.min_coverage*100:.0f}% coverage: {len(pass_indices)} / {n_genes}')

    # 2. Sort by chosen method
    metric_values = variance if args.method == 'variance' else coverage
    filtered_metrics = metric_values[pass_indices]
    
    # Sort indices of the filtered list
    sort_in_filtered = np.argsort(filtered_metrics)[::-1]
    
    # Take top N
    selected_in_filtered = sort_in_filtered[:args.topn]
    top_idx = pass_indices[selected_in_filtered]

    out_csv_name = f'top{len(top_idx)}_{args.method}_mincov{args.min_coverage}_{split_label}_seed{args.seed}.csv'

    # Build result dataframe
    # FIX: Use len(top_idx) to prevent crash if genes list is shorter than topn.
    df = pd.DataFrame({
        'gene_idx':    top_idx,
        'gene_name':   [genes[i] for i in top_idx],
        'variance':    variance[top_idx],
        'coverage':    coverage[top_idx],
        'rank':        np.arange(1, len(top_idx) + 1),
    })

    # Save
    out_csv = out_dir / out_csv_name
    df.to_csv(out_csv, index=False)

    print(f'\nTop {len(top_idx)} genes selected:')
    print(f'  Coverage range: [{df["coverage"].min():.3f}, '
          f'{df["coverage"].max():.3f}]')
    print(f'  Coverage mean : {df["coverage"].mean():.3f}')
    print(f'\nCoverage distribution of selected genes:')
    
    # FIX: Shifted the bins slightly so that '10-15%' actually represents 10-15%.
    bins   = [0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    labels = ['10-15%', '15-20%', '20-30%', '30-50%', '>50%']
    for i in range(len(labels)):
        n = ((df['coverage'] >= bins[i]) & (df['coverage'] < bins[i+1])).sum()
        print(f'  {labels[i]}: {n} genes')

    print(f'\nTop 10 genes by {args.method}:')
    print(df.head(10)[['rank', 'gene_name', 'variance', 'coverage']].to_string(index=False))
    print(f'\nSaved → {out_csv}')
    print(f'Total jobs to run: {len(df)}')


if __name__ == '__main__':
    main()
