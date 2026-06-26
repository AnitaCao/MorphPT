"""
make_random_split.py
─────────────────────
Create random train/val/test splits for Visium HD datasets.
Saves to meta_random_split.csv alongside existing meta.csv (spatial split).
Also recomputes expr_stats_random.npz on new train cells.

Does NOT overwrite meta.csv or expr_stats.npz.

Usage:
  python make_random_split.py --dataset all
  python make_random_split.py --dataset lung
  python make_random_split.py --dataset crc lung pancreas
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
}


def make_random_split(dataset: str,
                      train_frac: float = 0.70,
                      val_frac:   float = 0.15,
                      seed:       int   = 42):
    cache = CACHE_DIRS[dataset]
    print(f'\n{"="*50}')
    print(f'Dataset: {dataset}  cache: {cache}')

    # Load existing meta
    meta = pd.read_csv(cache / 'meta.csv')
    n    = len(meta)
    print(f'  Total cells: {n:,}')
    print(f'  Original spatial split: {meta["split"].value_counts().to_dict()}')

    # Random permutation
    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    split_arr                       = np.array(['train'] * n, dtype=object)
    split_arr[perm[n_train:n_train + n_val]] = 'val'
    split_arr[perm[n_train + n_val:]]        = 'test'

    meta_rand         = meta.copy()
    meta_rand['split'] = split_arr

    counts = meta_rand['split'].value_counts().to_dict()
    print(f'  New random split:')
    print(f'    train: {counts.get("train", 0):,}  '
          f'({counts.get("train",0)/n*100:.1f}%)')
    print(f'    val  : {counts.get("val",   0):,}  '
          f'({counts.get("val",  0)/n*100:.1f}%)')
    print(f'    test : {counts.get("test",  0):,}  '
          f'({counts.get("test", 0)/n*100:.1f}%)')

    # Save new meta (do NOT overwrite original)
    out_meta = cache / 'meta_random_split.csv'
    meta_rand.to_csv(out_meta, index=False)
    print(f'  Saved → {out_meta}')

    # Recompute gene stats on new train cells
    expr       = np.load(str(cache / 'expr.npy'), mmap_mode='r')
    train_mask = meta_rand['split'].values == 'train'
    train_idx  = meta_rand[train_mask]['mmap_idx'].values
    X_train    = np.array(expr[train_idx], dtype=np.float32)

    gene_mean  = X_train.mean(axis=0).astype(np.float32)
    gene_std   = np.clip(X_train.std(axis=0).astype(np.float32), 1e-5, None)

    out_stats  = cache / 'expr_stats_random.npz'
    np.savez(str(out_stats), gene_mean=gene_mean, gene_std=gene_std)
    print(f'  Saved → {out_stats}')
    print(f'  Gene mean range: [{gene_mean.min():.3f}, {gene_mean.max():.3f}]')
    print(f'  Gene std  range: [{gene_std.min():.3f},  {gene_std.max():.3f}]')

    return meta_rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+',
                    default=['crc', 'lung', 'pancreas'],
                    choices=['crc', 'lung', 'pancreas', 'all'])
    ap.add_argument('--train_frac', type=float, default=0.70)
    ap.add_argument('--val_frac',   type=float, default=0.15)
    ap.add_argument('--seed',       type=int,   default=42)
    args = ap.parse_args()

    datasets = args.dataset
    if 'all' in datasets:
        datasets = ['crc', 'lung', 'pancreas']

    for ds in datasets:
        make_random_split(ds,
                          train_frac=args.train_frac,
                          val_frac=args.val_frac,
                          seed=args.seed)

    print(f'\n{"="*50}')
    print('Done. Files created:')
    for ds in datasets:
        cache = CACHE_DIRS[ds]
        print(f'  {cache}/meta_random_split.csv')
        print(f'  {cache}/expr_stats_random.npz')
    print('\nTo use random split in training, update VisiumHDPredictionDataset')
    print('to load meta_random_split.csv and expr_stats_random.npz.')


if __name__ == '__main__':
    main()