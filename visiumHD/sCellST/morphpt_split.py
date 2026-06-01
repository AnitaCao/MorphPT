"""Reproduce MorphPT's random cell-level split EXACTLY.

So that competing methods (sCellST: frozen-RN50 / MoCo) train, evaluate, and
self-supervise on the *identical* train/val/test cells as MorphPT, the split
must be regenerated with the same RNG, the same index space, and the same
fractions used in visiumHD/train_lora_multi_probing.py:

    meta     = read cache_<ds>/meta.csv
    meta     = meta.merge(splits/<layout>/splits.csv, on='mmap_idx')   # if present
    all_meta = concat([meta[split==s] for s in (train, val, test)])    # layout order
    np.random.seed(seed)                  # legacy MT19937 (NOT default_rng/PCG64)
    idx = np.arange(n); np.random.shuffle(idx)
    train = idx[:0.70n] ; val = idx[0.70n:0.85n] ; test = idx[0.85n:]

The cell_id sets are identical between the MorphPT cache and the delivery meta,
so callers map the returned cell_ids onto delivery patches/expression by cell_id.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def morphpt_random_split_cellids(
    cache_dir,
    split_layout: str = 'default',
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Return (train_cell_ids, val_cell_ids, test_cell_ids) as str arrays,
    ordered exactly as MorphPT's shuffled split for the given seed."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / 'meta.csv'
    if not meta_path.exists():
        raise FileNotFoundError(f'MorphPT cache meta not found: {meta_path}')

    meta = pd.read_csv(meta_path)
    layout_dir = cache_dir / 'splits' / split_layout
    splits_csv = layout_dir / 'splits.csv'
    if splits_csv.exists():
        splits = pd.read_csv(splits_csv)
        meta = meta.merge(splits, on='mmap_idx', how='inner')
    elif 'split' not in meta.columns:
        meta['split'] = 'train'

    all_meta = pd.concat(
        [meta[meta['split'] == s] for s in ['train', 'val', 'test']],
        axis=0,
    ).reset_index(drop=True)

    n = len(all_meta)
    # Legacy global RNG, seeded immediately before the shuffle — matches MorphPT.
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]

    cid = all_meta['cell_id'].astype(str).to_numpy()
    return cid[tr], cid[va], cid[te]
