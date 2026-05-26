#!/usr/bin/env python3
"""
precompute_gene_split_stats_v2.py
────────────────────────────────
FIXED: Properly reorders meta to match training script's ordering,
and uses mmap_idx for expression matrix indexing.

Training script reorders all_meta by split (train→val→test), so when 
it saves test_idx, those indices refer to positions in the REORDERED 
meta. Our original analysis used unordered meta → wrong cells!

Fix: Always reorder meta the same way before applying split indices.

Output: cache_crc/per_gene_split_stats_seed_42.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')
CACHE_DIR = PROJECT / 'cache_crc'

EXP_DIR = PROJECT / 'experiments' / 'lora_probing_crc_multi_10.0x_mse'
SPLITS_FILE = EXP_DIR / 'splits_seed_42.npz'

GENE_STATS_FILE = CACHE_DIR / 'per_gene' / 'top400_variance_mincov0.1.csv'
OUTPUT_FILE = CACHE_DIR / 'per_gene_split_stats_seed_42.csv'


def main():
    print('=' * 70)
    print(' Precompute per-gene split statistics (v2, FIXED ordering)')
    print('=' * 70)

    # Load splits
    t0 = time.time()
    splits = np.load(SPLITS_FILE, allow_pickle=True)
    train_idx = splits['train_idx']
    val_idx   = splits['val_idx']
    test_idx  = splits['test_idx']
    test_tiles = splits['test_tiles']

    print(f'\nSplits loaded ({time.time()-t0:.1f}s):')
    print(f'  Train: {len(train_idx):>7,} cells')
    print(f'  Val:   {len(val_idx):>7,} cells')
    print(f'  Test:  {len(test_idx):>7,} cells')
    print(f'  Test tiles: {test_tiles.tolist()}')

    # =========================================================
    # CRITICAL: Reorder meta exactly how training script did it
    # =========================================================
    meta = pd.read_csv(CACHE_DIR / 'meta.csv')
    
    # Reorder by split (train → val → test)
    all_meta_list = []
    for spl in ['train', 'val', 'test']:
        all_meta_list.append(meta[meta['split'] == spl].copy())
    all_meta = pd.concat(all_meta_list, axis=0).reset_index(drop=True)
    
    print(f'\nMeta reordered by split:')
    for spl in ['train', 'val', 'test']:
        n = (all_meta['split'] == spl).sum()
        print(f'  {spl}: {n:,} rows')
    print(f'  Total: {len(all_meta):,} rows')

    # Sanity check
    assert len(all_meta) == len(meta[meta['split'] != 'excluded']), \
        'Meta length mismatch after reordering'

    # Verify test tiles align with test_meta
    test_meta = all_meta.iloc[test_idx]
    test_tile_dist = test_meta['tile_id'].value_counts().sort_index()
    
    print(f'\nVerification — test tile distribution:')
    print(f'  Expected tiles: {test_tiles.tolist()}')
    print(f'  Actual tiles in test_meta: {sorted(test_meta["tile_id"].unique())}')
    
    if set(test_meta['tile_id'].unique()) != set(test_tiles.tolist()):
        print(f'  ⚠️ STILL MISALIGNED! Expected only {test_tiles} but got {sorted(test_meta["tile_id"].unique())}')
        print(f'  This means the reordering fix did not work.')
        return
    else:
        print(f'  ✓ Test tiles match!')
        print(f'\n  Cells per test tile:')
        for tile_id, n in test_tile_dist.items():
            print(f'    Tile {tile_id:>3}: {n:>5,} cells')

    # Load gene stats
    df_genes = pd.read_csv(GENE_STATS_FILE)
    gene_indices = df_genes['gene_idx'].values.astype(int)
    gene_names = df_genes['gene_name'].values
    n_genes = len(gene_indices)
    print(f'\nAnalyzing {n_genes} genes')

    # Load expression matrix
    # expr.npy row i corresponds to cell with mmap_idx = i
    t0 = time.time()
    print(f'\nLoading expression matrix into RAM...')
    expr_file = CACHE_DIR / 'expr.npy'
    expr_full = np.load(expr_file)
    print(f'  Loaded shape: {expr_full.shape} in {time.time()-t0:.1f}s')

    # Subset to 400 genes FIRST to avoid memory explosion
    expr_400 = expr_full[:, gene_indices]
    del expr_full
    print(f'  Subset to {n_genes} genes. shape: {expr_400.shape}')

    # =========================================================
    # CRITICAL: Use mmap_idx to correctly map reordered rows 
    # to expression matrix rows
    # =========================================================
    if 'mmap_idx' in all_meta.columns:
        print(f'\nUsing mmap_idx for correct expression matrix indexing')
        # Map split indices (into all_meta) → mmap_idx (into expr)
        train_mmap = all_meta.iloc[train_idx]['mmap_idx'].values
        val_mmap   = all_meta.iloc[val_idx]['mmap_idx'].values
        test_mmap  = all_meta.iloc[test_idx]['mmap_idx'].values
        
        train_expr = expr_400[train_mmap]
        val_expr   = expr_400[val_mmap]
        test_expr  = expr_400[test_mmap]
    else:
        print(f'\nWARNING: No mmap_idx column found. Assuming direct alignment.')
        train_expr = expr_400[train_idx]
        val_expr   = expr_400[val_idx]
        test_expr  = expr_400[test_idx]

    print(f'  Train expr subset: {train_expr.shape}')
    print(f'  Val expr subset:   {val_expr.shape}')
    print(f'  Test expr subset:  {test_expr.shape}')

    # Compute per-split counts
    t0 = time.time()
    print(f'\nComputing per-split non-zero counts...')
    train_nonzero = (train_expr > 0).sum(axis=0)
    val_nonzero   = (val_expr > 0).sum(axis=0)
    test_nonzero  = (test_expr > 0).sum(axis=0)
    print(f'  Done in {time.time()-t0:.1f}s')

    # Per-tile stats
    t0 = time.time()
    print(f'\nComputing per-tile coverage...')
    
    tile_stats = {}
    for tile_id in test_tiles:
        # Boolean mask over test rows
        tile_mask_in_test = (test_meta['tile_id'] == int(tile_id)).values
        n_tile_cells = int(tile_mask_in_test.sum())
        
        if n_tile_cells == 0:
            print(f'  Tile {tile_id}: NO CELLS — skipping')
            tile_stats[int(tile_id)] = {
                'n_cells': 0,
                'nonzero': np.zeros(n_genes, dtype=int),
                'coverage': np.zeros(n_genes),
            }
            continue
        
        tile_expr = test_expr[tile_mask_in_test]
        tile_nonzero = (tile_expr > 0).sum(axis=0)
        tile_stats[int(tile_id)] = {
            'n_cells': n_tile_cells,
            'nonzero': tile_nonzero,
            'coverage': tile_nonzero / n_tile_cells,
        }
        print(f'  Tile {tile_id:>3}: {n_tile_cells:>6,} cells')
    
    print(f'  Done in {time.time()-t0:.1f}s')

    # Build output DataFrame
    df_out = pd.DataFrame({
        'gene_idx':       gene_indices,
        'gene_name':      gene_names,
        'n_train_cells':  len(train_idx),
        'n_val_cells':    len(val_idx),
        'n_test_cells':   len(test_idx),
        'train_nonzero':  train_nonzero,
        'val_nonzero':    val_nonzero,
        'test_nonzero':   test_nonzero,
        'train_coverage': train_nonzero / len(train_idx),
        'val_coverage':   val_nonzero   / len(val_idx),
        'test_coverage':  test_nonzero  / len(test_idx),
    })

    # Per-tile
    for tile_id in test_tiles:
        tid = int(tile_id)
        df_out[f'tile_{tid}_nonzero']  = tile_stats[tid]['nonzero']
        df_out[f'tile_{tid}_coverage'] = tile_stats[tid]['coverage']
        df_out[f'tile_{tid}_n_cells']  = tile_stats[tid]['n_cells']

    tile_cov_cols = [f'tile_{int(t)}_coverage' for t in test_tiles]
    df_out['tile_coverage_mean'] = df_out[tile_cov_cols].mean(axis=1)
    df_out['tile_coverage_std']  = df_out[tile_cov_cols].std(axis=1)
    df_out['tile_coverage_cv']   = df_out['tile_coverage_std'] / (
        df_out['tile_coverage_mean'] + 0.001)

    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f'\n{"=" * 70}')
    print(f' Saved: {OUTPUT_FILE}')
    print(f' Shape: {df_out.shape}')
    print(f'{"=" * 70}')

    print(f'\nSummary:')
    for col in ['train_nonzero', 'train_coverage', 'test_coverage']:
        s = df_out[col]
        print(f'  {col:<20}: mean={s.mean():>10.2f}  '
              f'min={s.min():>10.2f}  max={s.max():>10.2f}')


if __name__ == '__main__':
    main()