#!/usr/bin/env python3
"""
retile.py
─────────
Re-assign tile_id by binning x_centroid, y_centroid onto a finer regular grid.

Use case: when a dataset has too few tiles for a useful spatial split (e.g. pancreas
has 14 tiles, where the 4-test-tile policy loses 32% of cells and forces test tiles
into the periphery).

Properties:
- All cells are kept (no drop). mmap_idx, cell_id, split column are preserved.
- expr.npy alignment is unchanged → no need to recompute it.
- Gene selection (select_top_genes.py) does NOT need to be re-run, because it uses
  meta["split"] (unchanged) rather than tile_id.
- Original meta.csv is backed up to meta_original.csv on first run.
- "Main" tiles (>= --min_cells) are renumbered 0..N-1; sparse "fragment" tiles get
  IDs at the end of the list. Pick test tiles from the main group.

Usage:
  python retile.py --dataset pancreas --target_n 25
  python retile.py --dataset pancreas --nx 6 --ny 5     # explicit grid
  python retile.py --dataset pancreas --target_n 25 --dry_run

After retiling: pick 3 main tile IDs from the printed table, then run training
with `--test_tiles "T1,T2,T3" --n_test_tiles 3`.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   type=str, default='pancreas')
    p.add_argument('--target_n',  type=int, default=25,
                   help='Target number of "main" tiles (>= --min_cells)')
    p.add_argument('--min_cells', type=int, default=200,
                   help='Tiles with fewer cells are sorted to the end of the ID list as "fragments"')
    p.add_argument('--nx',        type=int, default=None,
                   help='Force grid width (overrides target_n auto-search)')
    p.add_argument('--ny',        type=int, default=None,
                   help='Force grid height (overrides target_n auto-search)')
    p.add_argument('--out_meta',  type=str, default='meta.csv',
                   help='Filename to write the retiled meta to (within cache_dir)')
    p.add_argument('--dry_run',   action='store_true',
                   help='Print the plan without writing files')
    return p.parse_args()


def assign_grid(x, y, nx, ny):
    x_edges = np.linspace(x.min(), x.max() + 1e-6, nx + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-6, ny + 1)
    xb = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    yb = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    return xb, yb


def find_grid(x, y, target_n, min_cells, max_dim=12):
    """Search nx, ny in [2..max_dim] for the (nx, ny) that yields a count of
    main tiles closest to target_n. Tie-break by total grid size also closest to target_n."""
    best = None
    for nx in range(2, max_dim + 1):
        for ny in range(2, max_dim + 1):
            xb, yb = assign_grid(x, y, nx, ny)
            keys   = xb * ny + yb
            counts = pd.Series(keys).value_counts()
            n_main = int((counts >= min_cells).sum())
            cand   = (abs(n_main - target_n), abs(nx * ny - target_n), nx, ny, n_main)
            if best is None or cand < best:
                best = cand
    return best[2], best[3], best[4]


def main():
    args = get_args()
    cache       = PROJECT / f'cache_{args.dataset}'
    meta_path   = cache / 'meta.csv'
    backup_path = cache / 'meta_original.csv'
    out_path    = cache / args.out_meta

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = pd.read_csv(meta_path)
    print(f'Dataset       : {args.dataset}')
    print(f'Cells         : {len(meta):,}')
    print(f'Original tiles: {meta["tile_id"].nunique()}')

    x = meta['x_centroid'].values.astype(float)
    y = meta['y_centroid'].values.astype(float)

    # Decide grid
    if args.nx is not None and args.ny is not None:
        nx, ny = args.nx, args.ny
        xb, yb = assign_grid(x, y, nx, ny)
        keys   = xb * ny + yb
        n_main = int((pd.Series(keys).value_counts() >= args.min_cells).sum())
    else:
        nx, ny, n_main = find_grid(x, y, args.target_n, args.min_cells)
        xb, yb = assign_grid(x, y, nx, ny)

    print(f'Grid          : {nx} × {ny} = {nx * ny} bins')
    print(f'Main tiles    : {n_main} (>= {args.min_cells} cells)')

    # Apply grid and renumber: main first, fragments last
    raw    = xb * ny + yb
    counts = pd.Series(raw).value_counts()
    main_ids     = sorted(counts[counts >= args.min_cells].index.tolist())
    fragment_ids = sorted(counts[counts <  args.min_cells].index.tolist())

    tile_map = {}
    for i, t in enumerate(main_ids):
        tile_map[t] = i
    for j, t in enumerate(fragment_ids):
        tile_map[t] = len(main_ids) + j

    new_tile_id = np.array([tile_map[t] for t in raw], dtype=np.int64)

    new_meta = meta.copy()
    new_meta['tile_id_old'] = new_meta['tile_id']
    new_meta['tile_id']     = new_tile_id
    new_meta['x_bin']       = xb
    new_meta['y_bin']       = yb

    # Report
    print(f'\nNew tile distribution (sorted by tile_id):')
    print(f'{"tile_id":>8} {"n_cells":>10} {"x_mean":>10} {"y_mean":>10}   group')
    print('-' * 60)
    summary = (new_meta.groupby('tile_id')
                       .agg(n_cells=('cell_id', 'size'),
                            x_mean=('x_centroid', 'mean'),
                            y_mean=('y_centroid', 'mean'))
                       .reset_index()
                       .sort_values('tile_id'))
    for _, row in summary.iterrows():
        marker = 'main' if row['tile_id'] < len(main_ids) else 'fragment'
        print(f'{int(row["tile_id"]):>8d} {int(row["n_cells"]):>10,} '
              f'{row["x_mean"]:>10.0f} {row["y_mean"]:>10.0f}   {marker}')

    if args.dry_run:
        print('\n[DRY RUN] No files written.')
        return

    if not backup_path.exists():
        shutil.copy(meta_path, backup_path)
        print(f'\nBacked up original → {backup_path.name}')
    else:
        print(f'\nBackup {backup_path.name} already exists; not overwriting.')

    new_meta.to_csv(out_path, index=False)
    print(f'Wrote new meta → {out_path}')
    print(f'\nNext: pick 3 main tile IDs from the table above, then run')
    print(f'  python train.py --dataset {args.dataset} \\')
    print(f'      --top_csv cache_{args.dataset}/per_gene/top400_variance_mincov0.1_train_seed42.csv \\')
    print(f'      --test_tiles "T1,T2,T3" --n_test_tiles 3 --n_seeds 3')


if __name__ == '__main__':
    main()
