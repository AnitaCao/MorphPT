#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from morphpt_split import morphpt_random_split_cellids  # script-dir import


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build cell_SSL-compatible H5 files from delivery patch PNGs (training subset only by default).')
    p.add_argument('--data_root', type=str, default='/hpc/group/jilab/boxuan/visiumHD', help='Delivery root with <dataset>/meta and image folders')
    p.add_argument('--output_root', type=str, default='/hpc/group/jilab/tc459/sCellST/hest_data', help='Output root used by cell_SSL, contains cell_images/')
    p.add_argument('--datasets', type=str, default='mouse_brain,mouse_embryo,mouse_intestine,mouse_kidney', help='Comma-separated dataset names')
    p.add_argument('--scale', type=str, default='10.0', help='Patch scale in meta folder (e.g. 10.0)')

    # Split args MUST stay in sync with extract_scellst_embeddings_from_delivery.py
    # so the SSL training subset matches the supervised train split exactly.
    p.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'all'], help='Which split to write to the H5 (SSL must use train only)')
    p.add_argument('--seed', type=int, default=42, help='Split seed; must match MorphPT extraction seed')
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--morphpt_cache_root', type=str, default='/hpc/group/jilab/tc459/MorphPT', help='Root containing cache_<dataset>; used to reproduce MorphPT split by cell_id')
    p.add_argument('--split_layout', type=str, default='default', help='Layout under cache_<dataset>/splits/ ordering cells before the MorphPT shuffle')

    p.add_argument('--max_cells_per_dataset', type=int, default=0, help='If >0, randomly subsample this many cells per dataset (applied after split)')
    p.add_argument('--subsample_seed', type=int, default=1234, help='Seed for the optional max_cells subsample (kept separate from split seed)')
    p.add_argument('--overwrite', action='store_true', default=False)
    return p.parse_args()


def resolve_patch_path(dataset_root: Path, raw_img_path: str) -> Path:
    p = Path(str(raw_img_path))
    if p.is_absolute():
        return p
    return dataset_root / p


def filter_meta_to_expr(meta: pd.DataFrame, cells_path: Path) -> pd.DataFrame:
    # Mirror the extractor: align meta rows to expression cells and drop any
    # cell_id missing from cells.txt, preserving row order before the split.
    cells = np.loadtxt(cells_path, dtype=str)
    cell_to_col = {c: i for i, c in enumerate(cells)}
    meta = meta.copy()
    meta['col_idx'] = meta['cell_id'].map(cell_to_col)
    meta = meta[meta['col_idx'].notna()].copy()
    return meta


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    out_cell_images = out_root / 'cell_images'
    out_cell_images.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    if not datasets:
        raise ValueError('No datasets provided.')

    for ds in datasets:
        ds_root = data_root / ds
        meta_path = ds_root / 'meta' / f'{args.scale}x' / f'{ds}.csv'
        if not meta_path.exists():
            raise FileNotFoundError(f'Meta file not found: {meta_path}')

        df = pd.read_csv(meta_path)
        if 'raw_img_path' not in df.columns:
            raise ValueError(f'Missing raw_img_path column in {meta_path}')

        # Align to expression cells exactly as the supervised extractor does, then
        # reproduce the same random split and keep only the requested subset.
        cells_path = ds_root / 'expr' / 'cells.txt'
        if not cells_path.exists():
            raise FileNotFoundError(f'cells.txt not found (needed to match MorphPT split): {cells_path}')
        df = filter_meta_to_expr(df, cells_path).reset_index(drop=True)

        n = len(df)

        # Reproduce MorphPT's random split exactly and select cells by cell_id,
        # so the SSL train subset == MorphPT's train cells (no val/test leakage).
        cache_dir = Path(args.morphpt_cache_root) / f'cache_{ds}'
        tr_cells, va_cells, te_cells = morphpt_random_split_cellids(
            cache_dir, split_layout=args.split_layout, seed=args.seed,
            train_frac=args.train_frac, val_frac=args.val_frac,
        )
        cellid_to_pos = {c: i for i, c in enumerate(df['cell_id'].astype(str))}

        def _to_positions(cell_ids):
            pos = [cellid_to_pos[c] for c in cell_ids.astype(str) if c in cellid_to_pos]
            return np.asarray(pos, dtype=int)

        split_map = {
            'train': _to_positions(tr_cells),
            'val': _to_positions(va_cells),
            'test': _to_positions(te_cells),
        }

        if args.split == 'all':
            sel = np.arange(n)
        else:
            sel = np.sort(split_map[args.split])
        df = df.iloc[sel].reset_index(drop=True)
        print(f'[{ds}] total={n} | split={args.split} -> {len(df)} cells '
              f"(train/val/test = {len(split_map['train'])}/{len(split_map['val'])}/{len(split_map['test'])})")

        if args.max_cells_per_dataset > 0 and len(df) > args.max_cells_per_dataset:
            sub_rng = np.random.default_rng(args.subsample_seed)
            keep = sub_rng.choice(len(df), args.max_cells_per_dataset, replace=False)
            df = df.iloc[np.sort(keep)].reset_index(drop=True)
            print(f'[{ds}] subsampled to {len(df)} cells (seed={args.subsample_seed})')

        abs_paths = [resolve_patch_path(ds_root, p) for p in df['raw_img_path'].astype(str)]
        if not abs_paths:
            raise ValueError(f'No patch paths found for dataset {ds}')

        h5_path = out_cell_images / f'{ds}.h5'
        if h5_path.exists() and not args.overwrite:
            print(f'[skip] {h5_path} exists (use --overwrite to rebuild)')
            continue

        first = np.array(Image.open(abs_paths[0]).convert('RGB'), dtype=np.uint8)
        h, w, c = first.shape
        if c != 3:
            raise ValueError(f'Expected RGB image, got shape {first.shape} for {abs_paths[0]}')

        print(f'Building {h5_path} from {len(abs_paths)} patches of size {h}x{w}...')
        with h5py.File(h5_path, 'w') as f:
            d_img = f.create_dataset(
                'img',
                shape=(len(abs_paths), h, w, 3),
                dtype=np.uint8,
                # Per-image chunks: writing one image at a time must only (de)compress
                # its own ~345KB chunk, not a 512-image (~177MB) chunk every write.
                # Also gives fast random-access single-cell reads during MoCo training.
                chunks=(1, h, w, 3),
                compression='gzip',
                compression_opts=1,
            )
            d_cell = f.create_dataset('cell_id', shape=(len(abs_paths),), dtype=h5py.string_dtype('utf-8'))

            # Record provenance so the SSL H5 can be traced back to its split.
            f.attrs['split'] = args.split
            f.attrs['split_seed'] = args.seed
            f.attrs['train_frac'] = args.train_frac
            f.attrs['val_frac'] = args.val_frac
            f.attrs['n_total_cells'] = n

            for i, p in enumerate(tqdm(abs_paths, desc=ds)):
                arr = np.array(Image.open(p).convert('RGB'), dtype=np.uint8)
                if arr.shape != (h, w, 3):
                    raise ValueError(
                        f'Inconsistent patch size in {ds}: expected {(h, w, 3)}, got {arr.shape} at {p}'
                    )
                d_img[i] = arr
                d_cell[i] = str(df.iloc[i]['cell_id']) if 'cell_id' in df.columns else str(i)

        print(f'[ok] wrote {h5_path}')

    print('Done.')


if __name__ == '__main__':
    main()
