#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Extract sCellST embeddings directly from delivery data (no MorphPT cache).')
    p.add_argument('--dataset', type=str, required=True, help='Dataset name, e.g. mouse_brain')
    p.add_argument('--data_root', type=str, default='/hpc/group/jilab/boxuan/visiumHD', help='Root containing <dataset>/meta and <dataset>/expr')
    p.add_argument('--output_root', type=str, default='/hpc/group/jilab/tc459/MorphPT/prepared/visiumHD_delivery', help='Writable root for extracted split tensors and metadata')
    p.add_argument('--scale', type=str, default='10.0', help='Context scale for patches, e.g. 2.5 or 10.0')

    p.add_argument('--split_mode', type=str, default='random', choices=['random'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--val_frac', type=float, default=0.15)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)

    p.add_argument('--scellst_repo', type=str, default='/hpc/group/jilab/tc459/sCellST')
    p.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet18'])
    p.add_argument('--weights_mode', type=str, default='imagenet', choices=['imagenet', 'moco'])
    p.add_argument('--moco_ckpt', type=str, default='', help='Required when weights_mode=moco')
    p.add_argument('--out_subdir', type=str, default='', help='Override output subdir under <dataset>/splits/')
    return p.parse_args()


def build_encoder(args: argparse.Namespace) -> torch.nn.Module:
    scellst_repo = Path(args.scellst_repo)
    if not scellst_repo.exists():
        raise FileNotFoundError(f'sCellST repo not found: {scellst_repo}')
    sys.path.append(str(scellst_repo))

    from scellst.module.image_encoder import InstanceEmbedder  # pylint: disable=import-outside-toplevel

    if args.weights_mode == 'imagenet':
        weights = f'imagenet-{args.arch.replace("resnet", "rn")}'
    else:
        if not args.moco_ckpt:
            raise ValueError('--moco_ckpt is required when --weights_mode moco')
        ckpt = Path(args.moco_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f'MoCo checkpoint not found: {ckpt}')
        weights = str(ckpt)

    return InstanceEmbedder(archi=args.arch, weights=weights)


def infer_out_subdir(args: argparse.Namespace) -> str:
    if args.out_subdir:
        return args.out_subdir
    if args.weights_mode == 'imagenet':
        return f'embeddings_scellst_rn50_delivery_s{args.seed}' if args.arch == 'resnet50' else f'embeddings_scellst_rn18_delivery_s{args.seed}'
    return f'embeddings_scellst_moco_delivery_s{args.seed}'


class DeliveryPatchDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, expr_cell_major: np.ndarray):
        self.meta_df = meta_df.reset_index(drop=True)
        self.expr = expr_cell_major
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int):
        row = self.meta_df.iloc[idx]
        img = Image.open(row['raw_img_abs']).convert('RGB')
        x = self.tf(img)
        y = torch.from_numpy(self.expr[idx]).float()
        return x, y, row['cell_id']


def split_indices(n: int, seed: int, train_frac: float, val_frac: float):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]
    return tr, va, te


def main() -> None:
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_root = Path(args.data_root) / args.dataset
    if not ds_root.exists():
        raise FileNotFoundError(f'Dataset root not found: {ds_root}')

    meta_path = ds_root / 'meta' / f'{args.scale}x' / f'{args.dataset}.csv'
    expr_root = ds_root / 'expr'
    mtx_path = expr_root / 'expr.mtx'
    genes_path = expr_root / 'genes.txt'
    cells_path = expr_root / 'cells.txt'

    for p in [meta_path, mtx_path, genes_path, cells_path]:
        if not p.exists():
            raise FileNotFoundError(f'Missing required file: {p}')

    meta = pd.read_csv(meta_path)
    genes = np.loadtxt(genes_path, dtype=str)
    cells = np.loadtxt(cells_path, dtype=str)

    X = sio.mmread(mtx_path).tocsc()  # genes x cells
    cell_to_col = {c: i for i, c in enumerate(cells)}

    # Build absolute patch paths and align rows to expression columns.
    def _abs_path(p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str(ds_root / pp)

    meta['raw_img_abs'] = meta['raw_img_path'].astype(str).map(_abs_path)
    meta['col_idx'] = meta['cell_id'].map(cell_to_col)
    meta = meta[meta['col_idx'].notna()].copy()
    meta['col_idx'] = meta['col_idx'].astype(int)

    n = len(meta)

    out_subdir = infer_out_subdir(args)
    out_ds_root = Path(args.output_root) / args.dataset
    out_dir = out_ds_root / 'splits' / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or generate the canonical 70/15/15 random split ───────────────
    # The H5 builder saves splits_seed.npz one level up (<output_root>/<dataset>/).
    # If it exists, load it so SSL encoder and supervised predictor share the
    # exact same split. Otherwise generate it fresh and save it.
    split_npz = out_ds_root / 'splits_seed.npz'
    if split_npz.exists():
        print(f'[split] Loading existing split from {split_npz}')
        saved = np.load(split_npz)
        tr_idx = saved['train_idx']
        va_idx = saved['val_idx']
        te_idx = saved['test_idx']
        assert len(tr_idx) + len(va_idx) + len(te_idx) == n, (
            f'Saved split total {len(tr_idx)+len(va_idx)+len(te_idx)} != meta rows {n}. '
            f'Delete {split_npz} and rerun.'
        )
    else:
        tr_idx, va_idx, te_idx = split_indices(n, args.seed, args.train_frac, args.val_frac)
        np.savez(
            split_npz,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=te_idx,
            n_total=np.int64(n),
            seed=np.int64(args.seed),
        )
        print(f'[split] Generated new split ({len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test) → {split_npz}')

    encoder = build_encoder(args).to(device)
    encoder.eval()

    print('=======================================================')
    print('Extracting sCellST embeddings from delivery data')
    print(f'Dataset root: {ds_root}')
    print(f'Output dataset root: {out_ds_root}')
    print(f'Meta: {meta_path}')
    print(f'Expr: {mtx_path}')
    print(f'Split: random seed={args.seed}, train/val/test={len(tr_idx)}/{len(va_idx)}/{len(te_idx)}')
    print(f'Encoder: {args.arch} | Weights: {args.weights_mode}')
    print(f'Output dir: {out_dir}')
    print('=======================================================')

    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    split_map = {'train': tr_idx, 'val': va_idx, 'test': te_idx}

    # Save gene names alongside split tensors for downstream alignment.
    np.save(out_dir / 'gene_names.npy', genes)

    for split_name, idx in split_map.items():
        sub = meta.iloc[idx].copy().reset_index(drop=True)

        # Slice expression and convert to cell-major dense array once per split.
        cols = sub['col_idx'].to_numpy(dtype=int)
        expr_split = X[:, cols].T.toarray().astype(np.float32)  # cells x genes

        dset = DeliveryPatchDataset(sub, expr_split)
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        feats = []
        ys = []
        cell_ids = []

        for imgs, y, cid in tqdm(loader, desc=f'{args.dataset}:{split_name}'):
            x = imgs.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                f = encoder(x)
            feats.append(f.float().cpu())
            ys.append(y.float().cpu())
            cell_ids.extend(list(cid))

        feat_t = torch.cat(feats, dim=0)
        y_t = torch.cat(ys, dim=0)

        torch.save(feat_t, out_dir / f'{split_name}_features.pt')
        torch.save(y_t, out_dir / f'{split_name}_expr.pt')
        pd.DataFrame({'cell_id': cell_ids}).to_csv(out_dir / f'{split_name}_cell_ids.csv', index=False)

        print(f'[{split_name}] features={tuple(feat_t.shape)} expr={tuple(y_t.shape)}')

    print('Done.')


if __name__ == '__main__':
    main()
