#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')
sys.path.append(str(PROJECT))

from data.visium_dataset import VisiumHDPredictionDataset


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Extract sCellST-style RN50 embeddings from MorphPT cache splits.')
    p.add_argument('--dataset', type=str, required=True, help='Dataset suffix, e.g. mouse_brain')
    p.add_argument('--split_mode', type=str, default='spatial', choices=['spatial', 'random'], help='Match MorphPT split strategy')
    p.add_argument('--split_layout', type=str, default='default')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)

    p.add_argument('--scellst_repo', type=str, default='/hpc/group/jilab/tc459/sCellST')
    p.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet18'])
    p.add_argument('--weights_mode', type=str, default='imagenet', choices=['imagenet', 'moco'])
    p.add_argument('--moco_ckpt', type=str, default='', help='Required when weights_mode=moco; path to moco_model_best.pth.tar')
    p.add_argument('--out_subdir', type=str, default='', help='Override output subdir. Defaults to embeddings_scellst_rn50 or embeddings_scellst_moco')
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

    model = InstanceEmbedder(archi=args.arch, weights=weights)
    return model


def infer_out_subdir(args: argparse.Namespace) -> str:
    if args.out_subdir:
        return args.out_subdir
    if args.weights_mode == 'imagenet':
        return 'embeddings_scellst_rn50' if args.arch == 'resnet50' else 'embeddings_scellst_rn18'
    return 'embeddings_scellst_moco'


def main() -> None:
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = PROJECT / f'cache_{args.dataset}'
    if not cache_dir.exists():
        raise FileNotFoundError(f'Cache dir not found: {cache_dir}')

    out_subdir = infer_out_subdir(args)
    out_dir = cache_dir / 'splits' / args.split_layout / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_encoder(args).to(device)
    encoder.eval()

    print('=======================================================')
    print('Extracting sCellST embeddings from MorphPT cache')
    print(f'Dataset: {args.dataset} | Split: {args.split_mode} | Layout: {args.split_layout}')
    print(f'Encoder: {args.arch} | Weights: {args.weights_mode}')
    print(f'Output dir: {out_dir}')
    print('=======================================================')

    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    for split in ['train', 'val', 'test']:
        ds = VisiumHDPredictionDataset(
            cache_dir=cache_dir,
            split=split,
            scales=['10.0x'],
            fuse='identity',
            augment=False,
            split_type=args.split_mode,
            split_layout=args.split_layout,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        all_feats = []
        all_expr = []

        for imgs, expr, _ in tqdm(loader, desc=f'{args.dataset}:{split}'):
            x = imgs.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                feat = encoder(x)
            all_feats.append(feat.float().cpu())
            all_expr.append(expr.float().cpu())

        feat_t = torch.cat(all_feats, dim=0)
        expr_t = torch.cat(all_expr, dim=0)

        torch.save(feat_t, out_dir / f'{split}_features.pt')
        torch.save(expr_t, out_dir / f'{split}_expr.pt')
        print(f'[{split}] features={tuple(feat_t.shape)} expr={tuple(expr_t.shape)}')

    print('Done.')


if __name__ == '__main__':
    main()
