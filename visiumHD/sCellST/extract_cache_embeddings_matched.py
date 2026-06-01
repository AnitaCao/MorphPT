#!/usr/bin/env python3
"""Unified cache-based feature extractor for sCellST competing-method arms.

All arms share IDENTICAL targets/genes/split so the encoder is the only variable:
  - Reads MorphPT cache (224x224 images, log1p+z-scored expr) via VisiumHDPredictionDataset.
  - Extracts per-cell features with one of:
        rn50_imagenet : sCellST InstanceEmbedder, ImageNet RN50 (frozen)
        rn50_moco     : sCellST InstanceEmbedder, MoCo RN50 checkpoint (frozen)
        morphpt_frozen: MorphPT ViT-B backbone (frozen, no LoRA, identity, 10x)
  - Partitions cells into train/val/test by reproducing MorphPT's exact split
    (morphpt_split.py) and matching by cell_id (order-independent).
  - Saves {split}_features.pt and {split}_expr.pt under
        cache_<ds>/splits/<layout>/<out_subdir>/
    NO gene_names.npy is written, so the trainer selects genes by gene_idx
    directly into the cache's 2847-gene expr (exactly like MorphPT).

Run rn50_* arms in scellst_env; run morphpt_frozen in cellpt_env (MorphPT model deps).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')
sys.path.append(str(PROJECT))

from data.visium_dataset import VisiumHDPredictionDataset  # noqa: E402

from morphpt_split import morphpt_random_split_cellids  # noqa: E402 (script-dir import)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Cache-based, split-matched feature extraction for sCellST arms.')
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--encoder', type=str, required=True,
                   choices=['rn50_imagenet', 'rn50_moco', 'morphpt_frozen', 'morphpt_lora'])
    p.add_argument('--split_layout', type=str, default='default')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out_subdir', type=str, default='', help='Override output subdir name')

    # sCellST RN50 / MoCo options
    p.add_argument('--scellst_repo', type=str, default='/hpc/group/jilab/tc459/sCellST')
    p.add_argument('--moco_ckpt', type=str, default='', help='Required when encoder=rn50_moco')

    # MorphPT frozen-backbone options (un-adapted pretrained checkpoint)
    p.add_argument('--morphpt_ckpt', type=str,
                   default=str(PROJECT / 'experiments/router_nobreast_vitb_gate_r16_mlp_cw/best.pt'))

    # MorphPT LoRA-adapted backbone options (per-tissue best_model checkpoint).
    # Defaults mirror train_lora_multi_probing.py so module shapes match the saved weights.
    p.add_argument('--morphpt_lora_ckpt', type=str, default='',
                   help='Per-tissue best_model_*.pt from the LoRA probing run (required for encoder=morphpt_lora)')
    p.add_argument('--model_name', type=str, default='vit_base_patch16_dinov3.lvd1689m')
    p.add_argument('--lora_blocks', type=str, default='0,2,4,6,8,10,11')
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_targets', type=str, default='qkv,proj,mlp_fc1,mlp_fc2')
    return p.parse_args()


def default_out_subdir(encoder: str) -> str:
    return {
        'rn50_imagenet': 'embeddings_scellst_rn50_cache_matched',
        'rn50_moco': 'embeddings_scellst_moco_cache_matched',
        'morphpt_frozen': 'embeddings_morphpt_frozen_cache_matched',
        'morphpt_lora': 'embeddings_morphpt_lora_cache_matched',
    }[encoder]


def build_extract_fn(args: argparse.Namespace, device: torch.device):
    """Return (extract_fn, label). extract_fn maps a (B,3,224,224) batch -> (B,D) float cpu-ready tensor."""
    if args.encoder.startswith('rn50'):
        sys.path.append(args.scellst_repo)
        from scellst.module.image_encoder import InstanceEmbedder  # noqa: E402
        if args.encoder == 'rn50_imagenet':
            weights = 'imagenet-rn50'
        else:
            if not args.moco_ckpt or not Path(args.moco_ckpt).exists():
                raise ValueError('--moco_ckpt is required and must exist for encoder=rn50_moco')
            weights = str(args.moco_ckpt)
        encoder = InstanceEmbedder(archi='resnet50', weights=weights).to(device).eval()

        def _fn(x):
            return encoder(x)

        return _fn, f'InstanceEmbedder(resnet50, {weights})'

    # morphpt_frozen: MorphPT ViT-B backbone, frozen, no LoRA, identity fusion.
    from models.visium_regression import VisiumRegressor  # noqa: E402

    if args.encoder == 'morphpt_lora':
        if not args.morphpt_lora_ckpt or not Path(args.morphpt_lora_ckpt).exists():
            raise ValueError('--morphpt_lora_ckpt required and must exist for morphpt_lora')
        lckpt = Path(args.morphpt_lora_ckpt)
        lmodel = VisiumRegressor(
            model_name=args.model_name, img_size=224, out_dim=2, pretrained=False,
            fuse='identity', freeze_backbone=True,
            lora_blocks=args.lora_blocks, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets, unfreeze_lora=False,
            ckpt_path=None, head_type='mlp',
        ).to(device).eval()
        lsd = torch.load(str(lckpt), map_location='cpu', weights_only=False)
        if isinstance(lsd, dict):
            lsd = lsd.get('model', lsd.get('state_dict', lsd))
        lbsd = {k[len('backbone.'):]: v for k, v in lsd.items() if k.startswith('backbone.')}
        lmiss, lunexp = lmodel.backbone.load_state_dict(lbsd, strict=False)
        print(f'[morphpt_lora] loaded={len(lbsd)} missing={len(lmiss)} unexpected={len(lunexp)}')
        if len(lbsd) == 0:
            raise RuntimeError(f'No backbone.* keys in {lckpt}')
        return (lambda x: lmodel.backbone(x)), f'MorphPT LoRA-adapted [{lckpt.parent.name}]'
    ckpt = Path(args.morphpt_ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f'MorphPT checkpoint not found: {ckpt}')
    model = VisiumRegressor(
        model_name='vit_base_patch16_dinov3.lvd1689m',
        img_size=224,
        out_dim=2,            # dummy head; we intercept backbone features
        pretrained=False,
        fuse='identity',
        freeze_backbone=True,
        lora_blocks='none',   # frozen config: no LoRA, matches train_lora "frozen" arm
        unfreeze_lora=False,
        ckpt_path=str(ckpt),
        head_type='mlp',
    ).to(device).eval()

    def _fn(x):
        return model.backbone(x)

    return _fn, f'MorphPT ViT-B frozen backbone ({ckpt.name})'


def main() -> None:
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = PROJECT / f'cache_{args.dataset}'
    if not cache_dir.exists():
        raise FileNotFoundError(f'Cache dir not found: {cache_dir}')

    out_subdir = args.out_subdir or default_out_subdir(args.encoder)
    out_dir = cache_dir / 'splits' / args.split_layout / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_fn, enc_label = build_extract_fn(args, device)
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    # Single pass over ALL cells; partition afterwards by matched split.
    ds = VisiumHDPredictionDataset(
        cache_dir=str(cache_dir),
        split='all',
        scales=['10.0x'],
        fuse='identity',
        augment=False,
        split_type='random',
        split_layout=args.split_layout,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    print('=======================================================')
    print('Cache-based split-matched extraction')
    print(f'Dataset: {args.dataset} | Encoder: {args.encoder}')
    print(f'Encoder detail: {enc_label}')
    print(f'Cells (all): {len(ds)} | Output: {out_dir}')
    print('=======================================================')

    feats, exprs, cell_ids = [], [], []
    for imgs, expr, meta in tqdm(loader, desc=f'{args.dataset}:{args.encoder}'):
        x = imgs.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            f = extract_fn(x)
        feats.append(f.float().cpu())
        exprs.append(expr.float().cpu())
        cell_ids.extend([str(c) for c in meta['cell_id']])

    feat_t = torch.cat(feats, dim=0)
    expr_t = torch.cat(exprs, dim=0)
    print(f'Extracted features={tuple(feat_t.shape)} expr={tuple(expr_t.shape)}')

    # Reproduce MorphPT split and partition by cell_id.
    tr_cells, va_cells, te_cells = morphpt_random_split_cellids(
        cache_dir, split_layout=args.split_layout, seed=args.seed,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    pos = {c: i for i, c in enumerate(cell_ids)}

    def _idx(cs):
        return np.asarray([pos[c] for c in cs.astype(str) if c in pos], dtype=int)

    split_idx = {'train': _idx(tr_cells), 'val': _idx(va_cells), 'test': _idx(te_cells)}
    matched = sum(len(v) for v in split_idx.values())
    if matched != len(cell_ids):
        print(f'[Warning] matched {matched}/{len(cell_ids)} cells to MorphPT split.')

    np.savez(out_dir / 'splits_seed.npz',
             train_cell_ids=tr_cells, val_cell_ids=va_cells, test_cell_ids=te_cells)

    for split, idx in split_idx.items():
        idx_t = torch.from_numpy(idx)
        torch.save(feat_t[idx_t].clone(), out_dir / f'{split}_features.pt')
        torch.save(expr_t[idx_t].clone(), out_dir / f'{split}_expr.pt')
        print(f'[{split}] features={tuple(feat_t[idx_t].shape)} expr={tuple(expr_t[idx_t].shape)}')

    print('Done.')


if __name__ == '__main__':
    main()
