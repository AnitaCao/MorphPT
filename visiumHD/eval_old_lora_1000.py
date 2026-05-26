#!/usr/bin/env python3
"""
eval_old_lora_1000.py
──────────────────────
Run the trained MorphPT+LoRA model on the test set and save predictions.

Output (in OLD_EXP_DIR):
  test_y_true.npy   (N_test, 1000) z-score standardized ground truth
  test_y_pred.npy   (N_test, 1000) z-score standardized predictions
  eval_info.json    metadata (n_cells, n_genes, mean_r, etc.)

After this runs, you can load predictions anywhere:
  y_true = np.load(OLD_EXP_DIR / 'test_y_true.npy')
  y_pred = np.load(OLD_EXP_DIR / 'test_y_pred.npy')

Usage:
  python eval_old_lora_1000.py
  # or override paths:
  python eval_old_lora_1000.py --exp_dir /path/to/experiment
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')
sys.path.append(str(PROJECT))

from data.visium_dataset import VisiumHDPredictionDataset
from models.visium_regression import VisiumRegressor


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir',    type=str,
                   default=str(PROJECT / 'experiments/visium_morphpt_lora_10x_mlp'))
    p.add_argument('--ckpt_name',  type=str, default='best.pth',
                   help='Checkpoint filename inside exp_dir')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--workers',    type=int, default=8)
    return p.parse_args()


def pearson_per_gene(y_pred, y_true):
    """Compute per-gene Pearson r.  Arrays: (N, G)."""
    pm = y_pred - y_pred.mean(axis=0)
    tm = y_true - y_true.mean(axis=0)
    num = (pm * tm).sum(axis=0)
    den = np.sqrt((pm**2).sum(axis=0) * (tm**2).sum(axis=0)) + 1e-8
    return num / den


def main():
    args     = get_args()
    exp_dir  = Path(args.exp_dir)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'{"="*60}')
    print(f'Evaluating: {exp_dir.name}')
    print(f'{"="*60}')
    print(f'Device     : {device}')
    print(f'Experiment : {exp_dir}')

    if not exp_dir.exists():
        raise FileNotFoundError(f'Experiment dir not found: {exp_dir}')

    ckpt_path = exp_dir / args.ckpt_name
    if not ckpt_path.exists():
        print(f'\n{args.ckpt_name} not found. Available files:')
        for f in exp_dir.glob('*.pth'):
            print(f'  {f.name}')
        raise FileNotFoundError(ckpt_path)

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f'\nLoading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    print(f'Model config:')
    for k in ['model', 'fuse', 'head_type', 'lora_rank', 'lora_alpha',
              'lora_blocks', 'lora_targets', 'scales', 'cache_dir',
              'split_type', 'freeze_backbone', 'unfreeze_lora']:
        if k in ckpt_args:
            print(f'  {k:<20}: {ckpt_args[k]}')

    # ── Build test dataset ─────────────────────────────────────────────────
    cache_dir = ckpt_args.get('cache_dir',
                              '/hpc/group/jilab/rz179/MorphPT_MOE/cache_visium')
    scales    = [s.strip() for s in ckpt_args.get('scales', '10.0x').split(',')]

    print(f'\nBuilding test dataset from: {cache_dir}')
    test_ds = VisiumHDPredictionDataset(
        cache_dir  = cache_dir,
        split      = 'test',
        scales     = scales,
        fuse       = ckpt_args.get('fuse', 'identity'),
        augment    = False,
        split_type = ckpt_args.get('split_type', 'spatial'),
    )
    num_genes = test_ds.gene_mean.shape[0]
    print(f'  Test cells : {len(test_ds):,}')
    print(f'  Num genes  : {num_genes}')

    test_loader = DataLoader(
        test_ds,
        batch_size         = args.batch_size,
        shuffle            = False,
        num_workers        = args.workers,
        pin_memory         = True,
        persistent_workers = (args.workers > 0),
    )

    # ── Build model & load weights ─────────────────────────────────────────
    print(f'\nBuilding model...')
    model = VisiumRegressor(
        model_name      = ckpt_args['model'],
        img_size        = ckpt_args.get('img_size', 224),
        out_dim         = num_genes,
        pretrained      = False,
        fuse            = ckpt_args['fuse'],
        freeze_backbone = bool(ckpt_args.get('freeze_backbone', 1)),
        lora_blocks     = ckpt_args['lora_blocks'],
        lora_rank       = ckpt_args['lora_rank'],
        lora_alpha      = ckpt_args['lora_alpha'],
        lora_dropout    = ckpt_args.get('lora_dropout', 0.05),
        lora_targets    = ckpt_args['lora_targets'],
        unfreeze_lora   = bool(ckpt_args.get('unfreeze_lora', 1)),
        ckpt_path       = ckpt_args.get('ckpt_path'),
        head_type       = ckpt_args['head_type'],
        gate_dropout    = ckpt_args.get('gate_dropout', 0.1),
    ).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Total params: {n_params/1e6:.1f}M')

    # ── Run inference ──────────────────────────────────────────────────────
    print(f'\nRunning inference on {len(test_ds):,} test cells...')
    t0 = time.time()

    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16) if torch.cuda.is_available() else None

    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, expr, _ in tqdm(test_loader, ncols=80):
            imgs = imgs.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    pred = model(imgs)
            else:
                pred = model(imgs)
            all_preds.append(pred.float().cpu())
            all_true.append(expr.float())

    y_pred = torch.cat(all_preds, dim=0).numpy()   # (N, G)
    y_true = torch.cat(all_true,  dim=0).numpy()   # (N, G)

    dt = time.time() - t0
    print(f'Inference done in {dt:.1f}s')
    print(f'  y_true shape: {y_true.shape}')
    print(f'  y_pred shape: {y_pred.shape}')

    # ── Quick sanity check: compute per-gene Pearson r ─────────────────────
    r = pearson_per_gene(y_pred, y_true)
    mean_r   = float(np.mean(r))
    median_r = float(np.median(r))

    # Compare with saved Pearson r if available
    saved_r_path = exp_dir / 'per_gene_pearson_test.pt'
    if saved_r_path.exists():
        saved_r = torch.load(saved_r_path, map_location='cpu').numpy()
        max_diff = float(np.abs(r - saved_r).max())
        print(f'\nSanity check vs saved per_gene_pearson_test.pt:')
        print(f'  saved mean_r     : {saved_r.mean():.4f}')
        print(f'  computed mean_r  : {mean_r:.4f}')
        print(f'  max abs diff     : {max_diff:.2e}')
        if max_diff > 1e-3:
            print(f'  ⚠️  WARNING: differences >1e-3, something may be off')

    print(f'\nFinal metrics:')
    print(f'  mean Pearson r    : {mean_r:.4f}')
    print(f'  median Pearson r  : {median_r:.4f}')
    print(f'  r > 0.1           : {(r > 0.1).sum():>4} / {len(r)}')
    print(f'  r > 0.2           : {(r > 0.2).sum():>4} / {len(r)}')
    print(f'  r > 0.3           : {(r > 0.3).sum():>4} / {len(r)}')
    print(f'  r > 0.5           : {(r > 0.5).sum():>4} / {len(r)}')

    # ── Save ───────────────────────────────────────────────────────────────
    print(f'\nSaving predictions...')
    np.save(exp_dir / 'test_y_true.npy', y_true.astype(np.float32))
    np.save(exp_dir / 'test_y_pred.npy', y_pred.astype(np.float32))
    print(f'  → {exp_dir}/test_y_true.npy   ({y_true.nbytes/1e6:.1f} MB)')
    print(f'  → {exp_dir}/test_y_pred.npy   ({y_pred.nbytes/1e6:.1f} MB)')

    info = {
        'exp_dir':       str(exp_dir),
        'ckpt':          str(ckpt_path),
        'cache_dir':     str(cache_dir),
        'n_test_cells':  int(y_true.shape[0]),
        'n_genes':       int(y_true.shape[1]),
        'mean_pearson':  mean_r,
        'median_pearson':median_r,
        'split_type':    ckpt_args.get('split_type', 'spatial'),
        'scales':        scales,
        'fuse':          ckpt_args.get('fuse'),
        'target_normalization': 'z-score per gene (train statistics)',
    }
    (exp_dir / 'eval_info.json').write_text(json.dumps(info, indent=2))
    print(f'  → {exp_dir}/eval_info.json')

    print(f'\nDone. You can now load predictions in your notebook:')
    print(f'  y_true = np.load("{exp_dir}/test_y_true.npy")')
    print(f'  y_pred = np.load("{exp_dir}/test_y_pred.npy")')


if __name__ == '__main__':
    main()
