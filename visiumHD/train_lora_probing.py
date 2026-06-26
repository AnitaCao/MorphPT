#!/usr/bin/env python3
"""
Single-gene LoRA probing with Hybrid Spatial/Random Split.
Mirrors train_mlp_probing.py exactly, but uses VisiumRegressor + LoRA.

Usage:
  python train_lora_probing.py --gene_idx 42 --gene_name MYC
  python train_lora_probing.py --gene_idx 42 --gene_name MYC --loss_type mixed
"""
import os
import argparse
import json
import copy
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT = Path(os.environ.get("MORPHPT_ROOT", Path(__file__).resolve().parents[1]))  # repo root; override with MORPHPT_ROOT to point at your data/cache
sys.path.append(str(PROJECT))

from data.visium_dataset import VisiumHDPredictionDataset
from models.visium_regression import VisiumRegressor

def pearson_r(pred, target):
    if pred.size(0) < 2:
        return 0.0
    pm = pred - pred.mean()
    tm = target - target.mean()
    num = (pm * tm).sum()
    den = (pm.pow(2).sum() * tm.pow(2).sum()).clamp(min=1e-8).sqrt()
    return (num / den).item()

def mixed_loss_single(pred, target, epoch, corr_start=3, lambda_corr=0.2, lambda_var=0.05):
    mse = F.mse_loss(pred, target)
    if epoch < corr_start:
        return mse, {"loss_mse": mse.item()}
    pm = pred - pred.mean()
    tm = target - target.mean()
    num = (pm * tm).sum()
    den = (pm.pow(2).sum() * tm.pow(2).sum()).clamp(min=1e-8).sqrt()
    l_corr = 1.0 - num / den
    eps = 1e-6
    l_var = torch.abs(torch.log(pred.std().clamp(min=eps)) - torch.log(target.std().clamp(min=eps)))
    loss = mse + lambda_corr * l_corr + lambda_var * l_var
    return loss, {"loss_mse": mse.item(), "loss_corr": l_corr.item(), "loss_var": l_var.item()}

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='crc')
    p.add_argument('--gene_idx', type=int, required=True)
    p.add_argument('--gene_name', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lora_lr_scale', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=10)
    
    p.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mixed'])
    p.add_argument('--lambda_corr', type=float, default=0.2)
    p.add_argument('--lambda_var', type=float, default=0.05)
    p.add_argument('--loss_start_epoch', type=int, default=3)
    
    p.add_argument('--model_name', type=str, default='vit_base_patch16_dinov3.lvd1689m')
    p.add_argument('--ckpt_path', type=str, default='experiments/router_nobreast_vitb_gate_r16_mlp_cw/best.pt')
    p.add_argument('--lora_blocks', type=str, default="0,2,4,6,8,10,11")
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_targets', type=str, default="qkv,proj,mlp_fc1,mlp_fc2")
    p.add_argument('--scales', type=str, default="10.0x")
    p.add_argument('--fuse', type=str, default="identity", choices=["identity", "gate"])
    
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--base_seed', type=int, default=42)
    p.add_argument('--n_test_tiles', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.1)
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gene_idx = args.gene_idx
    gene_name = args.gene_name
    scales_list = [s.strip() for s in args.scales.split(',')]
    scale_tag = '+'.join(scales_list)
    
    if args.loss_type == 'mixed':
        loss_suffix = f"mixed_c{args.lambda_corr}_v{args.lambda_var}"
    else:
        loss_suffix = "mse"
    
    out_dir = PROJECT / "experiments" / f"lora_single_{args.dataset}_{scale_tag}_{loss_suffix}" / gene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"==============================================")
    print(f" Single-Gene LoRA Probing - Hybrid Split")
    print(f" Gene: {gene_name} (idx={gene_idx})")
    print(f" Scales: {scales_list} | Fuse: {args.fuse}")
    print(f" Loss: {args.loss_type} | Seeds: {args.n_seeds}")
    print(f"==============================================")
    
    cache_dir = PROJECT / f"cache_{args.dataset}"
    meta = pd.read_csv(cache_dir / "meta.csv")
    
    ds_train_master = VisiumHDPredictionDataset(cache_dir, split="all", scales=scales_list, fuse=args.fuse, split_type="spatial", augment=False)
    ds_eval_master  = VisiumHDPredictionDataset(cache_dir, split="all", scales=scales_list, fuse=args.fuse, split_type="spatial", augment=False)
    
    # Align ordering with MLP script
    all_meta_ordered = []
    for spl in ["train", "val", "test"]:
        all_meta_ordered.append(meta[meta["split"] == spl].copy())
    all_meta = pd.concat(all_meta_ordered, axis=0).reset_index(drop=True)
    
    ds_train_master._mmap_rows = all_meta["mmap_idx"].values.astype(np.int64)
    ds_train_master._cell_ids  = all_meta["cell_id"].values
    ds_eval_master._mmap_rows  = all_meta["mmap_idx"].values.astype(np.int64)
    ds_eval_master._cell_ids   = all_meta["cell_id"].values
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    all_seed_results = []
    t0_overall = time.time()
    
    for seed_idx in range(args.n_seeds):
        current_seed = args.base_seed + seed_idx
        print(f"\n── SEED {current_seed} ({seed_idx+1}/{args.n_seeds}) ──")
        
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        
        unique_tiles = all_meta['tile_id'].unique()
        test_tiles = np.random.choice(unique_tiles, size=args.n_test_tiles, replace=False)
        test_mask = all_meta['tile_id'].isin(test_tiles).values
        
        test_idx = np.where(test_mask)[0]
        train_pool_idx = np.where(~test_mask)[0]
        np.random.shuffle(train_pool_idx)
        n_val = int(len(train_pool_idx) * args.val_frac)
        val_idx = train_pool_idx[:n_val]
        train_idx = train_pool_idx[n_val:]
        
        print(f"  Tiles: {test_tiles.tolist()} | Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        train_ds = copy.copy(ds_train_master)
        train_ds._mmap_rows = ds_train_master._mmap_rows[train_idx]
        train_ds._cell_ids  = ds_train_master._cell_ids[train_idx]
        val_ds = copy.copy(ds_eval_master)
        val_ds._mmap_rows = ds_eval_master._mmap_rows[val_idx]
        val_ds._cell_ids  = ds_eval_master._cell_ids[val_idx]
        test_ds = copy.copy(ds_eval_master)
        test_ds._mmap_rows = ds_eval_master._mmap_rows[test_idx]
        test_ds._cell_ids  = ds_eval_master._cell_ids[test_idx]
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Model
        checkpoint = PROJECT / args.ckpt_path
        model = VisiumRegressor(
            model_name=args.model_name, img_size=224, out_dim=1,
            fuse=args.fuse, freeze_backbone=True,
            lora_blocks=args.lora_blocks, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets, unfreeze_lora=True,
            ckpt_path=str(checkpoint) if checkpoint.exists() else None,
            head_type="mlp",
        )
        # Replace head to match multi-task's hidden_dim=1024 for fair comparison
        d_embed = model.head[0].in_features
        model.head = nn.Sequential(
            nn.Linear(d_embed, 1024),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 1)
        )
        model = model.to(device)
        
        param_groups = model.get_param_groups(lr=args.lr, weight_decay=1e-4, gate_wd=0.1, lora_lr_scale=args.lora_lr_scale)
        optimizer = torch.optim.AdamW(param_groups)
        
        warmup_epochs = 5
        sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=warmup_epochs)
        sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])
        
        best_val_r = -float('inf')
        epochs_no_improve = 0
        best_state = None
        t0 = time.time()
        
        for ep in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            n_train = 0
            for imgs, y_full, _ in train_loader:
                x = imgs.to(device, non_blocking=True)
                y = y_full[:, gene_idx].float().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                    pred = model(x).squeeze(-1)
                    if args.loss_type == 'mixed':
                        loss, _ = mixed_loss_single(pred, y, ep, args.loss_start_epoch, args.lambda_corr, args.lambda_var)
                    else:
                        loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item() * len(y)
                n_train += len(y)
            scheduler.step()
            
            model.eval()
            all_preds, all_gts = [], []
            with torch.no_grad():
                for imgs, y_full, _ in val_loader:
                    x = imgs.to(device, non_blocking=True)
                    y = y_full[:, gene_idx].float()
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                        pred = model(x).squeeze(-1)
                    all_preds.append(pred.float().cpu())
                    all_gts.append(y)
            val_r = pearson_r(torch.cat(all_preds), torch.cat(all_gts))
            
            if ep % 10 == 0 or ep == 1:
                print(f"    Ep {ep:>3d} | Loss: {train_loss/n_train:.4f} | Val R: {val_r:.4f}")
            
            if val_r > best_val_r:
                best_val_r = val_r
                epochs_no_improve = 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(best_state, out_dir / f'best_model_seed_{current_seed}.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"    Early stopping at epoch {ep}")
                    break
        
        # Test
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for imgs, y_full, _ in test_loader:
                x = imgs.to(device, non_blocking=True)
                y = y_full[:, gene_idx].float()
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                    pred = model(x).squeeze(-1)
                all_preds.append(pred.float().cpu())
                all_gts.append(y)
        test_r = pearson_r(torch.cat(all_preds), torch.cat(all_gts))
        
        np.save(out_dir / f'test_y_pred_seed_{current_seed}.npy', torch.cat(all_preds).numpy())
        np.save(out_dir / f'test_y_true_seed_{current_seed}.npy', torch.cat(all_gts).numpy())
        
        seed_time = time.time() - t0
        print(f"  Val R: {best_val_r:.4f} | Test R: {test_r:.4f} | {seed_time:.1f}s")
        
        all_seed_results.append({
            'seed': current_seed, 'val_r': best_val_r, 'test_r': test_r,
            'test_tiles': test_tiles.tolist(), 'time': seed_time
        })
        np.savez(out_dir / f'splits_seed_{current_seed}.npz',
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, test_tiles=test_tiles)
    
    test_rs = [r['test_r'] for r in all_seed_results]
    summary = {
        'gene_name': gene_name, 'gene_idx': gene_idx,
        'loss_type': args.loss_type, 'scales': scales_list, 'fuse': args.fuse,
        'test_r_mean': float(np.mean(test_rs)), 'test_r_std': float(np.std(test_rs)),
        'seeds': all_seed_results,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    
    total = time.time() - t0_overall
    print(f"\n{'='*50}")
    print(f" {gene_name}: Test R = {np.mean(test_rs):.4f} ± {np.std(test_rs):.4f} ({total:.1f}s)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
