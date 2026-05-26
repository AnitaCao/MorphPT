#!/usr/bin/env python3
"""
Single-gene MLP probing with Hybrid Spatial/Random Split.
Accepts gene_idx and gene_name directly as arguments.

Usage:
  python train_mlp_probing.py --gene_idx 42 --gene_name MYC
  python train_mlp_probing.py --gene_idx 42 --gene_name MYC --loss_type mixed
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')

class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def pearson_r(pred, target):
    if pred.size(0) < 2:
        return 0.0
    pm = pred - pred.mean()
    tm = target - target.mean()
    num = (pm * tm).sum()
    den = (pm.pow(2).sum() * tm.pow(2).sum()).clamp(min=1e-8).sqrt()
    return (num / den).item()

def mixed_loss_single(pred, target, epoch, corr_start=3, lambda_corr=0.2, lambda_var=0.05):
    """Simplified MixedLoss for single-gene (1D output)."""
    mse = F.mse_loss(pred, target)
    if epoch < corr_start:
        return mse, {"loss_mse": mse.item()}
    
    # Correlation term
    pm = pred - pred.mean()
    tm = target - target.mean()
    num = (pm * tm).sum()
    den = (pm.pow(2).sum() * tm.pow(2).sum()).clamp(min=1e-8).sqrt()
    corr = num / den
    l_corr = 1.0 - corr
    
    # Variance term
    eps = 1e-6
    p_std = pred.std().clamp(min=eps)
    t_std = target.std().clamp(min=eps)
    l_var = torch.abs(torch.log(p_std) - torch.log(t_std))
    
    loss = mse + lambda_corr * l_corr + lambda_var * l_var
    return loss, {"loss_mse": mse.item(), "loss_corr": l_corr.item(), "loss_var": l_var.item()}

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='crc')
    p.add_argument('--gene_idx', type=int, required=True, help='Global gene index')
    p.add_argument('--gene_name', type=str, required=True, help='Gene name for labeling')
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=15)
    
    p.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mixed'])
    p.add_argument('--lambda_corr', type=float, default=0.2)
    p.add_argument('--lambda_var', type=float, default=0.05)
    p.add_argument('--loss_start_epoch', type=int, default=3)
    
    # Hybrid Split
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
    
    if args.loss_type == 'mixed':
        loss_suffix = f"mixed_c{args.lambda_corr}_v{args.lambda_var}"
    else:
        loss_suffix = "mse"
    
    out_dir = PROJECT / "experiments" / f"mlp_single_{args.dataset}_{loss_suffix}" / gene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"==============================================")
    print(f" Single-Gene MLP Probing - Hybrid Split")
    print(f" Gene: {gene_name} (idx={gene_idx})")
    print(f" Loss: {args.loss_type} | Seeds: {args.n_seeds}")
    print(f"==============================================")
    
    # Load pre-extracted features
    cache_dir = PROJECT / f"cache_{args.dataset}"
    emb_dir = cache_dir / "embeddings_morphpt"
    meta = pd.read_csv(cache_dir / "meta.csv")
    
    print("Loading features into RAM...")
    feats_list, expr_list, df_list = [], [], []
    for spl in ["train", "val", "test"]:
        feats = torch.load(emb_dir / f"{spl}_features.pt", map_location='cpu')
        expr_all = torch.load(emb_dir / f"{spl}_expr.pt", map_location='cpu')
        expr = expr_all[:, gene_idx].float()
        feats_list.append(feats)
        expr_list.append(expr)
        df_list.append(meta[meta["split"] == spl].copy().reset_index(drop=True))
    
    all_feats = torch.cat(feats_list, dim=0)
    all_exprs = torch.cat(expr_list, dim=0)
    all_meta = pd.concat(df_list, axis=0).reset_index(drop=True)
    in_dim = all_feats.shape[1]
    
    print(f"Pool: {all_feats.shape[0]} cells × {in_dim} features")
    
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
        
        train_ds = TensorDataset(all_feats[train_idx].clone(), all_exprs[train_idx].clone())
        val_ds   = TensorDataset(all_feats[val_idx].clone(),   all_exprs[val_idx].clone())
        test_ds  = TensorDataset(all_feats[test_idx].clone(),  all_exprs[test_idx].clone())
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        model = MLPHead(in_features=in_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
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
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(x)
                if args.loss_type == 'mixed':
                    loss, _ = mixed_loss_single(pred, y, ep, args.loss_start_epoch, args.lambda_corr, args.lambda_var)
                else:
                    loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
                n_train += len(y)
            scheduler.step()
            
            model.eval()
            all_preds, all_gts = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    all_preds.append(model(x).cpu())
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
            for x, y in test_loader:
                all_preds.append(model(x.to(device)).cpu())
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
    
    # Summary
    test_rs = [r['test_r'] for r in all_seed_results]
    val_rs = [r['val_r'] for r in all_seed_results]
    summary = {
        'gene_name': gene_name, 'gene_idx': gene_idx,
        'loss_type': args.loss_type,
        'test_r_mean': float(np.mean(test_rs)), 'test_r_std': float(np.std(test_rs)),
        'val_r_mean': float(np.mean(val_rs)),
        'seeds': all_seed_results,
        'hyperparams': {
            'hidden_dim': args.hidden_dim, 'lr': args.lr,
            'lambda_corr': args.lambda_corr, 'lambda_var': args.lambda_var
        }
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    
    total = time.time() - t0_overall
    print(f"\n{'='*50}")
    print(f" {gene_name}: Test R = {np.mean(test_rs):.4f} ± {np.std(test_rs):.4f} ({total:.1f}s)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
