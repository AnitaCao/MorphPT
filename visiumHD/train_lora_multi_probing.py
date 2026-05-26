#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
import sys
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')
sys.path.append(str(PROJECT))

from data.visium_dataset import VisiumHDPredictionDataset
from models.visium_regression import VisiumRegressor

class MixedGeneLoss(nn.Module):
    def __init__(
        self,
        lambda_corr=0.2,
        lambda_var=0.05,
        corr_start_epoch=3,
        var_start_epoch=3,
        eps=1e-6,
        min_batch_std=1e-4,
    ):
        super().__init__()
        self.lambda_corr = lambda_corr
        self.lambda_var = lambda_var
        self.corr_start_epoch = corr_start_epoch
        self.var_start_epoch = var_start_epoch
        self.eps = eps
        self.min_batch_std = min_batch_std

    def forward(self, pred, target, epoch):
        metrics = {}

        mse = F.mse_loss(pred, target)
        loss = mse
        metrics["loss_mse"] = mse.detach()

        batch_t_std = target.std(dim=0, unbiased=False)
        valid_mask = batch_t_std > self.min_batch_std

        if epoch >= self.corr_start_epoch and valid_mask.any():
            p = pred[:, valid_mask]
            t = target[:, valid_mask]

            p = p - p.mean(dim=0, keepdim=True)
            t = t - t.mean(dim=0, keepdim=True)

            num = (p * t).sum(dim=0)
            den = torch.sqrt((p.pow(2).sum(dim=0) * t.pow(2).sum(dim=0)).clamp_min(self.eps))
            corr = num / den
            corr_loss = 1.0 - corr.mean()

            loss = loss + self.lambda_corr * corr_loss
            metrics["loss_corr"] = corr_loss.detach()
            metrics["mean_batch_corr"] = corr.mean().detach()
        else:
            metrics["loss_corr"] = torch.tensor(0.0, device=pred.device)
            metrics["mean_batch_corr"] = torch.tensor(0.0, device=pred.device)

        if epoch >= self.var_start_epoch and valid_mask.any():
            p_std = pred[:, valid_mask].std(dim=0, unbiased=False).clamp_min(self.eps)
            t_std = target[:, valid_mask].std(dim=0, unbiased=False).clamp_min(self.eps)

            var_loss = torch.abs(torch.log(p_std) - torch.log(t_std)).mean()

            loss = loss + self.lambda_var * var_loss
            metrics["loss_var"] = var_loss.detach()
            metrics["pred_std_mean"] = p_std.mean().detach()
            metrics["true_std_mean"] = t_std.mean().detach()
        else:
            metrics["loss_var"] = torch.tensor(0.0, device=pred.device)

        metrics["loss_total"] = loss.detach()
        return loss, metrics

class FixedCovGaussianLoss(nn.Module):
    def __init__(self, covariance, normalize_by_dim=True):
        super().__init__()
        covariance = covariance.float()
        dim = covariance.size(0)

        chol = torch.linalg.cholesky(covariance)
        precision = torch.cholesky_inverse(chol)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()

        self.register_buffer("precision", precision)
        self.register_buffer("logdet", logdet)
        self.dim = dim
        self.normalize_by_dim = normalize_by_dim

    def forward(self, pred, target, epoch=None):
        resid = pred.float() - target.float()
        precision = self.precision.float()
        mahal = (resid @ precision * resid).sum(dim=1)

        if self.normalize_by_dim:
            loss_gaussian = 0.5 * mahal.mean() / self.dim
        else:
            loss_gaussian = 0.5 * mahal.mean()

        metrics = {
            "loss_gaussian": loss_gaussian.detach(),
            "mahal_mean": mahal.mean().detach(),
            "loss_mse": F.mse_loss(pred, target).detach(),
            "loss_total": loss_gaussian.detach(),
        }
        return loss_gaussian, metrics

def estimate_fixed_covariance(dataset, row_indices, gene_indices, shrinkage=0.1, ridge=1e-4, chunk_size=8192):
    expr = dataset._expr
    gene_mean = dataset.gene_mean.numpy()[gene_indices].astype(np.float64)
    gene_std = dataset.gene_std.numpy()[gene_indices].astype(np.float64)
    gene_std = np.maximum(gene_std, 1e-8)
    rows = dataset._mmap_rows[row_indices]
    genes = np.asarray(gene_indices, dtype=np.int64)

    n = 0
    dim = len(genes)
    x_sum = np.zeros(dim, dtype=np.float64)
    xx_sum = np.zeros((dim, dim), dtype=np.float64)

    for start in range(0, len(rows), chunk_size):
        batch_rows = rows[start:start + chunk_size]
        x = np.asarray(expr[batch_rows][:, genes], dtype=np.float64)
        x = (x - gene_mean) / gene_std
        x_sum += x.sum(axis=0)
        xx_sum += x.T @ x
        n += x.shape[0]

    if n < 2:
        raise ValueError("Need at least two training cells to estimate covariance.")

    cov = (xx_sum - np.outer(x_sum, x_sum) / n) / (n - 1)
    diag = np.clip(np.diag(cov), ridge, None)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(diag)
    cov.flat[::dim + 1] += ridge
    return torch.from_numpy(cov.astype(np.float32))

def pearson_r_multi(pred, target):
    if pred.size(0) < 2:
        return torch.zeros(pred.size(1), device=pred.device)
    
    pm = pred - pred.mean(dim=0, keepdim=True)
    tm = target - target.mean(dim=0, keepdim=True)
    
    num = (pm * tm).sum(dim=0)
    den = torch.sqrt((pm.pow(2).sum(dim=0) * tm.pow(2).sum(dim=0)) + 1e-8)
    return num / den

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='crc')
    p.add_argument('--split_layout', type=str, default='default', help='Custom layout folder name under splits/')
    p.add_argument('--split_mode', type=str, default='spatial', choices=['spatial', 'random'], help='Data split strategy')
    p.add_argument('--top_csv', type=str, 
                   default='cache_crc/per_gene/top400_genes_by_coverage.csv')
    p.add_argument('--batch_size', type=int, default=256, help='Reduced for LoRA compatibility')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4) # Standard for LoRA
    p.add_argument('--lora_lr_scale', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=15)
    
    p.add_argument('--loss_type', type=str, default='mixed', choices=['mse', 'mixed', 'fixed_cov_gaussian'])
    p.add_argument('--lambda_corr', type=float, default=0.2)
    p.add_argument('--lambda_var', type=float, default=0.05)
    p.add_argument('--loss_start_epoch', type=int, default=3)
    p.add_argument('--cov_shrinkage', type=float, default=0.1, help='Diagonal shrinkage for fixed covariance Gaussian loss')
    p.add_argument('--cov_ridge', type=float, default=1e-4, help='Ridge added to covariance diagonal for fixed covariance Gaussian loss')
    p.add_argument('--cov_chunk_size', type=int, default=8192, help='Rows per chunk when estimating target covariance')
    
    # LoRA / Model Parameters
    p.add_argument('--model_name', type=str, default='vit_base_patch16_dinov3.lvd1689m')
    p.add_argument('--ckpt_path', type=str, default='experiments/router_nobreast_vitb_gate_r16_mlp_cw/best.pt')
    p.add_argument('--hidden_dim', type=int, default=1024, help='Hidden in MLP head')
    p.add_argument('--lora_blocks', type=str, default="0,2,4,6,8,10,11")
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--lora_targets', type=str, default="qkv,proj,mlp_fc1,mlp_fc2")
    p.add_argument('--scales', type=str, default="10.0x", help='Comma-separated scales, e.g. "10.0x" or "2.5x,10.0x"')
    p.add_argument('--fuse', type=str, default="identity", choices=["identity", "gate"])
    p.add_argument('--augment', action='store_true', default=False, help='If flagged, apply vision augmentations at training')
    p.add_argument('--exp_tag', type=str, default='', help='Optional tag to prepend to output directory name')
    
    # Hybrid Split Parameters
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--base_seed', type=int, default=42)
    p.add_argument('--n_test_tiles', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.1)
    
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    csv_path = PROJECT / args.top_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find ranking CSV at {csv_path}")
        
    df = pd.read_csv(csv_path)
    gene_indices = df['gene_idx'].values.astype(int)
    gene_names = df['gene_name'].values
    out_features = len(gene_indices)
    
    print(f"=======================================================")
    print(f" Multi-Task LORA Probing - {args.split_mode.title()} Split")
    print(f" Dataset: {args.dataset} | Layout: {args.split_layout}")
    print(f" Target Genes: {out_features} | LoRA Blocks: {args.lora_blocks}")
    if args.split_mode == 'spatial':
        print(f" Test Tiles: {args.n_test_tiles} | Internal Val: {args.val_frac*100:.0f}%")
    else:
        print(f" Train: 70% | Val: 15% | Test: 15%")
    print(f"=======================================================")
    
    # Setup output dir
    scales_list = [s.strip() for s in args.scales.split(',')]
    scale_tag = '+'.join(scales_list)
    
    if args.loss_type == 'mixed':
        loss_suffix = f"mixed_c{args.lambda_corr}_v{args.lambda_var}"
    elif args.loss_type == 'fixed_cov_gaussian':
        loss_suffix = f"fixedcov_s{args.cov_shrinkage}_r{args.cov_ridge}"
    else:
        loss_suffix = "mse"
        
    prefix = f"{args.exp_tag}_" if args.exp_tag else ""
    split_tag = args.split_layout if args.split_mode == 'spatial' else 'random'
    out_dir = PROJECT / "experiments" / f"{prefix}lora_probing_{args.dataset}_top{out_features}_multi_{split_tag}_{scale_tag}_{loss_suffix}_seed{args.base_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Memmap Dataset instantiation
    cache_dir = PROJECT / f"cache_{args.dataset}"
    print(f"Mapping raw dataset via VisiumHDPredictionDataset from {cache_dir}...")
    
    # Make master datasets. One with augment for training, one clean for eval. Routing to split_layout folder.
    ds_train_master = VisiumHDPredictionDataset(cache_dir, split="all", scales=scales_list, fuse=args.fuse, split_type="spatial", augment=args.augment, split_layout=args.split_layout)
    ds_eval_master  = VisiumHDPredictionDataset(cache_dir, split="all", scales=scales_list, fuse=args.fuse, split_type="spatial", augment=False, split_layout=args.split_layout)
    
    # Grab the aligned meta mapping structure
    meta = pd.read_csv(cache_dir / "meta.csv")
    layout_dir = cache_dir / "splits" / args.split_layout
    if layout_dir.exists():
        splits = pd.read_csv(layout_dir / "splits.csv")
        meta = meta.merge(splits, on="mmap_idx", how="inner")
    elif "split" not in meta.columns:
        meta["split"] = "train"
    
    # Force all_meta and master datasets to mimic the exact concatenated ordering used in Frozen MLP!
    all_meta_ordered = []
    for spl in ["train", "val", "test"]:
        df_spl = meta[meta["split"] == spl].copy()
        all_meta_ordered.append(df_spl)
    all_meta = pd.concat(all_meta_ordered, axis=0).reset_index(drop=True)
    
    # Overwrite the default dataset mappings to strictly match our concatenated timeline
    ds_train_master._mmap_rows = all_meta["mmap_idx"].values.astype(np.int64)
    ds_train_master._cell_ids  = all_meta["cell_id"].values
    
    ds_eval_master._mmap_rows = all_meta["mmap_idx"].values.astype(np.int64)
    ds_eval_master._cell_ids  = all_meta["cell_id"].values
    
    all_seed_results = []
    t0_overall = time.time()
    
    # --- MULTI-SEED LOOP ---
    for seed_idx in range(args.n_seeds):
        current_seed = args.base_seed + seed_idx
        
        print(f"\n────────────────────────────────────────────────────────")
        print(f" ► RUNNING SEED {current_seed} ({seed_idx + 1}/{args.n_seeds})")
        print(f"────────────────────────────────────────────────────────")
        
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        
        if args.split_mode == 'random':
            print(f"  Using random cell-level splits (70/15/15)")
            n_cells = len(all_meta)
            indices = np.arange(n_cells)
            np.random.shuffle(indices)
            
            n_train = int(n_cells * 0.70)
            n_val   = int(n_cells * 0.15)
            
            train_idx = indices[:n_train]
            val_idx   = indices[n_train:n_train+n_val]
            test_idx  = indices[n_train+n_val:]
            test_tiles = []
        elif layout_dir.exists():
            print(f"  Using deterministic pre-computed splits from layout: '{args.split_layout}'")
            train_idx = np.where(all_meta['split'] == 'train')[0]
            val_idx   = np.where(all_meta['split'] == 'val')[0]
            test_idx  = np.where(all_meta['split'] == 'test')[0]
            
            layout_json = layout_dir / "split_stats.json"
            if layout_json.exists():
                try:
                    test_tiles = json.loads(layout_json.read_text()).get("test_tiles", [])
                except Exception:
                    test_tiles = all_meta[all_meta['split'] == 'test']['tile_id'].unique()
            else:
                test_tiles = all_meta[all_meta['split'] == 'test']['tile_id'].unique()
        else:
            unique_tiles = all_meta['tile_id'].unique()
            test_tiles = np.random.choice(unique_tiles, size=args.n_test_tiles, replace=False)
            test_mask  = all_meta['tile_id'].isin(test_tiles).values
            train_pool_mask = ~test_mask
            
            test_idx       = np.where(test_mask)[0]
            train_pool_idx = np.where(train_pool_mask)[0]
            
            np.random.shuffle(train_pool_idx)
            n_val = int(len(train_pool_idx) * args.val_frac)
            val_idx   = train_pool_idx[:n_val]
            train_idx = train_pool_idx[n_val:]
        
        print(f"  Test Tiles ({args.n_test_tiles}): {list(test_tiles)}")
        print(f"  Cell Splits -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # We manually slice the mmap_row mappings to ensure memory isolation and dataset properties logic
        train_ds = copy.copy(ds_train_master)
        train_ds._mmap_rows = ds_train_master._mmap_rows[train_idx]
        train_ds._cell_ids  = ds_train_master._cell_ids[train_idx]
        
        val_ds = copy.copy(ds_eval_master)
        val_ds._mmap_rows = ds_eval_master._mmap_rows[val_idx]
        val_ds._cell_ids  = ds_eval_master._cell_ids[val_idx]
        
        test_ds = copy.copy(ds_eval_master)
        test_ds._mmap_rows = ds_eval_master._mmap_rows[test_idx]
        test_ds._cell_ids  = ds_eval_master._cell_ids[test_idx]
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # ── Model & Optimizer Setup ──
        ckpt_path_str = None
        if args.ckpt_path and args.ckpt_path.lower() not in ("none", "null", ""):
            checkpoint = PROJECT / args.ckpt_path if not Path(args.ckpt_path).is_absolute() else Path(args.ckpt_path)
            if checkpoint.is_file():
                ckpt_path_str = str(checkpoint)
            else:
                print(f"  [Warning] Specified ckpt_path '{args.ckpt_path}' not found as a file. Proceeding with base backbone weights.")

        model = VisiumRegressor(
            model_name=args.model_name,
            img_size=224,
            out_dim=out_features,
            fuse=args.fuse,
            freeze_backbone=True,
            lora_blocks=args.lora_blocks,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets,
            unfreeze_lora=True,
            ckpt_path=ckpt_path_str,
            head_type="mlp",
        )
        # Adapt regression head explicitly to our multi-task objective to strictly copy MLPMultiHead
        d_embed = model.head[0].in_features if isinstance(model.head, nn.Sequential) else model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(d_embed, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(args.hidden_dim, out_features)
        )
        model = model.to(device)
        
        param_groups = model.get_param_groups(
            lr=args.lr,
            weight_decay=1e-4,
            gate_wd=0.1,
            lora_lr_scale=args.lora_lr_scale,
        )
        optimizer = torch.optim.AdamW(param_groups)
        
        criterion_mixed = MixedGeneLoss(
            lambda_corr=args.lambda_corr, 
            lambda_var=args.lambda_var, 
            corr_start_epoch=args.loss_start_epoch, 
            var_start_epoch=args.loss_start_epoch
        )
        criterion_fixed_cov = None
        if args.loss_type == 'fixed_cov_gaussian':
            print("  Estimating fixed target covariance from training cells...")
            fixed_cov = estimate_fixed_covariance(
                ds_train_master,
                train_idx,
                gene_indices,
                shrinkage=args.cov_shrinkage,
                ridge=args.cov_ridge,
                chunk_size=args.cov_chunk_size,
            )
            criterion_fixed_cov = FixedCovGaussianLoss(fixed_cov).to(device)
            print(f"  Fixed covariance Gaussian loss: dim={out_features}, shrinkage={args.cov_shrinkage}, ridge={args.cov_ridge}")
        
        warmup_epochs = 5
        sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=warmup_epochs)
        sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup_epochs])
        
        best_mean_val_r = -float('inf')
        epochs_no_improve = 0
        best_model_state = None
        best_val_r_per_gene = None
        
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        t0_seed = time.time()
        
        for ep in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            n_train = 0
            last_metrics = None
            
            for imgs, y_full, _ in train_loader:
                x = imgs.to(device, non_blocking=True)
                y = y_full[:, gene_indices].float().to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                    pred = model(x)
                    if args.loss_type == 'mixed':
                        loss, metrics = criterion_mixed(pred, y, epoch=ep)
                        last_metrics = metrics
                    elif args.loss_type == 'fixed_cov_gaussian':
                        loss, metrics = criterion_fixed_cov(pred, y, epoch=ep)
                        last_metrics = metrics
                    else:
                        loss = F.mse_loss(pred, y)
                        last_metrics = {"loss_mse": loss.detach()}
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                train_loss += loss.item() * len(y)
                n_train += len(y)
                
            scheduler.step()
            train_total_loss = train_loss / max(1, n_train)
            
            # ── Validation ──
            model.eval()
            val_loss = 0.0
            n_val_cells = 0
            all_preds, all_gts = [], []
            
            with torch.no_grad():
                for imgs, y_full, _ in val_loader:
                    x = imgs.to(device, non_blocking=True)
                    y = y_full[:, gene_indices].float().to(device, non_blocking=True)
                    
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                        pred = model(x)
                        vloss = F.mse_loss(pred, y)
                        
                    val_loss += vloss.item() * len(y)
                    n_val_cells += len(y)
                    all_preds.append(pred.float().cpu())
                    all_gts.append(y.float().cpu())
                    
            val_mse = val_loss / max(1, n_val_cells)
            all_preds = torch.cat(all_preds)
            all_gts = torch.cat(all_gts)
            
            val_r_per_gene = pearson_r_multi(all_preds, all_gts)
            mean_val_r = val_r_per_gene.mean().item()
            
            if ep % 5 == 0 or ep == 1:
                if args.loss_type == 'mse':
                    print(f"    Ep {ep:>3d} | Train Loss: {train_total_loss:.4f} | Val R: {mean_val_r:.4f}")
                elif args.loss_type == 'fixed_cov_gaussian':
                    l_mse = last_metrics.get('loss_mse', 0.0)
                    l_gauss = last_metrics.get('loss_gaussian', 0.0)
                    l_mahal = last_metrics.get('mahal_mean', 0.0)
                    print(f"    Ep {ep:>3d} | Train Gaussian: {train_total_loss:.4f} [Batch MSE: {l_mse:.4f} | Gaussian: {l_gauss:.4f} | Mahalanobis: {l_mahal:.4f}] | Val R: {mean_val_r:.4f}")
                elif ep < criterion_mixed.corr_start_epoch:
                    print(f"    Ep {ep:>3d} | Train Loss: {train_total_loss:.4f} | Val R: {mean_val_r:.4f}")
                else:
                    l_mse = last_metrics.get('loss_mse', 0.0)
                    l_corr = last_metrics.get('loss_corr', 0.0)
                    l_var = last_metrics.get('loss_var', 0.0)
                    print(f"    Ep {ep:>3d} | Train Total: {train_total_loss:.4f} [MSE: {l_mse:.4f} | Corr: {l_corr:.4f} | Var: {l_var:.4f}] | Val R: {mean_val_r:.4f}")
            
            if mean_val_r > best_mean_val_r:
                best_mean_val_r = mean_val_r
                epochs_no_improve = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_val_r_per_gene = val_r_per_gene
                torch.save(best_model_state, out_dir / f'best_model_seed_{current_seed}.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"    Early stopping triggered at epoch {ep}!")
                    break
                    
        total_time_seed = time.time() - t0_seed
        print(f"  Training Complete in {total_time_seed:.1f}s | Best Inner Validation Pearson: {best_mean_val_r:.4f}")
        
        # ── Test Evaluation ──
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()
        
        all_t_preds, all_t_gts = [], []
        with torch.no_grad():
            for imgs, y_full, _ in test_loader:
                x = imgs.to(device, non_blocking=True)
                y = y_full[:, gene_indices].float().to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                    pred = model(x)
                all_t_preds.append(pred.float().cpu())
                all_t_gts.append(y.float().cpu())
                
        all_t_preds = torch.cat(all_t_preds)
        all_t_gts = torch.cat(all_t_gts)
        test_r_per_gene = pearson_r_multi(all_t_preds, all_t_gts)
        
        np.save(out_dir / f'test_y_pred_seed_{current_seed}.npy', all_t_preds.numpy())
        np.save(out_dir / f'test_y_true_seed_{current_seed}.npy', all_t_gts.numpy())
        
        mean_test_r = test_r_per_gene.mean().item()
        
        print(f"  Final Test (OOD Specific Tiles) Pearson: {mean_test_r:.4f}")
        
        res_df = pd.DataFrame({
            'gene_idx': gene_indices,
            'gene_name': gene_names,
            f'val_pearson_s{current_seed}': best_val_r_per_gene.numpy(),
            f'test_pearson_s{current_seed}': test_r_per_gene.numpy()
        })
        
        seed_info = {
            'seed':          int(current_seed),
            'test_tiles':    [int(t) for t in test_tiles],
            'n_train_cells': int(len(train_idx)),
            'n_val_cells':   int(len(val_idx)),
            'n_test_cells':  int(len(test_idx)),
            'best_val_r':    float(best_mean_val_r),
            'test_r':        float(mean_test_r),
            'training_time': float(total_time_seed),
            'hyperparams': {
                'hidden_dim': args.hidden_dim,
                'batch_size': args.batch_size,
                'lr':         args.lr,
                'lora_lr_scale': args.lora_lr_scale,
                'loss_type':  args.loss_type,
                'lambda_corr': args.lambda_corr,
                'lambda_var': args.lambda_var,
                'loss_start_epoch': args.loss_start_epoch,
                'cov_shrinkage': args.cov_shrinkage,
                'cov_ridge': args.cov_ridge,
                'augment': args.augment,
                'exp_tag': args.exp_tag,
                'ckpt_path': args.ckpt_path
            }
        }
        info_file = out_dir / f'seed_{current_seed}_info.json'
        info_file.write_text(json.dumps(seed_info, indent=2))
        np.savez(out_dir / f'splits_seed_{current_seed}.npz',
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, test_tiles=test_tiles)
            
        all_seed_results.append((current_seed, mean_test_r, res_df))

    total_time = time.time() - t0_overall
    print(f"\n=======================================================")
    print(f" LORA SUMMARY ACROSS ALL {args.n_seeds} SEEDS (Time: {total_time/60:.1f} min)")
    test_means = [r[1] for r in all_seed_results]
    print(f" Individual Seed Test R: {[f'{v:.4f}' for v in test_means]}")
    print(f" OVERALL OOD TEST R (Mean ± Std): {np.mean(test_means):.4f} ± {np.std(test_means):.4f}")
    
    if not all_seed_results:
        return
        
    final_df = all_seed_results[0][2].copy()
    for r in all_seed_results[1:]:
        final_df = pd.merge(final_df, r[2], on=['gene_idx', 'gene_name'], how='left')
        
    test_cols = [c for c in final_df.columns if 'test_pearson_s' in c]
    val_cols  = [c for c in final_df.columns if 'val_pearson_s' in c]
    
    final_df['test_pearson_mean'] = final_df[test_cols].mean(axis=1)
    final_df['test_pearson_std']  = final_df[test_cols].std(axis=1)
    
    csv_out_path = out_dir / "multi_lora_hybrid_results.csv"
    final_df.to_csv(csv_out_path, index=False)
    print(f"Saved LORA independent and averaged seed results to {csv_out_path}")

if __name__ == "__main__":
    main()
