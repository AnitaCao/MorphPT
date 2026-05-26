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
    p.add_argument('--top_csv', type=str, 
                   default='cache_crc/per_gene/top400_genes_by_coverage.csv')
    p.add_argument('--batch_size', type=int, default=256, help='Reduced for LoRA compatibility')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4) # Standard for LoRA
    p.add_argument('--lora_lr_scale', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=15)
    
    p.add_argument('--loss_type', type=str, default='mixed', choices=['mse', 'mixed'])
    p.add_argument('--lambda_corr', type=float, default=0.2)
    p.add_argument('--lambda_var', type=float, default=0.05)
    p.add_argument('--loss_start_epoch', type=int, default=3)
    
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
    
    # Hybrid Split Parameters
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--base_seed', type=int, default=42)
    p.add_argument('--n_test_tiles', type=int, default=3)               # ← CHANGED default 4 → 3
    p.add_argument('--test_tiles', type=str, default=None,              # ← NEW
                   help='Comma-separated tile IDs for the test set, '
                        'e.g. "3,17,23". If set, fixes test tiles across all seeds '
                        '(train/val split still varies per seed). Otherwise random per seed.')
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
    
    # ─── NEW: parse explicit test tiles upfront so we can validate + tag the out_dir ───
    if args.test_tiles is not None:
        explicit_test_tiles = np.array([int(t) for t in args.test_tiles.split(',')])
        if len(explicit_test_tiles) != args.n_test_tiles:
            print(f"[Note] --test_tiles has {len(explicit_test_tiles)} tiles; "
                  f"overriding --n_test_tiles={args.n_test_tiles} → {len(explicit_test_tiles)}")
            args.n_test_tiles = len(explicit_test_tiles)
        tile_suffix = "_tiles" + "-".join(str(int(t)) for t in explicit_test_tiles)
    else:
        explicit_test_tiles = None
        tile_suffix = ""
    
    print(f"=======================================================")
    print(f" Multi-Task LORA Probing - Hybrid Spatial/Random Split")
    print(f" Target Genes: {out_features} | LoRA Blocks: {args.lora_blocks}")
    print(f" Test Tiles: {args.n_test_tiles} | Internal Val: {args.val_frac*100:.0f}%")
    if explicit_test_tiles is not None:
        print(f" Explicit test tiles: {explicit_test_tiles.tolist()}")
    print(f"=======================================================")
    
    # Setup output dir
    scales_list = [s.strip() for s in args.scales.split(',')]
    scale_tag = '+'.join(scales_list)
    
    if args.loss_type == 'mixed':
        loss_suffix = f"mixed_c{args.lambda_corr}_v{args.lambda_var}"
    else:
        loss_suffix = "mse"
        
    out_dir = PROJECT / "experiments" / (
        f"lora_probing_{args.dataset}_top{out_features}_multi_"
        f"{scale_tag}_{loss_suffix}_seed{args.base_seed}{tile_suffix}"   # ← CHANGED: tile_suffix tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Memmap Dataset instantiation
    cache_dir = PROJECT / f"cache_{args.dataset}"
    print(f"Mapping raw dataset via VisiumHDPredictionDataset from {cache_dir}...")
    
    # Make master datasets. One with augment for training, one clean for eval.
    ds_train_master = VisiumHDPredictionDataset(cache_dir, split="all", scales=scales_list, fuse=args.fuse, split_type="spatial", augment=args.augment)
    ds_eval_master  = VisiumHDPredictionDataset(cache_dir, split="all", scales=scales_list, fuse=args.fuse, split_type="spatial", augment=False)
    
    # Grab the aligned meta mapping structure (exactly aligned to _mmap_rows)
    meta = pd.read_csv(cache_dir / "meta.csv")
    
    # Force all_meta and ds_train_master to mimic the exact concatenated ordering used in Frozen MLP!
    all_meta_ordered = []
    for spl in ["train", "val", "test"]:
        df_spl = meta[meta["split"] == spl].copy()
        all_meta_ordered.append(df_spl)
    all_meta = pd.concat(all_meta_ordered, axis=0).reset_index(drop=True)
    
    # ─── NEW: validate explicit test tiles against meta ───
    if explicit_test_tiles is not None:
        avail = set(all_meta['tile_id'].unique().tolist())
        missing = sorted(set(int(t) for t in explicit_test_tiles) - avail)
        if missing:
            raise ValueError(
                f"Requested test tiles {missing} are not present in {cache_dir/'meta.csv'}. "
                f"Available tile IDs: {sorted(avail)}"
            )
    
    # Overwrite the default dataset mappings to strictly match our concatenated timeline
    # so random splits generate identical seeds as the Frozen script
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
        
        # ─── CHANGED: explicit test tiles override random sampling ───
        if explicit_test_tiles is not None:
            test_tiles = explicit_test_tiles.copy()
        else:
            unique_tiles = all_meta['tile_id'].unique()
            test_tiles = np.random.choice(unique_tiles, size=args.n_test_tiles, replace=False)
        test_mask = all_meta['tile_id'].isin(test_tiles).values
        train_pool_mask = ~test_mask
        
        test_idx       = np.where(test_mask)[0]
        train_pool_idx = np.where(train_pool_mask)[0]
        
        np.random.shuffle(train_pool_idx)
        n_val = int(len(train_pool_idx) * args.val_frac)
        val_idx   = train_pool_idx[:n_val]
        train_idx = train_pool_idx[n_val:]
        
        print(f"  Test Tiles ({args.n_test_tiles}): {test_tiles.tolist()}")
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
        checkpoint = PROJECT / args.ckpt_path
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
            ckpt_path=str(checkpoint) if checkpoint.exists() else None,
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
                if args.loss_type == 'mse' or ep < criterion_mixed.corr_start_epoch:
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
            'test_tiles_explicit': bool(explicit_test_tiles is not None),     # ← NEW: provenance
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
                'augment': args.augment
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
