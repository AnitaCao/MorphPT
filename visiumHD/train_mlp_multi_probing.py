#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')

class MLPMultiHead(nn.Module):
    def __init__(self, in_features, hidden_dim=1024, out_features=400, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x):
        return self.net(x) # Output (B, 400)

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
    """
    Calculate Pearson correlation coefficient for each gene independently.
    pred: (N, G)
    target: (N, G)
    Returns: (G,) tensor of Pearson correlations
    """
    if pred.size(0) < 2:
        return torch.zeros(pred.size(1), device=pred.device)
    
    pm = pred - pred.mean(dim=0, keepdim=True)
    tm = target - target.mean(dim=0, keepdim=True)
    
    num = (pm * tm).sum(dim=0)
    # Using + 1e-8 before sqrt bypasses singularity and ensures safe gradients
    den = torch.sqrt((pm.pow(2).sum(dim=0) * tm.pow(2).sum(dim=0)) + 1e-8)
    
    corrs = num / den
    return corrs

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='crc')
    p.add_argument('--split_layout', type=str, default='default', help='Custom layout folder name under splits/')
    p.add_argument('--top_csv', type=str, 
                   default='cache_crc/per_gene/top400_genes_by_coverage.csv')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension size for the MLP')
    p.add_argument('--loss_type', type=str, default='mixed', choices=['mse', 'mixed'], help='Loss formulation to use')
    p.add_argument('--lambda_corr', type=float, default=0.2, help='Weight for correlation loss')
    p.add_argument('--lambda_var', type=float, default=0.05, help='Weight for variance matching loss')
    p.add_argument('--loss_start_epoch', type=int, default=3, help='Epoch to start applying mixed loss components')
    
    # Hybrid Split Parameters
    p.add_argument('--n_seeds', type=int, default=3, help='Number of independent seeds to run')
    p.add_argument('--base_seed', type=int, default=42, help='Base random seed')
    p.add_argument('--n_test_tiles', type=int, default=4, help='Number of tiles to reserve purely for Test (OOD)')
    p.add_argument('--val_frac', type=float, default=0.1, help='Fraction of remaining pool to use for early stopping Val (IID)')
    
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    csv_path = PROJECT / args.top_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find ranking CSV at {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Ensure correct format (avoid nan errors if CSVs change)
    gene_indices = df['gene_idx'].values.astype(int)
    gene_names = df['gene_name'].values
    out_features = len(gene_indices)
    
    print(f"=======================================================")
    print(f" Multi-Task MLP Probing - Hybrid Spatial/Random Split")
    print(f" Dataset: {args.dataset} | Layout: {args.split_layout}")
    print(f" Number of target genes: {out_features}")
    print(f" Test Tiles: {args.n_test_tiles} | Internal Val: {args.val_frac*100:.0f}%")
    print(f" Running {args.n_seeds} independent seeds")
    print(f"=======================================================")
    
    # Setup output dir
    if args.loss_type == 'mixed':
        loss_suffix = f"mixed_c{args.lambda_corr}_v{args.lambda_var}"
    else:
        loss_suffix = "mse"
        
    out_dir = PROJECT / "experiments" / f"mlp_probing_{args.dataset}_multi_{args.split_layout}_{loss_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pre-extracted data routing
    cache_dir = PROJECT / f"cache_{args.dataset}"
    layout_dir = cache_dir / "splits" / args.split_layout
    
    if layout_dir.exists() and (layout_dir / "embeddings_morphpt").exists():
        emb_dir = layout_dir / "embeddings_morphpt"
        print(f"Loading pre-extracted features from decoupled layout: '{args.split_layout}' -> {emb_dir}")
        meta = pd.read_csv(cache_dir / "meta.csv")
        splits = pd.read_csv(layout_dir / "splits.csv")
        meta = meta.merge(splits, on="mmap_idx", how="inner")
    else:
        emb_dir = cache_dir / "embeddings_morphpt"
        print(f"Loading pre-extracted features from root cache -> {emb_dir}")
        meta = pd.read_csv(cache_dir / "meta.csv")
        if "split" not in meta.columns:
            meta["split"] = "train"
    
    print("Loading all data subsets into RAM for dynamic slicing...")
    feats_list, expr_list, df_list = [], [], []
    
    for spl in ["train", "val", "test"]:
        feat_path = emb_dir / f"{spl}_features.pt"
        expr_path = emb_dir / f"{spl}_expr.pt"
        if not feat_path.exists():
            continue
            
        feats = torch.load(feat_path, map_location='cpu')
        expr_all = torch.load(expr_path, map_location='cpu')
        
        # Only take target genes
        expr = expr_all[:, gene_indices].float()
        
        feats_list.append(feats)
        expr_list.append(expr)
        
        df_spl = meta[meta["split"] == spl].copy().reset_index(drop=True)
        df_list.append(df_spl)
        
    # Cat into gigantic memory pool
    all_feats = torch.cat(feats_list, dim=0)
    all_exprs = torch.cat(expr_list, dim=0)
    all_meta  = pd.concat(df_list, axis=0).reset_index(drop=True)
    in_dim    = all_feats.shape[1]
    
    print(f"Combined Pool Dimension: {all_feats.shape[0]} cells × {in_dim} features")
    
    all_seed_results = []
    
    # --- MULTI-SEED LOOP ---
    t0_overall = time.time()
    for seed_idx in range(args.n_seeds):
        current_seed = args.base_seed + seed_idx
        
        print(f"\n────────────────────────────────────────────────────────")
        print(f" ► RUNNING SEED {current_seed} ({seed_idx + 1}/{args.n_seeds})")
        print(f"────────────────────────────────────────────────────────")
        
        # ── 1. Dynamic Split Logic ──
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        
        # Select purely OOD tiles for Test
        if layout_dir.exists():
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
        
        print(f"  Test Tiles ({args.n_test_tiles}): {test_tiles.tolist()}")
        print(f"  Cell Splits -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # ── 2. Instantiate Loaders ──
        # Use Subset to guarantee zero-copy views (advanced tensor indexing like all_feats[idx] creates copies in RAM)
        full_ds = TensorDataset(all_feats, all_exprs)
        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        val_ds   = torch.utils.data.Subset(full_ds, val_idx)
        test_ds  = torch.utils.data.Subset(full_ds, test_idx)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # ── 3. Model & Optimizer Setup ──
        model = MLPMultiHead(in_features=in_dim, hidden_dim=args.hidden_dim, out_features=out_features, dropout=0.15).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
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
        
        t0_seed = time.time()
        
        # ── 4. Training Loop ──
        for ep in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            n_train = 0
            
            last_metrics = None
            
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    pred = model(x)
                    
                    if args.loss_type == 'mixed':
                        loss, metrics = criterion_mixed(pred, y, epoch=ep)
                        last_metrics = metrics
                    else:
                        loss = F.mse_loss(pred, y)
                        last_metrics = {"loss_mse": loss.detach()}
                    
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
                n_train += len(y)
                
            scheduler.step()
            train_total_loss = train_loss / max(1, n_train)
            
            # Validation
            model.eval()
            val_loss = 0.0
            n_val_cells = 0
            all_preds, all_gts = [], []
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        pred = model(x)
                        loss = F.mse_loss(pred, y)
                    val_loss += loss.item() * len(y)
                    n_val_cells += len(y)
                    all_preds.append(pred.cpu())
                    all_gts.append(y.cpu())
                    
            val_mse = val_loss / max(1, n_val_cells)
            all_preds = torch.cat(all_preds)
            all_gts = torch.cat(all_gts)
            
            val_r_per_gene = pearson_r_multi(all_preds, all_gts)
            mean_val_r = val_r_per_gene.mean().item()
            
            # Print occasionally
            if ep % 5 == 0 or ep == 1:
                if args.loss_type == 'mse' or ep < criterion_mixed.corr_start_epoch:
                    print(f"    Ep {ep:>3d} | Train Loss: {train_total_loss:.4f} | Val R: {mean_val_r:.4f}")
                else:
                    l_mse = last_metrics.get('loss_mse', 0.0)
                    l_corr = last_metrics.get('loss_corr', 0.0)
                    l_var = last_metrics.get('loss_var', 0.0)
                    print(f"    Ep {ep:>3d} | Train Total: {train_total_loss:.4f} [Batch MSE: {l_mse:.4f} | Corr: {l_corr:.4f} | Var: {l_var:.4f}] | Val R: {mean_val_r:.4f}")
            
            # Checkpoint & Early Stop
            if mean_val_r > best_mean_val_r:
                best_mean_val_r = mean_val_r
                epochs_no_improve = 0
                # Copy to RAM to save VRAM
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_val_r_per_gene = val_r_per_gene
                torch.save(best_model_state, out_dir / f'best_model_seed_{current_seed}.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"    Early stopping triggered at epoch {ep}!")
                    break
                    
        total_time_seed = time.time() - t0_seed
        print(f"  Training Complete in {total_time_seed:.1f}s")
        print(f"  Best Inner Validation Pearson: {best_mean_val_r:.4f}")
        
        # ── 5. Final Test Evaluation (OOD) ──
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()
        
        all_t_preds, all_t_gts = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    pred = model(x)
                all_t_preds.append(pred.cpu())
                all_t_gts.append(y.cpu())
                
        all_t_preds = torch.cat(all_t_preds)
        all_t_gts = torch.cat(all_t_gts)
        test_r_per_gene = pearson_r_multi(all_t_preds, all_t_gts)
        
        np.save(out_dir / f'test_y_pred_seed_{current_seed}.npy', all_t_preds.numpy())
        np.save(out_dir / f'test_y_true_seed_{current_seed}.npy', all_t_gts.numpy())
        
        mean_test_r = test_r_per_gene.mean().item()
        print(f"  Final Test (OOD Specific Tiles) Pearson: {mean_test_r:.4f}")
        
        # Save independent run results
        res_df = pd.DataFrame({
            'gene_idx': gene_indices,
            'gene_name': gene_names,
            f'val_pearson_s{current_seed}': best_val_r_per_gene.numpy(),
            f'test_pearson_s{current_seed}': test_r_per_gene.numpy()
        })
        
        # Save seed-specific info for reproducibility
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
                'loss_type':  args.loss_type,
                'lambda_corr': args.lambda_corr,
                'lambda_var': args.lambda_var,
                'loss_start_epoch': args.loss_start_epoch
            }
        }
        info_file = out_dir / f'seed_{current_seed}_info.json'
        info_file.write_text(json.dumps(seed_info, indent=2))
        print(f"  Saved info → {info_file}")
        
        # Save exact splits
        np.savez(out_dir / f'splits_seed_{current_seed}.npz',
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            test_tiles=test_tiles)
            
        all_seed_results.append((current_seed, mean_test_r, res_df))

    # ================= SUMMARY =================
    total_time = time.time() - t0_overall
    print(f"\n=======================================================")
    print(f" SUMMARY ACROSS ALL {args.n_seeds} SEEDS (Time: {total_time/60:.1f} min)")
    test_means = [r[1] for r in all_seed_results]
    print(f" Individual Seed Test R: {[f'{v:.4f}' for v in test_means]}")
    print(f" OVERALL OOD TEST R (Mean ± Std): {np.mean(test_means):.4f} ± {np.std(test_means):.4f}")
    print(f"=======================================================\n")
    
    # Export Aggregate Metrics
    if not all_seed_results:
        return
        
    final_df = all_seed_results[0][2].copy()
    for r in all_seed_results[1:]:
        final_df = pd.merge(final_df, r[2], on=['gene_idx', 'gene_name'], how='left')
        
    test_cols = [c for c in final_df.columns if 'test_pearson_s' in c]
    val_cols  = [c for c in final_df.columns if 'val_pearson_s' in c]
    
    final_df['test_pearson_mean'] = final_df[test_cols].mean(axis=1)
    final_df['test_pearson_std']  = final_df[test_cols].std(axis=1)
    
    csv_out_path = out_dir / "multi_mlp_hybrid_results.csv"
    final_df.to_csv(csv_out_path, index=False)
    print(f"Saved independent and averaged seed results to {csv_out_path}")

if __name__ == "__main__":
    main()
