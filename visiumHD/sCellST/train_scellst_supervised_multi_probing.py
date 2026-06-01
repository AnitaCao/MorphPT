#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')


def pearson_r_multi(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.size(0) < 2:
        return torch.zeros(pred.size(1), device=pred.device)

    pm = pred - pred.mean(dim=0, keepdim=True)
    tm = target - target.mean(dim=0, keepdim=True)
    num = (pm * tm).sum(dim=0)
    den = torch.sqrt((pm.pow(2).sum(dim=0) * tm.pow(2).sum(dim=0)).clamp_min(1e-8))
    return num / den


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train sCellST predictor on MorphPT HE->gene task with fixed train/val/test splits.')
    p.add_argument('--dataset', type=str, required=True, help='Dataset suffix, e.g. mouse_brain')
    p.add_argument('--split_mode', type=str, default='spatial', choices=['spatial', 'random'], help='Match MorphPT split strategy')
    p.add_argument('--split_layout', type=str, default='default', help='Split layout folder name under cache_<dataset>/splits/')
    p.add_argument('--gene_panel_csv', type=str, default='', help='Optional custom gene panel CSV. Defaults to visiumHD/gene_panel_<dataset>.csv')
    p.add_argument('--top_k', type=int, default=0, help='Use first top_k genes from gene panel (0 means use all rows)')
    p.add_argument('--embeddings_subdir', type=str, default='embeddings_morphpt', help='Subdir under split layout containing train/val/test *_features.pt and *_expr.pt')
    p.add_argument('--embeddings_dir', type=str, default='', help='Optional absolute/relative path to embedding directory with train/val/test tensors')

    p.add_argument('--hidden_dim', type=str, default='256,256,256', help='Comma-separated hidden dims for sCellST predictor MLP')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=20)

    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--base_seed', type=int, default=42)
    p.add_argument('--scellst_repo', type=str, default='/hpc/group/jilab/tc459/sCellST', help='Path to local sCellST repo for importing predictor module')
    p.add_argument('--exp_tag', type=str, default='')
    return p.parse_args()


def load_gene_panel(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, Path]:
    if args.gene_panel_csv:
        panel_path = Path(args.gene_panel_csv)
    else:
        panel_path = PROJECT / 'visiumHD' / f'gene_panel_{args.dataset}.csv'
    if not panel_path.exists():
        raise FileNotFoundError(f'Gene panel not found: {panel_path}')

    panel = pd.read_csv(panel_path)
    if 'gene_idx' not in panel.columns or 'gene_name' not in panel.columns:
        raise ValueError(f'Gene panel must contain gene_idx and gene_name columns: {panel_path}')

    if args.top_k > 0:
        panel = panel.iloc[: args.top_k].copy()

    return panel['gene_idx'].to_numpy(dtype=int), panel['gene_name'].to_numpy(), panel_path


def load_split_tensors(cache_dir: Path, split_layout: str, embeddings_subdir: str, embeddings_dir: str = ''):
    emb_dir = Path(embeddings_dir) if embeddings_dir else (cache_dir / 'splits' / split_layout / embeddings_subdir)
    req = {
        'train_x': emb_dir / 'train_features.pt',
        'val_x': emb_dir / 'val_features.pt',
        'test_x': emb_dir / 'test_features.pt',
        'train_y': emb_dir / 'train_expr.pt',
        'val_y': emb_dir / 'val_expr.pt',
        'test_y': emb_dir / 'test_expr.pt',
    }
    missing = [str(p) for p in req.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing extracted embeddings. Run visiumHD/extract_embeddings.py first. Missing files:\n'
            + '\n'.join(missing)
        )

    data = {k: torch.load(v, map_location='cpu').float() for k, v in req.items()}
    return emb_dir, data


def make_predictor(input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float, scellst_repo: Path) -> nn.Module:
    if not scellst_repo.exists():
        raise FileNotFoundError(f'sCellST repo not found: {scellst_repo}')
    sys.path.append(str(scellst_repo))
    from scellst.module.gene_predictor import GenePredictor  # pylint: disable=import-outside-toplevel

    # Targets are z-scored in MorphPT extraction, so use identity output activation.
    return GenePredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        final_activation='identity',
        hidden_dim=hidden_dims,
        dropout_rate=dropout,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    gts = []
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model({'output_embedding': x})['output_prediction']
            mse = F.mse_loss(pred, y)
            mse_sum += mse.item() * y.size(0)
            n += y.size(0)
            preds.append(pred.float().cpu())
            gts.append(y.float().cpu())

    all_preds = torch.cat(preds, dim=0)
    all_gts = torch.cat(gts, dim=0)
    r = pearson_r_multi(all_preds, all_gts)
    return mse_sum / max(1, n), r, all_preds, all_gts


def main() -> None:
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_dims = [int(x.strip()) for x in args.hidden_dim.split(',') if x.strip()]
    scellst_repo = Path(args.scellst_repo)

    cache_dir = PROJECT / f'cache_{args.dataset}'
    if not cache_dir.exists():
        raise FileNotFoundError(f'Cache dir not found: {cache_dir}')

    gene_indices, gene_names, panel_path = load_gene_panel(args)
    emb_dir, split_data = load_split_tensors(cache_dir, args.split_layout, args.embeddings_subdir, args.embeddings_dir)

    train_x = split_data['train_x']
    val_x = split_data['val_x']
    test_x = split_data['test_x']

    # If extractor saved explicit gene names, map panel names to this order.
    # This is required for direct-delivery data where gene_idx is dataset-specific.
    embed_gene_names_path = emb_dir / 'gene_names.npy'
    if embed_gene_names_path.exists():
        embed_gene_names = np.load(embed_gene_names_path, allow_pickle=False).astype(str)
        name_to_idx = {g: i for i, g in enumerate(embed_gene_names)}
        mapped_idx = np.array([name_to_idx[g] for g in gene_names if g in name_to_idx], dtype=int)
        mapped_names = np.array([g for g in gene_names if g in name_to_idx])
        if len(mapped_idx) == 0:
            raise ValueError('No overlap between gene panel names and embedding gene names.')
        if len(mapped_idx) < len(gene_names):
            print(f'[Warning] {len(gene_names) - len(mapped_idx)} panel genes missing in embedding genes; using {len(mapped_idx)} matched genes.')
        gene_indices = mapped_idx
        gene_names = mapped_names

    train_y = split_data['train_y'][:, gene_indices]
    val_y = split_data['val_y'][:, gene_indices]
    test_y = split_data['test_y'][:, gene_indices]

    in_dim = train_x.shape[1]
    out_dim = len(gene_indices)

    prefix = f"{args.exp_tag}_" if args.exp_tag else ''
    split_tag = args.split_layout if args.split_mode == 'spatial' else 'random'
    out_dir = PROJECT / 'experiments' / f"{prefix}scellst_supervised_{args.dataset}_top{out_dim}_{split_tag}_seed{args.base_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=======================================================')
    print(f'sCellST supervised probing on MorphPT features')
    print(f'Dataset: {args.dataset} | Split: {args.split_mode} | Layout: {args.split_layout}')
    print(f'Embedding dir: {emb_dir}')
    print(f'Gene panel: {panel_path} | Targets: {out_dim}')
    print(f'Shapes train/val/test: {tuple(train_x.shape)} / {tuple(val_x.shape)} / {tuple(test_x.shape)}')
    print('=======================================================')

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_seed_results = []
    t0_all = time.time()

    for seed_i in range(args.n_seeds):
        seed = args.base_seed + seed_i
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = make_predictor(in_dim, out_dim, hidden_dims, args.dropout, scellst_repo).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_r = -float('inf')
        best_state = None
        best_val_r_per_gene = None
        no_improve = 0

        t0_seed = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            tr_loss_sum = 0.0
            n_tr = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model({'output_embedding': x})['output_prediction']
                loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                tr_loss_sum += loss.item() * y.size(0)
                n_tr += y.size(0)

            tr_loss = tr_loss_sum / max(1, n_tr)
            val_mse, val_r_per_gene, _, _ = evaluate(model, val_loader, device)
            mean_val_r = val_r_per_gene.mean().item()

            if ep % 5 == 0 or ep == 1:
                print(f'Seed {seed} | Ep {ep:>3d} | Train MSE {tr_loss:.4f} | Val MSE {val_mse:.4f} | Val R {mean_val_r:.4f}')

            if mean_val_r > best_val_r:
                best_val_r = mean_val_r
                no_improve = 0
                best_val_r_per_gene = val_r_per_gene.clone()
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(best_state, out_dir / f'best_model_seed_{seed}.pt')
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f'Seed {seed}: early stop at epoch {ep}')
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_mse, test_r_per_gene, test_pred, test_true = evaluate(model, test_loader, device)
        mean_test_r = test_r_per_gene.mean().item()

        np.save(out_dir / f'test_y_pred_seed_{seed}.npy', test_pred.numpy())
        np.save(out_dir / f'test_y_true_seed_{seed}.npy', test_true.numpy())

        info = {
            'seed': int(seed),
            'dataset': args.dataset,
            'split_mode': args.split_mode,
            'split_layout': args.split_layout,
            'embeddings_subdir': args.embeddings_subdir,
            'gene_panel_csv': str(panel_path),
            'n_train_cells': int(len(train_x)),
            'n_val_cells': int(len(val_x)),
            'n_test_cells': int(len(test_x)),
            'input_dim': int(in_dim),
            'output_dim': int(out_dim),
            'best_val_r': float(best_val_r),
            'test_r': float(mean_test_r),
            'test_mse': float(test_mse),
            'training_time_s': float(time.time() - t0_seed),
            'hyperparams': {
                'hidden_dim': hidden_dims,
                'dropout': args.dropout,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'patience': args.patience,
            },
        }
        (out_dir / f'seed_{seed}_info.json').write_text(json.dumps(info, indent=2))

        seed_df = pd.DataFrame({
            'gene_idx': gene_indices,
            'gene_name': gene_names,
            f'val_pearson_s{seed}': best_val_r_per_gene.numpy(),
            f'test_pearson_s{seed}': test_r_per_gene.numpy(),
        })
        all_seed_results.append((seed, mean_test_r, seed_df))

        print(f'Seed {seed} done | Best Val R {best_val_r:.4f} | Test R {mean_test_r:.4f} | Test MSE {test_mse:.4f}')

    print('=======================================================')
    test_means = [x[1] for x in all_seed_results]
    print(f'Overall test R mean +- std: {np.mean(test_means):.4f} +- {np.std(test_means):.4f}')
    print(f'Total runtime: {(time.time() - t0_all)/60:.1f} min')

    final_df = all_seed_results[0][2].copy()
    for _, _, df_i in all_seed_results[1:]:
        final_df = pd.merge(final_df, df_i, on=['gene_idx', 'gene_name'], how='left')

    test_cols = [c for c in final_df.columns if c.startswith('test_pearson_s')]
    final_df['test_pearson_mean'] = final_df[test_cols].mean(axis=1)
    final_df['test_pearson_std'] = final_df[test_cols].std(axis=1)

    out_csv = out_dir / 'scellst_supervised_results.csv'
    final_df.to_csv(out_csv, index=False)
    print(f'Saved results: {out_csv}')


if __name__ == '__main__':
    main()
