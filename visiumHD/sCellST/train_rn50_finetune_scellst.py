#!/usr/bin/env python3
"""RN50 fine-tuned (last block) + sCellST head, trained END-TO-END on cache images.

Generic-CNN control that completes the 2x2 (generic vs MorphPT) x (frozen vs adapted):
  - Backbone: sCellST InstanceEmbedder(resnet50, ImageNet) — IDENTICAL to the frozen
    RN50-cache arm — but with the last block (layer4, index 7) unfrozen and fine-tuned.
  - Head: sCellST GenePredictor (3x256 LeakyReLU, dropout 0.1, identity) — same as
    every other sCellST arm.
  - Data: MorphPT cache (224px, log1p+z-scored 2847-gene expr), MorphPT-matched split
    (morphpt_split.py), top-200 genes by gene_idx. End-to-end MSE training.

Output mirrors train_scellst_supervised_multi_probing.py so it aggregates with the
other arms: experiments/<exp_tag>_scellst_supervised_<ds>_top200_random_seed42/.
Run in scellst_env (torchvision + scellst.GenePredictor + VisiumHDPredictionDataset).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')
sys.path.append(str(PROJECT))
from data.visium_dataset import VisiumHDPredictionDataset  # noqa: E402
from morphpt_split import morphpt_random_split_cellids  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(description='RN50 last-block fine-tune + sCellST head, end-to-end.')
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--split_layout', type=str, default='default')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--gene_panel_csv', type=str, default='')
    p.add_argument('--top_k', type=int, default=200)
    p.add_argument('--ft_modules', type=str, default='7',
                   help='Comma-separated InstanceEmbedder.model indices to unfreeze (7=layer4, 6=layer3)')
    p.add_argument('--head_hidden', type=str, default='256,256,256')
    p.add_argument('--head_dropout', type=float, default=0.1)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_backbone', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--base_seed', type=int, default=42)
    p.add_argument('--scellst_repo', type=str, default='/hpc/group/jilab/tc459/sCellST')
    p.add_argument('--exp_tag', type=str, default='rn50_ft_lastblock')
    return p.parse_args()


def pearson_r_multi(pred, target):
    if pred.size(0) < 2:
        return torch.zeros(pred.size(1))
    pm = pred - pred.mean(0, keepdim=True)
    tm = target - target.mean(0, keepdim=True)
    num = (pm * tm).sum(0)
    den = torch.sqrt((pm.pow(2).sum(0) * tm.pow(2).sum(0)).clamp_min(1e-8))
    return num / den


class RN50FineTune(nn.Module):
    def __init__(self, ft_idx, head_hidden, head_dropout, out_dim, scellst_repo):
        super().__init__()
        sys.path.append(scellst_repo)
        from scellst.module.image_encoder import InstanceEmbedder
        from scellst.module.gene_predictor import GenePredictor
        self.embedder = InstanceEmbedder(archi='resnet50', weights='imagenet-rn50')
        in_dim = self.embedder.get_output_dim()
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.ft_idx = ft_idx
        for i in ft_idx:
            for p in self.embedder.model[i].parameters():
                p.requires_grad = True
        self.head = GenePredictor(
            input_dim=in_dim, output_dim=out_dim, final_activation='identity',
            hidden_dim=head_hidden, dropout_rate=head_dropout,
        )

    def set_modes(self):
        self.embedder.eval()
        for i in self.ft_idx:
            self.embedder.model[i].train()
        self.head.train()

    def forward(self, x):
        feat = self.embedder(x)
        return self.head({'output_embedding': feat})['output_prediction']


@torch.no_grad()
def evaluate(model, loader, gene_idx, device):
    model.eval()
    preds, gts, mse_sum, n = [], [], 0.0, 0
    for imgs, expr, _ in loader:
        x = imgs.to(device, non_blocking=True)
        y = expr[:, gene_idx].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            pred = model(x).float()
        mse_sum += F.mse_loss(pred, y).item() * y.size(0)
        n += y.size(0)
        preds.append(pred.cpu()); gts.append(y.cpu())
    P, G = torch.cat(preds), torch.cat(gts)
    return mse_sum / max(1, n), pearson_r_multi(P, G), P, G


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = PROJECT / f'cache_{args.dataset}'
    panel_path = Path(args.gene_panel_csv) if args.gene_panel_csv else \
        cache_dir / 'splits' / args.split_layout / 'top200_variance_mincov0.1_train_default_seed42.csv'
    panel = pd.read_csv(panel_path)
    if args.top_k > 0:
        panel = panel.iloc[:args.top_k]
    gene_idx = torch.as_tensor(panel['gene_idx'].to_numpy(dtype=int))
    gene_names = panel['gene_name'].to_numpy()
    out_features = len(gene_idx)
    ft_idx = [int(i) for i in args.ft_modules.split(',') if i.strip()]
    head_hidden = [int(h) for h in args.head_hidden.split(',') if h.strip()]

    ds = VisiumHDPredictionDataset(cache_dir=str(cache_dir), split='all', scales=['10.0x'],
                                   fuse='identity', augment=False, split_type='random',
                                   split_layout=args.split_layout)
    cell_ids = np.asarray([str(c) for c in ds._cell_ids])
    pos = {c: i for i, c in enumerate(cell_ids)}
    tr_c, va_c, te_c = morphpt_random_split_cellids(cache_dir, split_layout=args.split_layout,
                                                    seed=args.seed, train_frac=args.train_frac,
                                                    val_frac=args.val_frac)
    idx = {k: [pos[c] for c in v.astype(str) if c in pos] for k, v in
           {'train': tr_c, 'val': va_c, 'test': te_c}.items()}

    def mk_loader(split, shuffle):
        return DataLoader(Subset(ds, idx[split]), batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True, drop_last=False)
    train_loader = mk_loader('train', True)
    val_loader = mk_loader('val', False)
    test_loader = mk_loader('test', False)

    out_dir = PROJECT / 'experiments' / f'{args.exp_tag}_scellst_supervised_{args.dataset}_top{out_features}_random_seed{args.base_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=======================================================')
    print(f'RN50 fine-tune (modules {ft_idx}) + sCellST head | {args.dataset}')
    print(f'train/val/test = {len(idx["train"])}/{len(idx["val"])}/{len(idx["test"])} | genes={out_features}')
    print(f'Output: {out_dir}')
    print('=======================================================')

    all_results = []
    for si in range(args.n_seeds):
        seed = args.base_seed + si
        torch.manual_seed(seed); np.random.seed(seed)
        model = RN50FineTune(ft_idx, head_hidden, args.head_dropout, out_features, args.scellst_repo).to(device)
        bb = [p for p in model.embedder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': model.head.parameters(), 'lr': args.lr_head},
            {'params': bb, 'lr': args.lr_backbone},
        ], weight_decay=args.weight_decay)
        print(f'Seed {seed}: trainable backbone params={sum(p.numel() for p in bb):,}')

        best_val, best_state, best_val_per_gene, no_imp = -1e9, None, None, 0
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.set_modes()
            for imgs, expr, _ in train_loader:
                x = imgs.to(device, non_blocking=True)
                y = expr[:, gene_idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    pred = model(x)
                    loss = F.mse_loss(pred.float(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            val_mse, val_r, _, _ = evaluate(model, val_loader, gene_idx, device)
            mvr = val_r.mean().item()
            if ep % 5 == 0 or ep == 1:
                print(f'Seed {seed} Ep {ep:>3d} | Val MSE {val_mse:.4f} | Val R {mvr:.4f}')
            if mvr > best_val:
                best_val, no_imp = mvr, 0
                best_val_per_gene = val_r.clone()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= args.patience:
                    print(f'Seed {seed}: early stop at epoch {ep}'); break

        model.load_state_dict(best_state)
        test_mse, test_r, test_pred, test_true = evaluate(model, test_loader, gene_idx, device)
        mtr = test_r.mean().item()
        np.save(out_dir / f'test_y_pred_seed_{seed}.npy', test_pred.numpy())
        np.save(out_dir / f'test_y_true_seed_{seed}.npy', test_true.numpy())
        json.dump({'seed': int(seed), 'dataset': args.dataset, 'arm': args.exp_tag,
                   'ft_modules': ft_idx, 'n_train': len(idx['train']), 'n_val': len(idx['val']),
                   'n_test': len(idx['test']), 'output_dim': int(out_features),
                   'best_val_r': float(best_val), 'test_r': float(mtr), 'test_mse': float(test_mse),
                   'training_time_s': float(time.time() - t0)},
                  open(out_dir / f'seed_{seed}_info.json', 'w'), indent=2)
        df = pd.DataFrame({'gene_idx': gene_idx.numpy(), 'gene_name': gene_names,
                           f'val_pearson_s{seed}': best_val_per_gene.numpy(),
                           f'test_pearson_s{seed}': test_r.numpy()})
        all_results.append((seed, mtr, df))
        print(f'Seed {seed} done | Best Val R {best_val:.4f} | Test R {mtr:.4f}')

    means = [r[1] for r in all_results]
    print(f'Overall test R mean +- std: {np.mean(means):.4f} +- {np.std(means):.4f}')
    final = all_results[0][2]
    for _, _, d in all_results[1:]:
        final = pd.merge(final, d, on=['gene_idx', 'gene_name'], how='left')
    tcols = [c for c in final.columns if c.startswith('test_pearson_s')]
    final['test_pearson_mean'] = final[tcols].mean(axis=1)
    final['test_pearson_std'] = final[tcols].std(axis=1)
    final.to_csv(out_dir / 'scellst_supervised_results.csv', index=False)
    print(f'Saved {out_dir / "scellst_supervised_results.csv"}')


if __name__ == '__main__':
    main()
