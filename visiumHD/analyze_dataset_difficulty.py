#!/usr/bin/env python3
"""
analyze_dataset_difficulty.py
──────────────────────────────
Quantify task difficulty for each Visium HD dataset using data-driven metrics.

Metrics:
  1. Spatial autocorrelation — how similar are neighboring cells' expression
  2. Per-gene CV (coefficient of variation) — expression variability across cells
  3. Per-cell statistics — genes detected, total counts, sparsity
  4. Nuclear morphology diversity — CV of nuclear diameter
  5. Coverage distribution — fraction of cells expressing each gene

Outputs (in --out_dir):
  difficulty_summary.csv      key metrics per dataset
  spatial_autocorr.csv        spatial autocorrelation per dataset
  gene_cv.csv                 per-gene CV per dataset
  coverage_distribution.csv   per-gene coverage per dataset
  difficulty_analysis.pdf     summary figure

Usage:
  python analyze_dataset_difficulty.py
  python analyze_dataset_difficulty.py --out_dir /path/to/output
"""

import argparse
import gc
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.spatial import cKDTree

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')
MORPH   = Path('/hpc/group/jilab/hz/MorphPT/data/visiumHD')

DATASETS = [
    ('human_crc',        'cache_crc',       'CRC',        '#C44E52'),
    ('human_lungcancer', 'cache_lung',       'Lung',       '#4C72B0'),
    ('human_pancreas',   'cache_pancreas',   'Pancreas',   '#55A868'),
]

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str,
                   default=str(PROJECT / 'analysis/dataset_difficulty'))
    p.add_argument('--n_cells_autocorr', type=int, default=3000,
                   help='Cells to sample for spatial autocorrelation')
    p.add_argument('--n_query_autocorr', type=int, default=500,
                   help='Query cells for spatial autocorrelation')
    p.add_argument('--k_neighbors', type=int, default=6,
                   help='Number of spatial neighbors')
    p.add_argument('--batch_size', type=int, default=5000,
                   help='Batch size for expression loading')
    return p.parse_args()


# ── 1. Spatial autocorrelation ────────────────────────────────────────────
def compute_spatial_autocorr(dataset, cache_name, label, args):
    print(f'\n[Spatial autocorr] {label}...')
    cache   = PROJECT / cache_name
    expr    = np.load(str(cache / 'expr.npy'), mmap_mode='r')
    meta    = pd.read_csv(cache / 'meta_random_split.csv')
    meta['cell_id']    = meta['cell_id'].astype(str)
    
    if 'x_centroid' not in meta.columns or 'y_centroid' not in meta.columns:
        spatial = pd.read_csv(MORPH / dataset / 'spatial.csv')
        spatial['cell_id'] = spatial['cell_id'].astype(str)
        meta = meta.merge(spatial, on='cell_id', how='left')

    train = meta[meta['split'] == 'train'].reset_index(drop=True)
    n     = min(args.n_cells_autocorr, len(train))
    idx   = train['mmap_idx'].values[:n]
    X     = np.array(expr[idx], dtype=np.float32)

    coords = train[['x_centroid', 'y_centroid']].values[:n]
    tree   = cKDTree(coords)
    n_q    = min(args.n_query_autocorr, n)
    _, nb  = tree.query(coords[:n_q], k=args.k_neighbors + 1)

    # True spatial autocorrelation
    corrs = []
    for i in range(n_q):
        nb_idx = nb[i, 1:]
        xi = X[i]
        xn = X[nb_idx].mean(axis=0)
        if xi.std() > 1e-6 and xn.std() > 1e-6:
            corrs.append(float(np.corrcoef(xi, xn)[0, 1]))

    # Null baseline (shuffled)
    np.random.seed(42)
    Xs = X.copy()
    np.random.shuffle(Xs)
    corrs_null = []
    for i in range(n_q):
        nb_idx = nb[i, 1:]
        xi = Xs[i]
        xn = Xs[nb_idx].mean(axis=0)
        if xi.std() > 1e-6 and xn.std() > 1e-6:
            corrs_null.append(float(np.corrcoef(xi, xn)[0, 1]))

    result = {
        'dataset':        label,
        'autocorr_mean':  float(np.mean(corrs)),
        'autocorr_median':float(np.median(corrs)),
        'null_mean':      float(np.mean(corrs_null)),
        'gap':            float(np.mean(corrs) - np.mean(corrs_null)),
        'n_cells_used':   n_q,
        'k_neighbors':    args.k_neighbors,
    }

    print(f'  spatial autocorr = {result["autocorr_mean"]:.4f}')
    print(f'  null baseline    = {result["null_mean"]:.4f}')
    print(f'  gap              = {result["gap"]:.4f}')

    del X, Xs
    gc.collect()
    return result, corrs, corrs_null


# ── 2. Per-gene CV and coverage ────────────────────────────────────────────
def compute_gene_stats(dataset, cache_name, label, args):
    print(f'\n[Gene stats] {label}...')
    cache     = PROJECT / cache_name
    stats     = np.load(str(cache / 'expr_stats_random.npz'))
    gene_mean = stats['gene_mean']
    gene_std  = stats['gene_std']
    cv        = gene_std / np.clip(gene_mean, 1e-5, None)
    genes     = (cache / 'gene_list.txt').read_text().splitlines()

    # Per-gene coverage on test cells
    expr    = np.load(str(cache / 'expr.npy'), mmap_mode='r')
    meta    = pd.read_csv(cache / 'meta_random_split.csv')
    test_idx= meta[meta['split'] == 'test']['mmap_idx'].values

    # Batch compute coverage
    sum_expressed = np.zeros(expr.shape[1], dtype=np.float32)
    for i in range(0, len(test_idx), args.batch_size):
        batch = expr[test_idx[i:i+args.batch_size]]
        sum_expressed += (batch > 0).sum(axis=0)
    coverage = sum_expressed / len(test_idx)

    result = {
        'dataset':        label,
        'n_genes':        len(genes),
        'cv_mean':        float(cv.mean()),
        'cv_median':      float(np.median(cv)),
        'cv_std':         float(cv.std()),
        'coverage_mean':  float(coverage.mean()),
        'coverage_median':float(np.median(coverage)),
        'gene_mean_mean': float(gene_mean.mean()),
        'gene_std_mean':  float(gene_std.mean()),
    }

    print(f'  n_genes      = {result["n_genes"]}')
    print(f'  CV mean      = {result["cv_mean"]:.3f}')
    print(f'  coverage mean= {result["coverage_mean"]:.3f}')

    del expr
    gc.collect()
    return result, cv, coverage


# ── 3. Per-cell statistics ────────────────────────────────────────────────
def compute_cell_stats(dataset, cache_name, label, args):
    print(f'\n[Cell stats] {label}...')
    cache  = PROJECT / cache_name
    expr   = np.load(str(cache / 'expr.npy'), mmap_mode='r')
    meta   = pd.read_csv(cache / 'meta_random_split.csv')

    train_idx = meta[meta['split'] == 'train']['mmap_idx'].values
    n_sample  = min(20000, len(train_idx))
    sample_idx= train_idx[:n_sample]

    ngenes_list, total_list = [], []
    for i in range(0, n_sample, args.batch_size):
        X = expr[sample_idx[i:i+args.batch_size]]
        ngenes_list.append((X > 0).sum(axis=1))
        total_list.append(X.sum(axis=1))

    cell_ngenes = np.concatenate(ngenes_list)
    cell_total  = np.concatenate(total_list)
    sparsity    = 1 - cell_ngenes.mean() / expr.shape[1]

    # Nuclear morphology diversity
    diam_cv = None
    if 'biological_diameter_um' in meta.columns:
        diam    = meta.loc[meta['split']=='train',
                           'biological_diameter_um'].dropna()
        diam_cv = float(diam.std() / diam.mean())

    # Aspect ratio diversity
    ar_cv = None
    if 'aspect_ratio' in meta.columns:
        ar    = meta.loc[meta['split']=='train', 'aspect_ratio'].dropna()
        ar_cv = float(ar.std() / ar.mean())

    result = {
        'dataset':           label,
        'n_cells_train':     len(train_idx),
        'genes_per_cell_mean':   float(cell_ngenes.mean()),
        'genes_per_cell_median': float(np.median(cell_ngenes)),
        'total_counts_mean':     float(cell_total.mean()),
        'sparsity':              float(sparsity),
        'nuclear_diameter_cv':   diam_cv,
        'aspect_ratio_cv':       ar_cv,
    }

    print(f'  n_cells_train      = {result["n_cells_train"]:,}')
    print(f'  genes/cell (mean)  = {result["genes_per_cell_mean"]:.1f}')
    print(f'  sparsity           = {result["sparsity"]:.1%}')
    if diam_cv:
        print(f'  nuclear diam CV    = {result["nuclear_diameter_cv"]:.3f}')

    del expr
    gc.collect()
    return result, cell_ngenes, cell_total


# ── 4. Summary figure ─────────────────────────────────────────────────────
def make_figure(autocorr_results, gene_results, cell_results,
                all_corrs, all_null, all_cv, all_coverage,
                all_ngenes, out_dir):
    print('\nGenerating figure...')
    colors = ['#C44E52', '#4C72B0', '#55A868']
    labels = [r['dataset'] for r in autocorr_results]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: Spatial autocorrelation ────────────────────────────────────
    # (a) Gap bar chart
    ax = fig.add_subplot(gs[0, 0])
    gaps = [r['gap'] for r in autocorr_results]
    bars = ax.bar(labels, gaps, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Spatial autocorrelation gap\n(real − null)')
    ax.set_title('(a) Spatial autocorrelation\n(higher = stronger spatial pattern)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')

    # (b) Violin of per-cell autocorr
    ax = fig.add_subplot(gs[0, 1])
    parts = ax.violinplot(all_corrs, positions=range(len(labels)),
                          showmedians=True)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    # Null baseline violins (gray)
    parts2 = ax.violinplot(all_null, positions=range(len(labels)),
                           showmedians=False)
    for pc in parts2['bodies']:
        pc.set_facecolor('#cccccc')
        pc.set_alpha(0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Cell-neighbor expression correlation')
    ax.set_title('(b) Spatial autocorrelation distribution\n(gray=null baseline)')

    # (c) Real vs null scatter
    ax = fig.add_subplot(gs[0, 2])
    for i, (label, col) in enumerate(zip(labels, colors)):
        ax.scatter(autocorr_results[i]['null_mean'],
                   autocorr_results[i]['autocorr_mean'],
                   color=col, s=150, zorder=3, label=label)
        ax.annotate(f'  {label}',
                    (autocorr_results[i]['null_mean'],
                     autocorr_results[i]['autocorr_mean']),
                    fontsize=8)
    lim_min = min(r['null_mean'] for r in autocorr_results) - 0.01
    lim_max = max(r['autocorr_mean'] for r in autocorr_results) + 0.01
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=0.8)
    ax.set_xlabel('Null baseline (shuffled)')
    ax.set_ylabel('Real autocorrelation')
    ax.set_title('(c) Real vs null autocorrelation\n(above diagonal = spatial signal)')

    # ── Row 2: Gene-level statistics ──────────────────────────────────────
    # (d) CV distribution
    ax = fig.add_subplot(gs[1, 0])
    for i, (label, col, cv) in enumerate(zip(labels, colors, all_cv)):
        ax.hist(cv, bins=40, alpha=0.55, color=col, density=True,
                edgecolor='none',
                label=f'{label}  μ={cv.mean():.2f}')
    ax.set_xlabel('Per-gene CV (std/mean)')
    ax.set_ylabel('Density')
    ax.set_title('(d) Per-gene expression variability (CV)\n(higher = more variable across cells)')
    ax.legend(fontsize=7)

    # (e) Coverage distribution
    ax = fig.add_subplot(gs[1, 1])
    bins   = [0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    blabels= ['10-15%', '15-20%', '20-30%', '30-50%', '>50%']
    x      = np.arange(len(blabels))
    width  = 0.25
    for i, (label, col, cov) in enumerate(zip(labels, colors, all_coverage)):
        counts = [((cov >= bins[j]) & (cov < bins[j+1])).sum()
                  for j in range(len(blabels))]
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, counts, width=width*0.9,
                        color=col, alpha=0.85, label=label)
        for bar, cnt in zip(bars, counts):
            if cnt > 5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1, str(cnt),
                        ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(blabels, fontsize=7)
    ax.set_ylabel('Number of genes')
    ax.set_title('(e) Gene coverage distribution\n(% cells expressing each gene)')
    ax.legend(fontsize=7)

    # (f) Summary stats table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    table_data = [
        ['Metric', 'CRC', 'Lung', 'Pancreas'],
        ['n_genes',
         str(gene_results[0]['n_genes']),
         str(gene_results[1]['n_genes']),
         str(gene_results[2]['n_genes'])],
        ['n_cells (train)',
         f"{cell_results[0]['n_cells_train']:,}",
         f"{cell_results[1]['n_cells_train']:,}",
         f"{cell_results[2]['n_cells_train']:,}"],
        ['Spatial autocorr gap',
         f"{autocorr_results[0]['gap']:.4f}",
         f"{autocorr_results[1]['gap']:.4f}",
         f"{autocorr_results[2]['gap']:.4f}"],
        ['Gene CV (mean)',
         f"{gene_results[0]['cv_mean']:.3f}",
         f"{gene_results[1]['cv_mean']:.3f}",
         f"{gene_results[2]['cv_mean']:.3f}"],
        ['Coverage (mean)',
         f"{gene_results[0]['coverage_mean']:.1%}",
         f"{gene_results[1]['coverage_mean']:.1%}",
         f"{gene_results[2]['coverage_mean']:.1%}"],
        ['Genes/cell (mean)',
         f"{cell_results[0]['genes_per_cell_mean']:.0f}",
         f"{cell_results[1]['genes_per_cell_mean']:.0f}",
         f"{cell_results[2]['genes_per_cell_mean']:.0f}"],
        ['Sparsity',
         f"{cell_results[0]['sparsity']:.1%}",
         f"{cell_results[1]['sparsity']:.1%}",
         f"{cell_results[2]['sparsity']:.1%}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for j in range(4):
        tbl[0, j].set_facecolor('#dddddd')
    ax.set_title('(f) Dataset comparison summary', fontsize=9, pad=15)

    # ── Row 3: Per-cell statistics ─────────────────────────────────────────
    # (g) Genes per cell distribution
    ax = fig.add_subplot(gs[2, 0])
    for label, col, ng in zip(labels, colors, all_ngenes):
        ax.hist(ng, bins=50, alpha=0.55, color=col, density=True,
                edgecolor='none',
                label=f'{label}  μ={ng.mean():.0f}')
    ax.set_xlabel('Genes detected per cell')
    ax.set_ylabel('Density')
    ax.set_title('(g) Genes detected per cell\n(higher = richer expression profile)')
    ax.legend(fontsize=7)

    # (h) Nuclear diameter CV (morphology diversity)
    ax = fig.add_subplot(gs[2, 1])
    diam_cvs = [r['nuclear_diameter_cv'] for r in cell_results
                if r['nuclear_diameter_cv'] is not None]
    diam_labels = [r['dataset'] for r in cell_results
                   if r['nuclear_diameter_cv'] is not None]
    diam_colors = [colors[i] for i, r in enumerate(cell_results)
                   if r['nuclear_diameter_cv'] is not None]
    if diam_cvs:
        bars = ax.bar(diam_labels, diam_cvs, color=diam_colors,
                      alpha=0.85, width=0.5)
        for bar, val in zip(bars, diam_cvs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('CV of nuclear diameter')
        ax.set_title('(h) Nuclear morphology diversity\n(higher = more diverse cell shapes)')
    else:
        ax.text(0.5, 0.5, 'Nuclear diameter\nnot available',
                ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title('(h) Nuclear morphology diversity')

    # (i) Difficulty ranking summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    difficulty_text = (
        'Task Difficulty Analysis\n'
        '─────────────────────────\n\n'
        'CRC (Easiest):\n'
        f'  Spatial gap    = {autocorr_results[0]["gap"]:.4f}  ← strong\n'
        f'  Gene CV        = {gene_results[0]["cv_mean"]:.3f}\n'
        f'  n_cells_train  = {cell_results[0]["n_cells_train"]:,}\n\n'
        'Pancreas (Medium):\n'
        f'  Spatial gap    = {autocorr_results[2]["gap"]:.4f}\n'
        f'  Gene CV        = {gene_results[2]["cv_mean"]:.3f}\n'
        f'  n_cells_train  = {cell_results[2]["n_cells_train"]:,}  ← small\n\n'
        'Lung (Hardest):\n'
        f'  Spatial gap    = {autocorr_results[1]["gap"]:.4f}  ← weak\n'
        f'  Gene CV        = {gene_results[1]["cv_mean"]:.3f}\n'
        f'  n_cells_train  = {cell_results[1]["n_cells_train"]:,}\n\n'
        'Key: Lung has weakest spatial\n'
        'autocorrelation → expression\n'
        'pattern not predictable from\n'
        'morphology alone.'
    )
    ax.text(0.05, 0.95, difficulty_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
    ax.set_title('(i) Difficulty summary', fontsize=9)

    fig.suptitle('Dataset Difficulty Analysis — Visium HD Multi-Dataset',
                 fontsize=13, fontweight='bold')
    out = Path(out_dir) / 'difficulty_analysis.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out}')


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    args    = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    autocorr_results = []
    gene_results     = []
    cell_results     = []
    all_corrs        = []
    all_null         = []
    all_cv           = []
    all_coverage     = []
    all_ngenes       = []

    for dataset, cache_name, label, color in DATASETS:
        print(f'\n{"="*55}')
        print(f'Processing: {label}')
        print(f'{"="*55}')

        # Spatial autocorrelation
        ac_res, corrs, null = compute_spatial_autocorr(
            dataset, cache_name, label, args)
        autocorr_results.append(ac_res)
        all_corrs.append(corrs)
        all_null.append(null)

        # Gene stats
        g_res, cv, coverage = compute_gene_stats(
            dataset, cache_name, label, args)
        gene_results.append(g_res)
        all_cv.append(cv)
        all_coverage.append(coverage)

        # Cell stats
        c_res, cell_ngenes, _ = compute_cell_stats(
            dataset, cache_name, label, args)
        cell_results.append(c_res)
        all_ngenes.append(cell_ngenes)

        gc.collect()

    # ── Save CSVs ─────────────────────────────────────────────────────────
    pd.DataFrame(autocorr_results).to_csv(
        out_dir / 'spatial_autocorr.csv', index=False)
    pd.DataFrame(gene_results).to_csv(
        out_dir / 'gene_stats.csv', index=False)
    pd.DataFrame(cell_results).to_csv(
        out_dir / 'cell_stats.csv', index=False)

    # Summary
    summary_rows = []
    for i, (_, _, label, _) in enumerate(DATASETS):
        summary_rows.append({
            'dataset':              label,
            'n_genes':              gene_results[i]['n_genes'],
            'n_cells_train':        cell_results[i]['n_cells_train'],
            'spatial_autocorr':     autocorr_results[i]['autocorr_mean'],
            'spatial_null':         autocorr_results[i]['null_mean'],
            'spatial_gap':          autocorr_results[i]['gap'],
            'gene_cv_mean':         gene_results[i]['cv_mean'],
            'gene_coverage_mean':   gene_results[i]['coverage_mean'],
            'genes_per_cell_mean':  cell_results[i]['genes_per_cell_mean'],
            'sparsity':             cell_results[i]['sparsity'],
            'nuclear_diameter_cv':  cell_results[i]['nuclear_diameter_cv'],
            'aspect_ratio_cv':      cell_results[i]['aspect_ratio_cv'],
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / 'difficulty_summary.csv', index=False)

    # Per-gene files
    for i, (_, _, label, _) in enumerate(DATASETS):
        cache = PROJECT / DATASETS[i][1]
        genes = (cache / 'gene_list.txt').read_text().splitlines()
        pd.DataFrame({
            'gene':     genes,
            'cv':       all_cv[i],
            'coverage': all_coverage[i],
        }).sort_values('cv', ascending=False).to_csv(
            out_dir / f'gene_cv_{label.lower()}.csv', index=False)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('DIFFICULTY SUMMARY')
    print(f'{"="*65}')
    print(df_summary.to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────
    make_figure(autocorr_results, gene_results, cell_results,
                all_corrs, all_null, all_cv, all_coverage,
                all_ngenes, out_dir)

    print(f'\nAll outputs → {out_dir}')


if __name__ == '__main__':
    main()
