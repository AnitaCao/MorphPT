"""
explore_multidataset.py
────────────────────────
Multi-dataset EDA for all three Visium HD datasets.
Designed to run on cluster as a batch job (no interactive memory limits).
Processes one dataset at a time, frees memory between datasets.

Outputs (in --out_dir):
  summary_stats.csv          per-dataset key statistics
  top_genes_{dataset}.csv    top 20 genes per dataset
  gene_overlap.csv           pairwise gene overlap matrix
  shared_genes.csv           genes shared across all three datasets
  expr_eda_{dataset}.pdf     per-dataset expression plots
  spatial_{dataset}.pdf      spatial cell distribution
  comparison.pdf             cross-dataset comparison
  gene_overlap.pdf           gene overlap visualization

Usage:
  python explore_multidataset.py
  python explore_multidataset.py --datasets human_crc human_lungcancer
  python explore_multidataset.py --out_dir /path/to/output
"""

import argparse
import gc
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for cluster
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
VISIUM_ROOT = Path('/hpc/group/jilab/boxuan/visiumHD')
MORPH_ROOT  = Path('/hpc/group/jilab/hz/MorphPT/data/visiumHD')

COLORS = {
    'human_crc':        '#C44E52',
    'human_lungcancer': '#4C72B0',
    'human_pancreas':   '#55A868',
}
LABELS = {
    'human_crc':        'CRC',
    'human_lungcancer': 'Lung Cancer',
    'human_pancreas':   'Pancreas',
}

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})


# ── Args ───────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+',
                   default=['human_crc', 'human_lungcancer', 'human_pancreas'])
    p.add_argument('--out_dir', type=str,
                   default='/hpc/group/jilab/tc459/MorphPT/analysis/multidataset_eda')
    return p.parse_args()


# ── Load one dataset (matrices freed immediately after stats computed) ─────
def process_dataset(name: str, out_dir: Path) -> dict:
    print(f'\n{"="*55}')
    print(f'Processing: {name}')
    print(f'{"="*55}')

    root      = VISIUM_ROOT / name
    morph_dir = MORPH_ROOT  / name

    # ── Expression matrix ──────────────────────────────────────────────────
    print('  Loading expression matrix...')
    X     = sio.mmread(root / 'expr/expr.mtx').tocsr()
    genes = (root / 'expr/genes.txt').read_text().splitlines()
    cells = (root / 'expr/cells.txt').read_text().splitlines()

    if X.shape[0] == len(genes) and X.shape[1] == len(cells):
        X = X.T.tocsr()

    n_cells, n_genes = X.shape
    sparsity = 1 - X.nnz / (n_cells * n_genes)
    print(f'  Shape    : {n_cells:,} cells × {n_genes:,} genes')
    print(f'  Sparsity : {sparsity:.3%}  ({X.nnz:,} non-zero)')

    # Compute all stats before freeing X
    cell_total  = np.array(X.sum(axis=1), dtype=np.float32).ravel()
    cell_ngenes = np.diff(X.indptr).astype(np.int32)
    gene_total  = np.array(X.sum(axis=0), dtype=np.float32).ravel()
    gene_ncells = np.array((X > 0).sum(axis=0), dtype=np.int32).ravel()
    nz_sample   = np.random.choice(X.data, size=min(100000, X.nnz),
                                   replace=False).astype(np.float32)
    nz_min  = float(X.data.min())
    nz_max  = float(X.data.max())
    nz_mean = float(X.data.mean())
    nz_med  = float(np.median(X.data))
    is_int  = bool(np.all(X.data == X.data.astype(int)))

    print(f'  NZ values: min={nz_min:.3f}  max={nz_max:.3f}  '
          f'mean={nz_mean:.3f}  median={nz_med:.3f}')
    print(f'  Integers?: {is_int}')

    # Free large matrix
    del X
    gc.collect()
    print('  Matrix freed.')

    # ── Spatial coords (only x, y) ─────────────────────────────────────────
    sp_path = morph_dir / 'spatial.csv'
    if sp_path.exists():
        spatial = pd.read_csv(sp_path, usecols=['cell_id', 'x_centroid', 'y_centroid'])
        print(f'  Spatial  : {len(spatial):,} rows')
    else:
        spatial = None
        print('  Spatial  : NOT FOUND')

    # ── Meta CSV ───────────────────────────────────────────────────────────
    meta_path = root / f'meta/10.0x/{name}.csv'
    if meta_path.exists():
        keep = ['cell_id', 'biological_area_um2', 'biological_diameter_um',
                'aspect_ratio', 'coverage']
        avail = pd.read_csv(meta_path, nrows=0).columns.tolist()
        use   = [c for c in keep if c in avail]
        meta  = pd.read_csv(meta_path, usecols=use)
        print(f'  Meta     : {len(meta):,} rows  cols={use}')
    else:
        meta = None
        print('  Meta     : NOT FOUND')

    # ── Per-gene stats printout ────────────────────────────────────────────
    top20_idx  = np.argsort(gene_total)[::-1][:20]
    bot20_idx  = np.argsort(gene_total)[:20]

    print(f'\n  Per-cell counts : mean={cell_total.mean():.1f}  '
          f'median={np.median(cell_total):.1f}  '
          f'min={cell_total.min():.0f}  max={cell_total.max():.0f}')
    print(f'  Per-cell genes  : mean={cell_ngenes.mean():.1f}  '
          f'median={np.median(cell_ngenes):.1f}  '
          f'min={cell_ngenes.min():.0f}  max={cell_ngenes.max():.0f}')
    print(f'  Per-gene ncells : mean={gene_ncells.mean():.1f}  '
          f'min={gene_ncells.min():.0f}  max={gene_ncells.max():.0f}')

    print(f'\n  Top 20 genes:')
    print(f"  {'Gene':<20} {'TotalCount':>12} {'Coverage':>10}")
    for i in top20_idx:
        print(f'  {genes[i]:<20} {int(gene_total[i]):>12,} '
              f'{gene_ncells[i]/n_cells:>9.1%}')

    print(f'\n  Bottom 20 genes:')
    print(f"  {'Gene':<20} {'TotalCount':>12} {'Coverage':>10}")
    for i in bot20_idx:
        print(f'  {genes[i]:<20} {int(gene_total[i]):>12,} '
              f'{gene_ncells[i]/n_cells:>9.1%}')

    # ── Save CSVs ──────────────────────────────────────────────────────────
    # Top genes CSV
    top_df = pd.DataFrame({
        'gene':       [genes[i] for i in top20_idx],
        'total_count':[int(gene_total[i]) for i in top20_idx],
        'coverage':   [gene_ncells[i]/n_cells for i in top20_idx],
    })
    top_df.to_csv(out_dir / f'top_genes_{name}.csv', index=False)

    # Per-gene summary CSV
    gene_df = pd.DataFrame({
        'gene':        genes,
        'total_count': gene_total.astype(int),
        'n_cells':     gene_ncells,
        'coverage':    gene_ncells / n_cells,
    }).sort_values('total_count', ascending=False)
    gene_df.to_csv(out_dir / f'gene_stats_{name}.csv', index=False)

    # ── Per-dataset expression plots ───────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    col = COLORS[name]
    lbl = LABELS[name]

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(cell_total, bins=80, color=col, alpha=0.85, edgecolor='none')
    ax.set_xlabel('Total expression per cell')
    ax.set_ylabel('# cells')
    ax.set_title(f'(a) Per-cell total counts\n'
                 f'μ={cell_total.mean():.1f}  median={np.median(cell_total):.1f}')

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(cell_ngenes, bins=80, color=col, alpha=0.85, edgecolor='none')
    ax.set_xlabel('Genes detected per cell')
    ax.set_ylabel('# cells')
    ax.set_title(f'(b) Genes detected per cell\n'
                 f'μ={cell_ngenes.mean():.1f}  median={np.median(cell_ngenes):.1f}')

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(gene_ncells / n_cells * 100, bins=60, color=col, alpha=0.85, edgecolor='none')
    ax.set_xlabel('% cells expressing gene')
    ax.set_ylabel('# genes')
    ax.set_title(f'(c) Gene coverage distribution\nsparsity={sparsity:.1%}')

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(nz_sample, bins=80, color=col, alpha=0.85, edgecolor='none')
    ax.set_xlabel('Non-zero expression value')
    ax.set_ylabel('# entries (sampled)')
    ax.set_title(f'(d) Non-zero value distribution\n'
                 f'[{nz_min:.2f}, {nz_max:.2f}]  mean={nz_mean:.2f}')

    ax = fig.add_subplot(gs[1, 1])
    if spatial is not None:
        ax.scatter(spatial['x_centroid'], -spatial['y_centroid'],
                   s=0.2, alpha=0.2, color=col, rasterized=True)
        ax.set_xlabel('x centroid (µm)')
        ax.set_ylabel('y centroid (flipped)')
        ax.set_aspect('equal')
    ax.set_title(f'(e) Spatial distribution\n({n_cells:,} cells)')

    ax = fig.add_subplot(gs[1, 2])
    top_names  = [genes[i] for i in top20_idx[:15]]
    top_counts = [gene_total[i] for i in top20_idx[:15]]
    ax.barh(top_names[::-1], top_counts[::-1], color=col, alpha=0.85)
    ax.set_xlabel('Total expression count')
    ax.set_title('(f) Top 15 genes')
    ax.tick_params(axis='y', labelsize=7)

    fig.suptitle(f'Visium HD — {lbl} ({n_cells:,} cells, {n_genes:,} genes)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_path = out_dir / f'expr_eda_{name}.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved → {out_path}')

    # Morphology plot
    if meta is not None:
        morph_cols   = ['biological_area_um2', 'biological_diameter_um',
                        'aspect_ratio', 'coverage']
        morph_labels = ['Nuclear area (µm²)', 'Nuclear diameter (µm)',
                        'Aspect ratio', 'Coverage']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        for i, (mc, ml) in enumerate(zip(morph_cols, morph_labels)):
            if mc in meta.columns:
                vals = meta[mc].dropna()
                axes[i].hist(vals, bins=60, color=col, alpha=0.85, edgecolor='none')
                axes[i].set_xlabel(ml)
                axes[i].set_ylabel('# cells')
                axes[i].set_title(f'({chr(97+i)}) {ml}\nμ={vals.mean():.2f}')
        fig.suptitle(f'Cell Morphology — {lbl}', fontweight='bold')
        fig.tight_layout()
        out_path = out_dir / f'morphology_{name}.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved → {out_path}')

    return {
        'name':        name,
        'n_cells':     n_cells,
        'n_genes':     n_genes,
        'sparsity':    sparsity,
        'genes':       genes,
        'cell_total':  cell_total,
        'cell_ngenes': cell_ngenes,
        'gene_total':  gene_total,
        'gene_ncells': gene_ncells,
        'nz_sample':   nz_sample,
        'nz_min':      nz_min,
        'nz_max':      nz_max,
        'nz_mean':     nz_mean,
        'is_int':      is_int,
        'spatial':     spatial,
        'meta':        meta,
    }


# ── Cross-dataset comparison plot ─────────────────────────────────────────
def plot_comparison(all_stats: list, out_dir: Path):
    print('\nGenerating comparison plots...')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    # (a) Per-cell total counts
    ax = axes[0]
    for s in all_stats:
        ax.hist(s['cell_total'], bins=60, alpha=0.55,
                color=COLORS[s['name']], density=True, edgecolor='none',
                label=f"{LABELS[s['name']]}  μ={s['cell_total'].mean():.1f}")
    ax.set_xlabel('Total expression per cell')
    ax.set_ylabel('Density')
    ax.set_title('(a) Per-cell total counts')
    ax.legend(fontsize=7)

    # (b) Per-cell genes detected
    ax = axes[1]
    for s in all_stats:
        ax.hist(s['cell_ngenes'], bins=60, alpha=0.55,
                color=COLORS[s['name']], density=True, edgecolor='none',
                label=f"{LABELS[s['name']]}  μ={s['cell_ngenes'].mean():.1f}")
    ax.set_xlabel('Genes detected per cell')
    ax.set_ylabel('Density')
    ax.set_title('(b) Genes detected per cell')
    ax.legend(fontsize=7)

    # (c) Gene coverage distribution
    ax = axes[2]
    for s in all_stats:
        cov = s['gene_ncells'] / s['n_cells'] * 100
        ax.hist(cov, bins=50, alpha=0.55,
                color=COLORS[s['name']], density=True, edgecolor='none',
                label=f"{LABELS[s['name']]}  μ={cov.mean():.1f}%")
    ax.set_xlabel('% cells expressing gene')
    ax.set_ylabel('Density')
    ax.set_title('(c) Gene coverage distribution')
    ax.legend(fontsize=7)

    # (d) Non-zero value distribution (sampled)
    ax = axes[3]
    for s in all_stats:
        ax.hist(s['nz_sample'], bins=60, alpha=0.55,
                color=COLORS[s['name']], density=True, edgecolor='none',
                label=f"{LABELS[s['name']]}  [{s['nz_min']:.1f},{s['nz_max']:.1f}]")
    ax.set_xlabel('Non-zero expression value')
    ax.set_ylabel('Density')
    ax.set_title('(d) Non-zero value distribution')
    ax.legend(fontsize=7)

    # (e) Summary bar chart
    ax     = axes[4]
    stats_to_show = ['Sparsity (%)', 'Genes/cell', 'Genes/dataset']
    x     = np.arange(len(stats_to_show))
    width = 0.25
    for i, s in enumerate(all_stats):
        vals   = [s['sparsity']*100, s['cell_ngenes'].mean(), s['n_genes']]
        offset = (i - len(all_stats)/2) * width + width/2
        bars   = ax.bar(x + offset, vals, width=width*0.9,
                        color=COLORS[s['name']], alpha=0.85,
                        label=LABELS[s['name']])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5, f'{val:.0f}',
                    ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_to_show, fontsize=8)
    ax.set_title('(e) Key statistics comparison')
    ax.legend(fontsize=7)

    # (f) Overview table
    ax = axes[5]
    ax.axis('off')
    rows = [['Dataset', 'Cells', 'Genes', 'Sparsity', 'Genes/cell', 'NZ range']]
    for s in all_stats:
        rows.append([
            LABELS[s['name']],
            f"{s['n_cells']:,}",
            f"{s['n_genes']:,}",
            f"{s['sparsity']:.1%}",
            f"{s['cell_ngenes'].mean():.1f}",
            f"[{s['nz_min']:.2f},{s['nz_max']:.2f}]",
        ])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    for j in range(len(rows[0])):
        tbl[0, j].set_facecolor('#dddddd')
    for i, s in enumerate(all_stats):
        for j in range(len(rows[0])):
            tbl[i+1, j].set_facecolor(COLORS[s['name']] + '33')
    ax.set_title('(f) Dataset overview', fontsize=9, pad=20)

    fig.suptitle('Visium HD Multi-Dataset EDA — Comparison',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = out_dir / 'comparison.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out}')


# ── Gene overlap ───────────────────────────────────────────────────────────
def analyze_gene_overlap(all_stats: list, out_dir: Path):
    print('\nAnalyzing gene overlap...')
    gene_sets = {s['name']: set(s['genes']) for s in all_stats}
    names     = [s['name'] for s in all_stats]

    # Overlap matrix CSV
    rows = []
    for n1 in names:
        for n2 in names:
            overlap = len(gene_sets[n1] & gene_sets[n2])
            rows.append({'dataset1': LABELS[n1], 'dataset2': LABELS[n2],
                         'n_shared': overlap,
                         'pct_of_d1': overlap/len(gene_sets[n1])*100})
    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(out_dir / 'gene_overlap.csv', index=False)

    # Shared genes CSV
    if len(names) >= 2:
        shared = gene_sets[names[0]]
        for n in names[1:]:
            shared &= gene_sets[n]

        # Get CRC stats for shared genes
        crc_stats = next(s for s in all_stats if s['name'] == 'human_crc')
        shared_rows = []
        for g in sorted(shared):
            row = {'gene': g}
            for s in all_stats:
                if g in s['genes']:
                    idx = s['genes'].index(g)
                    row[f'coverage_{LABELS[s["name"]]}'] = \
                        s['gene_ncells'][idx] / s['n_cells']
                    row[f'total_{LABELS[s["name"]]}'] = \
                        int(s['gene_total'][idx])
                else:
                    row[f'coverage_{LABELS[s["name"]]}'] = 0.0
                    row[f'total_{LABELS[s["name"]]}'] = 0
            shared_rows.append(row)

        shared_df = pd.DataFrame(shared_rows)
        if 'coverage_CRC' in shared_df.columns:
            shared_df = shared_df.sort_values('coverage_CRC', ascending=False)
        shared_df.to_csv(out_dir / 'shared_genes.csv', index=False)
        print(f'  Shared genes (all datasets): {len(shared):,}')
        print(f'  Saved → {out_dir}/shared_genes.csv')

    # Overlap plot
    n   = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = len(gene_sets[names[i]] & gene_sets[names[j]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.imshow(mat, cmap='Blues')
    plt.colorbar(im, ax=ax, label='# shared genes')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([LABELS[nm] for nm in names], fontsize=8)
    ax.set_yticklabels([LABELS[nm] for nm in names], fontsize=8)
    ax.set_title('(a) Gene overlap matrix', fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{int(mat[i,j]):,}', ha='center', va='center',
                    fontsize=9,
                    color='white' if mat[i,j] > mat.max()*0.6 else 'black')

    ax = axes[1]
    bar_labels, bar_vals, bar_colors = [], [], []
    for s in all_stats:
        others = set()
        for s2 in all_stats:
            if s2['name'] != s['name']:
                others |= gene_sets[s2['name']]
        unique = len(gene_sets[s['name']] - others)
        bar_labels.append(f'Only\n{LABELS[s["name"]]}')
        bar_vals.append(unique)
        bar_colors.append(COLORS[s['name']])

    if len(names) >= 2:
        shared_all = gene_sets[names[0]]
        for nm in names[1:]:
            shared_all &= gene_sets[nm]
        bar_labels.append('All shared')
        bar_vals.append(len(shared_all))
        bar_colors.append('#555555')

    bars = ax.bar(range(len(bar_labels)), bar_vals, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1, str(val),
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('Number of genes')
    ax.set_title('(b) Unique vs shared genes', fontsize=9)

    fig.suptitle('Gene panel overlap across datasets', fontweight='bold')
    fig.tight_layout()
    out = out_dir / 'gene_overlap.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out}')


# ── Summary CSV ────────────────────────────────────────────────────────────
def save_summary(all_stats: list, out_dir: Path):
    rows = []
    for s in all_stats:
        cov = s['gene_ncells'] / s['n_cells'] * 100
        rows.append({
            'dataset':          s['name'],
            'label':            LABELS[s['name']],
            'n_cells':          s['n_cells'],
            'n_genes':          s['n_genes'],
            'sparsity_pct':     round(s['sparsity']*100, 2),
            'mean_genes_cell':  round(s['cell_ngenes'].mean(), 1),
            'median_genes_cell':round(float(np.median(s['cell_ngenes'])), 1),
            'mean_total_counts':round(s['cell_total'].mean(), 1),
            'nz_min':           round(s['nz_min'], 3),
            'nz_max':           round(s['nz_max'], 3),
            'nz_mean':          round(s['nz_mean'], 3),
            'values_are_int':   s['is_int'],
            'mean_gene_cov_pct':round(cov.mean(), 2),
            'genes_over10pct':  int((cov > 10).sum()),
            'genes_over20pct':  int((cov > 20).sum()),
            'genes_over50pct':  int((cov > 50).sum()),
        })
    df = pd.DataFrame(rows)
    out = out_dir / 'summary_stats.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved summary → {out}')
    print(df.to_string(index=False))
    return df


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args    = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for name in args.datasets:
        s = process_dataset(name, out_dir)
        all_stats.append(s)
        gc.collect()

    save_summary(all_stats, out_dir)

    if len(all_stats) > 1:
        plot_comparison(all_stats, out_dir)
        analyze_gene_overlap(all_stats, out_dir)

    print(f'\n{"="*55}')
    print(f'All outputs saved to: {out_dir}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
