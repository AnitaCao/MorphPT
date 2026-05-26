import nbformat
from pathlib import Path

# Load both notebooks
old_nb = nbformat.read('/hpc/group/jilab/tc459/MorphPT/visiumHD/aggregate_comparison.ipynb', as_version=4)
target_nb = nbformat.read('/hpc/group/jilab/tc459/MorphPT/visiumHD/aggregate_results_mouse_5seeds_random.ipynb', as_version=4)

# We want to basically replace the cells of target_nb with cells similar to old_nb, 
# but adapted for SEEDS instead of LAYOUTS, and random split instead of layout split.
new_cells = []

# Cell 0: Markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    "# MorphPT vs DINOv3+LoRA — mouse Visium HD encoder comparison (Random Split)\n\n"
    "Paired comparison across 4 tissues × 5 random splits, identical training protocol.\n"
    "The only varying factor is encoder initialization: MorphPT vs DINOv3 ViT-B/16.\n"
))

# Cell 1: Markdown Setup
new_cells.append(nbformat.v4.new_markdown_cell("## Setup"))

# Cell 2: Imports
new_cells.append(nbformat.v4.new_code_cell(
    "import json\n"
    "from pathlib import Path\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from scipy import stats\n\n"
    "PROJECT = Path('/hpc/group/jilab/tc459/MorphPT')\n\n"
    "plt.rcParams.update({\n"
    "    'figure.dpi':         110,\n"
    "    'savefig.dpi':        220,\n"
    "    'font.size':          10,\n"
    "    'axes.spines.top':    False,\n"
    "    'axes.spines.right':  False,\n"
    "    'axes.titlelocation': 'left',\n"
    "    'axes.titleweight':   'bold',\n"
    "    'axes.titlesize':     11,\n"
    "    'legend.frameon':     False,\n"
    "})"
))

# Cell 3: Config Markdown
new_cells.append(nbformat.v4.new_markdown_cell("## Configuration"))

# Cell 4: Config
new_cells.append(nbformat.v4.new_code_cell(
    "TISSUES  = ['mouse_brain', 'mouse_intestine', 'mouse_kidney', 'mouse_embryo']\n"
    "SEEDS  = [23, 123, 456, 789, 1234]\n"
    "ENCODERS = ['morphpt', 'dinov3_lora']\n"
    "EXP_TAG  = {'morphpt': '', 'dinov3_lora': 'dinov3_lora'}\n\n"
    "# Display labels\n"
    "ENC_LABEL = {'morphpt': 'MorphPT', 'dinov3_lora': 'DINOv3+LoRA'}\n"
    "ENC_COLOR = {'morphpt': '#C44E52', 'dinov3_lora': '#4C72B0'}\n\n"
    "# Tier sizes\n"
    "TIERS = [50, 100, 200, 300, 400, 500]\n\n"
    "# Training metadata\n"
    "N_TOP        = 500\n"
    "SCALES_TAG   = '10.0x'\n"
    "LOSS_SUFFIX  = 'mse'\n"
    "SELECT_SEED  = 42\n\n"
    "TISSUE_COLORS = {\n"
    "    'mouse_brain':     '#4C72B0',\n"
    "    'mouse_intestine': '#55A868',\n"
    "    'mouse_kidney':    '#C44E52',\n"
    "    'mouse_embryo':    '#8172B2',\n"
    "}\n\n"
    "def results_dir(encoder, tissue, seed):\n"
    "    prefix = f'{EXP_TAG[encoder]}_' if EXP_TAG[encoder] else ''\n"
    "    return PROJECT / 'experiments' / (\n"
    "        f'{prefix}lora_probing_{tissue}_top{N_TOP}_multi_random_'\n"
    "        f'{SCALES_TAG}_{LOSS_SUFFIX}_seed{seed}'\n"
    "    )\n\n"
    "def results_csv(encoder, tissue, seed):\n"
    "    return results_dir(encoder, tissue, seed) / 'multi_lora_hybrid_results.csv'\n\n"
    "def rank_csv(tissue, seed):\n"
    "    layout = f'seed{seed}'\n"
    "    return (PROJECT / f'cache_{tissue}' / 'splits' / layout /\n"
    "            f'top{N_TOP}_variance_mincov0.1_train_{layout}_seed{SELECT_SEED}.csv')"
))

# Cell 5: Markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Load results — both encoders\n\n"
    "For each (encoder, tissue, seed) load `multi_lora_hybrid_results.csv` and join with\n"
    "the training-set variance rank from the gene-selection CSV.\n"
))

# Cell 6: Load results
new_cells.append(nbformat.v4.new_code_cell(
    "rows, missing = [], []\n\n"
    "for encoder in ENCODERS:\n"
    "    for tissue in TISSUES:\n"
    "        for seed in SEEDS:\n"
    "            rcsv = results_csv(encoder, tissue, seed)\n"
    "            kcsv = rank_csv(tissue, seed)\n\n"
    "            if not rcsv.exists():\n"
    "                missing.append((encoder, tissue, seed, 'result',  str(rcsv)))\n"
    "                continue\n"
    "            if not kcsv.exists():\n"
    "                missing.append((encoder, tissue, seed, 'ranking', str(kcsv)))\n"
    "                continue\n\n"
    "            res    = pd.read_csv(rcsv)\n"
    "            ranks  = pd.read_csv(kcsv)\n"
    "            cov_cols = ['gene_idx', 'rank'] + (['coverage'] if 'coverage' in ranks.columns else [])\n"
    "            merged = res.merge(ranks[cov_cols], on='gene_idx', how='left')\n\n"
    "            test_col = next(c for c in merged.columns if c.startswith('test_pearson_s'))\n"
    "            val_col  = next(c for c in merged.columns if c.startswith('val_pearson_s'))\n\n"
    "            for _, r in merged.iterrows():\n"
    "                rows.append({\n"
    "                    'encoder':      encoder,\n"
    "                    'tissue':       tissue,\n"
    "                    'seed':         seed,\n"
    "                    'gene_idx':     int(r['gene_idx']),\n"
    "                    'gene_name':    r['gene_name'],\n"
    "                    'rank':         int(r['rank']) if pd.notna(r['rank']) else None,\n"
    "                    'coverage':     float(r['coverage']) if 'coverage' in merged.columns and pd.notna(r['coverage']) else np.nan,\n"
    "                    'test_pearson': float(r[test_col]),\n"
    "                    'val_pearson':  float(r[val_col]),\n"
    "                })\n\n"
    "df = pd.DataFrame(rows)\n"
    "print(f'Loaded: {len(df):,} (encoder, tissue, seed, gene) rows from '\n"
    "      f'{df.groupby([\"encoder\",\"tissue\",\"seed\"]).ngroups} runs')\n\n"
    "if missing:\n"
    "    print(f'\\nMissing ({len(missing)}):')\n"
    "    for m in missing:\n"
    "        print(' ', *m[:4])\n\n"
    "df.to_csv('mouse_encoder_comparison_results_random.csv', index=False)"
))

# Cell 7: Markdown
new_cells.append(nbformat.v4.new_markdown_cell("## Per-run summary — both encoders"))

# Cell 8: Per-run summary
new_cells.append(nbformat.v4.new_code_cell(
    "per_run = (df.groupby(['encoder', 'tissue', 'seed'])\n"
    "             .agg(mean_test_r=('test_pearson', 'mean'),\n"
    "                  median_test_r=('test_pearson', 'median'),\n"
    "                  n_genes=('gene_idx', 'count'))\n"
    "             .reset_index())\n\n"
    "# Wide pivot: tissue rows, (encoder × seed) columns\n"
    "pivot = (per_run.pivot_table(index='tissue', columns=['encoder', 'seed'],\n"
    "                              values='mean_test_r')\n"
    "                .reindex(TISSUES))\n"
    "print('Mean test Pearson r per (tissue × encoder × seed):\\n')\n"
    "print(pivot.round(4))\n\n"
    "# Per-tissue per-encoder summary\n"
    "summary = (per_run.groupby(['encoder', 'tissue'])['mean_test_r']\n"
    "                  .agg(['mean', 'std', 'min', 'max'])\n"
    "                  .reset_index())\n"
    "print('\\nPer-tissue per-encoder summary (mean ± std across 5 seeds):\\n')\n"
    "print(summary.round(4).to_string(index=False))"
))

# Cell 9: Plot 1 Markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Plot 1 — Headline: mean test r per tissue, by encoder\n\n"
    "Grouped bars, MorphPT vs DINOv3+LoRA per tissue. Error bars = std across 5 random seeds.\n"
    "Dots = individual seeds."
))

# Cell 10: Plot 1 Code
new_cells.append(nbformat.v4.new_code_cell(
    "agg = (per_run.groupby(['tissue', 'encoder'])['mean_test_r']\n"
    "              .agg(['mean', 'std', list])\n"
    "              .reset_index())\n\n"
    "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
    "xs   = np.arange(len(TISSUES))\n"
    "w    = 0.36\n\n"
    "rng = np.random.default_rng(0)\n"
    "for j, encoder in enumerate(ENCODERS):\n"
    "    sub = agg[agg['encoder'] == encoder].set_index('tissue').reindex(TISSUES)\n"
    "    offset = (j - 0.5) * w\n"
    "    ax.bar(xs + offset, sub['mean'], width=w,\n"
    "           yerr=sub['std'], capsize=3,\n"
    "           color=ENC_COLOR[encoder], edgecolor='black', linewidth=0.5,\n"
    "           alpha=0.85, label=ENC_LABEL[encoder])\n"
    "    # Overlay layout dots\n"
    "    for i, vals in enumerate(sub['list']):\n"
    "        if not isinstance(vals, list):\n"
    "            continue\n"
    "        xj = (i + offset) + rng.uniform(-0.06, 0.06, len(vals))\n"
    "        ax.scatter(xj, vals, s=14, color='black', alpha=0.65, zorder=3)\n"
    "    # Value annotations\n"
    "    for i, (m, s) in enumerate(zip(sub['mean'], sub['std'])):\n"
    "        if pd.notna(m):\n"
    "            ax.text(i + offset, m + (s if pd.notna(s) else 0) + 0.005,\n"
    "                    f'{m:.3f}', ha='center', va='bottom', fontsize=8.5)\n\n"
    "ax.set_xticks(xs)\n"
    "ax.set_xticklabels([t.replace('mouse_', '') for t in TISSUES])\n"
    "ax.set_ylabel('mean test Pearson r')\n"
    "ax.set_title('encoder comparison — mean test r across 500 genes, random splits')\n"
    "ax.legend(loc='upper left')\n"
    "ax.set_ylim(0, agg.groupby('tissue')['mean'].max().max() * 1.25)\n"
    "plt.tight_layout()\n"
    "plt.savefig('plot_encoder_comparison_bars_random.pdf', bbox_inches='tight')\n"
    "plt.show()"
))

# Cell 11: Plot 1b Markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Plot 1b — Per-tissue Δr (MorphPT − DINOv3) with paired Wilcoxon\n\n"
    "For each tissue, MorphPT vs DINOv3 across the 5 *matched* seeds gives 5 paired samples.\n"
    "Wilcoxon signed-rank test (one-sided, MorphPT > DINOv3). Pooled test uses all 20 pairs."
))

# Cell 12: Plot 1b code
new_cells.append(nbformat.v4.new_code_cell(
    "# Compute paired deltas per (tissue, seed)\n"
    "wide = per_run.pivot_table(index=['tissue', 'seed'],\n"
    "                            columns='encoder', values='mean_test_r').reset_index()\n"
    "wide['delta'] = wide['morphpt'] - wide['dinov3_lora']\n\n"
    "# Per-tissue stats\n"
    "delta_stats = []\n"
    "for tissue in TISSUES:\n"
    "    sub = wide[wide['tissue'] == tissue].dropna(subset=['morphpt', 'dinov3_lora'])\n"
    "    deltas = sub['delta'].values\n"
    "    if len(deltas) >= 2:\n"
    "        w_stat, p_one = stats.wilcoxon(deltas, alternative='greater',\n"
    "                                        zero_method='wilcox', mode='exact')\n"
    "    else:\n"
    "        w_stat, p_one = np.nan, np.nan\n"
    "    delta_stats.append({\n"
    "        'tissue':      tissue,\n"
    "        'n_pairs':     len(deltas),\n"
    "        'mean_delta':  deltas.mean() if len(deltas) else np.nan,\n"
    "        'std_delta':   deltas.std()  if len(deltas) else np.nan,\n"
    "        'wilcoxon_W':  w_stat,\n"
    "        'p_one_sided': p_one,\n"
    "    })\n"
    "delta_stats_df = pd.DataFrame(delta_stats)\n"
    "print('Per-tissue paired test (MorphPT − DINOv3, one-sided Wilcoxon):\\n')\n"
    "print(delta_stats_df.round(4).to_string(index=False))\n\n"
    "# Pooled across all 20 pairs\n"
    "pooled_deltas = wide['delta'].dropna().values\n"
    "w_pool, p_pool = stats.wilcoxon(pooled_deltas, alternative='greater',\n"
    "                                 zero_method='wilcox', mode='exact')\n"
    "print(f'\\nPooled across all {len(pooled_deltas)} (tissue, seed) pairs:')\n"
    "print(f'  mean Δr = {pooled_deltas.mean():+.4f}')\n"
    "print(f'  Wilcoxon W = {w_pool}, one-sided p = {p_pool:.3g}')"
))

# Cell 13: Plot 1b plot
new_cells.append(nbformat.v4.new_code_cell(
    "# Plot Δr per tissue with dot strip + p-value annotation\n"
    "fig, ax = plt.subplots(figsize=(7, 4.2))\n"
    "xs = np.arange(len(TISSUES))\n\n"
    "for i, tissue in enumerate(TISSUES):\n"
    "    sub = wide[wide['tissue'] == tissue].dropna(subset=['delta'])\n"
    "    deltas = sub['delta'].values\n"
    "    if len(deltas) == 0:\n"
    "        continue\n"
    "    xj = i + np.linspace(-0.15, 0.15, len(deltas))\n"
    "    ax.scatter(xj, deltas, s=42, color=TISSUE_COLORS[tissue],\n"
    "               edgecolor='black', linewidth=0.5, alpha=0.85, zorder=3)\n"
    "    m = deltas.mean()\n"
    "    ax.hlines(m, i - 0.25, i + 0.25, color='black', linewidth=2.2, zorder=4)\n\n"
    "    p = delta_stats_df.set_index('tissue').loc[tissue, 'p_one_sided']\n"
    "    p_text = f'p={p:.3f}' if pd.notna(p) and p >= 0.001 else 'p<0.001'\n"
    "    y_top = max(deltas.max(), m) + 0.005\n"
    "    ax.text(i, y_top + 0.003, p_text, ha='center', va='bottom',\n"
    "            fontsize=9, fontweight='bold')\n\n"
    "ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')\n"
    "ax.set_xticks(xs)\n"
    "ax.set_xticklabels([t.replace('mouse_', '') for t in TISSUES])\n"
    "ax.set_ylabel('Δ mean test r  (MorphPT − DINOv3)')\n"
    "ax.set_title(f'per-tissue MorphPT advantage;  pooled p = {p_pool:.3g}, n = {len(pooled_deltas)} pairs')\n"
    "plt.tight_layout()\n"
    "plt.savefig('plot_encoder_delta_random.pdf', bbox_inches='tight')\n"
    "plt.show()"
))

# Cell 14: Markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Tier computations — variance / coverage / performance × encoder\n\n"
    "For each (encoder, tissue, seed) compute mean test r over the top-N genes ranked by:\n"
    "- **variance** (a-priori, train-set variance — the realistic selection rule, no leakage)\n"
    "- **coverage** (a-priori, fraction of cells expressing — abundance control)\n"
    "- **performance** (post-hoc, ranked by test_pearson — oracle ceiling)"
))

# Cell 15: Tier computations
new_cells.append(nbformat.v4.new_code_cell(
    "def compute_tiers(df_in, rank_col):\n"
    "    rows = []\n"
    "    for (encoder, tissue, seed), sub in df_in.groupby(['encoder', 'tissue', 'seed']):\n"
    "        if rank_col == 'rank':\n"
    "            sub_sorted = sub.sort_values('rank', ascending=True)\n"
    "        elif rank_col == 'test_pearson':\n"
    "            sub_sorted = sub.sort_values('test_pearson', ascending=False)\n"
    "        elif rank_col == 'coverage':\n"
    "            sub_sorted = sub.dropna(subset=['coverage']).sort_values('coverage', ascending=False)\n"
    "        else:\n"
    "            raise ValueError(rank_col)\n"
    "        for tier in TIERS:\n"
    "            top = sub_sorted.head(tier)\n"
    "            rows.append({\n"
    "                'encoder':     encoder,\n"
    "                'tissue':      tissue,\n"
    "                'seed':        seed,\n"
    "                'tier':        tier,\n"
    "                'mean_test_r': top['test_pearson'].mean(),\n"
    "                'n_genes':     len(top),\n"
    "            })\n"
    "    return pd.DataFrame(rows)\n\n"
    "var_tier  = compute_tiers(df, 'rank')\n"
    "cov_tier  = compute_tiers(df, 'coverage')\n"
    "perf_tier = compute_tiers(df, 'test_pearson')\n\n"
    "def aggregate_tiers(tdf):\n"
    "    return (tdf.groupby(['encoder', 'tissue', 'tier'])\n"
    "               .agg(mean=('mean_test_r', 'mean'),\n"
    "                    std =('mean_test_r', 'std'))\n"
    "               .reset_index())\n\n"
    "var_agg  = aggregate_tiers(var_tier)\n"
    "cov_agg  = aggregate_tiers(cov_tier)\n"
    "perf_agg = aggregate_tiers(perf_tier)\n\n"
    "print('Variance-tier means at top-50 (encoder × tissue):\\n')\n"
    "print(var_agg[var_agg['tier'] == 50]\n"
    "        .pivot(index='tissue', columns='encoder', values='mean')[ENCODERS]\n"
    "        .reindex(TISSUES).round(3))"
))

# Cell 16: Markdown
new_cells.append(nbformat.v4.new_markdown_cell(
    "## Plot 2 — Variance-tier curves: MorphPT vs DINOv3 (4-panel)\n\n"
    "One panel per tissue. Variance-ranked tier curve for both encoders.\n"
))

# Cell 17: Plot 2
new_cells.append(nbformat.v4.new_code_cell(
    "fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)\n"
    "for ax, tissue in zip(axes, TISSUES):\n"
    "    for encoder in ENCODERS:\n"
    "        sub = var_agg[(var_agg['tissue'] == tissue) & (var_agg['encoder'] == encoder)]\n"
    "        ax.plot(sub['tier'], sub['mean'], marker='o', markersize=5,\n"
    "                color=ENC_COLOR[encoder], label=ENC_LABEL[encoder],\n"
    "                linewidth=2)\n"
    "        ax.fill_between(sub['tier'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],\n"
    "                        color=ENC_COLOR[encoder], alpha=0.15)\n"
    "    ax.set_title(tissue.replace('mouse_', ''))\n"
    "    ax.set_xlabel('Top N genes (variance)')\n"
    "    ax.grid(alpha=0.3)\n"
    "    if ax == axes[0]:\n"
    "        ax.set_ylabel('mean test Pearson r')\n"
    "        ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig('plot_variance_tiers_encoder_random.pdf', bbox_inches='tight')\n"
    "plt.show()"
))

target_nb.cells = new_cells
nbformat.write(target_nb, '/hpc/group/jilab/tc459/MorphPT/visiumHD/aggregate_results_mouse_5seeds_random.ipynb')

