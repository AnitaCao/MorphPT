#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Quick EDA for Visium HD Human CRC gene expression data.
Run interactively: python explore_expr.py
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT    = Path("/hpc/group/jilab/boxuan/visiumHD/human_crc")
EXPR    = ROOT / "expr/expr.mtx"
GENES   = ROOT / "expr/genes.txt"
CELLS   = ROOT / "expr/cells.txt"
META    = ROOT / "meta/10.0x/human_crc.csv"

# ── 1. Load ────────────────────────────────────────────────────────────────
print("Loading expression matrix...")
X = sio.mmread(EXPR)          # sparse COO → (genes × cells) or (cells × genes)?
X_csr = X.tocsr()

genes = pd.read_csv(GENES, header=None, names=["gene"])["gene"].tolist()
cells = pd.read_csv(CELLS, header=None, names=["cell_id"])["cell_id"].tolist()

print(f"Matrix shape (raw): {X_csr.shape}")
print(f"  genes.txt length : {len(genes)}")
print(f"  cells.txt length : {len(cells)}")

# Transpose if needed so X is (cells × genes)
if X_csr.shape[0] == len(genes) and X_csr.shape[1] == len(cells):
    print("  → Transposing to (cells × genes)")
    X_csr = X_csr.T.tocsr()
elif X_csr.shape[0] == len(cells) and X_csr.shape[1] == len(genes):
    print("  → Already (cells × genes)")
else:
    print("  ⚠ Shape mismatch — check genes.txt / cells.txt lengths")

n_cells, n_genes = X_csr.shape
print(f"\nFinal shape: {n_cells} cells × {n_genes} genes")

# ── 2. Sparsity ────────────────────────────────────────────────────────────
nnz       = X_csr.nnz
sparsity  = 1 - nnz / (n_cells * n_genes)
print(f"\nSparsity: {sparsity:.3%}  ({nnz:,} non-zero entries)")

# Per-cell stats
cell_total   = np.array(X_csr.sum(axis=1)).ravel()   # total counts per cell
cell_ngenes  = np.diff(X_csr.indptr)                 # genes detected per cell

# Per-gene stats
gene_total   = np.array(X_csr.sum(axis=0)).ravel()   # total counts per gene
gene_ncells  = np.array((X_csr > 0).sum(axis=0)).ravel()  # cells expressing each gene

print(f"\nPer-cell total counts:  mean={cell_total.mean():.1f}, median={np.median(cell_total):.1f}, "
      f"min={cell_total.min():.0f}, max={cell_total.max():.0f}")
print(f"Per-cell genes detected: mean={cell_ngenes.mean():.1f}, median={np.median(cell_ngenes):.1f}, "
      f"min={cell_ngenes.min():.0f}, max={cell_ngenes.max():.0f}")
print(f"\nPer-gene total counts:  mean={gene_total.mean():.1f}, median={np.median(gene_total):.1f}")
print(f"Per-gene cells expressing: mean={gene_ncells.mean():.1f}, "
      f"median={np.median(gene_ncells):.1f}, "
      f"min={gene_ncells.min():.0f}, max={gene_ncells.max():.0f}")

# Top 20 most expressed genes
top20_idx   = np.argsort(gene_total)[::-1][:20]
top20_genes = [(genes[i], int(gene_total[i]), int(gene_ncells[i])) for i in top20_idx]
print("\nTop 20 genes by total count:")
print(f"  {'Gene':<20} {'TotalCount':>12} {'CellsCoverage':>15}")
for g, tc, nc in top20_genes:
    print(f"  {g:<20} {tc:>12,} {nc/n_cells:>14.1%}")

# Bottom 20 (rarest genes)
bot20_idx   = np.argsort(gene_total)[:20]
bot20_genes = [(genes[i], int(gene_total[i]), int(gene_ncells[i])) for i in bot20_idx]
print("\nBottom 20 genes by total count:")
print(f"  {'Gene':<20} {'TotalCount':>12} {'CellsCoverage':>15}")
for g, tc, nc in bot20_genes:
    print(f"  {g:<20} {tc:>12,} {nc/n_cells:>14.1%}")

# ── 3. Value distribution (non-zero only) ─────────────────────────────────
nz_vals = X_csr.data
print(f"\nNon-zero value stats:")
print(f"  min={nz_vals.min():.3f}, max={nz_vals.max():.3f}, "
      f"mean={nz_vals.mean():.3f}, median={np.median(nz_vals):.3f}")
print(f"  Are values integers? {np.all(nz_vals == nz_vals.astype(int))}")
print(f"  Value range: {np.unique(nz_vals[:1000])[:20]}  (first 1000 unique)")

# ── 4. Meta CSV check ──────────────────────────────────────────────────────
print("\nLoading meta CSV...")
meta = pd.read_csv(META)
print(f"Meta shape: {meta.shape}")
print(f"Columns: {meta.columns.tolist()}")
print(meta.head(3).to_string())

# Check overlap with cells.txt
meta_ids = set(meta["cell_id"].astype(str))
expr_ids = set(cells)
print(f"\nMeta cells:  {len(meta_ids):,}")
print(f"Expr cells:  {len(expr_ids):,}")
print(f"Overlap:     {len(meta_ids & expr_ids):,}")
print(f"Only in meta (no expr): {len(meta_ids - expr_ids):,}")
print(f"Only in expr (no meta): {len(expr_ids - meta_ids):,}")

# ── 5. Plots ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# (a) Per-cell total counts distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(cell_total, bins=80, color="#4C72B0", edgecolor="none")
ax.set_xlabel("Total counts per cell")
ax.set_ylabel("# cells")
ax.set_title("(a) Per-cell total counts")
ax.spines[["top","right"]].set_visible(False)

# (b) Per-cell genes detected
ax = fig.add_subplot(gs[0, 1])
ax.hist(cell_ngenes, bins=80, color="#55A868", edgecolor="none")
ax.set_xlabel("Genes detected per cell")
ax.set_ylabel("# cells")
ax.set_title("(b) Genes detected per cell")
ax.spines[["top","right"]].set_visible(False)

# (c) Per-gene cell coverage
ax = fig.add_subplot(gs[0, 2])
ax.hist(gene_ncells / n_cells * 100, bins=60, color="#C44E52", edgecolor="none")
ax.set_xlabel("% cells expressing gene")
ax.set_ylabel("# genes")
ax.set_title("(c) Gene expression prevalence")
ax.spines[["top","right"]].set_visible(False)

# (d) Non-zero value distribution
ax = fig.add_subplot(gs[1, 0])
ax.hist(np.log1p(nz_vals), bins=80, color="#8172B2", edgecolor="none")
ax.set_xlabel("log1p(count)")
ax.set_ylabel("# entries")
ax.set_title("(d) Non-zero value distribution (log1p)")
ax.spines[["top","right"]].set_visible(False)

# (e) Per-gene total counts (log scale)
ax = fig.add_subplot(gs[1, 1])
ax.hist(np.log1p(gene_total), bins=60, color="#CCB974", edgecolor="none")
ax.set_xlabel("log1p(total counts)")
ax.set_ylabel("# genes")
ax.set_title("(e) Per-gene total counts (log1p)")
ax.spines[["top","right"]].set_visible(False)

# (f) Top 20 genes bar chart
ax = fig.add_subplot(gs[1, 2])
top_names  = [g for g, _, _ in top20_genes]
top_counts = [tc for _, tc, _ in top20_genes]
ax.barh(top_names[::-1], top_counts[::-1], color="#64B5CD", edgecolor="none")
ax.set_xlabel("Total counts")
ax.set_title("(f) Top 20 genes")
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=7)

fig.suptitle("Visium HD Human CRC — Gene Expression EDA", fontsize=13, fontweight="bold")
out_path = Path("expr_eda.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nPlot saved → {out_path.resolve()}")
plt.show()

