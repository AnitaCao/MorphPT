#!/usr/bin/env python3
"""
Plot feature-space visualizations for MorphPT vs frozen encoders.

Designed for server / Slurm execution, not notebooks.

Inputs:
  Parquet files with an embedding column, usually emb_fused.

Default expected files:
  results/embeddings/core500/morphpt_router_core500_by_fine.parquet
  results/embeddings/core500/frozen_dinov3_core500_by_fine.parquet
  results/embeddings/core500/frozen_simclr_core500_by_fine.parquet, optional

Outputs:
  UMAP and PCA figures as PNG/PDF (categorical, shade-by-coarse colouring).
  Separate legend files.
  Coordinate parquet for the plotted subset.
  feature_space_metrics.csv  -- ARI / NMI / silhouette per encoder.
  feature_space_ari.png/pdf  -- ARI bar chart.

Recommended:
  Start with --fit_per_class 150. Increase to 250 after it works.

Notes:
  - Coarse groups are derived from the fine label via FINE_TO_COARSE, so the
    script does not depend on a coarse column being present in the parquet.
  - Metrics (ARI/NMI/silhouette) are computed on the L2-normalized embeddings
    of the fit subset. Raise --fit_per_class for more stable numbers, or pass
    --skip_metrics while iterating on the figures.
  - emb_fused for a frozen encoder must be a documented naive fusion (e.g. mean
    of the two per-scale features); only MorphPT has a learned fusion.
"""

import os

# Set thread limits before importing numpy, sklearn, numba, or umap.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import gc
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

try:
    import umap
except ImportError as e:
    raise ImportError("Install umap-learn first: pip install umap-learn") from e


# --------------------------------------------------------------------------
# Canonical label structure and colours
# --------------------------------------------------------------------------

# fine class -> coarse group (from the extraction log)
FINE_TO_COARSE = {
    "Astrocytes": "Neuroglial", "Microglia": "Neuroglial",
    "Neurons": "Neuroglial", "Oligodendrocytes": "Neuroglial",
    "B cells": "Lymphoid", "NK cells": "Lymphoid", "T cells": "Lymphoid",
    "Colon cancer cells": "Cancer", "Liver cancer cells": "Cancer",
    "Lung cancer cells": "Cancer", "Ovary cancer cells": "Cancer",
    "Pancreas cancer cells": "Cancer", "Skin cancer cells": "Cancer",
    "Endothelial cells": "Tissue_Vascular", "Epithelial cells": "Tissue_Vascular",
    "Fibroblasts": "Tissue_Vascular", "Myeloid cells": "Tissue_Vascular",
    "Pericytes": "Tissue_Vascular", "Smooth muscle cells": "Tissue_Vascular",
    "Stem and progenitor cells": "Stem_Progenitor",
    "Stromal cells": "Stromal",
}

COARSE_ORDER = ["Cancer", "Lymphoid", "Neuroglial",
                "Stem_Progenitor", "Stromal", "Tissue_Vascular"]

COARSE_COLORS = {
    "Cancer":          "#9467BD",
    "Lymphoid":        "#2CA02C",
    "Neuroglial":      "#AEC7E8",
    "Stem_Progenitor": "#BCBD22",
    "Stromal":         "#E377C2",
    "Tissue_Vascular": "#D62728",
}

# User-provided cell-type palette (Cancer subtypes share one color key).
CELLTYPE_ORDER = [
    "B cells", "T cells", "NK cells", "Myeloid cells",
    "Astrocytes", "Oligodendrocytes", "Neurons", "Microglia",
    "Stem and progenitor cells", "Epithelial cells",
    "Stromal cells", "Endothelial cells", "Fibroblasts",
    "Pericytes", "Smooth muscle cells", "Cancer cells",
]
CELLTYPE_COLORS = {
    "B cells": "#2CA02C",
    "T cells": "#1F77B4",
    "NK cells": "#FF7F0E",
    "Myeloid cells": "#C49C94",
    "Astrocytes": "#98DF8A",
    "Oligodendrocytes": "#AEC7E8",
    "Neurons": "#FFBB78",
    "Microglia": "#FF9896",
    "Stem and progenitor cells": "#BCBD22",
    "Epithelial cells": "#F7B6D2",
    "Stromal cells": "#E377C2",
    "Endothelial cells": "#17BECF",
    "Fibroblasts": "#DBDB8D",
    "Pericytes": "#9EDAE5",
    "Smooth muscle cells": "#D62728",
    "Cancer cells": "#9467BD",
}

FINE_TO_PLOT_CELLTYPE = {
    "Astrocytes": "Astrocytes",
    "Microglia": "Microglia",
    "Neurons": "Neurons",
    "Oligodendrocytes": "Oligodendrocytes",
    "B cells": "B cells",
    "NK cells": "NK cells",
    "T cells": "T cells",
    "Colon cancer cells": "Cancer cells",
    "Liver cancer cells": "Cancer cells",
    "Lung cancer cells": "Cancer cells",
    "Ovary cancer cells": "Cancer cells",
    "Pancreas cancer cells": "Cancer cells",
    "Skin cancer cells": "Cancer cells",
    "Endothelial cells": "Endothelial cells",
    "Epithelial cells": "Epithelial cells",
    "Fibroblasts": "Fibroblasts",
    "Myeloid cells": "Myeloid cells",
    "Pericytes": "Pericytes",
    "Smooth muscle cells": "Smooth muscle cells",
    "Stem and progenitor cells": "Stem and progenitor cells",
    "Stromal cells": "Stromal cells",
}

FINE_COLORS = {
    fine: CELLTYPE_COLORS[FINE_TO_PLOT_CELLTYPE[fine]]
    for fine in FINE_TO_COARSE
}
FINE_LEGEND_COLORS = CELLTYPE_COLORS.copy()
FINE_ORDER = CELLTYPE_ORDER


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project_root",
        default="/hpc/group/jilab/tc459/MorphPT",
        help="Project root directory.",
    )
    ap.add_argument(
        "--emb_dir",
        default=None,
        help="Embedding directory. Defaults to PROJECT_ROOT/results/embeddings/core500.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output figure directory. Defaults to PROJECT_ROOT/results/figures/feature_space_core500_py.",
    )

    ap.add_argument(
        "--morphpt",
        default=None,
        help="MorphPT parquet path. Defaults to emb_dir/morphpt_router_core500_by_fine.parquet.",
    )
    ap.add_argument(
        "--dinov3",
        default=None,
        help="Frozen DINOv3 parquet path. Defaults to emb_dir/frozen_dinov3_core500_by_fine.parquet.",
    )
    ap.add_argument(
        "--simclr",
        default=None,
        help="Frozen SimCLR parquet path. Defaults to emb_dir/frozen_simclr_core500_by_fine.parquet if it exists.",
    )

    ap.add_argument("--emb_col", default="emb_2p5x")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument(
        "--fit_per_class",
        type=int,
        default=150,
        help="Balanced number of cells per fine class used to fit PCA/UMAP and metrics.",
    )
    ap.add_argument(
        "--plot_per_class",
        type=int,
        default=150,
        help="Balanced number of cells per fine class shown in the final plot.",
    )
    ap.add_argument(
        "--pca_pre_components",
        type=int,
        default=50,
        help="PCA dimensions before UMAP.",
    )

    ap.add_argument(
        "--align",
        choices=["row_order", "cell_id"],
        default="row_order",
        help="Use row_order when all files came from the same sampled parquet.",
    )

    ap.add_argument("--umap_neighbors", type=int, default=25)
    ap.add_argument("--umap_min_dist", type=float, default=0.10)
    ap.add_argument("--point_size", type=float, default=6.0)
    ap.add_argument("--point_alpha", type=float, default=0.70)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="Recompute cached coordinates.")
    ap.add_argument("--skip_metrics", action="store_true",
                    help="Skip ARI/NMI/silhouette computation (faster when iterating on plots).")
    ap.add_argument("--metric_kmeans_seeds", type=int, default=3,
                    help="Number of k-means seeds to average ARI/NMI over.")

    return ap.parse_args()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def available_columns(path):
    if pq is not None:
        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_parquet(path).columns)


def read_needed_parquet(path, encoder_name, emb_col):
    cols = available_columns(path)

    candidates = ["cell_id", "fine_label", "label", "target_label", "tissue", emb_col]
    use_cols = [c for c in candidates if c in cols]

    if emb_col not in use_cols:
        raise RuntimeError(f"{encoder_name}: missing {emb_col}. Available columns: {cols}")

    df = pd.read_parquet(path, columns=use_cols)

    if "cell_id" not in df.columns:
        df["cell_id"] = np.arange(len(df)).astype(str)
        warnings.warn(f"{encoder_name}: missing cell_id. Using row index as cell_id.")
    else:
        df["cell_id"] = df["cell_id"].astype(str)

    # Resolve the fine-label column.
    if "fine_label" in df.columns:
        df["fine_label_plot"] = df["fine_label"].astype(str)
    elif "label" in df.columns:
        df["fine_label_plot"] = df["label"].astype(str)
    elif "target_label" in df.columns:
        df["fine_label_plot"] = df["target_label"].astype(str)
    else:
        raise RuntimeError(f"{encoder_name}: no fine label column found. Columns: {list(df.columns)}")

    # Guard: if the resolved 'fine' column only holds coarse group names, the
    # parquet has no real fine-label column and the comparison would be wrong.
    fine_vals = set(df["fine_label_plot"].unique())
    if fine_vals and fine_vals <= set(COARSE_ORDER):
        raise RuntimeError(
            f"{encoder_name}: the resolved fine-label column only contains coarse "
            f"group names {sorted(fine_vals)}. The parquet likely lacks a true "
            f"fine-label column -- check the column names in {path}."
        )

    # Coarse group is derived from the fine label (canonical mapping), so the
    # script does not depend on a coarse column existing in the parquet.
    df["coarse_label_plot"] = df["fine_label_plot"].map(FINE_TO_COARSE)
    n_missing = int(df["coarse_label_plot"].isna().sum())
    if n_missing:
        bad = sorted(set(df.loc[df["coarse_label_plot"].isna(), "fine_label_plot"]))
        raise RuntimeError(
            f"{encoder_name}: fine labels missing from FINE_TO_COARSE: {bad}"
        )

    return df


def load_files(args):
    project_root = Path(args.project_root)
    emb_dir = Path(args.emb_dir) if args.emb_dir else project_root / "results/embeddings/core500"

    paths = {
        "MorphPT": Path(args.morphpt) if args.morphpt else emb_dir / "morphpt_router_core500_by_fine.parquet",
        "Frozen DINOv3": Path(args.dinov3) if args.dinov3 else emb_dir / "frozen_dinov3_core500_by_fine.parquet",
    }

    simclr_default = emb_dir / "frozen_simclr_core500_by_fine.parquet"
    if args.simclr:
        paths["Frozen SimCLR"] = Path(args.simclr)
    elif simclr_default.exists():
        paths["Frozen SimCLR"] = simclr_default

    dfs = {}
    for name, path in paths.items():
        if path.exists():
            df = read_needed_parquet(path, name, args.emb_col)
            dfs[name] = df
            print(f"{name:<14} {len(df):>7,} rows  path={path}")
        else:
            print(f"{name:<14} missing, skipped: {path}")

    if not dfs:
        raise RuntimeError("No embedding parquet files loaded.")

    return dfs


def align_dfs(dfs, align):
    names = list(dfs.keys())
    ref_name = names[0]
    ref = dfs[ref_name].copy()

    if align == "row_order":
        n_ref = len(ref)
        for name, df in dfs.items():
            if len(df) != n_ref:
                raise RuntimeError(
                    f"{name} has {len(df)} rows, but {ref_name} has {n_ref}. "
                    "Use --align cell_id if cell_id is unique."
                )
        print("Alignment: row_order")

        for name, df in dfs.items():
            agreement = (df["fine_label_plot"].to_numpy() == ref["fine_label_plot"].to_numpy()).mean()
            print(f"{name:<14} fine-label row agreement with {ref_name}: {agreement:.4f}")
            if agreement < 0.999:
                raise RuntimeError(
                    f"{name}: fine-label row agreement with {ref_name} is only "
                    f"{agreement:.4f}. The files are not the same cells in the same "
                    "order -- use --align cell_id, or check the label columns."
                )

    else:
        for name, df in dfs.items():
            dup = df["cell_id"].duplicated().sum()
            if dup > 0:
                raise RuntimeError(
                    f"{name} has {dup} duplicate cell_id values. "
                    "Use --align row_order if files came from the same sampled parquet."
                )

        common_ids = set(dfs[names[0]]["cell_id"])
        for name in names[1:]:
            common_ids &= set(dfs[name]["cell_id"])
        common_ids = sorted(common_ids)

        if not common_ids:
            raise RuntimeError("No common cell_id values across encoders.")

        aligned = {}
        for name, df in dfs.items():
            aligned[name] = df.set_index("cell_id").loc[common_ids].reset_index()
        dfs = aligned
        ref = dfs[ref_name].copy()
        print(f"Alignment: cell_id, common cells={len(common_ids):,}")

    print()
    print(f"Reference encoder: {ref_name}")
    print(f"Rows: {len(ref):,}")
    print(f"Fine classes: {ref['fine_label_plot'].nunique()}")
    print(ref["fine_label_plot"].value_counts().sort_index().to_string())

    return dfs, ref_name, ref


# --------------------------------------------------------------------------
# Sampling / embeddings / projections
# --------------------------------------------------------------------------

def balanced_row_indices(df, label_col, max_per_class, seed):
    parts = []
    for label, grp in df.groupby(label_col, sort=True):
        n = min(len(grp), max_per_class)
        parts.append(grp.sample(n=n, random_state=seed).index.to_numpy())

    idx = np.concatenate(parts)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return idx


def stack_embedding_column(series):
    arrs = []
    for v in series:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape {a.shape}")
        arrs.append(a)

    X = np.vstack(arrs).astype(np.float32)
    X = normalize(X, norm="l2", axis=1)
    return X


def pca_then_umap(X, name, args, out_dir):
    cache = out_dir / f"{safe_name(name)}_{args.emb_col}_umap_fit{args.fit_per_class}_seed{args.seed}.npz"
    if cache.exists() and not args.force:
        z = np.load(cache)
        return z["coords"]

    n_components = min(args.pca_pre_components, X.shape[0] - 1, X.shape[1])
    print(f"  PCA pre-reduction: {X.shape[1]} -> {n_components}")
    X_pca = PCA(n_components=n_components, random_state=args.seed).fit_transform(X)
    X_pca = normalize(X_pca.astype(np.float32), norm="l2", axis=1)

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
        low_memory=True,
        n_jobs=1,
    )
    coords = reducer.fit_transform(X_pca)
    np.savez_compressed(cache, coords=coords)
    return coords


def pca_2d(X, name, args, out_dir):
    cache = out_dir / f"{safe_name(name)}_{args.emb_col}_pca2_fit{args.fit_per_class}_seed{args.seed}.npz"
    if cache.exists() and not args.force:
        z = np.load(cache)
        return z["coords"]

    coords = PCA(n_components=2, random_state=args.seed).fit_transform(X)
    np.savez_compressed(cache, coords=coords)
    return coords


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def clustering_metrics(X, labels, n_seeds=3, base_seed=0):
    # X: (n, d) L2-normalized embeddings. labels: array of class strings.
    # k-means is run at k = number of true classes; ARI/NMI score how well the
    # unsupervised clustering recovers the labels. Silhouette needs no clustering.
    labels = np.asarray(labels).astype(str)
    k = int(len(np.unique(labels)))

    aris, nmis = [], []
    for s in range(max(1, n_seeds)):
        pred = KMeans(n_clusters=k, n_init=10, random_state=base_seed + s).fit_predict(X)
        aris.append(adjusted_rand_score(labels, pred))
        nmis.append(normalized_mutual_info_score(labels, pred))

    sil = silhouette_score(X, labels, metric="cosine")

    return {
        "k": k,
        "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris)),
        "nmi_mean": float(np.mean(nmis)), "nmi_std": float(np.std(nmis)),
        "silhouette": float(sil),
    }


def save_metrics_bar(metrics_df, out_dir, args):
    encoders = list(dict.fromkeys(metrics_df["encoder"]))
    fine = metrics_df[metrics_df["level"] == "fine"].set_index("encoder")
    coarse = metrics_df[metrics_df["level"] == "coarse"].set_index("encoder")

    x = np.arange(len(encoders))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.8 * len(encoders) + 2.2, 3.6))

    ax.bar(x - w / 2, [coarse.loc[e, "ari_mean"] for e in encoders], w,
           yerr=[coarse.loc[e, "ari_std"] for e in encoders], capsize=3,
           color="#2e6f9e", label="coarse (6-way)")
    ax.bar(x + w / 2, [fine.loc[e, "ari_mean"] for e in encoders], w,
           yerr=[fine.loc[e, "ari_std"] for e in encoders], capsize=3,
           color="#d1495b", label="fine (21-way)")

    for i, e in enumerate(encoders):
        ax.text(i - w / 2, coarse.loc[e, "ari_mean"] + 0.025,
                f"{coarse.loc[e, 'ari_mean']:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, fine.loc[e, "ari_mean"] + 0.025,
                f"{fine.loc[e, 'ari_mean']:.2f}", ha="center", fontsize=8)

    lo = min(0.0, float(metrics_df["ari_mean"].min()))
    ax.set_xticks(x)
    ax.set_xticklabels(encoders)
    ax.set_ylabel("Adjusted Rand Index (k-means vs labels)")
    ax.set_ylim(lo - 0.03, 1.05)
    ax.set_title("Cluster recovery of cell types from frozen features", loc="left")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = out_dir / "feature_space_ari"
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.with_suffix('.png')}")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def scatter_panel(ax, coords, labels, color_map, title, args):
    colors = [color_map.get(str(lab), "#cccccc") for lab in labels]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=args.point_size,
        alpha=args.point_alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_legend(order, color_map, out_path, args):
    handles = [
        Line2D([0], [0], marker="o", linestyle="", label=lab,
               markerfacecolor=color_map.get(lab, "#cccccc"),
               markeredgecolor="none", markersize=6)
        for lab in order
    ]
    height = max(2.5, 0.22 * len(handles))
    fig = plt.figure(figsize=(4.8, height))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", frameon=False)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_row_plot(results, coord_key, label_key, color_map, title, out_stem, args, out_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    axes = axes[0]

    for ax, (name, r) in zip(axes, results.items()):
        scatter_panel(ax, r[coord_key], r[label_key], color_map, name, args)

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()

    out = out_dir / out_stem
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out.with_suffix('.png')}")


def save_combined_umap(results, fine_color_map, coarse_color_map, args, out_dir):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.2), squeeze=False)

    for j, (name, r) in enumerate(results.items()):
        scatter_panel(axes[0, j], r["umap"], r["fine"], fine_color_map, name, args)
        scatter_panel(axes[1, j], r["umap"], r["coarse"], coarse_color_map, name, args)

    axes[0, 0].set_ylabel("Fine cell type", fontsize=12)
    axes[1, 0].set_ylabel("Coarse group", fontsize=12)

    fig.suptitle("Feature-space visualization on held-out DAPI core test set", y=1.01, fontsize=14)
    fig.tight_layout()

    out = out_dir / "umap_combined_fine_and_coarse"
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out.with_suffix('.png')}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir) if args.out_dir else project_root / f"results/figures/feature_space_core500_{safe_name(args.emb_col)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", out_dir)
    print("fit_per_class:", args.fit_per_class)
    print("plot_per_class:", args.plot_per_class)
    print("embedding column:", args.emb_col)
    print("metrics:", "skipped" if args.skip_metrics else f"on ({args.metric_kmeans_seeds} k-means seeds)")
    print()

    dfs = load_files(args)
    dfs, ref_name, ref = align_dfs(dfs, args.align)

    fit_idx = balanced_row_indices(ref, "fine_label_plot", args.fit_per_class, args.seed)

    fit_ref = ref.iloc[fit_idx].copy()
    plot_idx_local = balanced_row_indices(
        fit_ref.reset_index(drop=False),
        "fine_label_plot",
        args.plot_per_class,
        args.seed,
    )
    plot_idx = fit_ref.iloc[plot_idx_local].index.to_numpy()

    print()
    print(f"Fit cells: {len(fit_idx):,}")
    print(f"Plot cells: {len(plot_idx):,}")
    print(ref.iloc[fit_idx]["fine_label_plot"].value_counts().sort_index().to_string())
    print()

    results = {}
    metrics_rows = []

    for name, df in dfs.items():
        print(f"Processing {name}")

        X_fit = stack_embedding_column(df.iloc[fit_idx][args.emb_col])
        print(f"  X_fit shape: {X_fit.shape}")

        umap_fit = pca_then_umap(X_fit, name, args, out_dir)
        pca_fit = pca_2d(X_fit, name, args, out_dir)

        if not args.skip_metrics:
            fit_fine = df.iloc[fit_idx]["fine_label_plot"].astype(str).to_numpy()
            fit_coarse = df.iloc[fit_idx]["coarse_label_plot"].astype(str).to_numpy()
            for level, lab in [("fine", fit_fine), ("coarse", fit_coarse)]:
                m = clustering_metrics(
                    X_fit, lab,
                    n_seeds=args.metric_kmeans_seeds,
                    base_seed=args.seed,
                )
                metrics_rows.append({"encoder": name, "level": level, **m})
                print(f"  [{level:6s}] k={m['k']:2d}  "
                      f"ARI={m['ari_mean']:.3f}+/-{m['ari_std']:.3f}  "
                      f"NMI={m['nmi_mean']:.3f}  silhouette={m['silhouette']:.3f}")

        pos = {int(row_i): j for j, row_i in enumerate(fit_idx)}
        plot_pos = np.asarray([pos[int(i)] for i in plot_idx], dtype=np.int64)
        plot_df = df.iloc[plot_idx].copy()

        results[name] = {
            "umap": umap_fit[plot_pos],
            "pca": pca_fit[plot_pos],
            "fine": plot_df["fine_label_plot"].astype(str).to_numpy(),
            "coarse": plot_df["coarse_label_plot"].astype(str).to_numpy(),
            "cell_id": plot_df["cell_id"].astype(str).to_numpy(),
        }

        del X_fit, umap_fit, pca_fit, plot_df
        gc.collect()
        print()

    # ---- figures -------------------------------------------------------
    save_row_plot(
        results,
        coord_key="umap",
        label_key="fine",
        color_map=FINE_COLORS,
        title="UMAP of feature embeddings by fine cell type",
        out_stem="umap_by_fine_cell_type",
        args=args,
        out_dir=out_dir,
    )

    save_row_plot(
        results,
        coord_key="umap",
        label_key="coarse",
        color_map=COARSE_COLORS,
        title="UMAP of feature embeddings by coarse biological group",
        out_stem="umap_by_coarse_group",
        args=args,
        out_dir=out_dir,
    )

    save_combined_umap(results, FINE_COLORS, COARSE_COLORS, args, out_dir)

    save_row_plot(
        results,
        coord_key="pca",
        label_key="coarse",
        color_map=COARSE_COLORS,
        title="PCA of feature embeddings by coarse biological group",
        out_stem="pca_by_coarse_group",
        args=args,
        out_dir=out_dir,
    )

    save_legend(FINE_ORDER, FINE_LEGEND_COLORS, out_dir / "legend_fine_cell_type.png", args)
    save_legend(COARSE_ORDER, COARSE_COLORS, out_dir / "legend_coarse_group.png", args)

    # ---- plotted-subset coordinates -----------------------------------
    coord_rows = []
    for name, r in results.items():
        coord_rows.append(pd.DataFrame({
            "encoder": name,
            "cell_id": r["cell_id"],
            "fine_label": r["fine"],
            "coarse_label": r["coarse"],
            "umap_1": r["umap"][:, 0],
            "umap_2": r["umap"][:, 1],
            "pca_1": r["pca"][:, 0],
            "pca_2": r["pca"][:, 1],
        }))

    coords_df = pd.concat(coord_rows, ignore_index=True)
    coords_path = out_dir / "feature_space_coords_plot_subset.parquet"
    coords_df.to_parquet(coords_path, index=False)
    print(f"Saved: {coords_path}")

    # ---- metrics -------------------------------------------------------
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = out_dir / "feature_space_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved: {metrics_path}")
        save_metrics_bar(metrics_df, out_dir, args)
        print()
        print("Metrics (k-means vs labels on L2-normalized embeddings):")
        print(metrics_df.round(4).to_string(index=False))

    print()
    print("Done.")


if __name__ == "__main__":
    main()