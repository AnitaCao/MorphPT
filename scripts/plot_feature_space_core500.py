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
  UMAP and PCA figures as PNG/PDF.
  Coordinate parquet for the plotted subset.

Recommended:
  Start with --fit_per_class 150.
  Increase to 250 after it works.
"""

import os

# Set thread limits before importing numpy, sklearn, numba, or umap.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# Keep package caches on writable node-local storage during Slurm jobs.
_cache_root = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_root, "morphpt_mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_cache_root, "morphpt_numba_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

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
from matplotlib.colors import Normalize

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

try:
    import umap
except ImportError as e:
    raise ImportError("Install umap-learn first: pip install umap-learn") from e


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

    ap.add_argument("--emb_col", default="emb_fused")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument(
        "--fit_per_class",
        type=int,
        default=150,
        help="Balanced number of cells per fine class used to fit PCA/UMAP.",
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

    return ap.parse_args()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def available_columns(path):
    if pq is not None:
        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_parquet(path).columns)


def read_needed_parquet(path, encoder_name, emb_col):
    cols = available_columns(path)

    candidates = [
        "cell_id",
        "fine_label",
        "label",
        "target_label",
        "coarse_label",
        "coarse_label_str",
        "tissue",
        emb_col,
    ]
    use_cols = [c for c in candidates if c in cols]

    if emb_col not in use_cols:
        raise RuntimeError(f"{encoder_name}: missing {emb_col}. Available columns: {cols}")

    df = pd.read_parquet(path, columns=use_cols)

    if "cell_id" not in df.columns:
        df["cell_id"] = np.arange(len(df)).astype(str)
        warnings.warn(f"{encoder_name}: missing cell_id. Using row index as cell_id.")
    else:
        df["cell_id"] = df["cell_id"].astype(str)

    if "fine_label" in df.columns:
        df["fine_label_plot"] = df["fine_label"].astype(str)
    elif "label" in df.columns:
        df["fine_label_plot"] = df["label"].astype(str)
    elif "target_label" in df.columns:
        df["fine_label_plot"] = df["target_label"].astype(str)
    else:
        raise RuntimeError(f"{encoder_name}: no fine label column found. Columns: {list(df.columns)}")

    if "coarse_label_str" in df.columns:
        df["coarse_label_plot"] = df["coarse_label_str"].astype(str)
    elif "coarse_label" in df.columns:
        df["coarse_label_plot"] = df["coarse_label"].astype(str)
    else:
        df["coarse_label_plot"] = "unknown"
        warnings.warn(f"{encoder_name}: no coarse label column found.")

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
            if agreement < 0.95:
                warnings.warn(
                    f"{name}: low fine-label row agreement with {ref_name}. "
                    "Check whether row-order alignment is valid."
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
    cache = out_dir / (
        f"{safe_name(name)}_{args.emb_col}_umap"
        f"_fit{args.fit_per_class}"
        f"_pca{args.pca_pre_components}"
        f"_nn{args.umap_neighbors}"
        f"_md{args.umap_min_dist:g}"
        f"_seed{args.seed}.npz"
    )
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


def make_label_encoding(labels):
    labels = sorted(pd.unique(pd.Series(labels).astype(str)))
    mapping = {lab: i for i, lab in enumerate(labels)}
    return labels, mapping


def label_ids(labels, mapping):
    return np.asarray([mapping[str(x)] for x in labels], dtype=np.int32)


def scatter_panel(ax, coords, labels, mapping, title, args):
    ids = label_ids(labels, mapping)
    norm = Normalize(vmin=0, vmax=max(len(mapping) - 1, 1))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=ids,
        norm=norm,
        s=args.point_size,
        alpha=args.point_alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def save_legend(labels, mapping, out_path, args):
    norm = Normalize(vmin=0, vmax=max(len(mapping) - 1, 1))
    cmap = plt.get_cmap()

    handles = []
    for lab in labels:
        color = cmap(norm(mapping[lab]))
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                label=lab,
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
            )
        )

    height = max(2.5, 0.22 * len(handles))
    fig = plt.figure(figsize=(4.8, height))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", frameon=False)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_row_plot(results, coord_key, label_key, labels_sorted, mapping, title, out_stem, args, out_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    axes = axes[0]

    for ax, (name, r) in zip(axes, results.items()):
        scatter_panel(ax, r[coord_key], r[label_key], mapping, name, args)

    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()

    out = out_dir / out_stem
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out.with_suffix('.png')}")


def save_combined_umap(results, fine_labels_sorted, fine_mapping, coarse_labels_sorted, coarse_mapping, args, out_dir):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.2), squeeze=False)

    for j, (name, r) in enumerate(results.items()):
        scatter_panel(axes[0, j], r["umap"], r["fine"], fine_mapping, name, args)
        scatter_panel(axes[1, j], r["umap"], r["coarse"], coarse_mapping, name, args)

    axes[0, 0].set_ylabel("Fine cell type", fontsize=12)
    axes[1, 0].set_ylabel("Coarse group", fontsize=12)

    fig.suptitle("Feature-space visualization on held-out DAPI core test set", y=1.01, fontsize=14)
    fig.tight_layout()

    out = out_dir / "umap_combined_fine_and_coarse"
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out.with_suffix('.png')}")


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "results/figures/feature_space_core500_py"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", out_dir)
    print("fit_per_class:", args.fit_per_class)
    print("plot_per_class:", args.plot_per_class)
    print("embedding column:", args.emb_col)
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

    for name, df in dfs.items():
        print(f"Processing {name}")

        X_fit = stack_embedding_column(df.iloc[fit_idx][args.emb_col])
        print(f"  X_fit shape: {X_fit.shape}")

        umap_fit = pca_then_umap(X_fit, name, args, out_dir)
        pca_fit = pca_2d(X_fit, name, args, out_dir)

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

    all_fine = np.concatenate([r["fine"] for r in results.values()])
    fine_labels_sorted, fine_mapping = make_label_encoding(all_fine)

    all_coarse = np.concatenate([r["coarse"] for r in results.values()])
    coarse_labels_sorted, coarse_mapping = make_label_encoding(all_coarse)

    save_row_plot(
        results,
        coord_key="umap",
        label_key="fine",
        labels_sorted=fine_labels_sorted,
        mapping=fine_mapping,
        title="UMAP of feature embeddings by fine cell type",
        out_stem="umap_by_fine_cell_type",
        args=args,
        out_dir=out_dir,
    )

    save_row_plot(
        results,
        coord_key="umap",
        label_key="coarse",
        labels_sorted=coarse_labels_sorted,
        mapping=coarse_mapping,
        title="UMAP of feature embeddings by coarse biological group",
        out_stem="umap_by_coarse_group",
        args=args,
        out_dir=out_dir,
    )

    save_combined_umap(
        results,
        fine_labels_sorted,
        fine_mapping,
        coarse_labels_sorted,
        coarse_mapping,
        args,
        out_dir,
    )

    save_row_plot(
        results,
        coord_key="pca",
        label_key="coarse",
        labels_sorted=coarse_labels_sorted,
        mapping=coarse_mapping,
        title="PCA of feature embeddings by coarse biological group",
        out_stem="pca_by_coarse_group",
        args=args,
        out_dir=out_dir,
    )

    save_legend(fine_labels_sorted, fine_mapping, out_dir / "legend_fine_cell_type.png", args)
    save_legend(coarse_labels_sorted, coarse_mapping, out_dir / "legend_coarse_group.png", args)

    coord_rows = []
    for name, r in results.items():
        tmp = pd.DataFrame({
            "encoder": name,
            "cell_id": r["cell_id"],
            "fine_label": r["fine"],
            "coarse_label": r["coarse"],
            "umap_1": r["umap"][:, 0],
            "umap_2": r["umap"][:, 1],
            "pca_1": r["pca"][:, 0],
            "pca_2": r["pca"][:, 1],
        })
        coord_rows.append(tmp)

    coords_df = pd.concat(coord_rows, ignore_index=True)
    coords_path = out_dir / "feature_space_coords_plot_subset.parquet"
    coords_df.to_parquet(coords_path, index=False)
    print(f"Saved: {coords_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
