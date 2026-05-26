#!/usr/bin/env python3
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_selection import f_classif

try:
    import umap
except ImportError:
    raise RuntimeError("umap-learn not installed. Install with: pip install umap-learn")


def load_embeddings(parquet_path: str, emb_col: str):
    df = pd.read_parquet(parquet_path)

    required = {"label_id", "tissue", "cell_id", "x_centroid", "y_centroid", emb_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in embeddings parquet: {missing}. Got columns={list(df.columns)}")

    labels = df["label_id"].to_numpy().astype(int)
    tissues = df["tissue"].astype(str).to_numpy()
    cell_ids = df["cell_id"].astype(str).to_numpy()
    xs = df["x_centroid"].to_numpy().astype(np.float32)
    ys = df["y_centroid"].to_numpy().astype(np.float32)

    feats = np.stack(df[emb_col].to_numpy()).astype(np.float32)

    return df, feats, labels, tissues, cell_ids, xs, ys


def load_class_names(class_map_path: str | None, labels: np.ndarray):
    if class_map_path is None:
        # Fallback: label indices as strings
        uniq = np.unique(labels)
        return {int(i): str(int(i)) for i in uniq}

    with open(class_map_path, "r") as f:
        class_to_idx = json.load(f)

    # invert map
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    # guard: ensure all labels exist
    uniq = np.unique(labels)
    for i in uniq:
        if int(i) not in idx_to_class:
            idx_to_class[int(i)] = str(int(i))
    return idx_to_class


def compute_centroids(features: np.ndarray, labels: np.ndarray):
    uniq = np.unique(labels)
    cents = []
    counts = []
    for c in uniq:
        m = labels == c
        cents.append(features[m].mean(axis=0))
        counts.append(int(m.sum()))
    return uniq, np.stack(cents).astype(np.float32), np.array(counts, dtype=np.int64)


def run_umap(features: np.ndarray, n_neighbors=30, min_dist=0.1, seed=42):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(features)


def plot_umap_by_label(proj, labels, idx_to_class, out_png: Path, title: str):
    plt.figure(figsize=(12, 10))
    uniq = np.unique(labels)
    for c in uniq:
        m = labels == c
        name = idx_to_class[int(c)]
        pts = proj[m]
        plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7, label=name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_umap_by_tissue(proj, tissues, out_png: Path, title: str, max_legend=40):
    plt.figure(figsize=(12, 10))
    uniq = np.unique(tissues)
    for t in uniq:
        m = tissues == t
        pts = proj[m]
        plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7, label=str(t))
    if len(uniq) <= max_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_umap_grid_one_vs_all(proj, labels, idx_to_class, out_png: Path, grid_cols=4, title: str = ""):
    uniq = np.unique(labels)
    uniq = sorted([int(x) for x in uniq], key=lambda i: idx_to_class[i])

    n_classes = len(uniq)
    grid_rows = (n_classes + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4 * grid_rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= n_classes:
            ax.axis("off")
            continue
        c = uniq[i]
        name = idx_to_class[c]
        ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.25)
        m = labels == c
        ax.scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.9)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def clustermap_centroids(centroids: np.ndarray, class_ids: np.ndarray, idx_to_class: dict, out_png: Path,
                         top_n: int = 512, mode: str = "variance", features: np.ndarray | None = None,
                         labels: np.ndarray | None = None):
    if mode == "variance":
        score = np.var(centroids, axis=0)
    elif mode == "anova":
        if features is None or labels is None:
            raise ValueError("ANOVA mode requires features and labels")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Fv, _ = f_classif(features, labels)
        score = np.nan_to_num(Fv, nan=0.0)
    else:
        raise ValueError("mode must be 'variance' or 'anova'")

    D = centroids.shape[1]
    if D > top_n:
        idx = np.argsort(score)[::-1][:top_n]
        centroids_top = centroids[:, idx]
    else:
        centroids_top = centroids

    names = [idx_to_class[int(i)] for i in class_ids]
    df_top = pd.DataFrame(centroids_top, index=names)

    g = sns.clustermap(
        df_top,
        row_cluster=True,
        col_cluster=True,
        z_score=1,
        center=0,
        cmap="vlag",
        figsize=(20, 10),
        dendrogram_ratio=(0.1, 0.2),
        cbar_pos=(0.02, 0.32, 0.03, 0.2),
    )
    g.savefig(out_png, dpi=200)
    plt.close()


def plot_dendrogram_from_prototypes(centroids: np.ndarray, class_ids: np.ndarray, counts: np.ndarray, idx_to_class: dict, out_png: Path):
    # Use cosine distance between prototypes
    # Normalize then compute 1 - cosine similarity
    Cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sim = Cn @ Cn.T
    dist = 1.0 - sim
    # Convert to condensed form
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    Z = linkage(tri, method="average")

    labels = [f"{idx_to_class[int(i)]} (n={c})" for i, c in zip(class_ids, counts)]
    plt.figure(figsize=(18, 8))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def compute_medoids(features: np.ndarray, labels: np.ndarray, class_ids: np.ndarray, idx_to_class: dict,
                    df: pd.DataFrame, out_csv: Path):
    rows = []
    for c in class_ids:
        m = labels == c
        if not np.any(m):
            continue
        feats_c = features[m]
        centroid = feats_c.mean(axis=0, keepdims=True)
        dists = cdist(feats_c, centroid, metric="euclidean").reshape(-1)
        j = int(np.argmin(dists))

        # map back to original df row
        idxs = np.where(m)[0]
        orig_idx = int(idxs[j])
        r = df.iloc[orig_idx]

        rows.append({
            "label_id": int(c),
            "label": idx_to_class[int(c)],
            "cell_id": str(r["cell_id"]),
            "tissue": str(r["tissue"]),
            "x_centroid": float(r["x_centroid"]),
            "y_centroid": float(r["y_centroid"]),
            "distance_to_centroid": float(dists[j]),
            "row_index": orig_idx,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_parquet", required=True)
    ap.add_argument("--class_map", default=None, help="prepared/class_to_idx.json (optional)")
    ap.add_argument("--emb_col", default="emb_fused", choices=["emb_fused", "emb_2p5x", "emb_10x"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_n", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feats, labels, tissues, cell_ids, xs, ys = load_embeddings(args.embeddings_parquet, args.emb_col)
    idx_to_class = load_class_names(args.class_map, labels)

    print("Loaded:", args.embeddings_parquet)
    print("N:", len(df), "D:", feats.shape[1], "classes:", len(np.unique(labels)), "tissues:", len(np.unique(tissues)))

    # UMAP
    proj = run_umap(feats, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist, seed=args.seed)

    plot_umap_by_label(
        proj, labels, idx_to_class,
        out_dir / f"umap_{args.emb_col}_by_label.png",
        title=f"UMAP ({args.emb_col}) by label"
    )

    plot_umap_by_tissue(
        proj, tissues,
        out_dir / f"umap_{args.emb_col}_by_tissue.png",
        title=f"UMAP ({args.emb_col}) by tissue"
    )

    plot_umap_grid_one_vs_all(
        proj, labels, idx_to_class,
        out_dir / f"umap_{args.emb_col}_grid.png",
        title=f"UMAP grid ({args.emb_col})"
    )

    # Centroids
    class_ids, centroids, counts = compute_centroids(feats, labels)

    # Heatmaps
    clustermap_centroids(
        centroids, class_ids, idx_to_class,
        out_dir / f"heatmap_{args.emb_col}_variance_top{args.top_n}.png",
        top_n=args.top_n, mode="variance"
    )

    clustermap_centroids(
        centroids, class_ids, idx_to_class,
        out_dir / f"heatmap_{args.emb_col}_anova_top{args.top_n}.png",
        top_n=args.top_n, mode="anova",
        features=feats, labels=labels
    )

    # Class Support
    support_df = pd.DataFrame({
        "label_id": class_ids,
        "class_name": [idx_to_class[int(i)] for i in class_ids],
        "count": counts
    })
    support_df.to_csv(out_dir / "class_support.csv", index=False)

    # Dendrogram
    plot_dendrogram_from_prototypes(
        centroids, class_ids, counts, idx_to_class,
        out_dir / f"dendrogram_{args.emb_col}.png"
    )

    # Medoids
    compute_medoids(
        feats, labels, class_ids, idx_to_class,
        df, out_dir / f"medoids_{args.emb_col}.csv"
    )

    print("Done. Outputs in:", out_dir)


if __name__ == "__main__":
    main()


"""
python scripts/analyze_embeddings.py \
  --embeddings_parquet prepared/embeddings_small_balanced_frozen.parquet \
  --class_map prepared/class_to_idx.json \
  --emb_col emb_fused \     #emb_2p5x , emb_10x
  --out_dir analysis/frozen_fused

"""
