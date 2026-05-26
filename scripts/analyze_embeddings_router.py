#!/usr/bin/env python3
"""
Analyze embeddings: UMAP, cosine similarity heatmap, dendrogram, clustermap.

Supports both coarse (7-class) and fine (23-class) label analysis.
When fine_label column exists in the parquet, produces a 23×23 cosine
similarity heatmap with hierarchical clustering — use this to inform
coarse class grouping decisions.

Usage:
    python scripts/analyze_embeddings.py \
        --embeddings_parquet prepared/embeddings_gate_all_even_r16.parquet \
        --class_map prepared/splits_v3_seed1337/coarse_to_id.json \
        --emb_col emb_fused \
        --out_dir analysis/gate_all_even_r16
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_selection import f_classif

try:
    import umap
except ImportError:
    umap = None


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_embeddings(parquet_path: str, emb_col: str):
    df = pd.read_parquet(parquet_path)

    # Only emb_col and label are strictly required
    if emb_col not in df.columns:
        raise RuntimeError(f"Missing embedding column '{emb_col}'. Got: {list(df.columns)}")

    feats = np.stack(df[emb_col].to_numpy()).astype(np.float32)

    # Label: prefer fine_label if available
    if "fine_label" in df.columns:
        labels = df["fine_label"].astype(str).to_numpy()
        label_type = "fine"
    elif "label" in df.columns:
        labels = df["label"].astype(str).to_numpy()
        label_type = "coarse"
    elif "label_id" in df.columns:
        labels = df["label_id"].astype(str).to_numpy()
        label_type = "id"
    else:
        raise RuntimeError("No label column found")

    tissues = df["tissue"].astype(str).to_numpy() if "tissue" in df.columns else None

    print(f"Loaded {len(df):,} cells, {feats.shape[1]}d, "
          f"{len(np.unique(labels))} {label_type} classes")

    return df, feats, labels, tissues, label_type


# ═══════════════════════════════════════════════════════════════════
# Centroids
# ═══════════════════════════════════════════════════════════════════

def compute_centroids(features, labels):
    """Compute mean embedding per class, L2-normalized."""
    uniq = sorted(np.unique(labels))
    cents, counts = [], []
    for c in uniq:
        m = labels == c
        cent = features[m].mean(axis=0)
        cent = cent / (np.linalg.norm(cent) + 1e-12)  # normalize
        cents.append(cent)
        counts.append(int(m.sum()))
    return np.array(uniq), np.stack(cents).astype(np.float32), np.array(counts)


# ═══════════════════════════════════════════════════════════════════
# ★ Cosine similarity heatmap (the key plot for grouping decisions)
# ═══════════════════════════════════════════════════════════════════

def plot_cosine_similarity_heatmap(centroids, class_names, counts, out_png,
                                   fine_to_coarse=None, title=""):
    """
    23×23 (or N×N) cosine similarity matrix with hierarchical clustering.
    Annotates with coarse group colors if fine_to_coarse is provided.
    """
    sim = centroids @ centroids.T  # already L2-normalized
    np.fill_diagonal(sim, 1.0)

    # Labels with counts
    tick_labels = [f"{n} (n={c:,})" for n, c in zip(class_names, counts)]

    # Hierarchical clustering for row/col ordering
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    # Coarse group color sidebar
    row_colors = None
    if fine_to_coarse:
        coarse_palette = {}
        coarse_groups = sorted(set(fine_to_coarse.values()))
        colors = sns.color_palette("husl", len(coarse_groups))
        for g, c in zip(coarse_groups, colors):
            coarse_palette[g] = c
        row_colors = [coarse_palette.get(fine_to_coarse.get(n, "?"), (0.8, 0.8, 0.8))
                      for n in class_names]

    g = sns.clustermap(
        pd.DataFrame(sim, index=tick_labels, columns=tick_labels),
        row_linkage=Z, col_linkage=Z,
        cmap="RdBu_r", center=0, vmin=-0.2, vmax=1.0,
        figsize=(16, 14),
        row_colors=row_colors,
        col_colors=row_colors,
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(0.02, 0.80, 0.03, 0.15),
        linewidths=0.5,
    )

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                  rotation=45, ha="right", fontsize=8)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)

    if title:
        g.fig.suptitle(title, fontsize=14, y=1.02)

    # Add coarse group legend
    if fine_to_coarse and coarse_palette:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=coarse_palette[g], label=g)
                           for g in coarse_groups]
        g.ax_heatmap.legend(handles=legend_elements, title="Current Coarse",
                            bbox_to_anchor=(1.3, 1), loc="upper left", fontsize=7)

    g.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")


# ═══════════════════════════════════════════════════════════════════
# Dendrogram with proposed grouping overlay
# ═══════════════════════════════════════════════════════════════════

def plot_dendrogram(centroids, class_names, counts, out_png,
                    fine_to_coarse=None, title=""):
    sim = centroids @ centroids.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    # labels = [f"{n} (n={c:,})" for n, c in zip(class_names, counts)]
    labels = class_names.tolist()   

    fig, ax = plt.subplots(figsize=(10, 8))
    dend = dendrogram(Z, labels=labels, leaf_rotation=30, leaf_font_size=10, ax=ax)

    if fine_to_coarse:
        coarse_groups = sorted(set(fine_to_coarse.values()))
        colors = sns.color_palette("husl", len(coarse_groups))
        palette = {g: c for g, c in zip(coarse_groups, colors)}

        for lbl in ax.get_xticklabels():
            #name = lbl.get_text().split(" (n=")[0]
            name = lbl.get_text()
            coarse = fine_to_coarse.get(name, "?")
            lbl.set_color(palette.get(coarse, "black"))
            lbl.set_fontweight("bold")

    if title:
        ax.set_title(title, fontsize=14)
    ax.set_ylabel("Cosine distance")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = Path(str(out_png).replace(".png", f".{ext}"))
        plt.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close()
    print(f"  Saved: {out_png} and pdf")

# ═══════════════════════════════════════════════════════════════════
# UMAP
# ═══════════════════════════════════════════════════════════════════

def plot_umap(feats, labels, out_dir, emb_col, n_neighbors=30, min_dist=0.1,
              fine_to_coarse=None, max_points=50000, seed=42):
    if umap is None:
        print("  Skipping UMAP (umap-learn not installed)")
        return

    # Subsample for speed
    if len(feats) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(feats), max_points, replace=False)
        feats_s, labels_s = feats[idx], labels[idx]
    else:
        feats_s, labels_s = feats, labels

    print(f"  Running UMAP on {len(feats_s):,} points...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                         min_dist=min_dist, random_state=seed)
    proj = reducer.fit_transform(feats_s)

    # By fine label
    fig, ax = plt.subplots(figsize=(14, 11))
    uniq = sorted(np.unique(labels_s))
    for c in uniq:
        m = labels_s == c
        ax.scatter(proj[m, 0], proj[m, 1], s=6, alpha=0.6, label=c)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3, fontsize=7)
    ax.set_title(f"UMAP ({emb_col}) — {len(uniq)} classes")
    plt.tight_layout()
    plt.savefig(out_dir / f"umap_{emb_col}_by_label.png", dpi=200)
    plt.close()
    print(f"  Saved: umap_{emb_col}_by_label.png")

    # By coarse label (if mapping available)
    if fine_to_coarse:
        coarse_labels = np.array([fine_to_coarse.get(l, "?") for l in labels_s])
        fig, ax = plt.subplots(figsize=(14, 11))
        for c in sorted(np.unique(coarse_labels)):
            m = coarse_labels == c
            ax.scatter(proj[m, 0], proj[m, 1], s=6, alpha=0.6, label=c)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)
        ax.set_title(f"UMAP ({emb_col}) — coarse groups")
        plt.tight_layout()
        plt.savefig(out_dir / f"umap_{emb_col}_by_coarse.png", dpi=200)
        plt.close()
        print(f"  Saved: umap_{emb_col}_by_coarse.png")

    # Grid: one-vs-all
    n_classes = len(uniq)
    grid_cols = 5
    grid_rows = (n_classes + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols,
                              figsize=(4 * grid_cols, 3 * grid_rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= n_classes:
            ax.axis("off")
            continue
        c = uniq[i]
        ax.scatter(proj[:, 0], proj[:, 1], s=2, alpha=0.15, c="lightgray")
        m = labels_s == c
        ax.scatter(proj[m, 0], proj[m, 1], s=4, alpha=0.8)
        ax.set_title(c, fontsize=8)
        ax.axis("off")
    plt.suptitle(f"UMAP grid ({emb_col})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"umap_{emb_col}_grid.png", dpi=200)
    plt.close()
    print(f"  Saved: umap_{emb_col}_grid.png")


# ═══════════════════════════════════════════════════════════════════
# Embedding clustermap (feature dimensions)
# ═══════════════════════════════════════════════════════════════════

def plot_feature_clustermap(centroids, class_names, out_png, top_n=256):
    """Clustermap of top-variance embedding dimensions across classes."""
    var = np.var(centroids, axis=0)
    D = centroids.shape[1]
    if D > top_n:
        idx = np.argsort(var)[::-1][:top_n]
        centroids = centroids[:, idx]

    df = pd.DataFrame(centroids, index=class_names)
    g = sns.clustermap(
        df, row_cluster=True, col_cluster=True,
        z_score=1, center=0, cmap="vlag",
        figsize=(20, 10),
        dendrogram_ratio=(0.1, 0.2),
    )
    g.savefig(out_png, dpi=200)
    plt.close()
    print(f"  Saved: {out_png}")


# ═══════════════════════════════════════════════════════════════════
# Print similarity table (for quick terminal inspection)
# ═══════════════════════════════════════════════════════════════════

def print_top_similarities(centroids, class_names, top_k=10):
    """Print the most similar class pairs by cosine similarity."""
    sim = centroids @ centroids.T
    n = len(class_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)

    print(f"\n  Top {top_k} most similar pairs (cosine):")
    print(f"  {'Class A':<30} {'Class B':<30} {'Sim':>6}")
    print(f"  {'─' * 68}")
    for s, a, b in pairs[:top_k]:
        print(f"  {a:<30} {b:<30} {s:>6.3f}")

    print(f"\n  Bottom {top_k} least similar pairs:")
    print(f"  {'Class A':<30} {'Class B':<30} {'Sim':>6}")
    print(f"  {'─' * 68}")
    for s, a, b in pairs[-top_k:]:
        print(f"  {a:<30} {b:<30} {s:>6.3f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_parquet", required=True)
    ap.add_argument("--class_map", default=None, help="coarse_to_id.json (optional, for coloring)")
    ap.add_argument("--fine_to_coarse", default=None, help="fine_to_coarse.json")
    ap.add_argument("--emb_col", default="emb_fused",
                    choices=["emb_fused", "emb_2p5x", "emb_10x"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--umap_max_points", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_cancer", action="store_true", help="Drop cancer classes")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df, feats, labels, tissues, label_type = load_embeddings(
        args.embeddings_parquet, args.emb_col
    )

    # Fine-to-coarse mapping
    fine_to_coarse = None
    if args.fine_to_coarse and Path(args.fine_to_coarse).exists():
        with open(args.fine_to_coarse) as f:
            fine_to_coarse = json.load(f)

    if args.exclude_cancer:
        if fine_to_coarse:
            cancer_labels = {k for k, v in fine_to_coarse.items() if v == "Cancer"}
            mask = ~np.isin(labels, list(cancer_labels))
        else:
            cancer_labels = ["Colon cancer cells", "Liver cancer cells", "Lung cancer cells", "Ovary cancer cells", "Pancreas cancer cells", "Skin cancer cells"]
            mask = ~np.isin(labels, cancer_labels)
        
        df = df[mask].copy()
        feats = feats[mask]
        labels = labels[mask]
        if tissues is not None:
            tissues = tissues[mask]
        print(f"Excluded Cancer cells. Remaining cells: {len(df):,}")

    # ── Centroids ──
    print("\nComputing class centroids...")
    class_names, centroids, counts = compute_centroids(feats, labels)

    # ── ★ Cosine similarity heatmap ──
    print("\nCosine similarity heatmap:")
    plot_cosine_similarity_heatmap(
        centroids, class_names, counts,
        out_dir / f"cosine_sim_{args.emb_col}.png",
        fine_to_coarse=fine_to_coarse,
        title=f"Cosine Similarity Between Fine Class Centroids ({args.emb_col})",
    )

    # ── Dendrogram ──
    print("\nDendrogram:")
    plot_dendrogram(
        centroids, class_names, counts,
        out_dir / f"dendrogram_{args.emb_col}.png",
        fine_to_coarse=fine_to_coarse,
        title=f"Hierarchical Clustering of Fine Classes ({args.emb_col})",
    )

    # ── Top/bottom similarity pairs ──
    print_top_similarities(centroids, class_names)

    # ── Feature clustermap ──
    print("\nFeature clustermap:")
    plot_feature_clustermap(
        centroids, class_names,
        out_dir / f"feature_clustermap_{args.emb_col}.png",
    )

    # ── UMAP ──
    print("\nUMAP:")
    plot_umap(
        feats, labels, out_dir, args.emb_col,
        n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
        fine_to_coarse=fine_to_coarse,
        max_points=args.umap_max_points, seed=args.seed,
    )

    # ── Save support table ──
    support = pd.DataFrame({"class": class_names, "count": counts})
    if fine_to_coarse:
        support["coarse"] = support["class"].map(fine_to_coarse).fillna("?")
    support.to_csv(out_dir / "class_support.csv", index=False)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()