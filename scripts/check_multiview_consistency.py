#!/usr/bin/env python3
"""
Multi-view consistency check.
验证同一 cell 的 2.5x 和 10x embedding 是否比同类不同 cell 更近。

Usage:
    python scripts/check_multiview_consistency.py \
        --embeddings_parquet prepared/embeddings_small_balanced_frozen.parquet \
        --class_map prepared/class_to_idx.json \
        --out_dir analysis/multiview_check
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_embeddings(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    emb_2p5x = np.stack(df["emb_2p5x"].values).astype(np.float32)
    emb_10x = np.stack(df["emb_10x"].values).astype(np.float32)
    labels = df["label_id"].values.astype(int)
    return emb_2p5x, emb_10x, labels


def cosine_sim(a, b):
    """Row-wise cosine similarity."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_n * b_n).sum(axis=1)


def sample_pairs(labels, n_pairs=50000, seed=42):
    """Sample same-class and diff-class pairs."""
    rng = np.random.RandomState(seed)
    N = len(labels)

    same_class_pairs = []
    diff_class_pairs = []

    # group indices by label
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(int(l), []).append(i)

    all_labels = list(label_to_idx.keys())

    # same-class pairs
    for _ in range(n_pairs):
        l = rng.choice(all_labels)
        idxs = label_to_idx[l]
        if len(idxs) < 2:
            continue
        i, j = rng.choice(len(idxs), 2, replace=False)
        same_class_pairs.append((idxs[i], idxs[j]))

    # diff-class pairs
    for _ in range(n_pairs):
        l1, l2 = rng.choice(all_labels, 2, replace=False)
        i = rng.choice(label_to_idx[l1])
        j = rng.choice(label_to_idx[l2])
        diff_class_pairs.append((i, j))

    return same_class_pairs, diff_class_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_parquet", required=True)
    ap.add_argument("--class_map", default=None)
    ap.add_argument("--out_dir", default="analysis/multiview_check")
    ap.add_argument("--n_pairs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_to_class = None
    if args.class_map:
        with open(args.class_map) as f:
            c2i = json.load(f)
            idx_to_class = {v: k for k, v in c2i.items()}

    emb_2p5x, emb_10x, labels = load_embeddings(args.embeddings_parquet)
    N, D = emb_2p5x.shape
    print(f"Loaded {N} cells, dim={D}")

    # ── 1. Same cell, cross-view similarity ──
    same_cell_sim = cosine_sim(emb_2p5x, emb_10x)

    # ── 2. Same class, different cell, same view (2.5x) ──
    same_class_pairs, diff_class_pairs = sample_pairs(labels, n_pairs=args.n_pairs, seed=args.seed)

    sc_idx = np.array(same_class_pairs)
    dc_idx = np.array(diff_class_pairs)

    same_class_sim = cosine_sim(emb_2p5x[sc_idx[:, 0]], emb_2p5x[sc_idx[:, 1]])
    diff_class_sim = cosine_sim(emb_2p5x[dc_idx[:, 0]], emb_2p5x[dc_idx[:, 1]])

    # ── 3. Print summary ──
    print(f"\n{'='*60}")
    print(f"Multi-view Consistency Check")
    print(f"{'='*60}")
    print(f"Same cell cross-view (2.5x vs 10x):  {same_cell_sim.mean():.4f} ± {same_cell_sim.std():.4f}")
    print(f"Same class diff-cell (2.5x vs 2.5x): {same_class_sim.mean():.4f} ± {same_class_sim.std():.4f}")
    print(f"Diff class diff-cell (2.5x vs 2.5x): {diff_class_sim.mean():.4f} ± {diff_class_sim.std():.4f}")
    print()

    if same_cell_sim.mean() > same_class_sim.mean():
        print("✓ PASS: Same-cell cross-view > same-class diff-cell")
        print(f"  Gap: {same_cell_sim.mean() - same_class_sim.mean():.4f}")
    else:
        print("✗ FAIL: Same-cell cross-view ≤ same-class diff-cell")
        print("  Multi-view fusion may not be adding value!")

    if same_class_sim.mean() > diff_class_sim.mean():
        print("✓ PASS: Same-class > diff-class")
        print(f"  Gap: {same_class_sim.mean() - diff_class_sim.mean():.4f}")
    else:
        print("✗ FAIL: Same-class ≤ diff-class")
        print("  Embeddings are not class-discriminative!")

    # ── 4. Per-class cross-view similarity ──
    print(f"\nPer-class cross-view similarity (2.5x vs 10x):")
    unique_labels = np.unique(labels)
    per_class_stats = []
    for l in sorted(unique_labels):
        mask = labels == l
        sim = same_cell_sim[mask]
        name = idx_to_class[int(l)] if idx_to_class else str(l)
        print(f"  [{l:>2}] {name:<30s}  mean={sim.mean():.4f}  std={sim.std():.4f}  n={mask.sum()}")
        per_class_stats.append({"label_id": int(l), "class": name, "mean_sim": float(sim.mean()),
                                "std_sim": float(sim.std()), "n": int(mask.sum())})

    pd.DataFrame(per_class_stats).to_csv(out_dir / "per_class_crossview_sim.csv", index=False)

    # ── 5. Plot distributions ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: three distributions
    ax = axes[0]
    ax.hist(same_cell_sim, bins=100, alpha=0.7, density=True, label=f"Same cell cross-view\n(μ={same_cell_sim.mean():.3f})")
    ax.hist(same_class_sim, bins=100, alpha=0.7, density=True, label=f"Same class diff-cell\n(μ={same_class_sim.mean():.3f})")
    ax.hist(diff_class_sim, bins=100, alpha=0.7, density=True, label=f"Diff class\n(μ={diff_class_sim.mean():.3f})")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Multi-view Consistency")
    ax.legend(fontsize=8)
    ax.axvline(same_cell_sim.mean(), color="C0", ls="--", alpha=0.5)
    ax.axvline(same_class_sim.mean(), color="C1", ls="--", alpha=0.5)
    ax.axvline(diff_class_sim.mean(), color="C2", ls="--", alpha=0.5)

    # Panel B: per-class cross-view similarity
    ax = axes[1]
    stats_df = pd.DataFrame(per_class_stats).sort_values("mean_sim")
    ax.barh(range(len(stats_df)), stats_df["mean_sim"], xerr=stats_df["std_sim"],
            color="steelblue", alpha=0.8, capsize=2)
    ax.set_yticks(range(len(stats_df)))
    ax.set_yticklabels(stats_df["class"], fontsize=7)
    ax.set_xlabel("Cosine Similarity (2.5x vs 10x)")
    ax.set_title("Per-class Cross-view Agreement")
    ax.axvline(same_cell_sim.mean(), color="red", ls="--", alpha=0.5, label="Overall mean")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "multiview_consistency.png", dpi=200)
    plt.close()
    print(f"\nSaved: {out_dir / 'multiview_consistency.png'}")
    print(f"Saved: {out_dir / 'per_class_crossview_sim.csv'}")


if __name__ == "__main__":
    main()