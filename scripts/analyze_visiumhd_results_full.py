"""
analyze_results.py
──────────────────
Post-training analysis for all Visium HD gene expression prediction jobs.

Produces:
  1. comparison_table.csv       — all jobs, all metrics
  2. coverage_stratified.pdf    — heatmap + fixed-pair scatter
  3. delta_pearson.pdf          — per-gene delta between key model pairs
  4. learning_curves.pdf        — val Pearson over epochs
  5. pearson_distribution.pdf   — violin + stacked bar of per-gene r
  6. tile_performance.pdf       — per-tile mean Pearson on test tiles

Usage:
  python analyze_results.py
  python analyze_results.py --jobs 1,3,10
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
from pathlib import Path
from scipy.stats import pearsonr as scipy_pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT   = Path("/hpc/group/jilab/tc459/MorphPT")
CACHE     = PROJECT / "cache_visium"
EXPR_ROOT = Path("/hpc/group/jilab/boxuan/visiumHD/human_crc")
OUT_DIR   = PROJECT / "analysis/results_full"

# ── Job registry ───────────────────────────────────────────────────────────
JOBS = {
    1:  {"name": "visium_dinov3_25x",            "label": "DINOv3 frozen · Linear · 2.5×"},
    2:  {"name": "visium_morphpt_25x",            "label": "MorphPT frozen · Linear · 2.5×"},
    3:  {"name": "visium_morphpt_lora_25x",       "label": "MorphPT+LoRA · Linear · 2.5×"},
    4:  {"name": "visium_morphpt_10x",            "label": "MorphPT frozen · Linear · 10×"},
    5:  {"name": "visium_dinov3_10x",             "label": "DINOv3 frozen · Linear · 10×"},
    6:  {"name": "visium_morphpt_gate",           "label": "MorphPT frozen · Gate · Linear"},
    7:  {"name": "visium_morphpt_25x_mlp",        "label": "MorphPT frozen · MLP · 2.5×"},
    8:  {"name": "visium_morphpt_10x_mlp",        "label": "MorphPT frozen · MLP · 10×"},
    9:  {"name": "visium_morphpt_gate_mlp",       "label": "MorphPT frozen · Gate · MLP"},
    10: {"name": "visium_morphpt_lora_10x",       "label": "MorphPT+LoRA · Linear · 10×"},
    11: {"name": "visium_morphpt_lora_gate",      "label": "MorphPT+LoRA · Gate · Linear"},
    12: {"name": "visium_morphpt_lora_10x_mlp",   "label": "MorphPT+LoRA · MLP · 10×"},
    13: {"name": "visium_morphpt_lora_gate_mlp",  "label": "MorphPT+LoRA · Gate · MLP"},
    14: {"name": "visium_dinov3_lora_25x",        "label": "DINOv3+LoRA · Linear · 2.5×"},
    15: {"name": "visium_dinov3_lora_10x",        "label": "DINOv3+LoRA · Linear · 10×"},
}

DELTA_PAIRS = [
    (1,  3,  "DINOv3 vs MorphPT+LoRA (2.5×)"),
    (1,  5,  "DINOv3 2.5× vs DINOv3 10×"),
    (3,  10, "LoRA 2.5× vs LoRA 10×"),
    (4,  8,  "Frozen linear 10× vs Frozen MLP 10×"),
    (10, 11, "LoRA 10× vs LoRA gate (linear)"),
    (12, 13, "LoRA MLP 10× vs LoRA gate MLP"),
    (14, 3,  "DINOv3+LoRA vs MorphPT+LoRA (2.5×)"),   # key: init benefit
    (15, 10, "DINOv3+LoRA vs MorphPT+LoRA (10×)"),    # key: init benefit
]

SCATTER_PAIRS = [
    (1,  "DINOv3 frozen 2.5×",  "#4C72B0"),
    (3,  "MorphPT+LoRA 2.5×",   "#C44E52"),
    (5,  "DINOv3 frozen 10×",   "#55A868"),
    (10, "MorphPT+LoRA 10×",    "#8172B2"),
]

DIST_JOBS = [1, 3, 5, 8, 10, 12, 13, 14, 15]

STYLE = {
    "figure.dpi": 150,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
}
plt.rcParams.update(STYLE)


# ── Helpers ────────────────────────────────────────────────────────────────
def exp_dir(job_id):
    return PROJECT / "experiments" / JOBS[job_id]["name"]

def load_test_results(job_id):
    path = exp_dir(job_id) / "test_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

def load_train_log(job_id):
    path = exp_dir(job_id) / "train_log.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

def load_per_gene_pearson(job_id):
    path = exp_dir(job_id) / "per_gene_pearson_test.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu").numpy()

def load_gene_coverage():
    """
    Per-gene expression prevalence computed on TEST cells only,
    so coverage aligns exactly with the evaluation target.
    """
    expr     = np.load(str(CACHE / "expr.npy"), mmap_mode="r")
    meta     = pd.read_csv(CACHE / "meta.csv")
    test_idx = meta[meta["split"] == "test"]["mmap_idx"].values
    X_test   = expr[test_idx]
    coverage = (X_test > 0).mean(axis=0)   # (1000,)
    genes    = (EXPR_ROOT / "expr/genes.txt").read_text().splitlines()
    return coverage, genes


# ── 1. Comparison table ────────────────────────────────────────────────────
def make_comparison_table(job_ids):
    print("Building comparison table...")
    rows = []
    for jid in job_ids:
        res = load_test_results(jid)
        r   = load_per_gene_pearson(jid)
        if res is None or r is None:
            print(f"  Job {jid}: missing results — skipping")
            continue
        args = res.get("args", {})
        rows.append({
            "job":           jid,
            "name":          JOBS[jid]["name"],
            "encoder":       "MorphPT+LoRA" if args.get("unfreeze_lora") else
                             ("MorphPT" if args.get("ckpt_path") else "DINOv3"),
            "head":          args.get("head_type", "?"),
            "scales":        args.get("scales", "?"),
            "fuse":          args.get("fuse", "?"),
            "lora_unfrozen": bool(args.get("unfreeze_lora", 0)),
            "val_pearson":   res.get("best_val_pearson", float("nan")),
            "test_mean_r":   float(r.mean()),
            "test_median_r": float(np.median(r)),
            "test_zMSE":     res["test_zMSE"],
            "r>0.1":         int((r > 0.1).sum()),
            "r>0.2":         int((r > 0.2).sum()),
            "r>0.3":         int((r > 0.3).sum()),
        })

    df = pd.DataFrame(rows).sort_values("test_mean_r", ascending=False)
    out = OUT_DIR / "comparison_table.csv"
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    print(df[["job","name","test_mean_r","test_median_r",
              "r>0.1","r>0.2","r>0.3"]].to_string(index=False))
    return df


# ── 2. Coverage-stratified heatmap ────────────────────────────────────────
def plot_coverage_stratified(job_ids):
    print("\nPlotting coverage-stratified Pearson...")
    coverage, genes = load_gene_coverage()

    bins   = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 1.01]
    labels = ["<2%", "2–5%", "5–10%", "10–20%", "20–35%", ">35%"]
    bin_idx    = np.clip(np.digitize(coverage, bins) - 1, 0, len(labels) - 1)
    bin_counts = [int((bin_idx == i).sum()) for i in range(len(labels))]

    # Build heatmap matrix
    heatmap_rows, heatmap_labels = [], []
    for jid in job_ids:
        r = load_per_gene_pearson(jid)
        if r is None:
            continue
        means = [r[bin_idx == b].mean() if (bin_idx == b).sum() > 0 else np.nan
                 for b in range(len(labels))]
        heatmap_rows.append(means)
        heatmap_labels.append(f"J{jid}: {JOBS[jid]['label']}")

    heatmap = np.array(heatmap_rows)   # (n_jobs, n_bins)
    n_jobs  = len(heatmap_rows)

    # Dynamic figure height: 0.55 inches per job, min 4, max 14
    hm_height = max(4.0, min(14.0, n_jobs * 0.55))
    fig = plt.figure(figsize=(16, hm_height))
    gs  = plt.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.35)

    # Left: heatmap — allow negative vmin so weak models show properly
    ax_hm   = fig.add_subplot(gs[0])
    vmin    = min(-0.05, float(np.nanmin(heatmap)))
    vmax    = max(0.35,  float(np.nanmax(heatmap)))
    im      = ax_hm.imshow(heatmap, aspect="auto", cmap="RdYlGn",
                           vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax_hm, label="Mean Pearson r", shrink=0.8)

    ax_hm.set_xticks(range(len(labels)))
    ax_hm.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, bin_counts)],
                          fontsize=8)
    ax_hm.set_yticks(range(n_jobs))
    ax_hm.set_yticklabels(heatmap_labels, fontsize=7)
    ax_hm.set_xlabel("Gene expression prevalence (% test cells expressing)")
    ax_hm.set_title("(a) Coverage-stratified mean Pearson r\n"
                    "(rows=models, columns=coverage bins)", fontsize=9)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=6,
                           color="white" if abs(val) > 0.22 else "black")

    # Right: scatter for fixed model pairs
    ax_sc = fig.add_subplot(gs[1])
    for jid, lbl, col in SCATTER_PAIRS:
        if jid not in job_ids:
            continue
        r = load_per_gene_pearson(jid)
        if r is None:
            continue
        ax_sc.scatter(coverage * 100, r, alpha=0.3, s=6,
                      color=col, label=lbl, rasterized=True)

    ax_sc.axhline(0, color="gray", lw=0.8, ls="--")
    ax_sc.set_xlabel("% test cells expressing gene")
    ax_sc.set_ylabel("Per-gene Pearson r")
    ax_sc.set_title("(b) Coverage vs per-gene Pearson r\n(fixed model pairs)",
                    fontsize=9)
    ax_sc.legend(loc="upper left", fontsize=7, markerscale=2)

    fig.suptitle("Gene expression prediction: coverage-stratified performance",
                 fontweight="bold")
    out = OUT_DIR / "coverage_stratified.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 3. Per-gene delta plots ────────────────────────────────────────────────
def plot_delta_pearson(available_jobs):
    print("\nPlotting per-gene delta Pearson...")
    coverage, _ = load_gene_coverage()

    valid_pairs = [(a, b, label) for a, b, label in DELTA_PAIRS
                   if a in available_jobs and b in available_jobs]
    if not valid_pairs:
        print("  No valid pairs available yet — skipping")
        return

    ncols = 2
    nrows = (len(valid_pairs) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = np.array(axes).ravel()

    for i, (jid_a, jid_b, label) in enumerate(valid_pairs):
        r_a = load_per_gene_pearson(jid_a)
        r_b = load_per_gene_pearson(jid_b)
        if r_a is None or r_b is None:
            continue

        delta = r_b - r_a
        ax    = axes[i]
        sc    = ax.scatter(coverage * 100, delta, c=delta,
                           cmap="RdYlGn", vmin=-0.15, vmax=0.15,
                           alpha=0.5, s=8)
        ax.axhline(0, color="gray", lw=0.8, ls="--")

        from scipy.ndimage import uniform_filter1d
        sort_idx = np.argsort(coverage)
        smooth   = uniform_filter1d(delta[sort_idx], size=50)
        ax.plot(coverage[sort_idx] * 100, smooth, color="black", lw=1.5)

        ax.set_title(f"({chr(97+i)}) {label}\n"
                     f"Δ mean={delta.mean():+.4f}  "
                     f"better: {(delta>0).sum()}/1000", fontsize=8)
        ax.set_xlabel("% test cells expressing gene")
        ax.set_ylabel(f"ΔPearson (J{jid_b} − J{jid_a})")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    for j in range(len(valid_pairs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-gene Pearson improvement between model pairs",
                 fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "delta_pearson.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 4. Learning curves ────────────────────────────────────────────────────
def plot_learning_curves(job_ids):
    print("\nPlotting learning curves...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(job_ids)))

    for jid, col in zip(job_ids, colors):
        log = load_train_log(jid)
        if log is None:
            continue
        epochs   = [r["epoch"]          for r in log]
        val_corr = [r["va_mean_pearson"] for r in log]
        tr_zmse  = [r["tr_zMSE"]        for r in log]
        label    = f"J{jid}: {JOBS[jid]['label']}"
        axes[0].plot(epochs, val_corr, color=col, label=label, lw=1.5)
        axes[1].plot(epochs, tr_zmse,  color=col, label=label, lw=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val mean Pearson r")
    axes[0].set_title("(a) Validation Pearson r")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train zMSE")
    axes[1].set_title("(b) Training zMSE")
    axes[1].legend(fontsize=6, ncol=2)

    fig.suptitle("Learning curves — all jobs", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "learning_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 5. Per-gene Pearson distribution ──────────────────────────────────────
def plot_pearson_distribution(job_ids):
    print("\nPlotting per-gene Pearson distribution...")
    plot_ids = [j for j in DIST_JOBS if j in job_ids] or job_ids

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: violin
    ax    = axes[0]
    data, xlabels = [], []
    for jid in plot_ids:
        r = load_per_gene_pearson(jid)
        if r is None:
            continue
        data.append(r)
        xlabels.append(f"J{jid}\n{JOBS[jid]['label']}")

    if data:
        parts = ax.violinplot(data, positions=range(len(data)),
                              showmedians=True, showextrema=True)
        means = [d.mean() for d in data]
        norm  = plt.Normalize(min(means), max(means))
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(plt.cm.YlOrRd(norm(means[i])))
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(xlabels, fontsize=6)
        ax.axhline(0,   color="gray",    lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0.2, color="#C44E52", lw=0.8, ls=":",  alpha=0.7, label="r=0.2")
        ax.axhline(0.3, color="#8172B2", lw=0.8, ls=":",  alpha=0.7, label="r=0.3")
        ax.set_ylabel("Per-gene Pearson r")
        ax.set_title("(a) Distribution of per-gene Pearson r", fontsize=9)
        ax.legend(fontsize=7)

    # Right: stacked bar
    ax     = axes[1]
    tiers  = [(0.3, 1.0, "r > 0.3",      "#2d6a4f"),
              (0.2, 0.3, "0.2 < r ≤ 0.3", "#52b788"),
              (0.1, 0.2, "0.1 < r ≤ 0.2", "#95d5b2"),
              (0.0, 0.1, "0 < r ≤ 0.1",   "#d8f3dc"),
              (-1,  0.0, "r ≤ 0",          "#e0e0e0")]
    x       = np.arange(len(data))
    bottoms = np.zeros(len(data))
    for lo, hi, label, color in tiers:
        counts = np.array([int(((d > lo) & (d <= hi)).sum()) for d in data])
        ax.bar(x, counts, bottom=bottoms, color=color,
               label=label, width=0.6, edgecolor="white", linewidth=0.5)
        for xi, (cnt, bot) in enumerate(zip(counts, bottoms)):
            if cnt > 20:
                ax.text(xi, bot + cnt / 2, str(cnt),
                        ha="center", va="center", fontsize=6)
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel("Number of genes (out of 1000)")
    ax.set_title("(b) Genes per Pearson r tier", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1050)

    fig.suptitle("Per-gene Pearson r distribution across models", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "pearson_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 6. Tile-level performance ─────────────────────────────────────────────
def plot_tile_performance(job_ids):
    """
    For each test tile, compute mean Pearson r across genes and cells.
    Reveals whether val/test gap is driven by one or two easy tiles.
    Requires per_gene_pearson_test.pt AND spatial metadata.
    """
    print("\nPlotting tile-level performance...")

    meta     = pd.read_csv(CACHE / "meta.csv")
    test_meta = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    test_tiles = sorted(test_meta["tile_id"].unique())
    n_tiles    = len(test_tiles)

    if n_tiles == 0:
        print("  No test tiles found — skipping")
        return

    # Load expression for test cells
    expr     = np.load(str(CACHE / "expr.npy"), mmap_mode="r")
    test_idx = test_meta["mmap_idx"].values
    X_test   = np.array(expr[test_idx], dtype=np.float32)   # (N_test, 1000)

    # Load gene stats for denormalization
    stats     = np.load(str(CACHE / "expr_stats.npz"))
    gene_mean = stats["gene_mean"]
    gene_std  = stats["gene_std"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(job_ids)))

    # Collect per-tile metrics for all jobs
    tile_results = {}
    for jid in job_ids:
        pt_path = exp_dir(jid) / "per_gene_pearson_test.pt"
        pred_path = exp_dir(jid) / "predictions_test.npz"

        # Use cached predictions if available
        if pred_path.exists():
            npz   = np.load(str(pred_path))
            y_pred = npz["y_pred"]
        else:
            print(f"  Job {jid}: no predictions_test.npz — using per_gene_pearson only")
            y_pred = None

        if y_pred is None:
            # Fall back to per-gene Pearson (can't do tile breakdown without predictions)
            tile_results[jid] = None
            continue

        # Denormalize
        y_pred_denorm = y_pred * gene_std + gene_mean
        y_true_denorm = X_test * gene_std + gene_mean   # already aligned to test_idx

        tile_means = []
        for tid in test_tiles:
            mask = test_meta["tile_id"].values == tid
            yt   = y_true_denorm[mask]
            yp   = y_pred_denorm[mask]
            # Mean per-gene Pearson within this tile
            rs   = []
            for g in range(yt.shape[1]):
                if yt[:, g].std() > 1e-6 and yp[:, g].std() > 1e-6:
                    r, _ = scipy_pearsonr(yt[:, g], yp[:, g])
                    rs.append(r)
            tile_means.append(np.mean(rs) if rs else np.nan)
        tile_results[jid] = tile_means

    # Left: tile performance per model (bar chart)
    ax = axes[0]
    x  = np.arange(n_tiles)
    available = [(jid, col) for jid, col in zip(job_ids, colors)
                 if tile_results.get(jid) is not None]
    width = 0.8 / max(len(available), 1)

    for i, (jid, col) in enumerate(available):
        vals   = tile_results[jid]
        offset = (i - len(available) / 2) * width + width / 2
        ax.bar(x + offset, vals, width=width * 0.9,
               color=col, alpha=0.8, label=f"J{jid}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Tile {t}" for t in test_tiles], rotation=30, fontsize=8)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("(a) Per-tile mean Pearson r on test tiles", fontsize=9)
    ax.legend(fontsize=7, ncol=2)

    # Right: tile cell count + x/y position
    ax = axes[1]
    for tid in test_tiles:
        mask   = test_meta["tile_id"].values == tid
        tile_df = test_meta[mask]
        cx     = tile_df["x_centroid"].mean() if "x_centroid" in tile_df else 0
        cy     = tile_df["y_centroid"].mean() if "y_centroid" in tile_df else 0
        n_cells = mask.sum()
        ax.scatter(cx, cy, s=n_cells / 10, alpha=0.6, label=f"T{tid} (n={n_cells})")
        ax.annotate(f"T{tid}", (cx, cy), fontsize=7, ha="center")

    ax.set_xlabel("x centroid")
    ax.set_ylabel("y centroid")
    ax.set_title("(b) Test tile locations\n(bubble size ∝ cell count)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    # If no predictions cached, show a note
    if not any(v is not None for v in tile_results.values()):
        for ax in axes:
            ax.text(0.5, 0.5,
                    "Run spatial_maps.py --save_predictions first\n"
                    "to enable per-tile breakdown",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")

    fig.suptitle("Tile-level performance on test split", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "tile_performance.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, default=None,
                    help="Comma-separated job IDs (default: all with results)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    requested = ([int(j) for j in args.jobs.split(",")]
                 if args.jobs else list(JOBS.keys()))
    available = [j for j in requested
                 if (exp_dir(j) / "test_results.json").exists()]
    print(f"Jobs with results: {available}")
    if not available:
        print("No completed jobs found.")
        return

    make_comparison_table(available)
    plot_coverage_stratified(available)
    plot_delta_pearson(available)
    plot_learning_curves(available)
    plot_pearson_distribution(available)
    plot_tile_performance(available)

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()