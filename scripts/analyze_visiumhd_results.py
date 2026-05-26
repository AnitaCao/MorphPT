"""
analyze_results.py
──────────────────
Paper-focused analysis for Visium HD gene expression prediction.

Selected 8 models that tell the clean story:
  Frozen:  J1 (DINOv3 2.5×), J5 (DINOv3 10×), J8 (MorphPT MLP 10×)
  LoRA:    J3 (MorphPT 2.5×), J10 (MorphPT 10×), J12 (MorphPT MLP 10×)
           J14 (DINOv3 2.5×), J15 (DINOv3 10×)

Produces:
  1. comparison_table.csv        — all 15 jobs
  2. coverage_stratified.pdf     — heatmap (8 models) + scatter (4 fixed pairs)
  3. delta_pearson.pdf           — 4 key model pair comparisons
  4. pearson_distribution.pdf    — violin + stacked bar (8 models)
  5. tile_performance.pdf        — per-tile breakdown (needs predictions_test.npz)
  6. learning_curves.pdf         — all jobs (sanity check, not for paper)

Usage:
  python analyze_results.py
  python analyze_results.py --jobs 1,3,5,8,10,12,13,14,15  # subset
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr as scipy_pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT   = Path("/hpc/group/jilab/tc459/MorphPT")
CACHE     = PROJECT / "cache_visium"
EXPR_ROOT = Path("/hpc/group/jilab/boxuan/visiumHD/human_crc")
OUT_DIR   = PROJECT / "analysis/results_9models"

# ── Full job registry (all 15) ─────────────────────────────────────────────
JOBS = {
    1:  {"name": "visium_dinov3_25x",            "label": "DINOv3 frozen\n2.5×",          "group": "frozen"},
    2:  {"name": "visium_morphpt_25x",            "label": "MorphPT frozen\n2.5×",         "group": "frozen"},
    3:  {"name": "visium_morphpt_lora_25x",       "label": "MorphPT+LoRA\n2.5×",           "group": "lora"},
    4:  {"name": "visium_morphpt_10x",            "label": "MorphPT frozen\n10×",          "group": "frozen"},
    5:  {"name": "visium_dinov3_10x",             "label": "DINOv3 frozen\n10×",           "group": "frozen"},
    6:  {"name": "visium_morphpt_gate",    "label": "MorphPT frozen\nGate linear",  "group": "frozen"},
    7:  {"name": "visium_morphpt_25x_mlp",        "label": "MorphPT frozen\nMLP 2.5×",     "group": "frozen"},
    8:  {"name": "visium_morphpt_10x_mlp",        "label": "MorphPT frozen\nMLP 10×",      "group": "frozen"},
    9:  {"name": "visium_morphpt_gate_mlp",       "label": "MorphPT frozen\nGate MLP",     "group": "frozen"},
    10: {"name": "visium_morphpt_lora_10x",       "label": "MorphPT+LoRA\n10×",            "group": "lora"},
    11: {"name": "visium_morphpt_lora_gate",      "label": "MorphPT+LoRA\nGate linear",    "group": "lora"},
    12: {"name": "visium_morphpt_lora_10x_mlp",   "label": "MorphPT+LoRA\nMLP 10×",        "group": "lora"},
    13: {"name": "visium_morphpt_lora_gate_mlp",  "label": "MorphPT+LoRA\nGate MLP",       "group": "lora"},
    14: {"name": "visium_dinov3_lora_25x",        "label": "DINOv3+LoRA\n2.5×",            "group": "lora"},
    15: {"name": "visium_dinov3_lora_10x",        "label": "DINOv3+LoRA\n10×",             "group": "lora"},
}

# ── 8 selected models for paper figures ────────────────────────────────────
PAPER_JOBS = [1, 5, 8, 3, 14, 10, 15, 12, 13]

# ── Delta pairs — directly answer the 4 key questions ──────────────────────
DELTA_PAIRS = [
    (1,  3,  "J1→J3: DINOv3 frozen vs MorphPT+LoRA\n(2.5×, effect of LoRA + init)"),
    (3,  14, "J3→J14: MorphPT+LoRA vs DINOv3+LoRA\n(2.5×, effect of MorphPT init)"),
    (10, 15, "J10→J15: MorphPT+LoRA vs DINOv3+LoRA\n(10×, effect of MorphPT init)"),
    (12, 13, "J12→J13: MLP 10× vs Gate MLP\n(effect of multiview gate)"),
]

# ── Fixed scatter pairs for coverage plot ──────────────────────────────────
SCATTER_PAIRS = [
    (1,  "DINOv3 frozen 2.5×",   "#999999"),
    (5,  "DINOv3 frozen 10×",    "#4C72B0"),
    (15, "DINOv3+LoRA 10×",      "#55A868"),
    (12, "MorphPT+LoRA MLP 10×", "#C44E52"),
]

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

GROUP_COLORS = {"frozen": "#4C72B0", "lora": "#C44E52"}


# ── Helpers ────────────────────────────────────────────────────────────────
def exp_dir(jid):
    return PROJECT / "experiments" / JOBS[jid]["name"]

def load_test_results(jid):
    p = exp_dir(jid) / "test_results.json"
    return json.loads(p.read_text()) if p.exists() else None

def load_train_log(jid):
    p = exp_dir(jid) / "train_log.json"
    return json.loads(p.read_text()) if p.exists() else None

def load_per_gene_pearson(jid):
    p = exp_dir(jid) / "per_gene_pearson_test.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu").numpy()

def load_gene_coverage():
    """Per-gene prevalence computed on test cells only."""
    expr     = np.load(str(CACHE / "expr.npy"), mmap_mode="r")
    meta     = pd.read_csv(CACHE / "meta.csv")
    test_idx = meta[meta["split"] == "test"]["mmap_idx"].values
    coverage = (expr[test_idx] > 0).mean(axis=0)
    genes    = (EXPR_ROOT / "expr/genes.txt").read_text().splitlines()
    return coverage, genes

def safe_pearsonr(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    r, _ = scipy_pearsonr(a, b)
    return float(r)


# ── 1. Comparison table (all 15 jobs) ─────────────────────────────────────
def make_comparison_table(job_ids):
    print("Building comparison table...")
    rows = []
    for jid in job_ids:
        res = load_test_results(jid)
        r   = load_per_gene_pearson(jid)
        if res is None or r is None:
            print(f"  Job {jid}: missing — skipping")
            continue
        args = res.get("args", {})
        rows.append({
            "job":           jid,
            "name":          JOBS[jid]["name"],
            "group":         JOBS[jid]["group"],
            "encoder":       "MorphPT+LoRA" if args.get("unfreeze_lora") else
                             ("MorphPT" if args.get("ckpt_path") else "DINOv3"),
            "head":          args.get("head_type", "?"),
            "scales":        args.get("scales", "?"),
            "fuse":          args.get("fuse", "?"),
            "lora":          bool(args.get("unfreeze_lora", 0)),
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


# ── 2. Coverage-stratified heatmap (8 paper models) ───────────────────────
def plot_coverage_stratified(available_jobs):
    print("\nPlotting coverage-stratified heatmap...")
    coverage, genes = load_gene_coverage()

    bins      = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 1.01]
    labels    = ["<2%", "2–5%", "5–10%", "10–20%", "20–35%", ">35%"]
    bin_idx   = np.clip(np.digitize(coverage, bins) - 1, 0, len(labels) - 1)
    bin_counts= [int((bin_idx == b).sum()) for b in range(len(labels))]

    # Only show paper jobs
    plot_ids = [j for j in PAPER_JOBS if j in available_jobs]

    heatmap_rows, heatmap_labels, heatmap_groups = [], [], []
    for jid in plot_ids:
        r = load_per_gene_pearson(jid)
        if r is None:
            continue
        means = [r[bin_idx == b].mean() if (bin_idx == b).sum() > 0 else np.nan
                 for b in range(len(labels))]
        heatmap_rows.append(means)
        heatmap_labels.append(JOBS[jid]["label"].replace("\n", " "))
        heatmap_groups.append(JOBS[jid]["group"])

    heatmap  = np.array(heatmap_rows)
    n_models = len(heatmap_rows)
    hm_h     = max(4.0, min(10.0, n_models * 0.7))

    fig = plt.figure(figsize=(16, hm_h))
    gs  = plt.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.35)

    # Left: heatmap
    ax_hm = fig.add_subplot(gs[0])
    vmin  = min(-0.05, float(np.nanmin(heatmap)))
    vmax  = max(0.35,  float(np.nanmax(heatmap)))
    im    = ax_hm.imshow(heatmap, aspect="auto", cmap="RdYlGn",
                         vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax_hm, label="Mean Pearson r", shrink=0.8)

    ax_hm.set_xticks(range(len(labels)))
    ax_hm.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, bin_counts)],
                          fontsize=8)
    ax_hm.set_yticks(range(n_models))
    ax_hm.set_yticklabels(heatmap_labels, fontsize=8)
    ax_hm.set_xlabel("Gene prevalence in test cells (% expressing)")
    ax_hm.set_title("(a) Mean Pearson r by gene coverage bin\n"
                    "(rows sorted by overall performance)", fontsize=9)

    # Color y-tick labels by group
    for tick, grp in zip(ax_hm.get_yticklabels(), heatmap_groups):
        tick.set_color(GROUP_COLORS[grp])

    # Annotate cells
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            v = heatmap[i, j]
            if not np.isnan(v):
                ax_hm.text(j, i, f"{v:.2f}", ha="center", va="center",
                           fontsize=6.5,
                           color="white" if abs(v) > 0.22 else "black")

    # Right: scatter — 4 fixed pairs
    ax_sc = fig.add_subplot(gs[1])
    for jid, lbl, col in SCATTER_PAIRS:
        if jid not in available_jobs:
            continue
        r = load_per_gene_pearson(jid)
        if r is None:
            continue
        ax_sc.scatter(coverage * 100, r, alpha=0.25, s=6,
                      color=col, label=lbl, rasterized=True)

    ax_sc.axhline(0,   color="gray",    lw=0.8, ls="--", alpha=0.6)
    ax_sc.axhline(0.2, color="#C44E52", lw=0.8, ls=":",  alpha=0.6)
    ax_sc.axhline(0.3, color="#8172B2", lw=0.8, ls=":",  alpha=0.6)
    ax_sc.set_xlabel("% test cells expressing gene")
    ax_sc.set_ylabel("Per-gene Pearson r")
    ax_sc.set_title("(b) Per-gene Pearson r vs gene prevalence\n"
                    "(4 representative models)", fontsize=9)
    ax_sc.legend(loc="upper left", fontsize=7, markerscale=2)

    # Add group legend to heatmap
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=GROUP_COLORS["frozen"], label="Frozen backbone"),
                       Patch(facecolor=GROUP_COLORS["lora"],   label="LoRA finetuned")]
    ax_hm.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.suptitle("Gene expression prediction: coverage-stratified performance",
                 fontweight="bold")
    out = OUT_DIR / "coverage_stratified.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 3. Delta-per-gene plots (4 key pairs) ─────────────────────────────────
def plot_delta_pearson(available_jobs):
    print("\nPlotting per-gene delta Pearson...")
    coverage, _ = load_gene_coverage()

    valid_pairs = [(a, b, lbl) for a, b, lbl in DELTA_PAIRS
                   if a in available_jobs and b in available_jobs]
    if not valid_pairs:
        print("  No valid pairs — skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for i, (jid_a, jid_b, label) in enumerate(valid_pairs):
        r_a = load_per_gene_pearson(jid_a)
        r_b = load_per_gene_pearson(jid_b)
        if r_a is None or r_b is None:
            axes[i].set_visible(False)
            continue

        delta = r_b - r_a
        ax    = axes[i]

        sc = ax.scatter(coverage * 100, delta, c=delta,
                        cmap="RdYlGn", vmin=-0.15, vmax=0.15,
                        alpha=0.45, s=8, rasterized=True)
        ax.axhline(0, color="gray", lw=0.8, ls="--")

        # Smoothed trend line
        from scipy.ndimage import uniform_filter1d
        sort_idx = np.argsort(coverage)
        smooth   = uniform_filter1d(delta[sort_idx], size=50)
        ax.plot(coverage[sort_idx] * 100, smooth, color="black", lw=2.0,
                label="Smoothed trend")

        n_better   = int((delta > 0).sum())
        mean_delta = delta.mean()
        ax.set_title(f"({chr(97+i)}) {label}\n"
                     f"Mean Δr = {mean_delta:+.4f}  |  "
                     f"Genes improved: {n_better}/1000",
                     fontsize=8)
        ax.set_xlabel("% test cells expressing gene")
        ax.set_ylabel(f"ΔPearson r (J{jid_b} − J{jid_a})")
        plt.colorbar(sc, ax=ax, shrink=0.75, label="ΔPearson r")
        ax.legend(fontsize=7)

    fig.suptitle("Per-gene Pearson r improvement: key model comparisons",
                 fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "delta_pearson.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 4. Per-gene Pearson distribution (8 paper models) ────────────────────
def plot_pearson_distribution(available_jobs):
    print("\nPlotting per-gene Pearson distribution...")

    plot_ids = [j for j in PAPER_JOBS if j in available_jobs]
    data, xlabels, group_list = [], [], []
    for jid in plot_ids:
        r = load_per_gene_pearson(jid)
        if r is None:
            continue
        data.append(r)
        xlabels.append(JOBS[jid]["label"])
        group_list.append(JOBS[jid]["group"])

    if not data:
        print("  No data — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: violin
    ax    = axes[0]
    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=True)
    means = [d.mean() for d in data]
    norm  = plt.Normalize(min(means), max(means))
    for i, (pc, grp) in enumerate(zip(parts["bodies"], group_list)):
        base_col = GROUP_COLORS[grp]
        pc.set_facecolor(base_col)
        pc.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(xlabels, fontsize=7)
    for tick, grp in zip(ax.get_xticklabels(), group_list):
        tick.set_color(GROUP_COLORS[grp])

    ax.axhline(0,   color="gray",    lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0.2, color="#C44E52", lw=0.8, ls=":",  alpha=0.7, label="r = 0.2")
    ax.axhline(0.3, color="#8172B2", lw=0.8, ls=":",  alpha=0.7, label="r = 0.3")
    ax.set_ylabel("Per-gene Pearson r")
    ax.set_title("(a) Distribution of per-gene Pearson r\n"
                 "(line = median, blue = frozen, red = LoRA)", fontsize=9)
    ax.legend(fontsize=7)

    # Right: stacked bar
    ax      = axes[1]
    tiers   = [(0.3, 1.0, "r > 0.3",       "#2d6a4f"),
               (0.2, 0.3, "0.2 < r ≤ 0.3", "#52b788"),
               (0.1, 0.2, "0.1 < r ≤ 0.2", "#95d5b2"),
               (0.0, 0.1, "0 < r ≤ 0.1",   "#d8f3dc"),
               (-1,  0.0, "r ≤ 0",          "#e0e0e0")]
    x       = np.arange(len(data))
    bottoms = np.zeros(len(data))

    for lo, hi, label, color in tiers:
        counts = np.array([int(((d > lo) & (d <= hi)).sum()) for d in data])
        ax.bar(x, counts, bottom=bottoms, color=color, label=label,
               width=0.6, edgecolor="white", linewidth=0.5)
        for xi, (cnt, bot) in enumerate(zip(counts, bottoms)):
            if cnt > 25:
                ax.text(xi, bot + cnt / 2, str(cnt),
                        ha="center", va="center", fontsize=6)
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    for tick, grp in zip(ax.get_xticklabels(), group_list):
        tick.set_color(GROUP_COLORS[grp])
    ax.set_ylabel("Number of genes (out of 1000)")
    ax.set_title("(b) Genes per Pearson r tier\n"
                 "(blue = frozen, red = LoRA)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1060)

    fig.suptitle("Per-gene Pearson r distribution — selected models",
                 fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "pearson_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 5. Tile-level performance ─────────────────────────────────────────────
def plot_tile_performance(available_jobs):
    print("\nPlotting tile-level performance...")
    meta      = pd.read_csv(CACHE / "meta.csv")
    test_meta = meta[meta["split"] == "test"].copy().reset_index(drop=True)
    test_tiles= sorted(test_meta["tile_id"].unique())

    if len(test_tiles) == 0:
        print("  No test tiles — skipping")
        return

    stats     = np.load(str(CACHE / "expr_stats.npz"))
    gene_mean = stats["gene_mean"]
    gene_std  = stats["gene_std"]
    expr      = np.load(str(CACHE / "expr.npy"), mmap_mode="r")
    X_test    = np.array(expr[test_meta["mmap_idx"].values], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-tile Pearson bar chart for top models
    ax      = axes[0]
    top_ids = [j for j in [12, 13, 10, 15] if j in available_jobs]
    colors  = ["#C44E52", "#8172B2", "#4C72B0", "#55A868"]
    x       = np.arange(len(test_tiles))
    width   = 0.8 / max(len(top_ids), 1)

    any_pred = False
    for i, (jid, col) in enumerate(zip(top_ids, colors)):
        pred_path = exp_dir(jid) / "predictions_test.npz"
        if not pred_path.exists():
            continue
        any_pred  = True
        npz       = np.load(str(pred_path))
        y_pred    = npz["y_pred"] * gene_std + gene_mean
        y_true    = X_test       * gene_std + gene_mean

        tile_means = []
        for tid in test_tiles:
            mask = test_meta["tile_id"].values == tid
            yt, yp = y_true[mask], y_pred[mask]
            rs = [safe_pearsonr(yt[:, g], yp[:, g]) for g in range(yt.shape[1])
                  if yt[:, g].std() > 1e-6]
            tile_means.append(np.mean(rs) if rs else np.nan)

        offset = (i - len(top_ids) / 2) * width + width / 2
        ax.bar(x + offset, tile_means, width=width * 0.9,
               color=col, alpha=0.85,
               label=JOBS[jid]["label"].replace("\n", " "))

    if not any_pred:
        ax.text(0.5, 0.5,
                "Run spatial_maps.py --job 12 --split test\nto generate predictions_test.npz",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f"Tile {t}" for t in test_tiles], rotation=30, fontsize=8)
        ax.set_ylabel("Mean Pearson r")
        ax.set_title("(a) Per-tile mean Pearson r on test tiles", fontsize=9)
        ax.legend(fontsize=7)

    # Right: tile locations on tissue
    ax = axes[1]
    for tid in test_tiles:
        mask    = test_meta["tile_id"].values == tid
        tile_df = test_meta[mask]
        cx = tile_df["x_centroid"].mean() if "x_centroid" in tile_df else 0
        cy = tile_df["y_centroid"].mean() if "y_centroid" in tile_df else 0
        n  = mask.sum()
        ax.scatter(cx, -cy, s=n / 15, alpha=0.6, zorder=2)
        ax.annotate(f"T{tid}\n(n={n:,})", (cx, -cy),
                    ha="center", va="center", fontsize=6)

    ax.set_xlabel("x centroid")
    ax.set_ylabel("y centroid (flipped)")
    ax.set_title("(b) Test tile locations\n(bubble size ∝ cell count)", fontsize=9)

    fig.suptitle("Tile-level performance on test split", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "tile_performance.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── 6. Learning curves (sanity check, all jobs) ───────────────────────────
def plot_learning_curves(job_ids):
    print("\nPlotting learning curves (sanity check)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors    = plt.cm.tab20(np.linspace(0, 1, len(job_ids)))

    for jid, col in zip(job_ids, colors):
        log = load_train_log(jid)
        if log is None:
            continue
        epochs   = [r["epoch"]          for r in log]
        val_corr = [r["va_mean_pearson"] for r in log]
        tr_zmse  = [r["tr_zMSE"]        for r in log]
        lw  = 2.0 if jid in PAPER_JOBS else 0.8
        lbl = f"J{jid}: {JOBS[jid]['label'].replace(chr(10), ' ')}"
        axes[0].plot(epochs, val_corr, color=col, lw=lw, label=lbl)
        axes[1].plot(epochs, tr_zmse,  color=col, lw=lw, label=lbl)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val mean Pearson r")
    axes[0].set_title("(a) Validation Pearson r (bold = paper models)")
    axes[0].legend(fontsize=5, ncol=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train zMSE")
    axes[1].set_title("(b) Training zMSE")
    axes[1].legend(fontsize=5, ncol=2)

    fig.suptitle("Learning curves — all jobs (sanity check)", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "learning_curves.pdf"
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
    plot_pearson_distribution(available)
    plot_tile_performance(available)
    plot_learning_curves(available)

    print(f"\nAll outputs → {OUT_DIR}")
    print("\nPaper figures: coverage_stratified.pdf, delta_pearson.pdf, pearson_distribution.pdf, tile_performance.pdf")
    print("Sanity check : learning_curves.pdf")
    print("Table        : comparison_table.csv")


if __name__ == "__main__":
    main()