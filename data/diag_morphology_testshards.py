#!/usr/bin/env python3
"""
Morphology diagnostic: solidity + area distributions on test shards.
=====================================================================
Run this BEFORE committing to final MORPH_LIMITS thresholds.

Uses 10x crops (224px downsampled from 331px) for robust segmentation.
Metrics:
  - solidity   = area / convex_hull_area  (robust to boundary noise)
  - nucleus_area (px² at 224px resolution)

Segmentation pipeline per cell:
  1. CLAHE contrast enhancement
  2. Otsu threshold
  3. Morphological close (fill hollow nuclei) + open (denoise)
  4. RETR_CCOMP → outer contours only (ignore internal holes)
  5. Border-touch rejection (nucleus touching image edge → discard)

Output (--out_dir):
  solidity_by_coarse.png       — per-coarse histograms, fine class overlay
  solidity_per_fine_class.png  — violin plot, 21 classes sorted by median
  area_by_coarse.png           — area distributions per coarse group
  failure_rate_by_tissue.png   — segmentation failure rate per shard
  morphology_summary.csv       — per-fine-class stats (p5/p50/p95 solidity,
                                  p2/p50/p98 area)
  per_tissue_summary.csv       — per-tissue per-class stats (for threshold
                                  selection)

Usage:
  python data/diag_morphology_testshards.py \
    --test_shards prepared/splits_v2_seed1337_nobreast/test_shards \
    --n_shards 3 \
    --workers 16 \
    --out_dir results/morph_diag

  # All shards:
  python data/diag_morphology_testshards.py \
    --test_shards prepared/splits_v2_seed1337_nobreast/test_shards \
    --workers 16 \
    --out_dir results/morph_diag_full
"""

import argparse
import warnings
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ──────────────────────────────────────────────
# Class mappings
# ──────────────────────────────────────────────

FINE_TO_COARSE = {
    "Colon cancer cells":         "Cancer",
    "Liver cancer cells":         "Cancer",
    "Lung cancer cells":          "Cancer",
    "Ovary cancer cells":         "Cancer",
    "Pancreas cancer cells":      "Cancer",
    "Skin cancer cells":          "Cancer",
    "B cells":                    "Lymphoid",
    "NK cells":                   "Lymphoid",
    "T cells":                    "Lymphoid",
    "Astrocytes":                 "Neuroglial",
    "Microglia":                  "Neuroglial",
    "Neurons":                    "Neuroglial",
    "Oligodendrocytes":           "Neuroglial",
    "Endothelial cells":          "Tissue_Vascular",
    "Epithelial cells":           "Tissue_Vascular",
    "Fibroblasts":                "Tissue_Vascular",
    "Myeloid cells":              "Tissue_Vascular",
    "Pericytes":                  "Tissue_Vascular",
    "Smooth muscle cells":        "Tissue_Vascular",
    "Stem and progenitor cells":  "Stem_Progenitor",
    "Stromal cells":              "Stromal",
}

COARSE_ORDER = [
    "Cancer", "Lymphoid", "Neuroglial",
    "Tissue_Vascular", "Stem_Progenitor", "Stromal",
]

COARSE_COLORS = {
    "Cancer":          "#E24B4A",
    "Lymphoid":        "#378ADD",
    "Neuroglial":      "#1D9E75",
    "Tissue_Vascular": "#EF9F27",
    "Stem_Progenitor": "#7F77DD",
    "Stromal":         "#888780",
}


# ──────────────────────────────────────────────
# Morphology computation
# ──────────────────────────────────────────────

def compute_morphology(img_path: str):
    """
    Returns (area_px, solidity) for the dominant nucleus in a 10x crop.
    Returns (None, None) on failure or border-touching nucleus.

    solidity = contour_area / convex_hull_area ∈ (0, 1]
      1.0  → perfectly convex nucleus
      low  → irregular, fragmented, or severely deformed nucleus
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    h, w = img.shape

    # 1. CLAHE: stabilise DAPI contrast before Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 2. Otsu threshold
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Close (fill hollow nuclei) then open (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # 4. RETR_CCOMP: keep outer contours only (hierarchy[i][3] == -1)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return None, None

    outer = [
        contours[i] for i in range(len(contours))
        if hierarchy[0][i][3] == -1
    ]
    if not outer:
        return None, None

    # Select contour whose centroid is closest to image centre
    # (crop is centred on the target nucleus, so this picks the right one
    #  even in dense tissue where adjacent nuclei merge into large blobs)
    cx, cy = w / 2.0, h / 2.0

    def dist_to_center(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return float("inf")
        mx = M["m10"] / M["m00"]
        my = M["m01"] / M["m00"]
        return (mx - cx) ** 2 + (my - cy) ** 2

    c = min(outer, key=dist_to_center)
    area = cv2.contourArea(c)

    if area < 20:
        return None, None

    # 5. Border-touch rejection (2px tolerance — now safe because we select
    #    the centre contour, not the largest blob)
    border = 2
    pts = c.reshape(-1, 2)
    if (
        (pts[:, 0] <= border).any() or
        (pts[:, 0] >= w - 1 - border).any() or
        (pts[:, 1] <= border).any() or
        (pts[:, 1] >= h - 1 - border).any()
    ):
        return None, None

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return float(area), float(solidity)


def _process_row(args):
    idx, img_path = args
    area, sol = compute_morphology(img_path)
    return idx, area, sol


def compute_shard_morphology(parquet_path: str, workers: int) -> pd.DataFrame:
    """Load one shard, compute morphology for all cells, return augmented df."""
    df = pd.read_parquet(parquet_path)
    tissue = Path(parquet_path).stem

    tasks = [(i, row["img_path_10x"]) for i, row in df.iterrows()]
    areas = np.full(len(df), np.nan)
    sols  = np.full(len(df), np.nan)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_row, t): t[0] for t in tasks}
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"  {tissue[:45]}", leave=False):
            idx, area, sol = f.result()
            if area is not None:
                areas[idx] = area
                sols[idx]  = sol

    df["nucleus_area"] = areas
    df["solidity"]     = sols
    df["coarse"]       = df["label"].map(FINE_TO_COARSE)
    df["tissue"]       = tissue
    return df


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_solidity_by_coarse(df_valid: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, coarse in zip(axes, COARSE_ORDER):
        sub = df_valid[df_valid["coarse"] == coarse]
        color = COARSE_COLORS[coarse]

        if len(sub) == 0:
            ax.set_visible(False)
            continue

        for fine_label, grp in sub.groupby("label"):
            ax.hist(grp["solidity"], bins=40, range=(0, 1),
                    alpha=0.45, density=True, label=fine_label)

        p5 = sub["solidity"].quantile(0.05)
        ax.axvline(p5, color=color, linestyle="--", linewidth=1.5,
                   label=f"p5 = {p5:.2f}")

        ax.set_title(f"{coarse}  (n={len(sub):,})", fontsize=11)
        ax.set_xlabel("Solidity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, 1)

    plt.suptitle("Solidity distributions by coarse group\n"
                 "(dashed = p5 of group)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_solidity_violin(df_valid: pd.DataFrame, out_path: Path):
    fine_medians = (df_valid.groupby("label")["solidity"]
                   .median().sort_values())

    fig, ax = plt.subplots(figsize=(15, 6))

    for pos, fine_label in enumerate(fine_medians.index):
        coarse = FINE_TO_COARSE.get(fine_label, "DEFAULT")
        color  = COARSE_COLORS.get(coarse, "#888780")
        data   = df_valid[df_valid["label"] == fine_label]["solidity"].dropna()

        if len(data) < 5:
            continue

        parts = ax.violinplot(data, positions=[pos], widths=0.7,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(len(fine_medians)))
    ax.set_xticklabels(fine_medians.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Solidity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Solidity per fine class  (sorted by median)", fontsize=12)
    ax.axhline(0.65, color="gray", linestyle=":", linewidth=1, label="solidity=0.65")
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1, label="solidity=0.75")
    ax.legend(fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6)
               for c in COARSE_COLORS.values()]
    ax.legend(handles, COARSE_COLORS.keys(), fontsize=8,
              loc="lower right", title="Coarse group")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_area_by_coarse(df_valid: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, coarse in zip(axes, COARSE_ORDER):
        sub = df_valid[df_valid["coarse"] == coarse]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        for fine_label, grp in sub.groupby("label"):
            ax.hist(grp["nucleus_area"], bins=50, alpha=0.45,
                    density=True, label=fine_label)

        p2  = sub["nucleus_area"].quantile(0.02)
        p98 = sub["nucleus_area"].quantile(0.98)
        ax.axvline(p2,  color=COARSE_COLORS[coarse], linestyle=":",
                   linewidth=1.2, label=f"p2={p2:.0f}")
        ax.axvline(p98, color=COARSE_COLORS[coarse], linestyle="--",
                   linewidth=1.2, label=f"p98={p98:.0f}")

        ax.set_title(f"{coarse}  (n={len(sub):,})", fontsize=11)
        ax.set_xlabel("Area (px²)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Nucleus area distributions by coarse group\n"
                 "(dotted = p2, dashed = p98)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_failure_rate(df_all: pd.DataFrame, out_path: Path):
    tissue_stats = (df_all.groupby("tissue")
                   .apply(lambda g: pd.Series({
                       "total": len(g),
                       "failed": g["solidity"].isna().sum(),
                   }))
                   .reset_index())
    tissue_stats["fail_rate"] = tissue_stats["failed"] / tissue_stats["total"]
    tissue_stats = tissue_stats.sort_values("fail_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(14, max(4, len(tissue_stats) * 0.35)))
    colors = ["#E24B4A" if r > 0.15 else "#EF9F27" if r > 0.05 else "#639922"
              for r in tissue_stats["fail_rate"]]
    ax.barh(range(len(tissue_stats)), tissue_stats["fail_rate"],
            color=colors, height=0.7)
    ax.set_yticks(range(len(tissue_stats)))
    ax.set_yticklabels(tissue_stats["tissue"], fontsize=8)
    ax.set_xlabel("Failure / border-touch rate")
    ax.set_title("Segmentation failure rate per tissue shard\n"
                 "(red > 15%, orange 5-15%, green < 5%)", fontsize=11)
    ax.axvline(0.05, color="#EF9F27", linestyle="--", linewidth=1)
    ax.axvline(0.15, color="#E24B4A", linestyle="--", linewidth=1)
    ax.set_xlim(0, min(1.0, tissue_stats["fail_rate"].max() * 1.2 + 0.05))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_solidity_by_tissue(df_valid: pd.DataFrame, out_path: Path):
    """
    Per-tissue solidity box plot for each fine class.
    Helps visualise cross-tissue variability — the key question for
    deciding whether per-tissue thresholds are needed.
    """
    fine_classes = sorted(df_valid["label"].unique())
    n = len(fine_classes)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, nrows * 3.5), squeeze=False)

    for idx, fine_label in enumerate(fine_classes):
        ax = axes[idx // ncols][idx % ncols]
        sub = df_valid[df_valid["label"] == fine_label]
        coarse = FINE_TO_COARSE.get(fine_label, "DEFAULT")
        color  = COARSE_COLORS.get(coarse, "#888780")

        tissues = sorted(sub["tissue"].unique())
        data_per_tissue = [
            sub[sub["tissue"] == t]["solidity"].dropna().values
            for t in tissues
        ]
        # Drop empty
        pairs = [(t, d) for t, d in zip(tissues, data_per_tissue) if len(d) >= 5]
        if not pairs:
            ax.set_visible(False)
            continue

        tissues_ok, data_ok = zip(*pairs)
        bp = ax.boxplot(data_ok, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_xticks(range(1, len(tissues_ok) + 1))
        ax.set_xticklabels(
            [t.replace("Xenium_", "").replace("Xenium", "")[:22]
             for t in tissues_ok],
            rotation=60, ha="right", fontsize=6
        )
        ax.set_title(f"{fine_label}", fontsize=9)
        ax.set_ylabel("Solidity", fontsize=8)
        ax.set_ylim(0, 1.05)

    # Hide unused axes
    for idx in range(len(fine_classes), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle("Solidity per fine class — cross-tissue variability\n"
                 "(each box = one tissue shard)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# Summary tables
# ──────────────────────────────────────────────

def build_global_summary(df_valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, grp in df_valid.groupby("label"):
        coarse = FINE_TO_COARSE.get(label, "?")
        sol = grp["solidity"].dropna()
        area = grp["nucleus_area"].dropna()
        rows.append({
            "label":     label,
            "coarse":    coarse,
            "n":         len(grp),
            "n_valid":   len(sol),
            "sol_p5":    round(float(sol.quantile(0.05)),  3),
            "sol_p25":   round(float(sol.quantile(0.25)),  3),
            "sol_p50":   round(float(sol.quantile(0.50)),  3),
            "sol_p75":   round(float(sol.quantile(0.75)),  3),
            "sol_p95":   round(float(sol.quantile(0.95)),  3),
            "area_p2":   round(float(area.quantile(0.02)), 1),
            "area_p25":  round(float(area.quantile(0.25)), 1),
            "area_p50":  round(float(area.quantile(0.50)), 1),
            "area_p75":  round(float(area.quantile(0.75)), 1),
            "area_p98":  round(float(area.quantile(0.98)), 1),
        })
    df_sum = pd.DataFrame(rows).sort_values(["coarse", "label"])
    return df_sum


def build_per_tissue_summary(df_valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tissue, label), grp in df_valid.groupby(["tissue", "label"]):
        coarse = FINE_TO_COARSE.get(label, "?")
        sol  = grp["solidity"].dropna()
        area = grp["nucleus_area"].dropna()
        if len(sol) < 5:
            continue
        rows.append({
            "tissue":   tissue,
            "label":    label,
            "coarse":   coarse,
            "n":        len(grp),
            "sol_p5":   round(float(sol.quantile(0.05)),  3),
            "sol_p50":  round(float(sol.quantile(0.50)),  3),
            "sol_p95":  round(float(sol.quantile(0.95)),  3),
            "area_p2":  round(float(area.quantile(0.02)), 1),
            "area_p50": round(float(area.quantile(0.50)), 1),
            "area_p98": round(float(area.quantile(0.98)), 1),
        })
    return pd.DataFrame(rows).sort_values(["coarse", "label", "tissue"])


def print_global_summary(df_sum: pd.DataFrame):
    hdr = (f"{'Fine Class':<35} {'Coarse':<18} {'N':>6} "
           f"{'Sol p5':>7} {'Sol p50':>7} {'Sol p95':>7} "
           f"{'Area p2':>8} {'Area p50':>8} {'Area p98':>8}")
    print(f"\n{hdr}")
    print("─" * len(hdr))
    for _, row in df_sum.iterrows():
        print(f"{row['label']:<35} {row['coarse']:<18} {row['n']:>6,} "
              f"{row['sol_p5']:>7.3f} {row['sol_p50']:>7.3f} {row['sol_p95']:>7.3f} "
              f"{row['area_p2']:>8.1f} {row['area_p50']:>8.1f} {row['area_p98']:>8.1f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Morphology diagnostic for test shards."
    )
    ap.add_argument("--test_shards", required=True,
                    help="Directory containing test shard .parquet files.")
    ap.add_argument("--n_shards", type=int, default=None,
                    help="Number of shards to process (default: all).")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel workers for image processing.")
    ap.add_argument("--out_dir", default="results/morph_diag",
                    help="Output directory for plots and CSVs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob(str(Path(args.test_shards) / "*.parquet")))
    if args.n_shards:
        shards = shards[:args.n_shards]

    if not shards:
        raise RuntimeError(f"No parquet files found in {args.test_shards}")

    print(f"{'='*70}")
    print(f"Morphology diagnostic")
    print(f"{'='*70}")
    print(f"Shards:  {len(shards)}")
    print(f"Workers: {args.workers}")
    print(f"Output:  {out_dir}\n")

    # ── Compute morphology ──
    all_dfs = []
    for shard in shards:
        print(f"Processing: {Path(shard).stem}")
        df = compute_shard_morphology(shard, args.workers)
        all_dfs.append(df)

    df_all   = pd.concat(all_dfs, ignore_index=True)
    df_valid = df_all.dropna(subset=["solidity", "nucleus_area"])

    n_total   = len(df_all)
    n_valid   = len(df_valid)
    fail_rate = 1 - n_valid / n_total

    print(f"\n{'='*70}")
    print(f"Segmentation results")
    print(f"{'='*70}")
    print(f"  Total cells:     {n_total:,}")
    print(f"  Valid:           {n_valid:,}")
    print(f"  Failed / border: {n_total - n_valid:,}  ({fail_rate:.1%})")

    # ── Summary tables ──
    df_global_sum     = build_global_summary(df_valid)
    df_per_tissue_sum = build_per_tissue_summary(df_valid)

    print_global_summary(df_global_sum)

    df_global_sum.to_csv(out_dir / "morphology_summary.csv", index=False)
    df_per_tissue_sum.to_csv(out_dir / "per_tissue_summary.csv", index=False)
    print(f"\n  Saved: {out_dir / 'morphology_summary.csv'}")
    print(f"  Saved: {out_dir / 'per_tissue_summary.csv'}")

    # ── Plots ──
    print(f"\n{'='*70}")
    print("Generating plots")
    print(f"{'='*70}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_solidity_by_coarse(
            df_valid, out_dir / "solidity_by_coarse.png")
        plot_solidity_violin(
            df_valid, out_dir / "solidity_per_fine_class.png")
        plot_area_by_coarse(
            df_valid, out_dir / "area_by_coarse.png")
        plot_failure_rate(
            df_all,   out_dir / "failure_rate_by_tissue.png")
        plot_solidity_by_tissue(
            df_valid, out_dir / "solidity_cross_tissue.png")

    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Check solidity_cross_tissue.png — if distributions vary a lot")
    print(f"     across tissues for the same fine class, per-tissue thresholds")
    print(f"     are justified.")
    print(f"  2. Use morphology_summary.csv sol_p5 column as a starting point")
    print(f"     for MORPH_LIMITS min_solidity values.")
    print(f"  3. Check failure_rate_by_tissue.png — shards with > 15% failure")
    print(f"     may have image quality issues worth investigating separately.")


if __name__ == "__main__":
    main()