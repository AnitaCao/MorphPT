#!/usr/bin/env python3
"""
Morphology-based test set quality filter
=========================================
Filters test_shards by nucleus solidity and area computed from 10x crops.

Pipeline per cell:
  1. CLAHE contrast enhancement
  2. Otsu threshold
  3. Morphological close (fill hollow nuclei) + open (denoise)
  4. RETR_CCOMP → outer contours only (ignore internal holes)
  5. Border-touch rejection (10px tolerance)
  6. Solidity = area / convex_hull_area

Filtering strategy:
  - Per-tissue per-class: each tissue × fine_class pair filtered independently
  - Solidity threshold: per coarse group (from MORPH_LIMITS)
  - Area percentile: per coarse group, computed within each tissue × class subset
  - Lymphoid area upper limit: p90 instead of p98 to remove doublets/clumps

Outputs:
  <out_dir>/<tissue>.parquet            — filtered shards (one per tissue)
  <out_dir>/morphology_bad_ids.parquet  — removed cells with filter_reason:
                                            seg_failed | low_solidity | area_outlier
  <out_dir>/filter_summary.json         — overall stats + reason breakdown
  <out_dir>/class_filter_stats.csv      — per-tissue per-class stats
  <out_dir>/class_filter_summary.csv    — per-class aggregated stats

Usage:
  python scripts/filter_test_morphology.py \
    --test_shards prepared/splits_v4_seed1337/test_shards \
    --out_dir prepared/splits_v4_seed1337/test_shards_filtered \
    --workers 16

  # Dry run (stats + bad IDs only, no filtered shards written):
  python scripts/filter_test_morphology.py \
    --test_shards prepared/splits_v4_seed1337/test_shards \
    --out_dir prepared/splits_v4_seed1337/test_shards_filtered \
    --workers 16 \
    --dry_run
"""

import argparse
import json
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
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

# ──────────────────────────────────────────────
# Morphology thresholds (per coarse group)
# ──────────────────────────────────────────────
#
# min_solidity: derived from sol_p5 in morphology_summary.csv,
#               set slightly below the group minimum to keep ~95% of valid cells.
#
# area_range:   [lo_percentile, hi_percentile] computed per-tissue per-class.
#               Lymphoid uses [2, 90] instead of [2, 98] to remove doublets/clumps
#               (B cells area_p98=29k, T cells area_p98=25k — far above NK p98=17k).
#
#   Group            sol_p5 (from diag)   min_solidity chosen
#   Neuroglial       0.643–0.748          0.60  (clear separation from other groups)
#   Stromal          0.530                0.45
#   Cancer           0.493–0.534          0.43
#   Tissue_Vascular  0.480–0.578          0.43
#   Stem_Progenitor  0.454                0.40
#   Lymphoid         0.430–0.457          0.38  (small dense cells, lower bar)

MORPH_LIMITS = {
    "Neuroglial":      {"min_solidity": 0.60, "area_range": [2, 98]},
    "Stromal":         {"min_solidity": 0.45, "area_range": [2, 98]},
    "Cancer":          {"min_solidity": 0.43, "area_range": [2, 98]},
    "Tissue_Vascular": {"min_solidity": 0.43, "area_range": [2, 98]},
    "Stem_Progenitor": {"min_solidity": 0.40, "area_range": [2, 98]},
    "Lymphoid":        {"min_solidity": 0.38, "area_range": [2, 90]},  # p90 cuts doublets
    "DEFAULT":         {"min_solidity": 0.40, "area_range": [2, 98]},
}

# Minimum cells per tissue×class to apply percentile filtering.
# Groups smaller than this are kept entirely (percentile unreliable on tiny samples).
MIN_GROUP_SIZE = 10


# ──────────────────────────────────────────────
# Morphology computation
# ──────────────────────────────────────────────

def compute_morphology(img_path: str):
    """
    Returns (area_px, solidity) for dominant nucleus in 10x crop.
    Returns (None, None) on segmentation failure or border-touching nucleus.

    Uses 10x crop (stored at original crop size, typically 331px).
    border=10px tolerance: prevents over-rejection in dense tissue where
    neighbouring nuclei extend close to the crop edge.

    solidity = contour_area / convex_hull_area in (0, 1]
      1.0 -> perfectly convex
      low -> irregular, fragmented, or severely deformed
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    h, w = img.shape

    # 1. CLAHE: stabilise DAPI contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 2. Otsu threshold
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Close (fill hollow nuclei) then open (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # 4. RETR_CCOMP: outer contours only (hierarchy[i][3] == -1 -> no parent)
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

    # 6. Solidity
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return float(area), float(solidity)


def _process_row(args):
    idx, img_path = args
    area, sol = compute_morphology(img_path)
    return idx, area, sol


# ──────────────────────────────────────────────
# Per-shard filtering
# ──────────────────────────────────────────────

def filter_shard(parquet_path: str, out_dir: Path,
                 morph_limits: dict, dry_run: bool,
                 workers: int):
    """
    Returns:
      tissue        str
      n_before      int
      n_seg_failed  int
      n_after       int
      class_stats   list[dict]
      df_bad        pd.DataFrame  — removed cells with filter_reason column
    """
    df = pd.read_parquet(parquet_path)
    tissue = Path(parquet_path).stem

    label_col = "label" if "label" in df.columns else "fine_label"

    # --- Parallel morphology computation (10x crop) ---
    tasks = [(i, row["img_path_10x"]) for i, row in df.iterrows()]
    areas = np.full(len(df), np.nan)
    sols  = np.full(len(df), np.nan)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_row, t): t[0] for t in tasks}
        for f in as_completed(futs):
            idx, area, sol = f.result()
            if area is not None:
                areas[idx] = area
                sols[idx]  = sol

    df["nucleus_area"] = areas
    df["solidity"]     = sols
    df["coarse"]       = df[label_col].map(FINE_TO_COARSE)

    # Initialise filter_reason for all rows
    df["filter_reason"] = "kept"

    # --- Filter 1: segmentation failure / border touch ---
    seg_ok = ~np.isnan(areas)
    df.loc[~seg_ok, "filter_reason"] = "seg_failed"

    # --- Filter 2 & 3: per-tissue per-class solidity + area percentile ---
    keep        = np.zeros(len(df), dtype=bool)
    class_stats = []

    for label, grp in df[seg_ok].groupby(label_col):
        coarse = FINE_TO_COARSE.get(label, "DEFAULT")
        limits = morph_limits.get(coarse, morph_limits["DEFAULT"])

        min_sol        = limits["min_solidity"]
        lo_pct, hi_pct = limits["area_range"]

        # Solidity filter
        sol_ok  = grp["solidity"] >= min_sol
        grp_sol = grp[sol_ok]

        # Mark low-solidity cells
        df.loc[grp.index[~sol_ok], "filter_reason"] = "low_solidity"

        if len(grp_sol) < MIN_GROUP_SIZE:
            # Too few cells: keep all that passed solidity
            keep[grp_sol.index] = True
            class_stats.append({
                "tissue":   tissue, "label": label, "coarse": coarse,
                "n_total":  len(grp), "n_seg_ok": len(grp),
                "n_sol_ok": len(grp_sol), "n_kept": len(grp_sol),
                "area_lo":  None, "area_hi": None,
                "note":     "small_group_skip_area_filter",
            })
            continue

        # Area percentile (computed within this tissue x class subset)
        lo = np.percentile(grp_sol["nucleus_area"], lo_pct)
        hi = np.percentile(grp_sol["nucleus_area"], hi_pct)

        area_ok       = (grp_sol["nucleus_area"] >= lo) & (grp_sol["nucleus_area"] <= hi)
        idx_keep      = grp_sol.index[area_ok]
        idx_area_fail = grp_sol.index[~area_ok]

        keep[idx_keep] = True
        df.loc[idx_area_fail, "filter_reason"] = "area_outlier"

        class_stats.append({
            "tissue":   tissue,
            "label":    label,
            "coarse":   coarse,
            "n_total":  len(grp),
            "n_seg_ok": len(grp),
            "n_sol_ok": len(grp_sol),
            "n_kept":   len(idx_keep),
            "area_lo":  round(lo, 1),
            "area_hi":  round(hi, 1),
            "note":     "",
        })

    # Bad IDs: all rows not kept
    bad_cols = [c for c in [
        "cell_id", label_col, "coarse", "tissue",
        "nucleus_area", "solidity", "filter_reason",
    ] if c in df.columns]
    df_bad = df[~keep][bad_cols].copy().reset_index(drop=True)
    df_bad["tissue"] = tissue  # ensure tissue column always present

    df_filtered  = df[keep].reset_index(drop=True)
    n_before     = len(df)
    n_seg_failed = int((~seg_ok).sum())
    n_after      = len(df_filtered)

    if not dry_run:
        # Drop morphology helper columns before saving filtered shard
        save_df = df_filtered.drop(
            columns=["nucleus_area", "solidity", "coarse", "filter_reason"],
            errors="ignore",
        )
        save_df.to_parquet(out_dir / f"{tissue}.parquet", index=False)

    return tissue, n_before, n_seg_failed, n_after, class_stats, df_bad


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Morphology filter for test shards (solidity + area, per-tissue per-class)."
    )
    ap.add_argument("--test_shards", required=True,
                    help="Directory containing test shard .parquet files.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for filtered shards and bad IDs.")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel workers for image processing.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Compute stats and bad IDs only; do not write filtered shards.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # always create: needed for summary/bad IDs

    shards = sorted(glob(str(Path(args.test_shards) / "*.parquet")))
    if not shards:
        raise RuntimeError(f"No parquet files found in {args.test_shards}")

    print(f"{'='*70}")
    print(f"Morphology filter — {'DRY RUN' if args.dry_run else 'WRITING OUTPUT'}")
    print(f"{'='*70}")
    print(f"Shards:  {len(shards)}")
    print(f"Workers: {args.workers}")
    print(f"Output:  {out_dir}\n")
    print("MORPH_LIMITS:")
    for coarse, lim in MORPH_LIMITS.items():
        print(f"  {coarse:<20} min_solidity={lim['min_solidity']}  "
              f"area=[p{lim['area_range'][0]}, p{lim['area_range'][1]}]")
    print()

    # --- Process shards ---
    shard_summary   = []
    all_class_stats = []
    all_bad_dfs     = []
    total_before = total_after = total_seg_failed = 0

    for shard in tqdm(shards, desc="Filtering"):
        tissue, nb, nsf, na, cstats, df_bad = filter_shard(
            shard, out_dir, MORPH_LIMITS, args.dry_run, args.workers
        )
        shard_summary.append({
            "tissue":       tissue,
            "before":       nb,
            "seg_failed":   nsf,
            "after":        na,
            "removed":      nb - na,
            "removal_rate": round((nb - na) / nb, 3) if nb > 0 else 0,
        })
        all_class_stats.extend(cstats)
        all_bad_dfs.append(df_bad)
        total_before     += nb
        total_after      += na
        total_seg_failed += nsf

    # --- Save bad IDs (always, even in dry_run) ---
    df_all_bad   = pd.concat(all_bad_dfs, ignore_index=True)
    bad_ids_path = out_dir / "morphology_bad_ids.parquet"
    df_all_bad.to_parquet(bad_ids_path, index=False)
    reason_counts = df_all_bad["filter_reason"].value_counts()

    # --- Shard-level report ---
    print(f"\n{'='*75}")
    print(f"{'Tissue':<52} {'Before':>7} {'After':>7} {'Removed':>9}")
    print(f"{'─'*75}")
    for s in sorted(shard_summary, key=lambda x: -x["removed"]):
        print(f"  {s['tissue']:<50} {s['before']:>7,} {s['after']:>7,} "
              f"{s['removed']:>7,} ({100*s['removal_rate']:.1f}%)")
    print(f"{'─'*75}")
    print(f"  {'TOTAL':<50} {total_before:>7,} {total_after:>7,} "
          f"{total_before - total_after:>7,} "
          f"({100*(1 - total_after/total_before):.1f}%)")

    print(f"\n  Removal reasons:")
    for reason, count in reason_counts.items():
        print(f"    {reason:<25} {count:>8,}  ({100*count/len(df_all_bad):.1f}%)")

    # --- Per-class summary ---
    df_cs = pd.DataFrame(all_class_stats)
    if len(df_cs) > 0:
        class_agg = (
            df_cs.groupby(["label", "coarse"])
            .agg(
                n_total  =("n_total",  "sum"),
                n_seg_ok =("n_seg_ok", "sum"),
                n_sol_ok =("n_sol_ok", "sum"),
                n_kept   =("n_kept",   "sum"),
            )
            .reset_index()
        )
        class_agg["removal_rate"] = (
            (class_agg["n_total"] - class_agg["n_kept"]) / class_agg["n_total"]
        ).round(3)
        class_agg = class_agg.sort_values(["coarse", "label"])

        print(f"\n{'='*75}")
        print("Per-class summary (aggregated across tissues)")
        print(f"{'='*75}")
        print(f"  {'Fine Class':<35} {'Coarse':<18} {'Total':>7} "
              f"{'Sol OK':>7} {'Kept':>7} {'Removed%':>9}")
        print(f"  {'─'*85}")
        for _, row in class_agg.iterrows():
            removed_pct = 100 * row["removal_rate"]
            flag = " !" if removed_pct > 30 else ""
            print(f"  {row['label']:<35} {row['coarse']:<18} "
                  f"{row['n_total']:>7,} {row['n_sol_ok']:>7,} "
                  f"{row['n_kept']:>7,} {removed_pct:>8.1f}%{flag}")

    # --- Save summary files ---
    summary_data = {
        "params": {
            "test_shards":    args.test_shards,
            "out_dir":        str(out_dir),
            "workers":        args.workers,
            "dry_run":        args.dry_run,
            "morph_limits":   MORPH_LIMITS,
            "min_group_size": MIN_GROUP_SIZE,
        },
        "totals": {
            "before":        total_before,
            "after":         total_after,
            "removed":       total_before - total_after,
            "removal_rate":  round((total_before - total_after) / total_before, 4),
            "seg_failed":    total_seg_failed,
            "reason_counts": reason_counts.to_dict(),
        },
        "shards": shard_summary,
    }
    (out_dir / "filter_summary.json").write_text(json.dumps(summary_data, indent=2))

    if len(df_cs) > 0:
        df_cs.to_csv(out_dir / "class_filter_stats.csv", index=False)
        class_agg.to_csv(out_dir / "class_filter_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"Bad IDs:  {len(df_all_bad):,} cells -> {bad_ids_path}")
    if args.dry_run:
        print("Dry run complete — filtered shards NOT written.")
    else:
        print(f"Filtered shards -> {out_dir}")
    print(f"Summary:  {out_dir / 'filter_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()