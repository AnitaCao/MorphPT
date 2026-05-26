#!/usr/bin/env python3
"""
Select good test patches for visualization.
============================================
Scans results/all_patches/<tissue>/predictions.parquet for every tissue,
computes per-patch quality scores, and outputs the top N candidate patches.

Scoring criteria (all configurable via CLI flags):
  - Accuracy (soft)    : fraction of cells correctly predicted by pred_soft
  - Class diversity    : number of distinct true-label classes present
  - Model confidence   : mean router_margin (high = confident routing)
  - Cell count         : used as a filter only, not in the score

Final score = w_acc * norm_acc + w_div * norm_div + w_conf * norm_conf

Output
------
  <out_dir>/patch_scores.csv   – full ranked table of all patches
  <out_dir>/top_patches.csv    – top-N selected patches with all metadata
  Console summary              – printed table for quick inspection

Usage
-----
  python scripts/select_viz_patches.py \\
      --results_dir results/all_patches \\
      --out_dir results/viz_patches \\
      --top_n 4 \\
      --min_cells 300 \\
      --min_classes 3 \\
      --w_acc 0.5 --w_div 0.3 --w_conf 0.2

  # Force one patch per tissue (good for diverse visualization):
  python scripts/select_viz_patches.py \\
      --results_dir results/all_patches \\
      --top_n 4 --one_per_tissue

  # Filter to specific tissues:
  python scripts/select_viz_patches.py \\
      --results_dir results/all_patches \\
      --tissues Xenium_V1_hColon_Cancer_Base Xenium_V1_hLymphNode_nondiseased \\
      --top_n 4

Note: accuracy is computed on pred_soft (soft-ensemble strategy). Cell count
is used only as a minimum-cell filter, not as part of the composite score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Per-patch scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_patch_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the predictions DataFrame for one tissue, return a per-patch
    summary with accuracy (soft pred), class diversity, confidence,
    and representative image paths (grabbed in this single pass).

    Accuracy is computed using pred_soft (soft-ensemble strategy) so it
    reflects the model's best multi-expert performance.
    """
    # Choose the prediction column: prefer pred_soft, fall back to pred_top1
    pred_col = "pred_soft" if "pred_soft" in df.columns else "pred_top1"

    has_img_2p5x = "img_path_2p5x" in df.columns
    has_img_10x  = "img_path_10x"  in df.columns
    has_xy       = "x_centroid" in df.columns and "y_centroid" in df.columns

    records = []
    for patch_id, grp in df.groupby("patch_id"):
        n = len(grp)

        # Accuracy using soft predictions
        correct = (grp[pred_col] == grp["label"]).sum()
        acc = correct / n

        # Balanced accuracy: mean per-class recall (also using pred_soft)
        classes = grp["label"].unique()
        recalls = []
        for c in classes:
            mask = grp["label"] == c
            rec = (grp.loc[mask, pred_col] == c).mean()
            recalls.append(float(rec))
        bal_acc = float(np.mean(recalls)) if recalls else 0.0

        # Class diversity
        n_classes = len(classes)
        n_coarse = grp["coarse_label"].nunique() if "coarse_label" in grp.columns else np.nan

        # Router confidence
        mean_router_margin = grp["router_margin"].mean() if "router_margin" in grp.columns else np.nan

        # Expert confidence (NaN for passthrough-only patches)
        mean_expert_margin = (
            grp["expert_margin_top1"].dropna().mean()
            if "expert_margin_top1" in grp.columns else np.nan
        )

        # Label distribution (for display)
        label_counts = grp["label"].value_counts().to_dict()

        # Representative image paths and spatial centre (grabbed here to avoid a second pass)
        first = grp.iloc[0]
        records.append({
            "patch_id": patch_id,
            "pred_col_used": pred_col,
            "n_cells": n,
            "accuracy": round(float(acc), 4),
            "balanced_acc": round(bal_acc, 4),
            "n_classes": n_classes,
            "n_coarse": n_coarse,
            "mean_router_margin": round(float(mean_router_margin), 4)
            if not np.isnan(mean_router_margin) else np.nan,
            "mean_expert_margin": round(float(mean_expert_margin), 4)
            if not np.isnan(mean_expert_margin) else np.nan,
            "label_counts": json.dumps(
                {k: int(v) for k, v in sorted(label_counts.items(), key=lambda x: -x[1])}
            ),
            # Image paths
            "example_img_path_2p5x": first["img_path_2p5x"] if has_img_2p5x else None,
            "example_img_path_10x":  first["img_path_10x"]  if has_img_10x  else None,
            "patch_x_center": float(grp["x_centroid"].mean()) if has_xy else np.nan,
            "patch_y_center": float(grp["y_centroid"].mean()) if has_xy else np.nan,
        })

    return pd.DataFrame(records)


def score_patches(
    df_stats: pd.DataFrame,
    w_acc: float,
    w_div: float,
    w_conf: float,
) -> pd.DataFrame:
    """
    Compute a composite score for each patch and add it as a column.
    All components are normalised to [0, 1] across the full table before
    weighting so the weights are directly interpretable as importance.

    Score = w_acc * norm(accuracy) + w_div * norm(n_classes) + w_conf * norm(router_margin)
    Cell count is NOT part of the score — it is a filter only.
    """
    df = df_stats.copy()

    def _norm(col):
        lo, hi = df[col].min(), df[col].max()
        if hi == lo:
            return pd.Series(np.ones(len(df)), index=df.index)
        return (df[col] - lo) / (hi - lo)

    df["norm_acc"]        = _norm("accuracy")
    df["norm_n_classes"]  = _norm("n_classes")
    df["norm_router_conf"] = _norm("mean_router_margin").fillna(0.0)

    df["score"] = (
        w_acc  * df["norm_acc"]
        + w_div  * df["norm_n_classes"]
        + w_conf * df["norm_router_conf"]
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Select best test patches for visualization from eval_all_test_patches results."
    )
    ap.add_argument("--results_dir", default="results/all_patches",
                    help="Root directory with per-tissue result folders.")
    ap.add_argument("--out_dir", default="results/viz_patches",
                    help="Where to write patch_scores.csv and top_patches.csv.")
    ap.add_argument("--top_n", type=int, default=4,
                    help="Number of top patches to select.")
    ap.add_argument("--min_cells", type=int, default=200,
                    help="Minimum number of cells a patch must have.")
    ap.add_argument("--min_classes", type=int, default=2,
                    help="Minimum number of distinct fine classes in a patch.")
    ap.add_argument("--one_per_tissue", action="store_true",
                    help="At most one patch per tissue (encourages diversity).")
    ap.add_argument("--tissues", nargs="*", default=None,
                    help="Restrict to these tissue names (default: all).")
    # Scoring weights
    ap.add_argument("--w_acc",  type=float, default=0.5,
                    help="Weight for soft accuracy component.")
    ap.add_argument("--w_div",  type=float, default=0.3,
                    help="Weight for class diversity component.")
    ap.add_argument("--w_conf", type=float, default=0.2,
                    help="Weight for router confidence component.")
    ap.add_argument("--stats_cache", default=None,
                    help="Path to a cached patch_stats.csv. If it exists the slow "
                         "parquet scan is skipped; if it does not exist the cache is "
                         "created after the scan. Useful for quickly tweaking weights.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover tissue result directories ──────────────────────────────────
    tissue_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "predictions.parquet").exists()
    ])
    if args.tissues:
        tissue_dirs = [d for d in tissue_dirs if d.name in set(args.tissues)]

    if not tissue_dirs:
        print(f"[ERROR] No valid tissue directories found under {results_dir}")
        return

    print(f"Found {len(tissue_dirs)} tissue(s) with predictions.parquet\n")

    # ── Read tissue-level summaries ──────────────────────────────────────────
    tissue_summary = {}
    for td in tissue_dirs:
        sj = td / "summary.json"
        if sj.exists():
            with open(sj) as f:
                s = json.load(f)
            tissue_summary[td.name] = {
                "top1_macro_f1":   s.get("strategies", {}).get("top1", {}).get("macro_f1", np.nan),
                "top1_bal_acc":    s.get("strategies", {}).get("top1", {}).get("balanced_acc", np.nan),
                "router_acc":      s.get("router_coarse_acc", np.nan),
            }

    # ── Per-tissue, per-patch stats (or load from cache) ────────────────────
    cache_path = Path(args.stats_cache) if args.stats_cache else None

    if cache_path and cache_path.exists():
        print(f"Loading stats from cache: {cache_path}")
        df_all = pd.read_csv(cache_path)
        # Re-apply tissue filter if needed
        if args.tissues:
            df_all = df_all[df_all["tissue"].isin(set(args.tissues))].copy()
    else:
        all_patch_rows = []

        for td in tissue_dirs:
            tissue = td.name
            parquet_path = td / "predictions.parquet"
            print(f"  Processing: {tissue}")

            try:
                df = pd.read_parquet(parquet_path)
            except Exception as e:
                print(f"    [WARN] Could not read parquet: {e}")
                continue

            # Validate required columns (pred_soft preferred; pred_top1 as fallback)
            required = {"patch_id", "label", "router_margin"}
            if "pred_soft" not in df.columns and "pred_top1" not in df.columns:
                required.add("pred_soft")  # force the error message
            missing = required - set(df.columns)
            if missing:
                print(f"    [WARN] Missing columns {missing}, skipping.")
                continue

            stats = compute_per_patch_stats(df)
            stats.insert(0, "tissue", tissue)
            all_patch_rows.append(stats)

        if not all_patch_rows:
            print("[ERROR] No patch stats could be computed.")
            return

        df_all = pd.concat(all_patch_rows, ignore_index=True)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_all.to_csv(cache_path, index=False)
            print(f"\nStats cached to: {cache_path}")

    # ── Filter ───────────────────────────────────────────────────────────────
    df_filtered = df_all[
        (df_all["n_cells"]   >= args.min_cells) &
        (df_all["n_classes"] >= args.min_classes)
    ].copy()

    print(f"\nPatches after filtering "
          f"(min_cells={args.min_cells}, min_classes={args.min_classes}): "
          f"{len(df_filtered)} / {len(df_all)}\n")

    if df_filtered.empty:
        print("[WARN] All patches filtered out. Relaxing constraints...")
        df_filtered = df_all.copy()

    # ── Score ─────────────────────────────────────────────────────────────────
    df_scored = score_patches(df_filtered, args.w_acc, args.w_div, args.w_conf)
    df_scored = df_scored.sort_values("score", ascending=False).reset_index(drop=True)

    # ── Select top N ─────────────────────────────────────────────────────────
    if args.one_per_tissue:
        seen_tissues = set()
        selected_rows = []
        for _, row in df_scored.iterrows():
            if row["tissue"] not in seen_tissues:
                selected_rows.append(row)
                seen_tissues.add(row["tissue"])
                if len(selected_rows) >= args.top_n:
                    break
        df_top = pd.DataFrame(selected_rows).reset_index(drop=True)
    else:
        df_top = df_scored.head(args.top_n).reset_index(drop=True)

    # ── Attach tissue-level metrics ───────────────────────────────────────────
    df_top["tissue_macro_f1"] = df_top["tissue"].map(
        lambda t: tissue_summary.get(t, {}).get("top1_macro_f1", np.nan))
    df_top["tissue_bal_acc"] = df_top["tissue"].map(
        lambda t: tissue_summary.get(t, {}).get("top1_bal_acc", np.nan))

    # Image paths are already in df_all (grabbed during the single parquet scan)

    # ── Save ──────────────────────────────────────────────────────────────────
    full_score_path = out_dir / "patch_scores.csv"
    top_path        = out_dir / "top_patches.csv"

    display_cols = ["tissue", "patch_id", "score", "accuracy", "balanced_acc",
                    "n_classes", "n_coarse", "n_cells",
                    "mean_router_margin", "mean_expert_margin", "pred_col_used"]
    df_scored[display_cols].to_csv(full_score_path, index=False)

    df_top.to_csv(top_path, index=False)

    # ── Pretty print ──────────────────────────────────────────────────────────
    print("=" * 100)
    print(f"  TOP {args.top_n} PATCHES FOR VISUALIZATION")
    print(f"  Weights: acc={args.w_acc}  diversity={args.w_div}  confidence={args.w_conf}")
    if args.one_per_tissue:
        print("  Mode: one patch per tissue")
    print("=" * 100)

    pred_col_label = df_top["pred_col_used"].iloc[0] if "pred_col_used" in df_top.columns else "pred_soft"
    hdr = (f"  {'#':>3}  {'Tissue':<50} {'PatchID':>8}  "
           f"{'Score':>6}  {'Acc('+pred_col_label+')':>14}  {'Classes':>7}  {'Cells':>7}  {'RouterM':>8}")
    print(hdr)
    print("  " + "-" * 105)

    for rank, (_, row) in enumerate(df_top.iterrows(), 1):
        print(
            f"  {rank:>3}  {row['tissue']:<50} {int(row['patch_id']):>8}  "
            f"{row['score']:>6.4f}  {row['accuracy']:>14.4f}  "
            f"{int(row['n_classes']):>7}  {int(row['n_cells']):>7,}  "
            f"{row['mean_router_margin']:>8.4f}"
        )
        # Print class breakdown
        try:
            lc = json.loads(row["label_counts"])
            breakdown = "  ".join(f"{k}: {v}" for k, v in list(lc.items())[:6])
            print(f"       Classes: {breakdown}")
        except Exception:
            pass

    print("=" * 100)
    print(f"\nSaved:")
    print(f"  Full ranking : {full_score_path}")
    print(f"  Top patches  : {top_path}")


if __name__ == "__main__":
    main()
