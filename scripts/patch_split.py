#!/usr/bin/env python3
"""
Patch-based train/test split for CellPT MoE
============================================

Single pass per slide, minimal memory.

Outputs:
  router_shards/<slide>.parquet         all 7 coarse classes, capped per coarse per slide
  expert_<CoarseGroup>/<slide>.parquet  only fine classes in that group, capped per fine per slide
  test_shards/<slide>.parquet           all classes, natural distribution

Only coarse groups with 2+ fine classes get an expert directory:
  expert_Cancer/, expert_Lymphoid/, expert_Neuroglial/,
  expert_Tissue_Structural/, expert_Vascular/
  (Stromal and Stem_Progenitor have 1 fine class each → no expert needed)

Example:
  python patch_split_final.py \
    --data_dir /hpc/group/jilab/rz179/MorphPT_MOE/prepared/shards_multiview_parquet \
    --output_dir /hpc/group/jilab/rz179/MorphPT_MOE/prepared/splits_v3 \
    --seed 1337 \
    --per_slide_cap_coarse 20000 \
    --per_slide_cap_fine 10000
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================
# Stable hash
# =============================================================

def stable_hash(s: str) -> int:
    import hashlib
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (10**9)


# =============================================================
# 23 fine → 7 coarse (fixed)
# =============================================================

def build_fine_to_coarse() -> Dict[str, str]:
    cancer = {
        "Breast cancer cells", "Ovary cancer cells", "Colon cancer cells",
        "Skin cancer cells", "Lung cancer cells", "Pancreas cancer cells",
        "Liver cancer cells",
    }
    neuro = {"Microglia", "Oligodendrocytes", "Astrocytes", "Neurons"}
    stromal = {"Stromal cells"}
    stem = {"Stem and progenitor cells"}
    lymphoid = {"NK cells", "B cells", "T cells"}
    tissue_structural = {"Adipocytes", "Epithelial cells", "Pericytes", "Fibroblasts"}
    vascular = {"Endothelial cells", "Myeloid cells", "Smooth muscle cells"}

    m: Dict[str, str] = {}
    for k in neuro:              m[k] = "Neuroglial"
    for k in stromal:            m[k] = "Stromal"
    for k in stem:               m[k] = "Stem_Progenitor"
    for k in lymphoid:           m[k] = "Lymphoid"
    for k in tissue_structural:  m[k] = "Tissue_Structural"
    for k in vascular:           m[k] = "Vascular"
    for k in cancer:             m[k] = "Cancer"
    return m


def build_coarse_to_fines(ftc: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert: coarse → list of fine classes."""
    c2f: Dict[str, List[str]] = defaultdict(list)
    for fine, coarse in ftc.items():
        c2f[coarse].append(fine)
    return {k: sorted(v) for k, v in c2f.items()}


def coarse_to_id_from_map(ftc: Dict[str, str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(sorted(set(ftc.values())))}


# =============================================================
# Discover slides
# =============================================================

def discover_parquets(data_dir: str) -> List[Dict[str, str]]:
    pattern = os.path.join(data_dir, "tissue=*", "part.parquet")
    files = sorted(glob(pattern))
    return [{"path": f, "tissue": Path(f).parent.name.replace("tissue=", "")} for f in files]


# =============================================================
# Patch grid + farthest-point test selection
# =============================================================

def assign_patches(
    df: pd.DataFrame, x_col: str, y_col: str,
    grid_rows: int, grid_cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin: xmax = xmin + 1.0
    if ymax <= ymin: ymax = ymin + 1.0

    x_edges = np.linspace(xmin, xmax, grid_cols + 1)
    y_edges = np.linspace(ymin, ymax, grid_rows + 1)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, grid_cols - 1).astype(np.int16)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, grid_rows - 1).astype(np.int16)
    pid = (y_idx * grid_cols + x_idx).astype(np.int16)
    counts = np.bincount(pid.astype(np.int64), minlength=grid_rows * grid_cols)
    return pid, x_edges, y_edges, counts


def manhattan(a: int, b: int, gc: int) -> int:
    ra, ca = divmod(a, gc)
    rb, cb = divmod(b, gc)
    return abs(ra - rb) + abs(ca - cb)


def farthest_point_sample(valid: List[int], gc: int, k: int, rng) -> List[int]:
    if len(valid) <= k:
        return list(valid)
    seed = int(rng.choice(valid))
    sel = [seed]
    rem = set(valid) - {seed}
    while len(sel) < k and rem:
        best, best_d = None, -1
        for p in rem:
            d = min(manhattan(p, s, gc) for s in sel)
            if d > best_d or (d == best_d and rng.random() < 0.5):
                best_d, best = d, p
        if best is None: break
        sel.append(best)
        rem.discard(best)
    return sel


def auto_min_cells(n: int) -> int:
    return int(max(2000, min(20000, round(n / 200))))


def select_test_patches(
    counts: np.ndarray, n_test: int,
    gr: int, gc: int, rng, min_cells: int, tries: int = 50,
) -> Tuple[List[int], List[int]]:
    n_patches = gr * gc
    all_ids = list(range(n_patches))
    valid = [p for p in all_ids if counts[p] >= min_cells]

    if len(valid) < n_test:
        order = sorted(all_ids, key=lambda p: int(counts[p]), reverse=True)
        valid = [p for p in order if counts[p] > 0][:max(n_test, 10)]

    if len(valid) <= n_test:
        test = sorted(valid)
        return sorted(set(all_ids) - set(test)), test

    best, best_md = None, -1
    for _ in range(tries):
        cand = farthest_point_sample(valid, gc, n_test, rng)
        md = min(manhattan(a, b, gc) for i, a in enumerate(cand) for j, b in enumerate(cand) if j > i)
        if md > best_md:
            best_md, best = md, cand

    test = sorted(best if best else farthest_point_sample(valid, gc, n_test, rng))
    return sorted(set(all_ids) - set(test)), test


# =============================================================
# Groupwise capping
# =============================================================

def cap_groupwise(df: pd.DataFrame, col: str, cap: int, rng) -> pd.DataFrame:
    """Simple cap: randomly sample up to `cap` per group. Used for experts."""
    if cap <= 0:
        return df.reset_index(drop=True)
    parts = []
    for _, g in df.groupby(col, sort=False):
        if len(g) <= cap:
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=cap, replace=False)
            parts.append(g.loc[idx])
    return pd.concat(parts, ignore_index=True)


def cap_coarse_stratified(
    df: pd.DataFrame,
    coarse_col: str,
    fine_col: str,
    cap: int,
    min_quota: int,
    rng,
) -> pd.DataFrame:
    """
    Cap per coarse group, but protect small fine classes within each group.

    For each coarse group exceeding cap:
      1. Give each fine class min(n_i, min_quota) cells first
      2. Distribute remaining quota proportionally to leftover cells
      3. Total never exceeds cap
    """
    if cap <= 0:
        return df.reset_index(drop=True)

    parts = []
    for _, g_coarse in df.groupby(coarse_col, sort=False):
        if len(g_coarse) <= cap:
            parts.append(g_coarse)
            continue

        # Split by fine class
        fine_groups = {f: g for f, g in g_coarse.groupby(fine_col, sort=False)}

        # Phase 1: minimum quota per fine
        selected = []
        quota_used = 0
        leftover = {}  # fine → remaining rows after min quota

        for fine, g in fine_groups.items():
            q = min(len(g), min_quota)
            if len(g) <= q:
                selected.append(g)
            else:
                idx = rng.choice(g.index.to_numpy(), size=q, replace=False)
                selected.append(g.loc[idx])
                leftover[fine] = g.index.difference(idx)
            quota_used += q

        remaining = cap - quota_used
        if remaining <= 0:
            # Min quotas already exceed cap → keep min quotas, truncate if needed
            combined = pd.concat(selected, ignore_index=True)
            if len(combined) > cap:
                idx = rng.choice(combined.index.to_numpy(), size=cap, replace=False)
                combined = combined.loc[idx].reset_index(drop=True)
            parts.append(combined)
            continue

        # Phase 2: distribute remaining proportionally
        if leftover:
            pool_sizes = {f: len(idx) for f, idx in leftover.items()}
            total_pool = sum(pool_sizes.values())

            # Proportional allocation with floor
            alloc = {}
            alloc_sum = 0
            for f in leftover:
                a = int(remaining * pool_sizes[f] / total_pool)
                alloc[f] = min(a, pool_sizes[f])
                alloc_sum += alloc[f]

            # Distribute remainder 1-by-1 to largest pools
            shortfall = remaining - alloc_sum
            if shortfall > 0:
                by_pool = sorted(leftover.keys(),
                                 key=lambda f: pool_sizes[f] - alloc[f],
                                 reverse=True)
                for f in by_pool:
                    give = min(shortfall, pool_sizes[f] - alloc[f])
                    alloc[f] += give
                    shortfall -= give
                    if shortfall <= 0:
                        break

            # Sample from leftover
            for f, a in alloc.items():
                if a > 0:
                    idx = rng.choice(
                        leftover[f].to_numpy(), size=a, replace=False
                    )
                    selected.append(fine_groups[f].loc[idx])

        parts.append(pd.concat(selected, ignore_index=True))

    return pd.concat(parts, ignore_index=True)


# =============================================================
# Main
# =============================================================

def main():
    ap = argparse.ArgumentParser(description="Patch split → router + per-expert + test shards")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--grid", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--n_test_patches", type=int, default=5)
    ap.add_argument("--x_col", default="x_centroid")
    ap.add_argument("--y_col", default="y_centroid")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--per_slide_cap_coarse", type=int, default=20000,
                    help="Router: cap per coarse_id per slide. 0=disable.")
    ap.add_argument("--min_quota_per_fine", type=int, default=1000,
                    help="Router: minimum cells per fine class within each coarse cap.")
    ap.add_argument("--per_slide_cap_fine", type=int, default=10000,
                    help="Expert: cap per fine label per slide. 0=disable.")
    ap.add_argument("--keep_cols", nargs="*",
                    default=["cell_id", "x_centroid", "y_centroid", "label",
                             "img_path_2p5x", "img_path_10x",
                             "meta_csv_2p5x", "meta_csv_10x", "spatial_csv"],
                    help="Columns to keep in output shards (beyond computed ones).")
    ap.add_argument("--min_cells_mode", choices=["auto", "fixed"], default="auto")
    ap.add_argument("--min_cells_fixed", type=int, default=2000)
    args = ap.parse_args()

    gr, gc = args.grid
    n_patches = gr * gc
    assert args.n_test_patches < n_patches

    out = Path(args.output_dir)

    ftc = build_fine_to_coarse()
    c2id = coarse_to_id_from_map(ftc)
    c2f = build_coarse_to_fines(ftc)
    valid_fine = set(ftc.keys())

    # Identify which coarse groups need an expert (2+ fine classes)
    expert_groups = {c: fines for c, fines in c2f.items() if len(fines) >= 2}
    singleton_groups = {c: fines[0] for c, fines in c2f.items() if len(fines) == 1}

    # Create output directories
    (out / "router_shards").mkdir(parents=True, exist_ok=True)
    (out / "test_shards").mkdir(parents=True, exist_ok=True)
    for eg in expert_groups:
        (out / f"expert_{eg}" / "shards").mkdir(parents=True, exist_ok=True)

    # Save mappings
    (out / "fine_to_coarse.json").write_text(json.dumps(ftc, indent=2))
    (out / "coarse_to_id.json").write_text(json.dumps(c2id, indent=2))
    (out / "expert_groups.json").write_text(json.dumps({
        "experts_needed": {k: v for k, v in expert_groups.items()},
        "singletons_no_expert": singleton_groups,
    }, indent=2))

    slides = discover_parquets(args.data_dir)
    if not slides:
        raise RuntimeError(f"No slides found under {args.data_dir}")

    split_cols = [args.x_col, args.y_col, args.label_col]

    print(f"Slides: {len(slides)}")
    print(f"Fine classes: {len(valid_fine)}  |  Coarse groups: {len(c2id)}")
    print(f"Expert groups ({len(expert_groups)}): {list(expert_groups.keys())}")
    print(f"Singletons (no expert): {singleton_groups}")
    print(f"Router cap: {args.per_slide_cap_coarse}/coarse/slide (min_quota={args.min_quota_per_fine}/fine)")
    print(f"Expert cap: {args.per_slide_cap_fine}/fine/slide")
    print()

    manifest = {
        "seed": args.seed,
        "grid": [gr, gc],
        "n_test_patches": args.n_test_patches,
        "per_slide_cap_coarse": args.per_slide_cap_coarse,
        "min_quota_per_fine": args.min_quota_per_fine,
        "per_slide_cap_fine": args.per_slide_cap_fine,
        "fine_classes": sorted(valid_fine),
        "coarse_classes": sorted(c2id.keys()),
        "expert_groups": {k: v for k, v in expert_groups.items()},
        "singleton_groups": singleton_groups,
        "slides": [],
    }

    # Global counters
    g_router_coarse = defaultdict(int)
    g_router_fine = defaultdict(int)
    g_expert_fine = defaultdict(lambda: defaultdict(int))  # expert_group → fine → count
    g_test_fine = defaultdict(int)
    g_test_coarse = defaultdict(int)
    total_router = total_test = 0
    total_expert = defaultdict(int)  # expert_group → total count

    # All slides share the same schema (verified)
    ref_names = set(pq.read_schema(slides[0]["path"]).names)
    phase2_cols = sorted(set(args.keep_cols) & ref_names)
    missing_cols = set(args.keep_cols) - ref_names
    if missing_cols:
        print(f"  Note: requested columns not in parquet (ignored): {sorted(missing_cols)}")
    print(f"  Output columns: {phase2_cols} + [tissue, coarse_label, coarse_id, patch_id]")
    print()

    manifest["output_columns"] = phase2_cols

    # ── Single pass ──────────────────────────────────────────
    for s in slides:
        tissue = s["tissue"]
        p = Path(s["path"])

        # Phase 1: light read
        try:
            df_l = pd.read_parquet(p, columns=split_cols)
        except Exception as e:
            manifest["slides"].append({"tissue": tissue, "status": "error", "reason": str(e)})
            print(f"[err] {tissue}: {e}")
            continue

        keep = df_l[args.label_col].isin(valid_fine)
        if keep.sum() == 0:
            manifest["slides"].append({"tissue": tissue, "status": "skip"})
            print(f"[skip] {tissue}: no valid classes")
            continue

        df_l = df_l[keep].copy()
        df_l["coarse_label"] = df_l[args.label_col].map(ftc)
        df_l["coarse_id"] = df_l["coarse_label"].map(c2id).astype(np.int16)

        # Patch assignment
        pid, x_edges, y_edges, pcounts = assign_patches(df_l, args.x_col, args.y_col, gr, gc)
        df_l["patch_id"] = pid

        mc = auto_min_cells(len(df_l)) if args.min_cells_mode == "auto" else args.min_cells_fixed

        ss = args.seed + stable_hash(tissue)
        rng = np.random.default_rng(ss)

        train_patches, test_patches = select_test_patches(pcounts, args.n_test_patches, gr, gc, rng, mc)
        is_test = df_l["patch_id"].isin(set(test_patches))

        # Phase 2: read only columns needed for output shards
        df_f = pd.read_parquet(p, columns=phase2_cols)
        df_f = df_f[keep.values].copy()
        df_f["tissue"] = tissue
        df_f["coarse_label"] = df_l["coarse_label"].values
        df_f["coarse_id"] = df_l["coarse_id"].values
        df_f["patch_id"] = pid
        del df_l

        df_train = df_f[~is_test.values].reset_index(drop=True)
        df_test = df_f[is_test.values].reset_index(drop=True)
        del df_f

        # ── Router shard: stratified cap per coarse (protects small fine classes) ──
        rng_r = np.random.default_rng(ss + 101)
        df_router = cap_coarse_stratified(
            df_train, "coarse_id", args.label_col,
            args.per_slide_cap_coarse, args.min_quota_per_fine, rng_r
        )
        df_router.to_parquet(out / "router_shards" / f"{tissue}.parquet", index=False)

        for c, n in df_router["coarse_label"].value_counts().items():
            g_router_coarse[c] += n
        for c, n in df_router[args.label_col].value_counts().items():
            g_router_fine[c] += n
        nr = len(df_router)
        total_router += nr
        del df_router

        # ── Expert shards: one per coarse group, cap per fine ──
        slide_expert_info = {}
        for eg, fines in expert_groups.items():
            fine_set = set(fines)
            df_eg = df_train[df_train[args.label_col].isin(fine_set)]
            if len(df_eg) == 0:
                continue

            # Each expert group gets its own RNG offset
            eg_offset = stable_hash(eg) % 1000
            rng_e = np.random.default_rng(ss + 202 + eg_offset)
            df_eg_capped = cap_groupwise(df_eg, args.label_col, args.per_slide_cap_fine, rng_e)
            df_eg_capped.to_parquet(out / f"expert_{eg}" / "shards" / f"{tissue}.parquet", index=False)

            for c, n in df_eg_capped[args.label_col].value_counts().items():
                g_expert_fine[eg][c] += n
            ne = len(df_eg_capped)
            total_expert[eg] += ne
            slide_expert_info[eg] = ne
            del df_eg_capped

        del df_train

        # ── Test shard: natural distribution ──
        df_test.to_parquet(out / "test_shards" / f"{tissue}.parquet", index=False)

        for c, n in df_test[args.label_col].value_counts().items():
            g_test_fine[c] += n
        for c, n in df_test["coarse_label"].value_counts().items():
            g_test_coarse[c] += n
        nt = len(df_test)
        total_test += nt
        del df_test

        manifest["slides"].append({
            "tissue": tissue,
            "status": "ok",
            "n_router": nr,
            "n_test": nt,
            "n_experts": slide_expert_info,
            "min_cells": int(mc),
            "slide_seed": int(ss),
            "train_patches": train_patches,
            "test_patches": test_patches,
            "patch_counts": {str(i): int(pcounts[i]) for i in range(n_patches)},
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
        })

        expert_str = "  ".join(f"{k}={v:,}" for k, v in slide_expert_info.items())
        print(f"[ok] {tissue:<50} router={nr:>7,}  test={nt:>7,}  {expert_str}")

    # ── Distribution report ──────────────────────────────────
    n_ok = sum(1 for x in manifest["slides"] if x.get("status") == "ok")

    print(f"\n{'═'*80}")
    print("ROUTER (coarse distribution)")
    print(f"{'═'*80}")
    print(f"  {'Coarse':<22} {'Router':>10} {'Test':>10}")
    print(f"  {'─'*44}")
    for c in sorted(c2id.keys()):
        print(f"  {c:<22} {g_router_coarse.get(c,0):>10,} {g_test_coarse.get(c,0):>10,}")
    print(f"  {'─'*44}")
    print(f"  {'TOTAL':<22} {total_router:>10,} {total_test:>10,}")
    if g_router_coarse:
        vals = list(g_router_coarse.values())
        print(f"  Max/min: {max(vals):,} / {min(vals):,} = {max(vals)/max(min(vals),1):.1f}x")

    print(f"\n{'═'*80}")
    print("ROUTER (fine within coarse)")
    print(f"{'═'*80}")
    for coarse in sorted(c2id.keys()):
        fines = c2f.get(coarse, [])
        rc = g_router_coarse.get(coarse, 0)
        print(f"\n  {coarse} ({rc:,})")
        print(f"  {'Fine':<30} {'Count':>10} {'%':>7}")
        print(f"  {'─'*49}")
        for f in sorted(fines):
            n = g_router_fine.get(f, 0)
            pct = n / rc * 100 if rc > 0 else 0
            print(f"  {f:<30} {n:>10,} {pct:>6.1f}%")

    for eg in sorted(expert_groups):
        fines = expert_groups[eg]
        print(f"\n{'═'*80}")
        print(f"EXPERT: {eg}  ({len(fines)} fine classes)")
        print(f"{'═'*80}")
        print(f"  {'Fine Class':<30} {'Train':>10} {'Test':>10}")
        print(f"  {'─'*52}")
        eg_total = 0
        for f in sorted(fines):
            ne = g_expert_fine[eg].get(f, 0)
            nt = g_test_fine.get(f, 0)
            eg_total += ne
            print(f"  {f:<30} {ne:>10,} {nt:>10,}")
        print(f"  {'─'*52}")
        print(f"  {'TOTAL':<30} {eg_total:>10,}")

    # ── Save manifest ────────────────────────────────────────
    manifest["counts"] = {
        "router_total": total_router,
        "test_total": total_test,
        "experts": {k: int(v) for k, v in total_expert.items()},
        "n_slides_ok": n_ok,
    }
    manifest["distribution"] = {
        "router_coarse": dict(g_router_coarse),
        "router_fine": dict(g_router_fine),
        "expert_fine": {k: dict(v) for k, v in g_expert_fine.items()},
        "test_fine": dict(g_test_fine),
        "test_coarse": dict(g_test_coarse),
    }

    mp = out / f"split_manifest_seed{args.seed}.json"
    mp.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'═'*80}")
    print(f"DONE — {n_ok}/{len(slides)} slides")
    print(f"{'═'*80}")
    print(f"  Router:  {total_router:>10,}  ({out / 'router_shards'})")
    for eg in sorted(expert_groups):
        print(f"  Expert_{eg}: {total_expert[eg]:>10,}  ({out / f'expert_{eg}' / 'shards'})")
    print(f"  Test:    {total_test:>10,}  ({out / 'test_shards'})")
    print(f"  Manifest: {mp}")


if __name__ == "__main__":
    main()