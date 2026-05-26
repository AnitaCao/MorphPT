#!/usr/bin/env python3
"""
Patch-based train/test split for CellPT MoE
============================================

Hardcoded per-fine-class global targets.
rate = min(1.0, target / raw_train_total) per fine class.
Each slide sampled at that rate → slide diversity preserved proportionally.

Two-pass:
  Pass 1 (light): patches, test selection, count raw train per-fine
  Pass 2 (full): sample by rates → save shards

Outputs:
  router_shards/, expert_<Group>/shards/, test_shards/
  fine_to_coarse.json, coarse_to_id.json, expert_groups.json
  split_manifest_seed<seed>.json
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
# Hardcoded targets
# =============================================================

FINE_TARGETS = {
    # Cancer: 32K each, Liver keep all → ~201K
    "Breast cancer cells":       32_000,
    "Ovary cancer cells":        32_000,
    "Colon cancer cells":        32_000,
    "Skin cancer cells":         32_000,
    "Lung cancer cells":         32_000,
    "Pancreas cancer cells":     32_000,
    "Liver cancer cells":         8_400,
    # Lymphoid: 67K each → ~201K
    "T cells":                   67_000,
    "B cells":                   67_000,
    "NK cells":                  67_000,
    # Tissue_Structural: 85K large, small keep all → ~201K
    "Epithelial cells":          85_000,
    "Fibroblasts":               85_000,
    "Pericytes":                 23_200,
    "Adipocytes":                 8_400,
    # Vascular: 67K each → ~201K
    "Myeloid cells":             67_000,
    "Endothelial cells":         67_000,
    "Smooth muscle cells":       67_000,
    # Neuroglial: keep all → ~48K
    "Microglia":                  6_200,
    "Oligodendrocytes":          25_000,
    "Astrocytes":                 6_000,
    "Neurons":                   11_600,
    # Singletons
    "Stem and progenitor cells":100_000,
    "Stromal cells":             48_000,
}


# =============================================================
# Stable hash
# =============================================================

def stable_hash(s: str) -> int:
    import hashlib
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (10**9)


# =============================================================
# 23 fine → 7 coarse
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
# Groupwise capping (for experts)
# =============================================================

def cap_groupwise(df: pd.DataFrame, col: str, cap: int, rng) -> pd.DataFrame:
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


# =============================================================
# Sample by per-fine rates (for router)
# =============================================================

def sample_by_rates(df: pd.DataFrame, label_col: str,
                    fine_rates: Dict[str, float], rng) -> pd.DataFrame:
    parts = []
    for label, g in df.groupby(label_col, sort=False):
        rate = fine_rates.get(label, 1.0)
        n_take = int(round(len(g) * rate))
        n_take = max(0, min(n_take, len(g)))
        if n_take == 0:
            continue
        if n_take >= len(g):
            parts.append(g)
        else:
            idx = rng.choice(g.index.to_numpy(), size=n_take, replace=False)
            parts.append(g.loc[idx])
    if not parts:
        return pd.DataFrame(columns=df.columns)
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
    ap.add_argument("--per_slide_cap_fine", type=int, default=10000,
                    help="Expert: cap per fine label per slide.")
    ap.add_argument("--keep_cols", nargs="*",
                    default=["cell_id", "x_centroid", "y_centroid", "label",
                             "img_path_2p5x", "img_path_10x"])
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

    assert valid_fine == set(FINE_TARGETS.keys()), \
        f"Mismatch: {valid_fine.symmetric_difference(set(FINE_TARGETS.keys()))}"

    expert_groups = {c: fines for c, fines in c2f.items() if len(fines) >= 2}
    singleton_groups = {c: fines[0] for c, fines in c2f.items() if len(fines) == 1}

    (out / "router_shards").mkdir(parents=True, exist_ok=True)
    (out / "test_shards").mkdir(parents=True, exist_ok=True)
    for eg in expert_groups:
        (out / f"expert_{eg}" / "shards").mkdir(parents=True, exist_ok=True)

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
    ref_names = set(pq.read_schema(slides[0]["path"]).names)
    phase2_cols = sorted(set(args.keep_cols) & ref_names)

    print(f"Slides: {len(slides)}")
    print(f"Fine classes: {len(valid_fine)}  |  Coarse groups: {len(c2id)}")
    print(f"Expert groups ({len(expert_groups)}): {list(expert_groups.keys())}")
    print(f"Expert cap: {args.per_slide_cap_fine}/fine/slide")
    print(f"Output columns: {phase2_cols} + [tissue, coarse_label, coarse_id, patch_id]")
    print()

    # ══════════════════════════════════════════════════════════
    # PASS 1: Light scan — patches, test selection, raw train counts
    # ══════════════════════════════════════════════════════════
    print(f"{'═'*80}")
    print("PASS 1: Light scan")
    print(f"{'═'*80}")

    slide_info = []
    global_fine_train = defaultdict(int)

    for s in slides:
        tissue = s["tissue"]
        p = Path(s["path"])

        try:
            df_l = pd.read_parquet(p, columns=split_cols)
        except Exception as e:
            print(f"  [err] {tissue}: {e}")
            slide_info.append({"tissue": tissue, "status": "error", "reason": str(e)})
            continue

        keep = df_l[args.label_col].isin(valid_fine).to_numpy()
        if keep.sum() == 0:
            print(f"  [skip] {tissue}: no valid classes")
            slide_info.append({"tissue": tissue, "status": "skip"})
            continue

        df_l = df_l[keep].copy()
        df_l["coarse_label"] = df_l[args.label_col].map(ftc)
        df_l["coarse_id"] = df_l["coarse_label"].map(c2id).astype(np.int16)

        pid, x_edges, y_edges, pcounts = assign_patches(df_l, args.x_col, args.y_col, gr, gc)
        mc = auto_min_cells(len(df_l)) if args.min_cells_mode == "auto" else args.min_cells_fixed
        ss = args.seed + stable_hash(tissue)
        rng = np.random.default_rng(ss)

        train_patches, test_patches = select_test_patches(pcounts, args.n_test_patches, gr, gc, rng, mc)
        is_test = np.isin(pid, test_patches)

        for f, n in df_l.loc[~is_test, args.label_col].value_counts().items():
            global_fine_train[f] += n

        slide_info.append({
            "tissue": tissue,
            "status": "ok",
            "path": str(p),
            "keep": keep,
            "is_test": is_test,
            "pid": pid,
            "slide_seed": ss,
            "min_cells": mc,
            "train_patches": train_patches,
            "test_patches": test_patches,
            "pcounts": pcounts,
            "x_edges": x_edges,
            "y_edges": y_edges,
        })

        n_train = int((~is_test).sum())
        n_test = int(is_test.sum())
        print(f"  {tissue:<55} train={n_train:>9,}  test={n_test:>9,}")

    ok_slides = [s for s in slide_info if s.get("status") == "ok"]
    if not ok_slides:
        raise RuntimeError("No slides processed.")

    # ══════════════════════════════════════════════════════════
    # Compute sampling rates from hardcoded targets
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print("Router sampling plan")
    print(f"{'═'*80}")
    print(f"  {'Fine Class':<30} {'Raw':>10} {'Target':>10} {'Rate':>8}")
    print(f"  {'─'*62}")

    fine_rates = {}
    expected_total = 0
    for f in sorted(FINE_TARGETS.keys()):
        raw = global_fine_train.get(f, 0)
        tgt = FINE_TARGETS[f]
        actual_tgt = min(tgt, raw)  # can't exceed what we have
        rate = min(1.0, tgt / raw) if raw > 0 else 0.0
        fine_rates[f] = rate
        expected_total += actual_tgt
        flag = "keep" if rate >= 1.0 else f"{rate:.1%}"
        print(f"  {f:<30} {raw:>10,} {actual_tgt:>10,}  {flag:>8}")

    print(f"  {'─'*62}")
    print(f"  {'TOTAL':<30} {sum(global_fine_train.values()):>10,} {expected_total:>10,}")

    # Coarse summary
    coarse_expected = defaultdict(int)
    for f, tgt in FINE_TARGETS.items():
        raw = global_fine_train.get(f, 0)
        coarse_expected[ftc[f]] += min(tgt, raw)
    print(f"\n  Expected coarse distribution:")
    for c in sorted(coarse_expected.keys()):
        print(f"    {c:<22} {coarse_expected[c]:>10,}")
    vals = list(coarse_expected.values())
    print(f"    Ratio: {max(vals):,} / {min(vals):,} = {max(vals)/max(min(vals),1):.1f}x")
    print(f"    Expected total: {expected_total:,}")

    # ══════════════════════════════════════════════════════════
    # PASS 2: Save shards
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print("PASS 2: Save shards")
    print(f"{'═'*80}")

    manifest = {
        "seed": args.seed,
        "grid": [gr, gc],
        "n_test_patches": args.n_test_patches,
        "per_slide_cap_fine": args.per_slide_cap_fine,
        "fine_targets": FINE_TARGETS,
        "fine_rates": {f: round(r, 6) for f, r in fine_rates.items()},
        "fine_classes": sorted(valid_fine),
        "coarse_classes": sorted(c2id.keys()),
        "expert_groups": {k: v for k, v in expert_groups.items()},
        "singleton_groups": singleton_groups,
        "output_columns": phase2_cols,
        "slides": [],
    }

    g_router_coarse = defaultdict(int)
    g_router_fine = defaultdict(int)
    g_expert_fine = defaultdict(lambda: defaultdict(int))
    g_test_fine = defaultdict(int)
    g_test_coarse = defaultdict(int)
    total_router = total_test = 0
    total_expert = defaultdict(int)

    for info in slide_info:
        if info.get("status") != "ok":
            manifest["slides"].append({"tissue": info["tissue"], "status": info["status"]})
            continue

        tissue = info["tissue"]
        p = Path(info["path"])
        keep = info["keep"]
        is_test = info["is_test"]
        pid = info["pid"]
        ss = info["slide_seed"]

        df_f = pd.read_parquet(p, columns=phase2_cols)
        df_f = df_f[keep].copy()
        df_f["tissue"] = tissue
        df_f["coarse_label"] = df_f[args.label_col].map(ftc)
        df_f["coarse_id"] = df_f["coarse_label"].map(c2id).astype(np.int16)
        df_f["patch_id"] = pid

        df_train = df_f[~is_test].reset_index(drop=True)
        df_test = df_f[is_test].reset_index(drop=True)
        del df_f

        # ── Router: sample by fixed rates ──
        rng_r = np.random.default_rng(ss + 101)
        df_router = sample_by_rates(df_train, args.label_col, fine_rates, rng_r)
        df_router.to_parquet(out / "router_shards" / f"{tissue}.parquet", index=False)

        for c, n in df_router["coarse_label"].value_counts().items():
            g_router_coarse[c] += n
        for c, n in df_router[args.label_col].value_counts().items():
            g_router_fine[c] += n
        nr = len(df_router)
        total_router += nr
        del df_router

        # ── Expert shards ──
        slide_expert_info = {}
        for eg, fines in expert_groups.items():
            fine_set = set(fines)
            df_eg = df_train[df_train[args.label_col].isin(fine_set)]
            if len(df_eg) == 0:
                continue
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

        # ── Test shard ──
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
            "slide_seed": ss,
            "min_cells": info["min_cells"],
            "train_patches": info["train_patches"],
            "test_patches": info["test_patches"],
            "patch_counts": {str(i): int(info["pcounts"][i]) for i in range(n_patches)},
            "x_edges": info["x_edges"].tolist(),
            "y_edges": info["y_edges"].tolist(),
        })

        expert_str = "  ".join(f"{k}={v:,}" for k, v in slide_expert_info.items())
        print(f"  {tissue:<50} router={nr:>7,}  test={nt:>7,}  {expert_str}")

    # ══════════════════════════════════════════════════════════
    # Distribution report
    # ══════════════════════════════════════════════════════════
    n_ok = sum(1 for x in manifest["slides"] if x.get("status") == "ok")

    print(f"\n{'═'*80}")
    print("ROUTER (coarse)")
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

    # Save manifest
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