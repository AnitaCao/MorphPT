#!/usr/bin/env python3
"""
Build a balanced validation set from UNUSED train-patch cells,
with fallback supplement from router_shards for rare fine classes.

Pipeline:
  1. Read split manifest → know train/test patches per slide
  2. Read original parquets → filter to train patches only
  3. Read router_shards → get used cell_ids
  4. Exclude router_shards cells → val pool
  5. Identify rare fines (count < --rare_thresh) in val pool
  6. Supplement rare fines from router_shards (with guardrails)
  7. Write pruned router_shards (original minus stolen cells)
  8. Two-stage coarse-cap sampling → final val set

Outputs:
  --out                      Final val parquet (unused + supplement, sampled)
  --pruned_router_dir        router_shards minus supplement cells
  --supplement_out           Supplement-only parquet (for auditing)

Usage:
  python scripts/build_val_from_train.py \
    --manifest prepared/splits_v3_seed1337/split_manifest_seed1337.json \
    --data_dir prepared/shards_multiview_parquet \
    --router_dir prepared/splits_v3_seed1337/router_shards \
    --out prepared/splits_v3_seed1337/val_from_train.parquet \
    --pruned_router_dir prepared/splits_v3_seed1337/router_shards_pruned \
    --supplement_out prepared/splits_v3_seed1337/val_supplement.parquet \
    --cap 5000 \
    --seed 1337
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="split_manifest_seed*.json from split_data.py")
    ap.add_argument("--data_dir", required=True,
                    help="Original parquet dir (tissue=*/part.parquet)")
    ap.add_argument("--router_dir", required=True,
                    help="router_shards/ directory")
    ap.add_argument("--out", required=True,
                    help="Output val parquet path")
    ap.add_argument("--pruned_router_dir", default="",
                    help="Output dir for pruned router shards (default: router_dir + '_pruned')")
    ap.add_argument("--supplement_out", default="",
                    help="Output path for supplement-only parquet (for auditing)")
    ap.add_argument("--label_col", default="label",
                    help="Fine label column name")
    ap.add_argument("--cap", type=int, default=5_000,
                    help="Max samples per coarse class")
    ap.add_argument("--rare_thresh", type=int, default=100,
                    help="Fine classes with fewer than this in val pool get supplemented")
    ap.add_argument("--supplement_frac", type=float, default=0.10,
                    help="Fraction of router_shards to steal for rare fines")
    ap.add_argument("--supplement_min", type=int, default=100,
                    help="Min cells to steal per rare fine class")
    ap.add_argument("--supplement_max", type=int, default=1000,
                    help="Max cells to steal per rare fine class")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Default pruned dir
    if not args.pruned_router_dir:
        args.pruned_router_dir = str(Path(args.router_dir).parent / "router_shards_pruned")

    # ── Load manifest ──
    manifest = json.loads(Path(args.manifest).read_text())
    manifest_dir = Path(args.manifest).parent

    ftc_path = manifest_dir / "fine_to_coarse.json"
    if ftc_path.exists():
        fine_to_coarse = json.loads(ftc_path.read_text())
    else:
        raise RuntimeError(f"Missing {ftc_path}")

    coarse_to_id_path = manifest_dir / "coarse_to_id.json"
    if coarse_to_id_path.exists():
        coarse_to_id = json.loads(coarse_to_id_path.read_text())
    else:
        coarse_to_id = {n: i for i, n in enumerate(sorted(set(fine_to_coarse.values())))}

    valid_fine = set(fine_to_coarse.keys())
    grid = manifest["grid"]
    gr, gc = grid

    # ══════════════════════════════════════════════════════════
    # Step 1: Load router_shards cell_ids
    # ══════════════════════════════════════════════════════════
    router_dir = Path(args.router_dir)
    router_files = sorted(router_dir.glob("*.parquet"))
    print(f"Loading router_shards from {router_dir} ({len(router_files)} files)...")

    train_id_chunks = []
    for f in router_files:
        df_r = pd.read_parquet(f, columns=["cell_id"])
        train_id_chunks.append(df_r["cell_id"])
    train_ids = pd.Index(pd.concat(train_id_chunks)) if train_id_chunks else pd.Index([])
    print(f"  Router training cells: {len(train_ids):,}")

    # ══════════════════════════════════════════════════════════
    # Step 2: Extract unused train-patch cells → val pool
    # ══════════════════════════════════════════════════════════
    print(f"\nExtracting unused train-patch cells...")
    val_parts = []
    slides = manifest["slides"]

    for s in slides:
        if s.get("status") != "ok":
            continue

        tissue = s["tissue"]
        train_patches = s["train_patches"]

        pq_path = Path(args.data_dir) / f"tissue={tissue}" / "part.parquet"
        if not pq_path.exists():
            print(f"  [skip] {tissue}: parquet not found")
            continue

        # 只读必要列（如果文件缺少某些列则回退到读取全量）
        cols_to_read = ["cell_id", args.label_col, "x_centroid", "y_centroid", "img_path_10x", "img_path_2p5x"]
        try:
            df = pd.read_parquet(pq_path, columns=cols_to_read)
        except ValueError:
            df = pd.read_parquet(pq_path)
        df = df[df[args.label_col].isin(valid_fine)].copy()

        if len(df) == 0:
            continue

        # Assign patches (same logic as split_data.py)
        x = df["x_centroid"].to_numpy(dtype=np.float64)
        y = df["y_centroid"].to_numpy(dtype=np.float64)
        x_edges = np.array(s["x_edges"])
        y_edges = np.array(s["y_edges"])

        x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, gc - 1).astype(np.int16)
        y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, gr - 1).astype(np.int16)
        pid = (y_idx * gc + x_idx).astype(np.int16)

        is_train_patch = np.isin(pid, train_patches)
        df_train_all = df[is_train_patch].copy()

        not_in_router = ~df_train_all["cell_id"].isin(train_ids)
        df_unused = df_train_all[not_in_router].copy()

        if len(df_unused) == 0:
            continue

        df_unused["tissue"] = tissue
        df_unused["coarse_label"] = df_unused[args.label_col].map(fine_to_coarse)
        df_unused["coarse_id"] = df_unused["coarse_label"].map(coarse_to_id).astype(np.int16)

        val_parts.append(df_unused)
        print(f"  {tissue:<50} train_all={len(df_train_all):>9,}  "
              f"router={len(df_train_all)-len(df_unused):>9,}  "
              f"unused={len(df_unused):>9,}")

    if not val_parts:
        raise RuntimeError("No validation cells found")

    val_pool = pd.concat(val_parts, ignore_index=True)
    print(f"\nTotal val pool (unused): {len(val_pool):,} cells")

    # ══════════════════════════════════════════════════════════
    # Step 3: Identify rare fine classes
    # ══════════════════════════════════════════════════════════
    pool_fine_counts = val_pool[args.label_col].value_counts()
    all_fine_counts = {f: int(pool_fine_counts.get(f, 0)) for f in valid_fine}

    rare_fines = {f for f, c in all_fine_counts.items() if c < args.rare_thresh}

    print(f"\nVal pool per-fine distribution:")
    for f in sorted(all_fine_counts, key=lambda x: all_fine_counts[x]):
        cnt = all_fine_counts[f]
        tag = " ← RARE" if f in rare_fines else ""
        print(f"  {f:<30} {cnt:>8,}{tag}")

    # ══════════════════════════════════════════════════════════
    # Step 4: Supplement rare fines from router_shards
    # ══════════════════════════════════════════════════════════
    supplement_ids = set()

    router_cols = ["cell_id", "patch_id", args.label_col, "x_centroid", "y_centroid", "img_path_10x", "img_path_2p5x", "tissue", "coarse_label", "coarse_id"]

    if not rare_fines:
        print(f"\nNo rare fine classes (all >= {args.rare_thresh}). No supplement needed.")
        # Still copy router_shards to pruned_dir unchanged
        pruned_dir = Path(args.pruned_router_dir)
        pruned_dir.mkdir(parents=True, exist_ok=True)
        for f in router_files:
            try:
                df_r = pd.read_parquet(f, columns=router_cols)
            except ValueError:
                df_r = pd.read_parquet(f)
            df_r.to_parquet(pruned_dir / f.name, index=False)
    else:
        print(f"\n{'═' * 60}")
        print(f"SUPPLEMENT: {len(rare_fines)} rare fine classes (< {args.rare_thresh} in val pool)")
        print(f"  frac={args.supplement_frac}, min={args.supplement_min}, max={args.supplement_max}")
        print(f"{'═' * 60}")

        # Count rare fines in router_shards
        router_rare_counts = defaultdict(int)
        for f in router_files:
            df_r = pd.read_parquet(f, columns=["cell_id", args.label_col])
            for fine in rare_fines:
                router_rare_counts[fine] += int((df_r[args.label_col] == fine).sum())

        # Compute take per fine with guardrails
        take_per_fine = {}
        for fine in sorted(rare_fines):
            n_router = router_rare_counts[fine]
            if n_router == 0:
                print(f"  {fine:<30} router=0, cannot supplement")
                continue
            take = min(args.supplement_max,
                       max(args.supplement_min,
                           int(round(args.supplement_frac * n_router))))
            take = min(take, n_router)
            take_per_fine[fine] = take
            print(f"  {fine:<30} router={n_router:>8,}  take={take:>6,}  "
                  f"({take/n_router*100:.1f}%)")

        # Compute global sampling fraction per rare fine class
        fine_frac = {fine: take / router_rare_counts[fine] for fine, take in take_per_fine.items() if router_rare_counts[fine] > 0}

        # Sample and prune
        pruned_dir = Path(args.pruned_router_dir)
        pruned_dir.mkdir(parents=True, exist_ok=True)

        supplement_parts = []
        total_stolen = defaultdict(int)

        print(f"\nProcessing router shards...")
        for f in router_files:
            tissue = f.stem
            try:
                df_r = pd.read_parquet(f, columns=router_cols)
            except ValueError:
                df_r = pd.read_parquet(f)

            steal_mask = pd.Series(False, index=df_r.index)

            for fine, frac in fine_frac.items():
                fine_idx = df_r.index[df_r[args.label_col] == fine].to_numpy()
                if len(fine_idx) == 0:
                    continue

                n_take = int(round(len(fine_idx) * frac))
                n_take = min(n_take, len(fine_idx))
                
                # Enforce global quota cap to allow slight deviation but no overshoot
                n_take = min(n_take, take_per_fine[fine] - total_stolen[fine])

                if n_take > 0:
                    chosen = rng.choice(fine_idx, size=n_take, replace=False)
                    steal_mask.loc[chosen] = True
                    total_stolen[fine] += n_take

            n_stolen = int(steal_mask.sum())
            if n_stolen > 0:
                df_supp = df_r[steal_mask].copy()
                if "tissue" not in df_supp.columns:
                    df_supp["tissue"] = tissue
                if "coarse_label" not in df_supp.columns:
                    df_supp["coarse_label"] = df_supp[args.label_col].map(fine_to_coarse)
                if "coarse_id" not in df_supp.columns:
                    df_supp["coarse_id"] = df_supp["coarse_label"].map(coarse_to_id).astype(np.int16)
                supplement_parts.append(df_supp)
                supplement_ids.update(df_supp["cell_id"].tolist())

            # Write pruned shard
            df_pruned = df_r[~steal_mask]
            df_pruned.to_parquet(pruned_dir / f"{tissue}.parquet", index=False)

            if n_stolen > 0:
                print(f"  {tissue:<50} stolen={n_stolen:>5,}  "
                      f"pruned={len(df_pruned):>8,}")

        print(f"\nSupplement summary:")
        for fine in sorted(total_stolen):
            orig = all_fine_counts[fine]
            stolen = total_stolen[fine]
            print(f"  {fine:<30} pool={orig:>5,} + stolen={stolen:>5,} = {orig+stolen:>6,}")
        print(f"  Total supplement cells: {len(supplement_ids):,}")

        # Save supplement parquet
        if supplement_parts:
            df_supplement = pd.concat(supplement_parts, ignore_index=True)
            supp_path = Path(args.supplement_out) if args.supplement_out else \
                        Path(args.out).parent / "val_supplement.parquet"
            supp_path.parent.mkdir(parents=True, exist_ok=True)
            df_supplement.to_parquet(str(supp_path), index=False)
            print(f"  Saved supplement → {supp_path}")

            # Merge into val pool
            val_pool = pd.concat([val_pool, df_supplement], ignore_index=True)
            print(f"\nVal pool after supplement: {len(val_pool):,} cells")

    # ══════════════════════════════════════════════════════════
    # Step 5: Two-stage coarse-cap sampling
    # ══════════════════════════════════════════════════════════
    cap_val = args.cap
    print(f"\n{'═' * 60}")
    print(f"Two-stage sampling: per coarse class, cap={cap_val:,}")
    print(f"{'═' * 60}")

    sampled_parts = []
    for cls, cls_df in val_pool.groupby("coarse_label", sort=True):
        n_raw = len(cls_df)
        if n_raw <= cap_val:
            sampled_parts.append(cls_df)
            n_slides = cls_df["tissue"].nunique()
            print(f"  {cls:<22} keep all {n_raw:,}  ({n_slides} slides)")
            continue

        # Stage 1: exactly 1 per slide (diversity floor)
        stage1_parts = []
        stage1_ids = set()
        slide_groups = {sid: sdf for sid, sdf in cls_df.groupby("tissue")}

        for sid, sdf in slide_groups.items():
            pick = sdf.sample(n=1, random_state=rng.integers(1 << 31))
            stage1_parts.append(pick)
            stage1_ids.update(pick.index)

        n_stage1 = len(stage1_ids)
        remaining_quota = cap_val - n_stage1

        # Stage 2: fill proportionally from remaining cells
        stage2_parts = []
        if remaining_quota > 0:
            pool2 = cls_df.drop(index=stage1_ids)
            n_pool2 = len(pool2)
            if n_pool2 > 0:
                rate = remaining_quota / n_pool2
                for sid, sdf in pool2.groupby("tissue"):
                    n_take = max(0, int(round(len(sdf) * rate)))
                    n_take = min(n_take, len(sdf))
                    if n_take > 0:
                        idx = rng.choice(len(sdf), size=n_take, replace=False)
                        stage2_parts.append(sdf.iloc[idx])

        combined = pd.concat(stage1_parts + stage2_parts, ignore_index=True)

        # Trim if overshot
        if len(combined) > cap_val:
            stage2_all = pd.concat(stage2_parts, ignore_index=True) if stage2_parts else pd.DataFrame()
            if len(stage2_all) > remaining_quota:
                idx = rng.choice(len(stage2_all), size=remaining_quota, replace=False)
                stage2_all = stage2_all.iloc[idx]
            combined = pd.concat(stage1_parts + [stage2_all], ignore_index=True)

        sampled_parts.append(combined)
        n_slides = len(slide_groups)
        print(f"  {cls:<22} {n_raw:,} → {len(combined):,}  "
              f"(stage1={n_stage1}, stage2={len(combined)-n_stage1}, "
              f"{n_slides} slides)")

    val_df = pd.concat(sampled_parts, ignore_index=True)
    val_df = val_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ══════════════════════════════════════════════════════════
    # Final report
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"FINAL VAL SET: {len(val_df):,} cells")
    print(f"{'═' * 60}")

    print(f"\n  Coarse distribution:")
    for cls, cnt in val_df["coarse_label"].value_counts().sort_index().items():
        print(f"    {cls:<22} {cnt:>10,}")

    print(f"\n  Fine distribution (grouped by coarse):")
    for coarse in sorted(val_df["coarse_label"].unique()):
        sub = val_df[val_df["coarse_label"] == coarse]
        print(f"\n    {coarse} ({len(sub):,})")
        for fine, cnt in sub[args.label_col].value_counts().sort_values(ascending=False).items():
            src = "supp" if fine in rare_fines else "pool"
            n_slides = sub[sub[args.label_col] == fine]["tissue"].nunique()
            print(f"      {fine:<30} {cnt:>8,}  ({n_slides} slides, {src})")

    print(f"\n  Slide coverage: {val_df['tissue'].nunique()} slides")

    # ── Save ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_parquet(str(out_path), index=False)
    print(f"\nSaved val → {out_path}")

    if supplement_ids:
        print(f"Saved pruned router → {args.pruned_router_dir}")
        print(f"\nFor training, use:")
        print(f"  train: {args.pruned_router_dir}")
        print(f"  val:   {out_path}")
    else:
        print(f"\nNo supplement needed. Train with original router_shards.")


if __name__ == "__main__":
    main()