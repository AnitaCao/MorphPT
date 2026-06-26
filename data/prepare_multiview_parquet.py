#!/usr/bin/env python3
"""Build multi-view (2.5x/10x) parquet shards from CellImageNet metadata.

Input locations default to the environment variables below (or neutral
relative paths) and can be overridden via CLI flags; no site-specific paths
are baked in.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Defaults from env vars (override with CLI flags); neutral relative fallbacks.
META_2P5X_DIR = Path(os.environ.get("CELLIMAGENET_META_2P5X", "meta/2.5x"))
META_10X_DIR  = Path(os.environ.get("CELLIMAGENET_META_10X",  "meta/10x"))
SPATIAL_DIR   = Path(os.environ.get("CELLIMAGENET_SPATIAL",   "spatial"))
IMG_BASE      = Path(os.environ.get("CELLIMAGENET_IMG_BASE",  "image"))

OUT_DIR   = Path("./prepared")
SHARD_DIR = OUT_DIR / "shards_multiview_parquet"
ISSUE_DIR = OUT_DIR / "issues"

ID_COL = "cell_id"
LABEL_COL = "celltype"
RELIMG_COL = "image_path"
TISSUE_COL = "tissue"
X_COL = "x_centroid"
Y_COL = "y_centroid"

UNKNOWN_LABELS = {"unknown", "Unknown", "UNK", "Unassigned", "NA", "N/A", "Other", ""}
_META_USECOLS = [ID_COL, LABEL_COL, RELIMG_COL, TISSUE_COL]


def assert_paths_exist_sampled(df, col, tissue, n_check=1000, seed=1337, fail_rate=0.01):
    if len(df) == 0:
        return
    n = min(n_check, len(df))
    rs = (hash((seed, tissue, col)) & 0xFFFFFFFF)
    samp = df.sample(n=n, replace=False, random_state=rs)[col].tolist()
    missing = [p for p in samp if not Path(p).is_file()]
    miss_rate = len(missing) / max(1, n)
    if miss_rate > fail_rate:
        pd.DataFrame({col: missing}).to_csv(ISSUE_DIR / f"missing_sample_{col}_{tissue}.csv", index=False)
        raise RuntimeError(
            f"High missing rate for {col} in tissue={tissue}: {miss_rate:.3%} "
            f"(>{fail_rate:.3%}). See issues/missing_sample_{col}_{tissue}.csv"
        )


def read_meta(path: Path, view: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, usecols=_META_USECOLS)
    except Exception:
        df = pd.read_csv(path)

    for c in (ID_COL, LABEL_COL, RELIMG_COL):
        if c not in df.columns:
            raise RuntimeError(f"Missing col {c} in {path}. Got {list(df.columns)}")

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    tissue = df[TISSUE_COL].astype(str).str.strip() if TISSUE_COL in df.columns else path.stem
    df["tissue"] = tissue

    if df[ID_COL].duplicated().any():
        dups = df[df[ID_COL].duplicated(keep=False)].sort_values(ID_COL)
        dups.to_csv(ISSUE_DIR / f"dups_{view}_{path.stem}.csv", index=False)
        raise RuntimeError(f"[{view}] cell_id not unique in {path}. See issues/dups_{view}_{path.stem}.csv")

    rel = df[RELIMG_COL].astype(str).str.strip()
    df[f"img_path_{view}"] = rel.apply(lambda s: str((IMG_BASE / s).resolve()))
    df[f"meta_csv_{view}"] = str(path)

    out = df[[ID_COL, "tissue", LABEL_COL, f"img_path_{view}", f"meta_csv_{view}"]].copy()
    out = out.rename(columns={LABEL_COL: "label"})
    return out


def read_spatial(path: Path) -> pd.DataFrame:
    # Only read the required columns
    df = pd.read_csv(path, usecols=[ID_COL, X_COL, Y_COL])
    df.columns = [c.strip().strip('"') for c in df.columns]

    for c in (ID_COL, X_COL, Y_COL):
        if c not in df.columns:
            raise RuntimeError(f"Missing col {c} in {path}. Got {list(df.columns)}")

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    if df[ID_COL].duplicated().any():
        dups = df[df[ID_COL].duplicated(keep=False)].sort_values(ID_COL)
        dups.to_csv(ISSUE_DIR / f"dups_spatial_{path.stem}.csv", index=False)
        raise RuntimeError(f"[spatial] cell_id not unique in {path}. See issues/dups_spatial_{path.stem}.csv")

    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce").astype("float32")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce").astype("float32")
    df["spatial_csv"] = str(path)

    return df[[ID_COL, X_COL, Y_COL, "spatial_csv"]].copy()


def list_tissues():
    return sorted([p.stem for p in SPATIAL_DIR.glob("*.csv")])


def add_counts_from_existing_shard(out_path: Path, global_counts: dict):
    # Read only the label column for speed
    df_lbl = pd.read_parquet(out_path, columns=["label"])
    vc = df_lbl["label"].value_counts().to_dict()
    for k, v in vc.items():
        global_counts[k] = global_counts.get(k, 0) + int(v)


def main():
    global META_2P5X_DIR, META_10X_DIR, SPATIAL_DIR, IMG_BASE, OUT_DIR, SHARD_DIR, ISSUE_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta_2p5x_dir", default=str(META_2P5X_DIR),
                    help="dir of per-tissue 2.5x metadata CSVs")
    ap.add_argument("--meta_10x_dir", default=str(META_10X_DIR),
                    help="dir of per-tissue 10x metadata CSVs")
    ap.add_argument("--spatial_dir", default=str(SPATIAL_DIR),
                    help="dir of per-tissue spatial CSVs (defines tissue list)")
    ap.add_argument("--img_base", default=str(IMG_BASE),
                    help="base dir prepended to relative image paths")
    ap.add_argument("--out_dir", default=str(OUT_DIR),
                    help="output dir for shards / manifest / class map")
    args = ap.parse_args()

    META_2P5X_DIR = Path(args.meta_2p5x_dir)
    META_10X_DIR  = Path(args.meta_10x_dir)
    SPATIAL_DIR   = Path(args.spatial_dir)
    IMG_BASE      = Path(args.img_base)
    OUT_DIR       = Path(args.out_dir)
    SHARD_DIR     = OUT_DIR / "shards_multiview_parquet"
    ISSUE_DIR     = OUT_DIR / "issues"
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    ISSUE_DIR.mkdir(parents=True, exist_ok=True)

    tissues = list_tissues()
    if not tissues:
        raise RuntimeError(f"No spatial tissue csv found in {SPATIAL_DIR}")

    manifest_rows = []
    global_counts = {}
    removed_unknown = 0
    total_written = 0

    skipped_missing_inputs = 0
    skipped_already_exists = 0
    empty_after_join = 0

    for tissue in tissues:
        out_part_dir = SHARD_DIR / f"tissue={tissue}"
        out_path = out_part_dir / "part.parquet"

        # Early skip if shard exists, but still include its label counts
        if out_path.is_file():
            try:
                pf = pq.ParquetFile(str(out_path))
                n_rows = int(pf.metadata.num_rows)
            except Exception:
                n_rows = 0

            try:
                add_counts_from_existing_shard(out_path, global_counts)
            except Exception:
                # If label read fails, keep going, but note it
                pass

            skipped_already_exists += 1
            manifest_rows.append({
                "tissue": tissue,
                "status": "skipped_already_exists",
                "shard_path": str(out_path),
                "n_rows": n_rows,
            })
            continue

        p2 = META_2P5X_DIR / f"{tissue}.csv"
        p10 = META_10X_DIR / f"{tissue}.csv"
        ps = SPATIAL_DIR / f"{tissue}.csv"

        if (not p2.is_file()) or (not p10.is_file()) or (not ps.is_file()):
            skipped_missing_inputs += 1
            manifest_rows.append({
                "tissue": tissue,
                "status": "skipped_missing_inputs",
                "meta_2p5x": str(p2) if p2.exists() else "",
                "meta_10x": str(p10) if p10.exists() else "",
                "spatial": str(ps) if ps.exists() else "",
                "n_rows": 0
            })
            continue

        m2 = read_meta(p2, "2p5x")
        m10 = read_meta(p10, "10x")
        sp = read_spatial(ps)

        mask_u = m2["label"].isin(UNKNOWN_LABELS)
        removed_unknown += int(mask_u.sum())
        m2f = m2.loc[~mask_u].copy()

        df = (m2f.merge(sp, on=ID_COL, how="inner")
                 .merge(m10[[ID_COL, "img_path_10x", "meta_csv_10x"]], on=ID_COL, how="inner"))

        if len(df) == 0:
            empty_after_join += 1
            manifest_rows.append({
                "tissue": tissue,
                "status": "empty_after_join",
                "meta_2p5x": str(p2),
                "meta_10x": str(p10),
                "spatial": str(ps),
                "n_rows": 0
            })
            continue

        df["tissue"] = tissue

        assert_paths_exist_sampled(df, "img_path_2p5x", tissue)
        assert_paths_exist_sampled(df, "img_path_10x", tissue)

        out_part_dir.mkdir(parents=True, exist_ok=True)

        out_cols = [
            ID_COL, "tissue", X_COL, Y_COL, "label",
            "img_path_2p5x", "img_path_10x",
            "meta_csv_2p5x", "meta_csv_10x", "spatial_csv",
        ]
        df_out = df[out_cols].copy()
        df_out.to_parquet(out_path, index=False)

        vc = df_out["label"].value_counts().to_dict()
        for k, v in vc.items():
            global_counts[k] = global_counts.get(k, 0) + int(v)

        n = int(len(df_out))
        total_written += n
        manifest_rows.append({
            "tissue": tissue,
            "status": "ok",
            "meta_2p5x": str(p2),
            "meta_10x": str(p10),
            "spatial": str(ps),
            "shard_path": str(out_path),
            "n_rows": n,
            "n_classes": int(df_out["label"].nunique()),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_parquet(OUT_DIR / "manifest.parquet", index=False)

    classes = sorted(global_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(OUT_DIR / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    counts = {
        "n_tissues_total_spatial": int(len(tissues)),
        "n_tissues_ok": int((manifest_df["status"] == "ok").sum()),
        "n_tissues_skipped_missing_inputs": int(skipped_missing_inputs),
        "n_tissues_skipped_already_exists": int(skipped_already_exists),
        "n_tissues_empty_after_join": int(empty_after_join),
        "n_rows_written_this_run": int(total_written),
        "n_removed_unknown": int(removed_unknown),
        "n_classes": int(len(classes)),
        "counts_by_label": global_counts,
        "meta_2p5x_dir": str(META_2P5X_DIR),
        "meta_10x_dir": str(META_10X_DIR),
        "spatial_dir": str(SPATIAL_DIR),
        "img_base": str(IMG_BASE),
    }
    with open(OUT_DIR / "counts.json", "w") as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)

    print(f"OK shards at: {SHARD_DIR}")
    print(f"OK manifest:  {OUT_DIR / 'manifest.parquet'}")
    print(f"OK class map: {OUT_DIR / 'class_to_idx.json'}")
    print(f"OK counts:    {OUT_DIR / 'counts.json'}")


if __name__ == "__main__":
    main()
