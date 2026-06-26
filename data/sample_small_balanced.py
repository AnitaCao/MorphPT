#!/usr/bin/env python3
"""Build a small class-balanced subset parquet from multi-view shards.

All paths are CLI arguments (no site-specific defaults).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--shard_root", required=True,
                help="dir of multi-view shards (tissue=*/part.parquet)")
ap.add_argument("--out_parquet", required=True, help="output subset parquet")
ap.add_argument("--out_manifest", required=True, help="output JSON manifest")
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--target_max", type=int, default=10_000,
                help="max cells per class")
ap.add_argument("--drop_cancer", action=argparse.BooleanOptionalAction,
                default=True, help="exclude cancer labels (default: on)")
args = ap.parse_args()

SHARD_ROOT   = Path(args.shard_root)
OUT_PARQUET  = Path(args.out_parquet)
OUT_MANIFEST = Path(args.out_manifest)

SEED = args.seed
TARGET_MAX = args.target_max
DROP_CANCER = args.drop_cancer


def is_cancer_label(label: str) -> bool:
    return "cancer" in str(label).lower()

parts = sorted(SHARD_ROOT.glob("tissue=*/part.parquet"))
assert parts, f"No shards found under {SHARD_ROOT}"


# ---- Pass 0: global label counts (read label column only) ----
global_counts = {}
for p in parts:
    df = pd.read_parquet(p, columns=["label"])

    if DROP_CANCER:
        df = df[~df["label"].apply(is_cancer_label)]
    vc = df["label"].value_counts().to_dict()
    for k, v in vc.items():
        global_counts[k] = global_counts.get(k, 0) + int(v)

labels = sorted(global_counts.keys())
probs = {lbl: min(1.0, TARGET_MAX / global_counts[lbl]) for lbl in labels}

print("Labels (after filters):", len(labels))
print("Example probs:", list(probs.items())[:5])

# ---- Pass 1: Bernoulli sampling per label ----
rng = np.random.default_rng(SEED)
rng.shuffle(parts)

writer = None
picked_counts = {lbl: 0 for lbl in labels}

for p in parts:
    df = pd.read_parquet(p)  # full row for selected samples
    if DROP_CANCER:
        df = df[~df["label"].apply(is_cancer_label)]
    if len(df) == 0:
        continue

    # per-row keep probability based on label
    p_keep = df["label"].map(probs).astype(float).to_numpy()
    # deterministic RNG per shard
    shard_seed = (hash((SEED, str(p))) & 0xFFFFFFFF)
    rs = np.random.default_rng(shard_seed)
    keep = rs.random(len(df)) < p_keep

    df_keep = df.loc[keep].copy()
    if len(df_keep) == 0:
        continue

    # cap if a label overshoots (rare, only for small classes where p=1)
    # we enforce final caps in Pass 2, but keep this light
    vc = df_keep["label"].value_counts().to_dict()
    for k, v in vc.items():
        picked_counts[k] += int(v)

    if "x_centroid" in df_keep.columns:
        df_keep["x_centroid"] = df_keep["x_centroid"].astype("float32")
    if "y_centroid" in df_keep.columns:
        df_keep["y_centroid"] = df_keep["y_centroid"].astype("float32")

    table = pa.Table.from_pandas(df_keep, preserve_index=False)

    if writer is None:
        schema = table.schema
        writer = pq.ParquetWriter(str(OUT_PARQUET), schema, compression="snappy")
    else:
        table = table.cast(schema)

    writer.write_table(table)

if writer is not None:
    writer.close()




# ---- Pass 2: top-up to hit exactly TARGET_MAX where possible ----
rng2 = np.random.default_rng(SEED + 12345)
rng2.shuffle(parts)

df_out = pd.read_parquet(OUT_PARQUET)

used_ids = set(df_out["cell_id"].values)

vc_now = df_out["label"].value_counts().to_dict()

need = {lbl: max(0, TARGET_MAX - int(vc_now.get(lbl, 0))) for lbl in labels}
need_total = sum(need.values())
print("Need total top-up:", need_total)

if need_total > 0:
    # We will do another pass over shards and sample only missing labels
    out_path2 = OUT_PARQUET.with_suffix(".tmp.parquet")
    writer = None

    # Start by writing existing df_out
    table0 = pa.Table.from_pandas(df_out, preserve_index=False)
    writer = pq.ParquetWriter(str(out_path2), table0.schema, compression="snappy")
    writer.write_table(table0)

    for p in parts:
        if all(v == 0 for v in need.values()):
            break

        df = pd.read_parquet(p)
        # --- 去重：排除已经在 Pass 1 中选中的细胞 ---
        df = df[~df["cell_id"].isin(used_ids)]

        if DROP_CANCER:
            df = df[~df["label"].apply(is_cancer_label)]
        if len(df) == 0:
            continue

        # keep only rows whose label still needs samples
        df = df[df["label"].map(lambda x: need.get(x, 0) > 0)]
        if len(df) == 0:
            continue

        keep_idx = []
        for lbl, sub in df.groupby("label", sort=False):
            n_need = need.get(lbl, 0)
            if n_need <= 0:
                continue
            take = min(n_need, len(sub))
            if take <= 0:
                continue
            shard_seed = (hash((SEED, "topup", str(p), lbl)) & 0xFFFFFFFF)
            sub_s = sub.sample(n=take, replace=False, random_state=shard_seed)
            keep_idx.append(sub_s.index)
            need[lbl] -= take

        if not keep_idx:
            continue

        df_add = df.loc[np.concatenate([idx.values for idx in keep_idx])].copy()
        df_add["x_centroid"] = df_add["x_centroid"].astype("float32")   # ← 加这行
        df_add["y_centroid"] = df_add["y_centroid"].astype("float32") 
        table = pa.Table.from_pandas(df_add, preserve_index=False)
        writer.write_table(table)

    writer.close()

    # replace
    OUT_PARQUET.unlink()
    out_path2.rename(OUT_PARQUET)

# ---- Final manifest ----
df_final = pd.read_parquet(OUT_PARQUET, columns=["label", "tissue"])
final_counts = df_final["label"].value_counts().to_dict()

summary = {
    "seed": SEED,
    "target_max_per_class": TARGET_MAX,
    "drop_cancer": DROP_CANCER,
    "n_rows": int(len(df_final)),
    "n_labels": int(df_final["label"].nunique()),
    "n_tissues": int(df_final["tissue"].nunique()),
    "counts_by_label": {k: int(v) for k, v in final_counts.items()},
    "global_counts_used": {k: int(v) for k, v in global_counts.items()},
}
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MANIFEST, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Wrote:", OUT_PARQUET)
print("Wrote:", OUT_MANIFEST)
print(df_final["label"].value_counts().head(20))
