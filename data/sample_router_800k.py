#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IN_MANIFEST = Path("prepared/manifest.parquet")
OUT_PARQUET = Path("prepared/router_800k.parquet")
OUT_MANIFEST = Path("prepared/router_800k_manifest.json")

SEED = 1337
TARGET_N = 800_000

DROP_UNKNOWN = True
DROP_CANCER = True

UNKNOWN_LABELS = {"unknown", "Unknown", "UNK", "Unassigned", "NA", "N/A", "Other", ""}

def is_cancer_label(label: str) -> bool:
    return "cancer" in str(label).lower()

def load_manifest_ok():
    m = pd.read_parquet(IN_MANIFEST)
    m_ok = m[m["status"] == "ok"].copy()
    if "shard_path" not in m_ok.columns:
        raise RuntimeError("manifest.parquet missing shard_path column")
    return m_ok

def sample_rows_from_shard(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    n = min(int(n), len(df))
    return df.sample(n=n, replace=False, random_state=seed).copy()

def main():
    rng = np.random.default_rng(SEED)
    m_ok = load_manifest_ok()

    # Use manifest n_rows to weight shards for natural distribution half
    # If manifest doesn't have n_rows, we compute cheaply by reading metadata from parquet
    if "n_rows" not in m_ok.columns:
        n_rows = []
        for sp in m_ok["shard_path"].tolist():
            pf = pq.ParquetFile(sp)
            n_rows.append(pf.metadata.num_rows)
        m_ok["n_rows"] = n_rows

    shard_paths = m_ok["shard_path"].tolist()
    shard_weights = np.array(m_ok["n_rows"].values, dtype=np.float64)
    shard_weights = shard_weights / shard_weights.sum()

    n_natural = TARGET_N // 2
    n_balanced = TARGET_N - n_natural

    writer = None
    sampled_total = 0
    per_label = {}

    def write_df(df_out: pd.DataFrame):
        nonlocal writer, sampled_total, per_label
        if len(df_out) == 0:
            return
        # update counts
        vc = df_out["label"].value_counts().to_dict()
        for k, v in vc.items():
            per_label[k] = per_label.get(k, 0) + int(v)
        sampled_total += int(len(df_out))
        table = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(OUT_PARQUET), table.schema, compression="snappy")
        writer.write_table(table)

    # Part A: natural distribution sample
    # We sample shard ids with replacement by weight, then sample a chunk from each selected shard.
    # Chunk size kept moderate to avoid huge memory spikes.
    chunk = 25_000
    remaining = n_natural
    while remaining > 0:
        take = min(chunk, remaining)
        # pick one shard weighted, sample take rows from it
        shard_idx = rng.choice(len(shard_paths), p=shard_weights)
        sp = Path(shard_paths[shard_idx])
        df = pd.read_parquet(sp)

        if DROP_UNKNOWN:
            df = df[~df["label"].isin(UNKNOWN_LABELS)]
        if DROP_CANCER:
            df = df[~df["label"].apply(is_cancer_label)]

        seed = (hash((SEED, "natural", str(sp), remaining)) & 0xFFFFFFFF)
        df_s = sample_rows_from_shard(df, take, seed)
        write_df(df_s)
        remaining -= int(len(df_s))
        if len(df) == 0:
            # avoid getting stuck on empty
            remaining -= 0

    # Part B: class capped sample for balance
    # We do a pass over shards and take up to cap per label until reaching n_balanced.
    # cap tuned to prevent head domination
    cap_per_label = 25_000  # for router, you want more coverage than 10k but still capped
    remaining_by_label = {}
    remaining = n_balanced

    for sp in shard_paths:
        if remaining <= 0:
            break
        sp = Path(sp)
        df = pd.read_parquet(sp)

        if DROP_UNKNOWN:
            df = df[~df["label"].isin(UNKNOWN_LABELS)]
        if DROP_CANCER:
            df = df[~df["label"].apply(is_cancer_label)]

        if len(df) == 0:
            continue

        # init caps
        for lbl in df["label"].unique():
            if lbl not in remaining_by_label:
                remaining_by_label[lbl] = cap_per_label

        keep_idx = []
        for lbl, sub in df.groupby("label", sort=False):
            if remaining <= 0:
                break
            need_lbl = remaining_by_label.get(lbl, 0)
            if need_lbl <= 0:
                continue
            take = min(int(need_lbl), int(remaining), len(sub))
            if take <= 0:
                continue
            seed = (hash((SEED, "balanced", str(sp), str(lbl))) & 0xFFFFFFFF)
            sub_s = sub.sample(n=take, replace=False, random_state=seed)
            keep_idx.append(sub_s.index)
            remaining_by_label[lbl] -= take
            remaining -= take

        if keep_idx:
            out_df = df.loc[np.concatenate([idx.values for idx in keep_idx])].copy()
            out_df = out_df.sample(frac=1.0, random_state=(hash((SEED, "balanced_shuf", str(sp))) & 0xFFFFFFFF)).reset_index(drop=True)
            write_df(out_df)

    if writer is not None:
        writer.close()

    summary = {
        "seed": SEED,
        "target_n": int(TARGET_N),
        "sampled_total": int(sampled_total),
        "drop_unknown": DROP_UNKNOWN,
        "drop_cancer": DROP_CANCER,
        "natural_n_target": int(n_natural),
        "balanced_n_target": int(n_balanced),
        "balanced_cap_per_label": int(cap_per_label),
        "counts_by_label": {k: int(v) for k, v in sorted(per_label.items(), key=lambda x: -x[1])},
    }

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {OUT_PARQUET}")
    print(f"Wrote: {OUT_MANIFEST}")
    print(f"Sampled total: {sampled_total}")
    top = sorted(per_label.items(), key=lambda x: -x[1])[:10]
    print("Top sampled labels:")
    for k, v in top:
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
