#!/usr/bin/env python3
"""
Make a minimal, self-contained MorphPT example dataset.
========================================================

MorphPT consumes a *parquet manifest* (one row per cell) plus two grayscale
DAPI crops per cell: a 2.5x (smaller context scale, fine nuclear morphology) and
a 10x (larger context scale, broader tissue context) view. This script fabricates a tiny synthetic dataset in
exactly that format, plus all the label-space JSON files, so the inference and
fine-tuning vignettes run end-to-end without the full CellImageNet corpus.

Required parquet columns (see data/dataset.py :: CellParquetBase / MultiView):
    cell_id        unique id (str)
    label          fine cell-type label (str)
    coarse_label   broad morphology group (str)  [needed for router training]
    tissue         tissue source (str)
    x_centroid     float (spatial coord; metadata only)
    y_centroid     float
    img_path_2p5x  absolute path to the 2.5x grayscale PNG crop
    img_path_10x   absolute path to the 10x  grayscale PNG crop

Images may be any size; they are resized to 224x224 (bicubic), expanded
1->3 channels, and ImageNet-normalized at load time.

Also writes (mirroring prepared/splits_*/):
    coarse_to_id.json                 broad group -> id  (router class_map)
    fine_to_coarse.json               fine label  -> broad group
    expert_class_maps/<Group>.json    fine label  -> id, per multi-class group

Usage:
    python examples/make_minimal_dataset.py --out_dir examples/minimal_data --n 48
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Fine label -> broad morphology-informed group (subset of the real taxonomy).
FINE_TO_COARSE = {
    "Lung cancer cells": "Cancer",
    "Colon cancer cells": "Cancer",
    "Skin cancer cells": "Cancer",
    "T cells": "Lymphoid",
    "B cells": "Lymphoid",
    "NK cells": "Lymphoid",
    "Neurons": "Neuroglial",
    "Astrocytes": "Neuroglial",
    "Fibroblasts": "Tissue_Vascular",
    "Endothelial cells": "Tissue_Vascular",
    "Stromal cells": "Stromal",                  # passthrough (single type)
    "Stem and progenitor cells": "Stem_Progenitor",  # passthrough (single type)
}
COARSE_ORDER = ["Cancer", "Lymphoid", "Neuroglial",
                "Stem_Progenitor", "Stromal", "Tissue_Vascular"]
DEMO_LABELS = list(FINE_TO_COARSE.keys())
DEMO_TISSUES = ["lung", "colon", "skin", "lymph_node", "brain", "bone_marrow"]


def synth_nucleus(rng: np.random.Generator, size: int, n_blobs: int) -> Image.Image:
    """Synthesize a grayscale 'DAPI-like' crop: bright Gaussian blobs on a dark
    background. Purely illustrative — NOT biologically meaningful."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = rng.normal(8, 3, size=(size, size))
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0.25, 0.75, 2) * size
        sigma = rng.uniform(size * 0.06, size * 0.16)
        amp = rng.uniform(120, 220)
        img += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="examples/minimal_data")
    ap.add_argument("--n", type=int, default=48, help="number of synthetic cells")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir).resolve()
    crops_25 = out_dir / "crops" / "2p5x"
    crops_10 = out_dir / "crops" / "10x"
    crops_25.mkdir(parents=True, exist_ok=True)
    crops_10.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.n):
        label = DEMO_LABELS[i % len(DEMO_LABELS)]
        cell_id = f"demo_{i:04d}"
        # 2.5x = tight crop on the target nucleus (one dominant blob);
        # 10x  = wider crop showing neighborhood/context (several blobs).
        img_25 = synth_nucleus(rng, size=64, n_blobs=1)
        img_10 = synth_nucleus(rng, size=96, n_blobs=int(rng.integers(2, 5)))
        p25 = crops_25 / f"{cell_id}.png"
        p10 = crops_10 / f"{cell_id}.png"
        img_25.save(p25)
        img_10.save(p10)
        rows.append({
            "cell_id": cell_id,
            "label": label,
            "coarse_label": FINE_TO_COARSE[label],
            "tissue": DEMO_TISSUES[i % len(DEMO_TISSUES)],
            "x_centroid": float(rng.uniform(0, 1000)),
            "y_centroid": float(rng.uniform(0, 1000)),
            "img_path_2p5x": str(p25),
            "img_path_10x": str(p10),
        })

    df = pd.DataFrame(rows)
    parquet_path = out_dir / "cells.parquet"
    df.to_parquet(parquet_path, index=False)

    # ---- label-space JSONs (mirror prepared/splits_*/) ----
    (out_dir / "coarse_to_id.json").write_text(
        json.dumps({c: i for i, c in enumerate(COARSE_ORDER)}, indent=2))
    (out_dir / "fine_to_coarse.json").write_text(json.dumps(FINE_TO_COARSE, indent=2))

    cmap_dir = out_dir / "expert_class_maps"
    cmap_dir.mkdir(exist_ok=True)
    for group in COARSE_ORDER:
        fines = sorted(f for f, c in FINE_TO_COARSE.items() if c == group)
        if len(fines) > 1:  # only multi-class groups get an expert
            (cmap_dir / f"{group}.json").write_text(
                json.dumps({f: i for i, f in enumerate(fines)}, indent=2))

    print(f"Wrote {len(df)} cells")
    print(f"  manifest        : {parquet_path}")
    print(f"  crops           : {crops_25}/  and  {crops_10}/")
    print(f"  coarse_to_id    : {out_dir / 'coarse_to_id.json'}")
    print(f"  fine_to_coarse  : {out_dir / 'fine_to_coarse.json'}")
    print(f"  expert maps     : {cmap_dir}/  ({[p.stem for p in cmap_dir.glob('*.json')]})")


if __name__ == "__main__":
    main()
