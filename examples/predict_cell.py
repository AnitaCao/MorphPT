#!/usr/bin/env python3
"""
MorphPT single-cell prediction from image files.
=================================================

Predict the cell type of a cell directly from its two DAPI crop images — the
2.5x (fine nuclear morphology) view and the 10x (broader tissue context) view —
using the pretrained MorphPT model. Thin CLI over `morphpt.MorphPTPredictor`.

MorphPT is multi-view: a "cell" is the PAIR of co-registered crops of the same
nucleus, so both --img_2p5x and --img_10x are required.

Single cell:
    python examples/predict_cell.py \
        --weights_dir morphpt_weights \
        --img_2p5x examples/real_cells/<id>_2p5x.png \
        --img_10x  examples/real_cells/<id>_10x.png

Batch over the bundled real examples (reads examples/real_cells/cells.csv):
    python examples/predict_cell.py \
        --weights_dir morphpt_weights \
        --manifest examples/real_cells/cells.csv

`--weights_dir` is a directory with router/ and expert_<Group>/ subfolders
(e.g. a Hugging Face snapshot of jilab/MorphPT). To download automatically:
    from morphpt import MorphPTPredictor
    model = MorphPTPredictor.from_pretrained("jilab/MorphPT")
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from morphpt import MorphPTPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True,
                    help="dir with router/ and expert_<Group>/ subfolders")
    ap.add_argument("--img_2p5x")
    ap.add_argument("--img_10x")
    ap.add_argument("--manifest",
                    help="CSV with columns img_2p5x,img_10x[,true_label]; paths relative to examples/")
    args = ap.parse_args()

    model = MorphPTPredictor(args.weights_dir)
    print(f"Device: {model.device}  |  {len(model.fine_classes)} fine classes")
    base = REPO_ROOT / "examples"

    if args.manifest:
        df = pd.read_csv(args.manifest)
        a = [str(base / p) for p in df["img_2p5x"]]
        b = [str(base / p) for p in df["img_10x"]]
        outs = model.predict_batch(a, b)
        n_ok = 0
        print(f"\n{'cell_id':<46}{'true':<26}{'pred':<26}{'group':<16}{'conf':>6}")
        for (_, r), out in zip(df.iterrows(), outs):
            true = r.get("true_label", "")
            ok = (true == out["pred"]); n_ok += int(ok)
            print(f"{'OK ' if ok else 'xx '}{str(r.get('cell_id',''))[:42]:<43}"
                  f"{true:<26}{out['pred']:<26}{out['coarse_group']:<16}{out['confidence']:>6.3f}")
        if "true_label" in df.columns:
            print(f"\nCorrect: {n_ok}/{len(df)}  (accuracy {n_ok/len(df):.3f})")
    else:
        if not (args.img_2p5x and args.img_10x):
            ap.error("provide --img_2p5x and --img_10x, or --manifest")
        out = model.predict_one(args.img_2p5x, args.img_10x)
        print(f"\nPredicted cell type : {out['pred']}")
        print(f"Routed group        : {out['coarse_group']}")
        print(f"Confidence (Mr*maxPe): {out['confidence']:.3f}")
        print("Top-3:")
        for name, score in out["top3"]:
            print(f"  {name:<28} {score:.3f}")


if __name__ == "__main__":
    main()
