#!/usr/bin/env python3
"""
Assemble morphpt_moe_infer bundle and create .tar.gz for distribution.

Usage:
    python scripts/pack_morphpt_moe.py
    python scripts/pack_morphpt_moe.py --out /tmp/morphpt_moe_infer
"""

import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TEMPLATE_ROOT = PROJECT_ROOT / "release" / "morphpt_moe_template"


def cp(src: Path, dst: Path):
    if not src.exists():
        print(f"  [WARN] not found, skipping: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  + {dst.relative_to(dst.parents[len(dst.parts)-2])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "release" / "morphpt_moe_infer"))
    ap.add_argument("--no_archive", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"Assembling bundle: {out}\n")

    # ── checkpoints ───────────────────────────────────────────────────────────
    print("── Checkpoints")
    cp(PROJECT_ROOT / "experiments/router_nobreast_vitb_gate_r16_mlp_cw/best.pt",
       out / "checkpoints/router/best.pt")
    cp(PROJECT_ROOT / "prepared/splits_v2_seed1337_nobreast/coarse_to_id.json",
       out / "checkpoints/router/coarse_to_id.json")
    for group in ["Cancer", "Lymphoid", "Neuroglial", "Tissue_Vascular"]:
        cp(PROJECT_ROOT / f"experiments/moe_v4/expert_{group}/best.pt",
           out / f"checkpoints/expert_{group}/best.pt")
        cp(PROJECT_ROOT / f"prepared/splits_v2_seed1337_nobreast/expert_{group}/class_to_idx.json",
           out / f"checkpoints/expert_{group}/class_to_idx.json") 

    # ── model source (copied into morphpt_moe/models/) ────────────────────────
    print("\n── Model source")
    for f in ["lora.py", "model.py", "__init__.py"]:
        cp(PROJECT_ROOT / "models" / f, out / "morphpt_moe" / "models" / f)
    # dataset only needed if colleague wants batch-parquet inference
    cp(PROJECT_ROOT / "data" / "dataset.py", out / "morphpt_moe" / "data" / "dataset.py")

    # ── package files from template ───────────────────────────────────────────
    print("\n── Package files")
    for f in [
        "morphpt_moe/__init__.py",
        "morphpt_moe/pipeline.py",
        "morphpt_moe/infer.py",
        "run_infer.sh",
        "requirements.txt",
        "README.md",
    ]:
        cp(TEMPLATE_ROOT / f, out / f)

    # make run_infer.sh executable
    (out / "run_infer.sh").chmod(0o755)

    # ── example images: copy one cell from test set ───────────────────────────
    print("\n── Example images")
    example_dir = out / "example_images"
    example_dir.mkdir(exist_ok=True)
    # Try to grab one real example from test shards
    test_shards = PROJECT_ROOT / "prepared/splits_v2_seed1337_nobreast/test_shards"
    sample_found = False
    if test_shards.exists():
        try:
            import pandas as pd
            for parquet in sorted(test_shards.glob("*.parquet"))[:5]:
                df = pd.read_parquet(parquet, columns=["img_path_2p5x", "img_path_10x"])
                row = df.iloc[0]
                p1, p2 = Path(row["img_path_2p5x"]), Path(row["img_path_10x"])
                if p1.exists() and p2.exists():
                    cp(p1, example_dir / "example_2p5x.png")
                    cp(p2, example_dir / "example_10x.png")
                    sample_found = True
                    break
        except Exception as e:
            print(f"  [WARN] Could not copy example images: {e}")
    if not sample_found:
        print("  [WARN] No example images found — add manually to example_images/")

    # ── archive ───────────────────────────────────────────────────────────────
    if not args.no_archive:
        archive = out.parent / f"{out.name}.tar.gz"
        print(f"\n── Creating archive: {archive}")
        subprocess.run(
            ["tar", "-czf", str(archive), "-C", str(out.parent), out.name],
            check=True,
        )
        size_mb = archive.stat().st_size / 1e6
        print(f"  {archive}  ({size_mb:.0f} MB)")

    print(f"""
Done. Bundle layout:

  {out.name}/
  ├── checkpoints/
  │   ├── router/best.pt
  │   ├── expert_Cancer/best.pt
  │   ├── expert_Lymphoid/best.pt
  │   ├── expert_Neuroglial/best.pt
  │   └── expert_Tissue_Vascular/best.pt
  ├── morphpt_moe/
  │   ├── __init__.py
  │   ├── pipeline.py
  │   ├── infer.py
  │   ├── models/   (lora.py, model.py)
  │   └── data/     (dataset.py)
  ├── example_images/
  │   ├── example_2p5x.png
  │   └── example_10x.png
  ├── run_infer.sh          ← entry point
  ├── requirements.txt
  └── README.md
""")


if __name__ == "__main__":
    main()