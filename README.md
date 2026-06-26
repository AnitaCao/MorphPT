# MorphPT

[![🤗 Pretrained weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jilab%2FMorphPT-yellow)](https://huggingface.co/jilab/MorphPT)
[![License](https://img.shields.io/badge/Code%20License-MIT-blue)](LICENSE)

**MorphPT** is a domain-adapted vision model for **cell-type classification from
DAPI-stained nuclei**. Each cell is represented by paired multi-scale crops — a
**2.5×** view (fine nuclear morphology) and a **10×** view (broader tissue
context) — encoded by a shared DINOv3 ViT-B backbone with LoRA adaptation and fused by a learnable
**ScaleGate** module. A coarse **router** predicts one of six broad
morphology-informed groups, and group **experts** resolve fine cell types via
soft routing.

- 📦 **Pretrained weights (Hugging Face):** **https://huggingface.co/jilab/MorphPT**
- 🗂️ **CellImageNet dataset:** _forthcoming_
- 📄 **Paper:** _citation forthcoming_

> Code lives here; model weights are hosted on Hugging Face (the checkpoints are
> too large for git). See [Pretrained weights](#pretrained-weights).

---

## Installation

```bash
git clone https://github.com/AnitaCao/MorphPT.git
cd MorphPT
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Python ≥3.10 and a CUDA GPU are recommended.

## Pretrained weights

Download from Hugging Face:

```python
from huggingface_hub import snapshot_download
ckpt = snapshot_download("jilab/MorphPT")   # router/ + expert_<Group>/ at the root
```

Layout (`$CKPT` = the snapshot root):

```
router/best.pt                  router/coarse_to_id.json
expert_Cancer/best.pt           expert_Cancer/class_to_idx.json
expert_Lymphoid/best.pt         expert_Lymphoid/class_to_idx.json
expert_Neuroglial/best.pt       expert_Neuroglial/class_to_idx.json
expert_Tissue_Vascular/best.pt  expert_Tissue_Vascular/class_to_idx.json
```

## Quick start — predict a cell

A worked, runnable vignette (with 10 bundled **real** example cells) lives in
[`examples/`](examples/). Predict a single cell from its two crops:

```bash
python examples/predict_cell.py --weights_dir $CKPT \
    --img_2p5x examples/real_cells/<cell>_2p5x.png \
    --img_10x  examples/real_cells/<cell>_10x.png
```

Or programmatically (downloads the weights from the Hub):

```python
from morphpt import MorphPTPredictor
model = MorphPTPredictor.from_pretrained("jilab/MorphPT")
print(model.predict_one("cell_2p5x.png", "cell_10x.png"))
```

See [`examples/README.md`](examples/README.md) for batch prediction, the data
format, and a fine-tuning walkthrough.

## Data & input format

**CellImageNet** is a large-scale single-cell image database (~10M cells, 31
cell types, across diverse species/tissues/conditions) built from public 10x
Genomics Xenium data. Each cell has **two paired DAPI crops** — a 2.5× view
(fine nuclear morphology) and a 10× view (broader tissue context) — and a cell-type label.
_(Dataset link: TBD.)_

MorphPT consumes the **crops directly**; you only supply the two views per cell.
The only input you build is a small **manifest** listing, per cell, the two crop
paths and the label:

```csv
img_path_2p5x,img_path_10x,label
/data/crops/cell0_2p5x.png,/data/crops/cell0_10x.png,T cells
/data/crops/cell1_2p5x.png,/data/crops/cell1_10x.png,Neurons
```

- A `.csv` or `.parquet` manifest both work. The model reads the PNG crops at
  runtime (resized to 224×224, ImageNet-normalized).
- Only `img_path_2p5x`, `img_path_10x`, and the label column are required.
  `cell_id`, `tissue`, and `x/y` centroids are optional metadata (auto-filled
  if absent). For router training use `coarse_label` as the label column.
- This manifest is the universal input for `examples/predict_cell.py`,
  `trainer/train_*.py` (`--train_dir`), and `scripts/eval_moe_e2e.py`.

**Optional, not required:**
- `data/prepare_multiview_parquet.py` reproduces our CellImageNet manifest from
  the Xenium metadata. If you already have your own paired crops, skip it and
  write the 3-column manifest above directly.
- `data/preprocess_memmap.py` builds a memmap tensor **cache** for faster GPU
  throughput during large-scale training (pass `--cache_dir`). Training and
  evaluation read the PNG crops directly without it.

> Cropping/segmentation (raw slide → per-cell 2.5×/10× crops) is **out of
> scope**: MorphPT assumes the paired crops already exist.

## Repository layout

```
models/         backbone, LoRA, ScaleGate, MultiViewClassifier
data/           dataset (parquet + crops), preprocessing / caching
trainer/        router / expert / single-view training (DDP)
scripts/        end-to-end evaluation and analysis
examples/       minimal runnable vignette (inference + fine-tuning)
visiumHD/       downstream task: MorphPT router features -> spatial expression regression
configs/        configuration
slurm/          cluster launch templates
```

## Training

Router (six-way coarse stage):

```bash
python trainer/train_router.py \
    --train_dir  <manifest.parquet> --class_map <coarse_to_id.json> \
    --label_col coarse_label --fuse gate \
    --lora_rank 16 --lora_alpha 32 --batch_size 128 --epochs 35 \
    --lr_head 3e-4 --lr_lora 1e-4 --cons_weight 0.10 \
    --out_dir <out>
```

Group expert (initialized from the router checkpoint):

```bash
python trainer/train_expert.py \
    --train_dir <manifest.parquet> --class_map <expert_class_to_idx.json> \
    --label_col label --expert_name Cancer \
    --router_ckpt <router/best.pt> --fuse gate \
    --batch_size 128 --epochs 30 --lr_head 3e-4 --lr_lora 3e-5 \
    --out_dir <out>
```

`--train_dir` takes a manifest (`.csv`/`.parquet`) or a directory of parquet
shards and reads the crops directly; add `--cache_dir <memmap>` only if you want
the optional GPU-throughput cache. Multi-GPU runs use `torchrun`/SLURM (rank,
local rank, world size from the environment); see [`slurm/`](slurm/) for
templates.

## Evaluation

```bash
python scripts/eval_moe_e2e.py \
    --test_parquet <test.parquet> \
    --router_ckpt  $CKPT/router/best.pt \
    --expert_dir   $CKPT \
    --out_dir      <out>
```

Reports macro-F1, per-class metrics, confusion matrices, and confidence scores
for soft / top-1 / top-K / oracle routing.

## Downstream: VisiumHD spatial regression

[`visiumHD/`](visiumHD/) uses MorphPT's router features to predict spatial gene
expression, demonstrating transfer of the morphology-adapted representation. See
that directory for its scripts.

## Notes

- All scripts take paths via CLI flags or environment variables — no site-specific
  paths are baked in. The downstream `visiumHD/` scripts read a repo/data root from
  `MORPHPT_ROOT` (defaults to the repo root); the VisiumHD dataset/trainer read
  `VISIUM_ROOT`, `VISIUM_MORPHPT_DATA`, and `VISIUM_CACHE`. Set these to point at
  your own data.
- `CellImageNet` (the training corpus) is released separately (link above).
- Cropping/segmentation is out of scope; MorphPT operates on paired 2.5×/10×
  crops you provide.

## Citation

```bibtex
@article{cao2026visual,
  title   = {A visual foundation model for cell classification},
  author  = {Cao, Ting and Zhuang, Haotian and Zhang, Boxuan and
             Pang, Zhiping P. and Tang, Ruixiang and Liu, Dongfang and
             Ji, Zhicheng},
  year    = {2026}
}
```

## License

The **code** in this repository is released under [LICENSE](LICENSE). The
**pretrained weights** (hosted on [Hugging Face](https://huggingface.co/jilab/MorphPT))
embed Meta's DINOv3 backbone and are distributed under the **DINOv3 License** —
see the model card and `LICENSE.md` in the weights repository.
