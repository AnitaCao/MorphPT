---
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
library_name: pytorch
pipeline_tag: image-classification
tags:
  - biology
  - single-cell
  - cell-type-classification
  - microscopy
  - DAPI
  - vision-transformer
  - lora
  - dinov3
---

# MorphPT

**MorphPT** classifies cell types from **DAPI-stained nuclei**. Each cell is
described by two co-registered grayscale crops — a **2.5×** view (fine nuclear
morphology) and a **10×** view (broader tissue context) — encoded by a shared **DINOv3
ViT-B** backbone with **LoRA** adaptation and fused by a learnable **ScaleGate**
module. A coarse **router** predicts one of six broad morphology-informed
groups, and group **experts** resolve the fine cell type via soft routing.

Here **2.5×** and **10×** denote crop field-of-view *scales* (how much area
around the cell each crop spans), **not** microscope objective magnification:
the 2.5× crop is a tight view of the nucleus, while the 10× crop covers more
surrounding tissue.

- 💻 **Code:** _forthcoming_
- 🗂️ **CellImageNet benchmark dataset:** _forthcoming_
- 📄 **Paper:** "A visual foundation model for cell classification" — _citation forthcoming_

## Files

```
README.md                       # this card
LICENSE.md                      # DINOv3 License (required)
FINE_LABELS.md                  # fine cell types per coarse group (reference)
router/best.pt                  router/coarse_to_id.json     # 6-way coarse router
expert_Cancer/best.pt           expert_Cancer/class_to_idx.json
expert_Lymphoid/best.pt         expert_Lymphoid/class_to_idx.json
expert_Neuroglial/best.pt       expert_Neuroglial/class_to_idx.json
expert_Tissue_Vascular/best.pt  expert_Tissue_Vascular/class_to_idx.json
splits/                         # split spec + label maps (reproducibility)
```

Each checkpoint is self-contained and includes model weights, `args`, and label metadata needed for inference. The architecture is reconstructed from `args` at load time. The frozen DINOv3 ViT-B/16 backbone weights are embedded in each checkpoint.


## Label space

The released model covers **21 fine cell types** grouped into **6 broad
morphology-informed groups** (a human-tissue subset of CellImageNet):

| Group | Type of stage | Fine cell types |
|---|---|---|
| Cancer | expert | Colon / Liver / Lung / Ovary / Pancreas / Skin cancer cells |
| Lymphoid | expert | B cells, NK cells, T cells |
| Neuroglial | expert | Astrocytes, Microglia, Neurons, Oligodendrocytes |
| Tissue_Vascular | expert | Endothelial, Epithelial, Fibroblasts, Myeloid, Pericytes, Smooth muscle cells |
| Stromal | passthrough | Stromal cells |
| Stem_Progenitor | passthrough | Stem and progenitor cells |

Single-type groups (Stromal, Stem_Progenitor) are passthrough: the router label
maps directly to the cell type with no expert.

## Usage

Install the MorphPT package (code repository forthcoming), then:

```python
from morphpt import MorphPTPredictor

model = MorphPTPredictor.from_pretrained("jilab/MorphPT")   # downloads the weights
out = model.predict_one("cell_2p5x.png", "cell_10x.png")
print(out["pred"], out["coarse_group"], round(out["confidence"], 3))
```

Or download the weights and run the example CLI:

```bash
hf download jilab/MorphPT --repo-type model --local-dir morphpt_weights
python examples/predict_cell.py --weights_dir morphpt_weights \
    --img_2p5x cell_2p5x.png --img_10x cell_10x.png
```

Inputs are paired grayscale DAPI crops of the same cell. Crops are resized to
224×224 and ImageNet-normalized internally.

Prediction uses soft routing. For multi-type groups, expert probabilities are
weighted by the router probability for that group. For passthrough groups, the
router probability is assigned directly to the corresponding fine label. The
final cell type is chosen by the highest fine-type score:

```
s(y) = Σ_{g in experts} q(g) p_g(y)  +  Σ_{g in passthrough} q(g) 1[y = y_g]
```

## Training data

Trained on a **human-tissue subset of CellImageNet**, subsampled per class to
mitigate class imbalance. CellImageNet is a large-scale single-cell DAPI image
database (~10M cells; 28 human + 14 mouse tissues) built from public 10x
Genomics Xenium data; the full dataset is released separately as a benchmark.
The split specification used to train this model (seed, per-class subsample
targets, and label maps) is provided in this repository under `splits/` for
reproducibility. This checkpoint does **not** cover mouse tissues or cell types
outside the 21 listed above.

## Intended use and limitations

**Intended use.** MorphPT is intended for research use in fine-grained cell-type classification from DAPI nuclear-morphology crops in human tissue, including Xenium-style single-cell imagery. It may also be useful as a morphology-pretrained backbone for related research tasks.

**Not for clinical or diagnostic use.** This model is a research artifact, not a medical device, and should not be used for clinical diagnosis, treatment decisions, or other patient-care decisions.

**Scope and limitations.**
- Human tissue and DAPI input only. Mouse tissue, other stains, and other imaging modalities are unvalidated.
- Inputs must be correctly framed 2.5× and 10× crops of the same cell. Resizing fixes pixel size, not field of view. Mismatched crop scales may silently degrade predictions.
- The model was trained on a per-class-subsampled subset of CellImageNet. Rare cell types and out-of-distribution tissues may have weaker performance.
- Coarse routing is the main error source. Cells near coarse-group boundaries are likely to be less reliable.

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

This work builds on DINOv3 (Meta AI) and the `timm` library; please also cite
DINOv3 and `timm` (Ross Wightman, *PyTorch Image Models*).

## License

This model is a **derivative work of Meta's DINOv3** — the DINOv3 ViT-B backbone
weights are embedded in every checkpoint — and is therefore distributed under
the **DINOv3 License**, not a standard permissive license. A copy of the DINOv3
License is included in this repository as **`LICENSE.md`**, copied from the
official `facebookresearch/dinov3` repository. Authoritative page:
<https://ai.meta.com/resources/models-and-libraries/dinov3-license/>

By using these weights you agree to the DINOv3 License terms, including:
- redistribution of the weights or any derivative works must be **under the
  DINOv3 License and must include a copy of it**;
- products, websites, or publications built on this model must display
  **"Built with DINOv3"** and acknowledge the use of DINOv3;
- DINOv3's prohibited-use and export-control terms apply.

Components authored by the MorphPT team (LoRA adapters, ScaleGate, classification
heads, label maps, and inference code) are released for research use consistent
with the inherited DINOv3 License above.
