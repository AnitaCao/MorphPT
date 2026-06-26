# MorphPT — minimal usage vignette

A **self-contained, runnable** example showing how to (1) run the pretrained
MorphPT model on real cells to get cell-type predictions, and (2) fine-tune a
group expert on your own data.

Contents:

```
examples/
├── predict_cell.py          # inference: image(s) -> cell-type prediction
├── real_cells/              # 10 REAL example cells (DAPI crops) + cells.csv
│   ├── <cell_id>_2p5x.png    #   2.5x (fine nuclear morphology) crop
│   ├── <cell_id>_10x.png     #   10x  (broader tissue context) crop
│   └── cells.csv             #   manifest with ground-truth labels
├── make_minimal_dataset.py  # synthesize a tiny parquet (training data format)
└── README.md
```

---

## 0. Setup

MorphPT needs PyTorch, `timm` (DINOv3 ViT-B backbone), `pandas`, `pyarrow`,
`pillow`, `numpy`. Run from the repo root with it on `PYTHONPATH`:

```bash
cd /path/to/MorphPT
export PYTHONPATH=$PWD
export CKPT=/path/to/pretrained        # HF snapshot root: holds router/ and expert_<Group>/
```

A single GPU is recommended (CPU works but is slow).

---

## 1. What "one cell" means in MorphPT

MorphPT is a **multi-view** model. A *cell* is represented by **two
co-registered DAPI crops of the same nucleus**:

- a **2.5×** crop — smaller context scale, captures fine nuclear morphology, and
- a **10×** crop — larger context scale, captures broader tissue context.

Both views go through the shared DINOv3 ViT-B + LoRA backbone and are fused by
ScaleGate. So to use the pretrained model you provide **both crops** of a cell;
a single image cannot be fed to MorphPT. (The crops are just the same nucleus
imaged at two zoom levels — you already have both.)

Prediction follows the paper's two-stage soft-routing rule: a router predicts
one of six broad groups, group experts predict the fine cell type, and their
outputs are combined as `s(y) = Σ_g q(g)·p_expert_g(y)`.

---

## 2. Use the pretrained model (real cells, real images)

`real_cells/` ships 10 **real** cells (one per major type, spanning all six
broad groups) as raw grayscale PNG crops.

**Predict a single cell** from its two crop files:

```bash
python examples/predict_cell.py --weights_dir $CKPT \
    --img_2p5x examples/real_cells/Xenium_V1_hLung_cancer_n126294_2p5x.png \
    --img_10x  examples/real_cells/Xenium_V1_hLung_cancer_n126294_10x.png
```
```
Predicted cell type : Lung cancer cells
Routed group        : Cancer
Confidence (Mr*maxPe): 0.757
Top-3:
  Lung cancer cells            0.861
  Epithelial cells             0.094
  NK cells                     0.032
```

**Predict all bundled cells** (reads `real_cells/cells.csv`, compares to the
ground-truth labels):

```bash
python examples/predict_cell.py --weights_dir $CKPT \
    --manifest examples/real_cells/cells.csv
```

Expected output — all 10 correct:

```
cell_id                                       true                      pred                      group             conf
OK Xenium_V1_hColon_Cancer_Base_n385399       Colon cancer cells        Colon cancer cells        Cancer           0.459
OK Xenium_V1_hLung_cancer_n126294             Lung cancer cells         Lung cancer cells         Cancer           0.757
OK Xenium_V1_hPancreas_nondiseased_n74262     T cells                   T cells                   Lymphoid         0.763
OK Xenium_V1_FFPE_Human_Brain_Healthy_n5940   Neurons                   Neurons                   Neuroglial       0.989
OK Xenium_V1_FFPE_Human_Brain_Alzheimers_...  Oligodendrocytes          Oligodendrocytes          Neuroglial       0.720
OK Xenium_V1_hSkin_nondiseased_section_2_...  Epithelial cells          Epithelial cells          Tissue_Vascular  0.421
OK Xenium_V1_hLung_cancer_n80809              Endothelial cells         Endothelial cells         Tissue_Vascular  0.488
OK Xenium_V1_hKidney_cancer_n39388            Myeloid cells             Myeloid cells             Tissue_Vascular  0.585
OK Xenium_human_Lymph_Node_FFPE_n535242       Stromal cells             Stromal cells             Stromal          0.444
OK Xenium_V1_hColon_Cancer_Add_on_n7696       Stem and progenitor cells Stem and progenitor cells Stem_Progenitor  0.929

Correct: 10/10  (accuracy 1.000)
```

To predict your **own** cells, point `--img_2p5x`/`--img_10x` at your crops (any
size; resized to 224×224, ImageNet-normalized internally), or list them in a CSV
with `img_2p5x,img_10x` columns and use `--manifest`.

---

## 3. Data format for training / large-scale evaluation

For training and batched evaluation, MorphPT reads a **manifest** (`.csv` or
`.parquet`, one row per cell). Only three columns are required:

| column | required | meaning |
|---|---|---|
| `img_path_2p5x` | ✅ | path to the 2.5× grayscale PNG crop |
| `img_path_10x` | ✅ | path to the 10× grayscale PNG crop |
| `label` (or `coarse_label` for the router) | ✅ | cell-type label |
| `cell_id`, `tissue`, `x_centroid`, `y_centroid` | optional | metadata; auto-filled if absent |

Crops are read directly via `Image.open` ([`data/dataset.py`](../data/dataset.py)) —
you only provide the two views per cell. A memmap **cache** of pre-resized
tensors can optionally be built (`data/preprocess_memmap.py`, pass `--cache_dir`)
for faster GPU throughput, but it is **not required**; the crops-via-manifest
path above is the reference format. Cropping/segmentation is out of scope.

Generate a tiny **synthetic** manifest to see the format and drive the
fine-tuning smoke test below (crops here are random blobs — format demo only):

```bash
python examples/make_minimal_dataset.py --out_dir examples/minimal_data --n 48
```

For the full evaluation pipeline (all routing strategies, per-class metrics,
confusion matrices, confidence scores) on a real manifest, use
[`scripts/eval_moe_e2e.py`](../scripts/eval_moe_e2e.py).

---

## 4. Fine-tune / train a group expert

Each expert is **initialized from the trained router** (inheriting the DINOv3
backbone, LoRA adapters, ScaleGate, and per-view norms) and re-initializes only
the classification head for the group's fine cell types
([`trainer/train_expert.py`](../trainer/train_expert.py)).

Fine-tune a **Cancer** expert on the synthetic manifest (1-epoch smoke test):

```bash
python trainer/train_expert.py \
    --train_dir   examples/minimal_data/cells.parquet \
    --class_map   examples/minimal_data/expert_class_maps/Cancer.json \
    --label_col   label \
    --expert_name Cancer \
    --router_ckpt $CKPT/router/best.pt \
    --fuse gate --use_lora 1 --lora_rank 16 --lora_alpha 32 \
    --batch_size 4 --epochs 1 --val_frac 0.25 --eval_every 1 \
    --lr_head 3e-4 --lr_lora 3e-5 --workers 2 \
    --out_dir examples/runs/expert_Cancer_demo \
    --wandb_project ''        # empty disables Weights & Biases
```

Notes:
- `--train_dir` accepts a single `.parquet` or a directory of shards. Only cells
  whose `label` is in `--class_map` are kept, so the full manifest + the
  `Cancer.json` map trains on Cancer cells only; a `--val_frac` split is held
  out for checkpoint selection (best = mean of macro-F1 and accuracy).
- `--workers` must be `> 0` (loaders use `persistent_workers`).
- Outputs: `best.pt`, periodic `epoch_*.pt`, `args.json`, `train_log.json`,
  `per_class.json`, `confusion_matrix.json`.
- Real training uses more epochs (experts: 30) and the real, class-balanced
  shards; this command is only a smoke test.

The router (six-way coarse stage) is trained the same way with
[`trainer/train_router.py`](../trainer/train_router.py), using
`coarse_to_id.json` as `--class_map` and `coarse_label` as `--label_col`.
Multi-GPU runs use `torchrun`/SLURM (see [`slurm/`](../slurm/)).
