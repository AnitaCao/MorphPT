# MorphPT training split (human subset of CellImageNet)

Label spaces and the split specification for the released MorphPT model.

- `coarse_to_id.json`   — 6 broad morphology-informed groups → id (router classes)
- `fine_to_coarse.json` — 21 fine cell types → broad group
- `fine_class_to_idx.json` — fine cell type → id
- `expert_groups.json`  — which groups have experts vs. are passthrough
- `split_manifest_seed1337.json` — split parameters (seed, per-class subsample
  targets, per-slide caps, class lists) used to build the train/val/test split
  from CellImageNet. Reproduce the cell-level split by running the data-prep
  pipeline against CellImageNet with these parameters (seed 1337).
