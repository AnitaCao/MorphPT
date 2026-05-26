"""
visium_dataset_raw.py
─────────────────────
PIL-based Visium HD dataset — no preprocessing required.
Slower than memmap but useful for quick experiments or verification.

Key fixes vs previous version:
  1. Multiview augmentation: geometric transforms (flip/rotate) are sampled
     once and applied identically to both views. Color jitter is applied
     independently (different stain intensity is realistic per crop).
  2. Split parameters are explicit args — raw and memmap are directly
     comparable when called with the same grid_size/test_tiles/val_tiles/
     min_cells/seed/buffer_zone settings.
  3. buffer_zone support added (was missing before).
"""

import logging
from pathlib import Path
from typing import Literal
import random

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

logger = logging.getLogger(__name__)

_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
_STD  = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)


def _assign_tile_splits(df, grid_size, test_tiles, val_tiles,
                        min_cells, buffer_zone, seed):
    """
    Identical logic to build_memmap.py:compute_grid_splits.
    Must be kept in sync if that function changes.
    """
    rng = np.random.default_rng(seed)

    x_min, x_max = df["x_centroid"].min(), df["x_centroid"].max()
    y_min, y_max = df["y_centroid"].min(), df["y_centroid"].max()
    x_step = (x_max - x_min) / grid_size * 1.0001
    y_step = (y_max - y_min) / grid_size * 1.0001

    df = df.copy()
    df["x_bin"]   = np.floor((df["x_centroid"] - x_min) / x_step).astype(int).clip(0, grid_size - 1)
    df["y_bin"]   = np.floor((df["y_centroid"] - y_min) / y_step).astype(int).clip(0, grid_size - 1)
    df["tile_id"] = df["y_bin"] * grid_size + df["x_bin"]

    tile_counts = df["tile_id"].value_counts()
    eligible    = tile_counts[tile_counts >= min_cells].index.values

    needed = test_tiles + val_tiles
    if len(eligible) < needed:
        raise ValueError(f"Only {len(eligible)} eligible tiles, need {needed}.")

    chosen   = rng.choice(eligible, size=needed, replace=False)
    test_set = set(int(t) for t in chosen[:test_tiles])
    val_set  = set(int(t) for t in chosen[test_tiles:])

    buffer_set = set()
    if buffer_zone:
        for tid in test_set:
            tx, ty = tid // grid_size, tid % grid_size
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ntx, nty = tx + dx, ty + dy
                    if 0 <= ntx < grid_size and 0 <= nty < grid_size:
                        nb = ntx * grid_size + nty
                        if nb not in test_set and nb not in val_set:
                            buffer_set.add(nb)

    def _assign(tid):
        if tid in test_set:   return "test"
        if tid in val_set:    return "val"
        if tid in buffer_set: return "excluded"
        return "train"

    df["split"] = df["tile_id"].map(_assign)
    counts = df["split"].value_counts().to_dict()
    logger.info(f"Split counts: { {s: counts.get(s,0) for s in ['train','val','test','excluded']} }")
    return df


class VisiumHDPredictionDatasetRaw(Dataset):
    """
    Args:
        root_dir         : /hpc/group/jilab/boxuan/visiumHD/human_crc
        morphpt_data_dir : directory containing spatial.csv
        split            : "train", "val", "test", or "all"
        scales           : e.g. ["10.0x"] or ["2.5x", "10.0x"]
        fuse             : "identity" or "gate"
        img_variant      : "raw", "mask_target", "mask_context"
        img_size         : resize to (img_size, img_size)
        augment          : stochastic augmentation (train only)
        grid_size        : must match build_memmap.py --grid_size
        test_tiles        : must match build_memmap.py --test_tiles
        val_tiles        : must match build_memmap.py --val_tiles
        min_cells        : must match build_memmap.py --min_cells
        buffer_zone      : must match build_memmap.py --buffer_zone
        seed             : must match build_memmap.py --seed
    """

    def __init__(
        self,
        root_dir:         str | Path = "/hpc/group/jilab/boxuan/visiumHD/human_crc",
        morphpt_data_dir: str | Path = "/hpc/group/jilab/hz/MorphPT/data/visiumHD/human_crc",
        split:   Literal["train", "val", "test", "all"] = "train",
        scales:  list[str]  = None,
        fuse:    str        = "identity",
        img_variant: str    = "raw",
        img_size:    int    = 224,
        augment:     bool   = False,
        # Split params — must match build_memmap.py settings for comparability
        grid_size:   int    = 5,
        test_tiles:  int    = 4,
        val_tiles:   int    = 3,
        min_cells:   int    = 300,
        buffer_zone: bool   = False,
        seed:        int    = 42,
    ):
        if scales is None:
            scales = ["10.0x"]
        assert fuse in ("identity", "gate")
        if fuse == "gate"     and len(scales) != 2:
            raise ValueError(f"fuse='gate' needs 2 scales, got {scales}")
        if fuse == "identity" and len(scales) != 1:
            raise ValueError(f"fuse='identity' needs 1 scale, got {scales}")

        self.root_dir = Path(root_dir)
        self.scales   = scales
        self.fuse     = fuse
        self.img_size = img_size
        self.augment  = augment
        img_col       = f"{img_variant}_img_path"

        # ── Canonical cell list (intersection across all scales + spatial) ─
        spatial   = pd.read_csv(Path(morphpt_data_dir) / "spatial.csv")
        valid_ids = set(spatial["cell_id"].astype(str))

        scale_img_paths = {}
        for scale in scales:
            meta = pd.read_csv(self.root_dir / f"meta/{scale}/human_crc.csv")
            if img_col not in meta.columns:
                raise ValueError(f"'{img_col}' not in meta for scale {scale}")
            valid_ids &= set(meta["cell_id"].astype(str))
            scale_img_paths[scale] = dict(
                zip(meta["cell_id"].astype(str), meta[img_col])
            )

        canonical_ids = sorted(valid_ids)   # same sort as build_memmap
        df = pd.DataFrame({"cell_id": canonical_ids})
        df = df.merge(spatial[["cell_id", "x_centroid", "y_centroid"]],
                      on="cell_id", how="left")

        # ── Spatial split ──────────────────────────────────────────────────
        df = _assign_tile_splits(
            df, grid_size=grid_size, test_tiles=test_tiles,
            val_tiles=val_tiles, min_cells=min_cells,
            buffer_zone=buffer_zone, seed=seed,
        )

        if split == "all":
            df_sel = df[df["split"] != "excluded"].copy()
        else:
            df_sel = df[df["split"] == split].copy()
        df_sel = df_sel.reset_index(drop=True)
        logger.info(f"Split '{split}': {len(df_sel):,} cells")

        # ── Expression matrix ──────────────────────────────────────────────
        X_sparse   = sio.mmread(self.root_dir / "expr/expr.mtx").tocsr()
        cells_list = pd.read_csv(self.root_dir / "expr/cells.txt",
                                 header=None, names=["cell_id"])["cell_id"].tolist()
        if X_sparse.shape[0] != len(cells_list):
            X_sparse = X_sparse.T.tocsr()
        self.X_dense         = np.asarray(X_sparse.todense(), dtype=np.float32)
        self._expr_id_to_row = {str(cid): i for i, cid in enumerate(cells_list)}

        # Gene stats from train cells only
        train_ids  = df[df["split"] == "train"]["cell_id"].tolist()
        train_rows = [self._expr_id_to_row[cid] for cid in train_ids
                      if cid in self._expr_id_to_row]
        X_train        = self.X_dense[train_rows, :]
        self.gene_mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
        self.gene_std  = torch.tensor(
            np.clip(X_train.std(axis=0), 1e-5, None), dtype=torch.float32
        )

        # ── Pre-extract per-sample arrays ─────────────────────────────────
        self._cell_ids  = df_sel["cell_id"].values
        self._expr_rows = np.array([self._expr_id_to_row[cid]
                                    for cid in self._cell_ids])
        self._img_paths = {
            scale: np.array([
                str(self.root_dir / scale_img_paths[scale][cid])
                for cid in self._cell_ids
            ])
            for scale in scales
        }

        # ── Transforms ────────────────────────────────────────────────────
        # Base: resize + ToTensor only — NO normalize yet.
        # Normalize must come AFTER ColorJitter, which requires [0, 1] input.
        self._base_transform = T.Compose([
            T.Resize((img_size, img_size),
                     interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),   # → float32 [0, 1]
        ])
        # Color jitter applied independently per view (different stain intensity
        # per crop is realistic; do not synchronize color augmentation).
        # Applied while tensor is still in [0, 1], before normalize.
        self._color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ) if augment else None

    def _load_and_resize(self, path: str) -> torch.Tensor:
        """Load image, resize → float32 [0, 1] tensor. No normalize yet."""
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self._base_transform(img)   # (3, H, W) in [0, 1]

    def _apply_geometric(self, img: torch.Tensor,
                         hflip: bool, vflip: bool, angle: int) -> torch.Tensor:
        """Apply a pre-sampled geometric transform to a single image tensor."""
        if hflip:
            img = TF.hflip(img)
        if vflip:
            img = TF.vflip(img)
        if angle != 0:
            img = TF.rotate(img, angle)
        return img

    def __len__(self):
        return len(self._cell_ids)

    def __getitem__(self, idx: int):
        # ── Sample geometric augmentation ONCE for both views ──────────────
        # Both views of the same cell get identical flips/rotation so spatial
        # correspondence is preserved. Color jitter is per-view (independent).
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            angle = random.choice([0, 90, 180, 270])
        else:
            hflip = vflip = False
            angle = 0

        def load_view(scale: str) -> torch.Tensor:
            img = self._load_and_resize(self._img_paths[scale][idx])  # [0,1]
            img = self._apply_geometric(img, hflip, vflip, angle)
            if self._color_jitter is not None:
                img = self._color_jitter(img)        # still [0,1] ✓
            return (img - _MEAN) / _STD              # normalize last

        # ── Images ────────────────────────────────────────────────────────
        if self.fuse == "identity":
            imgs = load_view(self.scales[0])          # (3, H, W)
        else:
            imgs = torch.stack([
                load_view(self.scales[0]),            # (3, H, W)
                load_view(self.scales[1]),            # (3, H, W)
            ], dim=0)                                 # (2, 3, H, W)

        # ── Expression ────────────────────────────────────────────────────
        expr = torch.from_numpy(self.X_dense[self._expr_rows[idx]].copy())
        expr = (expr - self.gene_mean) / self.gene_std

        return imgs, expr, {"cell_id": self._cell_ids[idx]}