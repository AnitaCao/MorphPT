"""
visium_dataset.py
─────────────────
VisiumHDPredictionDataset backed by memmap built by build_memmap.py.

Row order guarantee: meta.csv is sorted by cell_id. All images_{scale}.npy
arrays are written in the same order. mmap_idx i always refers to the same
cell across scales — safe for multiview training.

Return shapes:
  fuse='identity'  → imgs (3, H, W),    expr (1000,)
  fuse='gate'      → imgs (2, 3, H, W), expr (1000,)
    Training loop does: x_cat = torch.cat([x[:,0], x[:,1]], dim=0)
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

logger = logging.getLogger(__name__)

_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
_STD  = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)


class VisiumHDPredictionDataset(Dataset):
    """
    Args:
        cache_dir    : directory produced by cache_visium_dataset.py
        split        : "train", "val", "test", or "all"
        scales       : list of scales, e.g. ["10.0x"] or ["2.5x", "10.0x"]
        fuse         : "identity" (single scale) or "gate" (dual scale)
        augment      : color jitter + flips (use for train split only)
        split_layout : custom layout folder name under splits/ (default: "default")
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split:     Literal["train", "val", "test", "all"] = "train",
        scales:    list[str] = None,
        fuse:      str = "identity",
        augment:   bool = False,
        split_type: str = "spatial",
        split_layout: str = "default",
    ):
        if scales is None:
            scales = ["10.0x"]
        assert fuse in ("identity", "gate")
        if fuse == "gate"     and len(scales) != 2:
            raise ValueError(f"fuse='gate' needs exactly 2 scales, got {scales}")
        if fuse == "identity" and len(scales) != 1:
            raise ValueError(f"fuse='identity' needs exactly 1 scale, got {scales}")

        self.cache_dir = Path(cache_dir)
        self.scales    = scales
        self.fuse      = fuse
        self.augment   = augment

        # ── Memmap arrays (zero-copy, shared across workers) ───────────────
        self._images = []
        for scale in scales:
            arr_path = self.cache_dir / f"images_{scale}.npy"
            if not arr_path.exists():
                raise FileNotFoundError(
                    f"{arr_path} not found. Run cache_visium_dataset.py --scales {scale}"
                )
            logger.info(f"Opening memmap: {arr_path}")
            self._images.append(np.load(str(arr_path), mmap_mode="r"))

        # ── Expression layer (zero-copy shared) ────────────────────────────
        self._expr = np.load(str(self.cache_dir / "expr.npy"), mmap_mode="r")
        
        # ── Layout Routing & Normalization Stats ───────────────────────────
        layout_dir = self.cache_dir / "splits" / split_layout
        
        if layout_dir.exists():
            logger.info(f"Using decoupled split layout: '{split_layout}' from {layout_dir}")
            stats = np.load(str(layout_dir / "expr_stats.npz"))
            meta  = pd.read_csv(self.cache_dir / "meta.csv")
            splits = pd.read_csv(layout_dir / "splits.csv")
            # Dynamic join connecting canonical mapping and lightweight splits
            meta = meta.merge(splits, on="mmap_idx", how="inner")
        else:
            logger.info(f"Layout dir '{layout_dir}' not found. Falling back to root cache metadata.")
            stats_file = ("expr_stats_random.npz" if split_type == "random" else "expr_stats.npz")
            stats = np.load(str(self.cache_dir / stats_file))
            meta_file = ("meta_random_split.csv" if split_type == "random" else "meta.csv")
            meta = pd.read_csv(self.cache_dir / meta_file)
            if "split" not in meta.columns:
                meta["split"] = "train"

        self.gene_mean = torch.from_numpy(stats["gene_mean"])
        self.gene_std  = torch.from_numpy(stats["gene_std"])

        # ── Split filter ───────────────────────────────────────────────────
        if split == "all":
            df = meta[meta["split"] != "excluded"].copy()
        else:
            df = meta[meta["split"] == split].copy()
        df = df.reset_index(drop=True)
        logger.info(f"Split '{split}': {len(df):,} cells")

        # Sanity check: mmap_idx must be within array bounds
        max_idx = df["mmap_idx"].max()
        assert max_idx < len(self._images[0]), \
            f"mmap_idx {max_idx} exceeds memmap length {len(self._images[0])}"

        self._mmap_rows = df["mmap_idx"].values.astype(np.int64)
        self._cell_ids  = df["cell_id"].values

        # ── Augmentation ───────────────────────────────────────────────────
        self._aug = None
        if augment:
            self._aug = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(90),
                T.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.05),
            ])

    def __len__(self):
        return len(self._mmap_rows)

    def _load_img(self, mmap_arr, row: int) -> torch.Tensor:
        """uint8 (H,W,3) → float32 normalized (3,H,W)."""
        img = torch.from_numpy(mmap_arr[row].copy())   # copy: avoids memmap stride issues
        img = img.permute(2, 0, 1).float().div_(255.0)
        if self._aug is not None:
            img = self._aug(img)
        return (img - _MEAN) / _STD

    def __getitem__(self, idx: int):
        row = int(self._mmap_rows[idx])

        # ── Images ────────────────────────────────────────────────────────
        if self.fuse == "identity":
            imgs = self._load_img(self._images[0], row)     # (3, H, W)
        else:
            # (2, 3, H, W) — model forward slices [:, 0] and [:, 1]
            imgs = torch.stack([
                self._load_img(self._images[0], row),       # scale[0]
                self._load_img(self._images[1], row),       # scale[1]
            ], dim=0)

        # ── Expression ────────────────────────────────────────────────────
        expr = torch.from_numpy(self._expr[row].copy())     # (1000,) already log1p
        expr = (expr - self.gene_mean) / self.gene_std

        return imgs, expr, {"cell_id": self._cell_ids[idx]}