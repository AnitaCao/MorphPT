from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode as IM

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random


@dataclass
class AugCfg:
    enable: bool = False
    hflip: float = 0.5
    vflip: float = 0.5
    rot_deg: float = 8.0
    translate: float = 0.05
    color_jitter: bool = False
    blur: bool = False


class PairedGeomAug:
    """Sample one set of geometric params and apply to both views."""
    def __init__(self, rot_deg=8.0, translate=0.05, hflip=0.5, vflip=0.5):
        self.rot_deg = float(rot_deg)
        self.translate = float(translate)
        self.hflip = float(hflip)
        self.vflip = float(vflip)

    def __call__(self, img_a: Image.Image, img_b: Image.Image):
        if random.random() < self.hflip:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)
        if random.random() < self.vflip:
            img_a = TF.vflip(img_a)
            img_b = TF.vflip(img_b)

        if self.rot_deg > 0:
            ang = random.uniform(-self.rot_deg, self.rot_deg)
            img_a = TF.rotate(img_a, ang, interpolation=IM.BILINEAR, fill=0)
            img_b = TF.rotate(img_b, ang, interpolation=IM.BILINEAR, fill=0)

        if self.translate > 0:
            w, h = img_a.size
            max_dx = self.translate * w
            max_dy = self.translate * h
            tx = int(random.uniform(-max_dx, max_dx))
            ty = int(random.uniform(-max_dy, max_dy))
            img_a = TF.affine(img_a, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0],
                              interpolation=IM.BILINEAR, fill=0)
            img_b = TF.affine(img_b, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0],
                              interpolation=IM.BILINEAR, fill=0)

        return img_a, img_b


def build_post_transform(size: int, mean=None, std=None):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=IM.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
        transforms.Normalize(mean, std),
    ])


def build_post_transform_cached(mean=None, std=None):
    """Normalize-only transform for pre-resized uint8 tensors from cache.

    Expects uint8 tensor [1, H, W]  (output of preprocess_tiles.py).
    Returns float32 [3, H, W] normalized with ImageNet stats.
    """
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std
    _mean = torch.tensor(mean).view(3, 1, 1)
    _std  = torch.tensor(std).view(3, 1, 1)

    def _apply(t: torch.Tensor) -> torch.Tensor:
        # t: uint8 [1, H, W]  ->  float32 [3, H, W] normalized
        x = t.float() / 255.0          # [1, H, W]
        x = x.expand(3, -1, -1)        # [3, H, W]  (no copy, view)
        x = (x - _mean) / _std
        return x

    return _apply


class CellParquetBase(Dataset):
    """
    Base class that loads a Parquet table (single file or directory of shards).
    """
    def __init__(
        self,
        parquet_path: str | Path,
        class_to_idx: Dict[str, int],
        label_col: str = "label",
        tissue_col: str = "tissue",
        x_col: str = "x_centroid",
        y_col: str = "y_centroid",
        cell_id_col: str = "cell_id",
        filter_unknown: bool = False,
        unknown_labels: Optional[set[str]] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.class_to_idx = class_to_idx
        self.label_col = label_col
        self.tissue_col = tissue_col
        self.x_col = x_col
        self.y_col = y_col
        self.cell_id_col = cell_id_col

        # Load: single file (parquet or csv) or directory of parquet shards
        if self.parquet_path.is_dir():
            shards = sorted(self.parquet_path.glob("*.parquet"))
            if not shards:
                raise RuntimeError(f"No .parquet files in {self.parquet_path}")
            df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        elif self.parquet_path.is_file():
            if self.parquet_path.suffix.lower() == ".csv":
                df = pd.read_csv(self.parquet_path)
            else:
                df = pd.read_parquet(self.parquet_path)
        else:
            raise RuntimeError(f"Path not found: {self.parquet_path}")

        # Only the label column is required. Metadata columns (cell_id, tissue,
        # x/y centroid) are auto-filled when absent, so a minimal manifest of
        # just image paths + label runs directly — no need to fabricate them.
        if label_col not in df.columns:
            raise RuntimeError(
                f"Required label column '{label_col}' not found in {self.parquet_path}. "
                f"Got columns: {list(df.columns)}")
        if cell_id_col not in df.columns:
            df[cell_id_col] = [f"cell_{i}" for i in range(len(df))]
        if tissue_col not in df.columns:
            df[tissue_col] = "unknown"
        if x_col not in df.columns:
            df[x_col] = 0.0
        if y_col not in df.columns:
            df[y_col] = 0.0

        # Optional filtering
        if filter_unknown:
            if unknown_labels is None:
                unknown_labels = {"unknown", "Unknown", "UNK", "Unassigned", "NA", "N/A", "Other", ""}
            df = df[~df[label_col].isin(unknown_labels)]

        # Keep only labels in class map
        allowed = set(class_to_idx.keys())
        df = df[df[label_col].isin(allowed)].reset_index(drop=True)

        self.df = df

        # Pre-extract frequently accessed columns as numpy arrays (avoids pandas iloc overhead)
        self._labels = self.df[label_col].values
        self._tissues = self.df[tissue_col].values


class CellParquetSingleView(CellParquetBase):
    """
    Single view dataset: returns x [C,H,W], y, and either tissue or meta dict.

    If ``cache_dir`` is provided, loads pre-processed uint8 tensors from disk
    (produced by ``data/preprocess_tiles.py``) instead of decoding PNGs at
    runtime.  The cache sub-directory is inferred from ``img_col``:
        cache_dir/<col_tag>/<stem>.pt
    where col_tag = img_col.replace('img_path_', ''), e.g. "2p5x" or "10x".
    """
    def __init__(
        self,
        parquet_path: str | Path,
        class_to_idx: Dict[str, int],
        img_col: str = "img_path_2p5x",
        size: int = 224,
        mean=None,
        std=None,
        aug: AugCfg = AugCfg(enable=False),
        return_meta: bool = False,
        cache_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        super().__init__(parquet_path, class_to_idx, **kwargs)
        if img_col not in self.df.columns:
            raise RuntimeError(f"Missing image column '{img_col}' in {parquet_path}")

        self.img_col = img_col
        self.return_meta = bool(return_meta)

        # Cache mode
        self._use_cache = cache_dir is not None
        if self._use_cache:
            tag = img_col.replace("img_path_", "")
            self._cache_sub = Path(cache_dir) / tag
            self._normalize = build_post_transform_cached(mean, std)
        else:
            self.post = build_post_transform(size, mean, std)

        # Pre-extract image paths as numpy array
        self._paths = self.df[img_col].values

        # single view aug: keep it simple and safe
        self.aug = aug
        self.sv_aug = None
        if aug.enable:
            ops = []
            if aug.hflip > 0:
                ops.append(transforms.RandomHorizontalFlip(p=aug.hflip))
            if aug.vflip > 0:
                ops.append(transforms.RandomVerticalFlip(p=aug.vflip))
            if aug.rot_deg > 0:
                ops.append(transforms.RandomApply(
                    [transforms.RandomRotation(aug.rot_deg, interpolation=IM.BILINEAR, fill=0)],
                    p=0.5
                ))
            self.sv_aug = transforms.Compose(ops) if ops else None

    def __len__(self):
        return len(self.df)

    def _load_tensor(self, p: str) -> torch.Tensor:
        """Load from cache: uint8 [1,H,W] -> normalized float32 [3,H,W]."""
        stem = Path(p).stem
        cache_path = self._cache_sub / f"{stem}.pt"
        t = torch.load(cache_path, weights_only=True)  # uint8 [1,H,W]
        return self._normalize(t)  # float32 [3,H,W]

    def __getitem__(self, i: int):
        p = self._paths[i]

        if self._use_cache:
            x = self._load_tensor(p)
            if self.sv_aug is not None:
                x = self.sv_aug(x)          # tensor aug (flip/rotate on tensor)
        else:
            img = Image.open(p).convert("L")
            if self.sv_aug is not None:
                img = self.sv_aug(img)
            x = self.post(img)

        y = self.class_to_idx[self._labels[i]]

        if not self.return_meta:
            return x, y, self._tissues[i]

        row = self.df.iloc[i]
        meta = {
            "tissue": self._tissues[i],
            "x": float(row[self.x_col]),
            "y": float(row[self.y_col]),
            "cell_id": row[self.cell_id_col],
            "path": p,
        }
        return x, y, meta


class CellParquetMultiView(CellParquetBase):
    """
    Multi view dataset: returns x [2,C,H,W], y, and either tissue or meta dict.
    Expects both img cols to exist in the parquet.

    If ``cache_dir`` is provided, loads pre-processed uint8 tensors from disk
    (produced by ``data/preprocess_tiles.py``) instead of decoding PNGs.
    Sub-directories are inferred from img_col names, e.g.:
        cache_dir/2p5x/<stem>.pt
        cache_dir/10x/<stem>.pt
    """
    def __init__(
        self,
        parquet_path: str | Path,
        class_to_idx: Dict[str, int],
        img_col_a: str = "img_path_2p5x",
        img_col_b: str = "img_path_10x",
        size: int = 224,
        mean=None,
        std=None,
        aug: AugCfg = AugCfg(enable=False),
        return_meta: bool = False,
        cache_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        super().__init__(parquet_path, class_to_idx, **kwargs)
        for c in (img_col_a, img_col_b):
            if c not in self.df.columns:
                raise RuntimeError(f"Missing image column '{c}' in {parquet_path}")

        self.img_col_a = img_col_a
        self.img_col_b = img_col_b
        self.return_meta = bool(return_meta)

        # Cache mode
        self._use_cache = cache_dir is not None
        if self._use_cache:
            tag_a = img_col_a.replace("img_path_", "")
            tag_b = img_col_b.replace("img_path_", "")
            self._cache_sub_a = Path(cache_dir) / tag_a
            self._cache_sub_b = Path(cache_dir) / tag_b
            self._normalize = build_post_transform_cached(mean, std)
        else:
            self.post = build_post_transform(size, mean, std)

        # Pre-extract image paths as numpy arrays
        self._paths_a = self.df[img_col_a].values
        self._paths_b = self.df[img_col_b].values

        self.aug = aug
        self.geo = None
        self.color = None
        self.blur = None

        if aug.enable:
            self.geo = PairedGeomAug(
                rot_deg=aug.rot_deg,
                translate=aug.translate,
                hflip=aug.hflip,
                vflip=aug.vflip
            )
            if aug.color_jitter:
                self.color = transforms.ColorJitter(brightness=0.20, contrast=0.20)
            if aug.blur:
                self.blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))

    def __len__(self):
        return len(self.df)

    def _load_tensor(self, p: str, cache_sub: Path) -> torch.Tensor:
        """Load from cache: uint8 [1,H,W] -> normalized float32 [3,H,W]."""
        stem = Path(p).stem
        t = torch.load(cache_sub / f"{stem}.pt", weights_only=True)  # uint8
        return self._normalize(t)

    def __getitem__(self, i: int):
        pa = self._paths_a[i]
        pb = self._paths_b[i]

        if self._use_cache:
            xa = self._load_tensor(pa, self._cache_sub_a)
            xb = self._load_tensor(pb, self._cache_sub_b)
            # Geometric aug on tensors (paired, consistent)
            if self.geo is not None:
                # Apply same random flip/rotate to both tensors
                if random.random() < self.geo.hflip:
                    xa = TF.hflip(xa); xb = TF.hflip(xb)
                if random.random() < self.geo.vflip:
                    xa = TF.vflip(xa); xb = TF.vflip(xb)
                if self.geo.rot_deg > 0:
                    ang = random.uniform(-self.geo.rot_deg, self.geo.rot_deg)
                    xa = TF.rotate(xa, ang, interpolation=IM.BILINEAR, fill=0)
                    xb = TF.rotate(xb, ang, interpolation=IM.BILINEAR, fill=0)
        else:
            ia = Image.open(pa).convert("L")
            ib = Image.open(pb).convert("L")

            if self.geo is not None:
                ia, ib = self.geo(ia, ib)

            if self.color is not None:
                ia = self.color(ia)
                ib = self.color(ib)

            if self.blur is not None:
                ia = self.blur(ia)
                ib = self.blur(ib)

            xa = self.post(ia)
            xb = self.post(ib)

        x = torch.stack([xa, xb], dim=0)  # [2,C,H,W]
        y = self.class_to_idx[self._labels[i]]

        if not self.return_meta:
            return x, y, self._tissues[i]

        row = self.df.iloc[i]
        meta = {
            "tissue": self._tissues[i],
            "x": float(row[self.x_col]),
            "y": float(row[self.y_col]),
            "cell_id": row[self.cell_id_col],
            "path_2p5x": pa,
            "path_10x": pb,
        }
        return x, y, meta


# ── Memmap-based dataset (fast, HPC-friendly) ─────────────────────────────────

class MemmapDataset(Dataset):
    """
    Dataset that reads from preprocessed memmap arrays (output of preprocess_memmap.py).

    Replaces PIL open + resize with a simple array index:
        x_mm[i]  →  uint8 [2, H, W]  →  float32 [2, 3, H, W] normalized

    Supports:
      - Multi-view (view="both", default): returns x [2, C, H, W]
      - Single-view (view="a" or "b"):     returns x [C, H, W]
      - Augmentation (aug): applied on float tensors after normalize

    Args:
        cache_dir   : Directory produced by preprocess_memmap.py
        split_name  : e.g. "router_shards", "val_balanced", "test_shards"
        view        : "both" | "a" | "b"
        mean, std   : ImageNet defaults if None
        aug         : AugCfg (geometric aug applied on tensors)
        return_meta : If True, __getitem__ returns meta dict instead of tissue str
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split_name: str,
        view: str = "both",
        mean=None,
        std=None,
        aug: AugCfg = AugCfg(enable=False),
        return_meta: bool = False,
        class_to_idx: dict | None = None,
        label_col: str | None = None,
    ):
        assert view in ("both", "a", "b"), f"view must be 'both'|'a'|'b', got {view}"
        cache_dir = Path(cache_dir)

        # ── Load meta ────────────────────────────────────────────────────
        meta_path = cache_dir / f"{split_name}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Meta not found: {meta_path}\n"
                f"Run preprocess_memmap.py --splits {split_name} first."
            )
        import json
        meta = json.loads(meta_path.read_text())
        n = meta["n"]
        shape = tuple(meta["shape"])  # (N, 2, H, W)

        # ── Memory-map image array (read-only, OS handles paging) ────────
        x_path = cache_dir / f"{split_name}_x.dat"
        self._x = np.memmap(str(x_path), dtype=np.uint8, mode="r", shape=shape)

        # ── Load small metadata arrays fully into RAM ─────────────────────
        self._y = np.load(str(cache_dir / f"{split_name}_y.npy"))
        self._tissues = np.load(
            str(cache_dir / f"{split_name}_tissue.npy"), allow_pickle=True
        )
        # Coarse/fine label strings (used by per_fine_routing_accuracy)
        coarse_path = cache_dir / f"{split_name}_coarse_label.npy"
        self._coarse_labels = np.load(str(coarse_path), allow_pickle=True) \
            if coarse_path.exists() else None
        fine_path = cache_dir / f"{split_name}_fine_label.npy"
        self._fine_labels = np.load(str(fine_path), allow_pickle=True) \
            if fine_path.exists() else None

        self.view = view
        self.return_meta = return_meta
        self.meta = meta
        
        # ── Remap labels if requested ────────────────────────────────────
        self._valid_indices = np.arange(len(self._y), dtype=np.int64)
        if class_to_idx is not None and label_col is not None:
            source_labels = None
            if label_col == meta.get("fine_col", "label"):
                source_labels = self._fine_labels
            elif label_col == meta.get("label_col", "coarse_label"):
                source_labels = self._coarse_labels
            elif label_col == "label" and "fine_col" not in meta:
                source_labels = self._fine_labels
                
            if source_labels is not None:
                new_y = np.empty_like(self._y)
                valid = []
                for i, lbl in enumerate(source_labels):
                    y_val = class_to_idx.get(lbl, -1)
                    new_y[i] = y_val
                    if y_val != -1:
                        valid.append(i)
                self._y = new_y
                self._valid_indices = np.array(valid, dtype=np.int64)
                filtered = len(new_y) - len(valid)
                if filtered > 0:
                    print(f"MemmapDataset: Filtered {filtered} samples not in class_map.")
            else:
                print(f"Warning: Could not find label array for column {label_col}. Using default y.")

        # ── Normalize constants ───────────────────────────────────────────
        _mean = torch.tensor(mean or IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
        _std  = torch.tensor(std  or IMAGENET_DEFAULT_STD ).view(3, 1, 1)
        self._mean = _mean
        self._std  = _std

        # ── Optional augmentation (tensor-level) ─────────────────────────
        self.aug = aug
        self._aug_ops = None
        if aug.enable:
            ops = []
            if aug.hflip > 0:
                ops.append(transforms.RandomHorizontalFlip(p=aug.hflip))
            if aug.vflip > 0:
                ops.append(transforms.RandomVerticalFlip(p=aug.vflip))
            if aug.rot_deg > 0:
                ops.append(transforms.RandomApply(
                    [transforms.RandomRotation(aug.rot_deg,
                                               interpolation=IM.BILINEAR, fill=0)],
                    p=0.5,
                ))
            self._aug_ops = transforms.Compose(ops) if ops else None

    def __len__(self):
        return len(self._valid_indices)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """uint8 [H,W] or [2,H,W] → normalized float32 [3,H,W] or [2,3,H,W]."""
        if arr.ndim == 2:
            # Single view: [H, W] → [3, H, W]
            t = torch.from_numpy(arr).float() / 255.0          # [H, W]
            t = t.unsqueeze(0).expand(3, -1, -1).clone()       # [3, H, W]
            t = (t - self._mean) / self._std
            return t
        else:
            # Both views: [2, H, W] → [2, 3, H, W]
            out = []
            for v in range(arr.shape[0]):
                t = torch.from_numpy(arr[v]).float() / 255.0   # [H, W]
                t = t.unsqueeze(0).expand(3, -1, -1).clone()   # [3, H, W]
                t = (t - self._mean) / self._std
                out.append(t)
            return torch.stack(out, dim=0)                      # [2, 3, H, W]

    def _apply_aug(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation. For 'both' view, same random params for both views."""
        if self._aug_ops is None:
            return x
        if self.view != "both":
            return self._aug_ops(x)
        # Paired aug: same random state for both views
        xa, xb = x[0], x[1]
        if self.aug.hflip > 0 and random.random() < self.aug.hflip:
            xa = TF.hflip(xa); xb = TF.hflip(xb)
        if self.aug.vflip > 0 and random.random() < self.aug.vflip:
            xa = TF.vflip(xa); xb = TF.vflip(xb)
        if self.aug.rot_deg > 0 and random.random() < 0.5:
            ang = random.uniform(-self.aug.rot_deg, self.aug.rot_deg)
            xa = TF.rotate(xa, ang, interpolation=IM.BILINEAR, fill=0)
            xb = TF.rotate(xb, ang, interpolation=IM.BILINEAR, fill=0)
        if self.aug.translate > 0 and random.random() < 0.5:
            _, h, w = xa.shape
            tx = int(random.uniform(-self.aug.translate * w, self.aug.translate * w))
            ty = int(random.uniform(-self.aug.translate * h, self.aug.translate * h))
            xa = TF.affine(xa, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0],
                           interpolation=IM.BILINEAR, fill=0)
            xb = TF.affine(xb, angle=0, translate=[tx, ty], scale=1.0, shear=[0, 0],
                           interpolation=IM.BILINEAR, fill=0)
        return torch.stack([xa, xb], dim=0)

    def __getitem__(self, i: int):
        idx = self._valid_indices[i]
        # .copy() is REQUIRED: breaks the memmap reference before handing
        # to DataLoader workers (avoids shared-memory issues)
        arr = self._x[idx].copy()   # uint8 [2, 224, 224]

        if self.view == "a":
            x = self._to_tensor(arr[0])    # [3, H, W]
        elif self.view == "b":
            x = self._to_tensor(arr[1])    # [3, H, W]
        else:
            x = self._to_tensor(arr)       # [2, 3, H, W]

        x = self._apply_aug(x)
        y = int(self._y[idx])

        if not self.return_meta:
            return x, y, self._tissues[idx]

        meta = {"tissue": self._tissues[idx], "index": idx}
        return x, y, meta