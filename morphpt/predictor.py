"""
MorphPT inference API.

`MorphPTPredictor` loads a trained MorphPT model (a coarse router + group
experts) and predicts a fine cell type from a cell's paired DAPI crops — the
2.5x (fine nuclear morphology) view and the 10x (broader tissue context) view.

The model structure (which groups have experts, the coarse/fine label spaces,
and the passthrough groups) is derived from the checkpoint files themselves, so
no hard-coded label mappings are needed:

    weights_dir/
        router/best.pt            (+ coarse_to_id.json)
        expert_<Group>/best.pt    (+ class_to_idx.json)   one per multi-type group
        splits/fine_to_coarse.json   (optional; gives passthrough fine labels)

Prediction uses soft routing:
    s(y) = Σ_{g∈experts} q(g)·p_g(y)  +  Σ_{g∈passthrough} q(g)·1[y = y_g]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from models.model import MultiViewClassifier
from models.lora import apply_lora_to_timm_vit
from data.dataset import build_post_transform

PathLike = Union[str, Path]
ImageLike = Union[str, Path, Image.Image]


def _load_one_model(ckpt_path: PathLike, device) -> tuple:
    """Reconstruct a MultiViewClassifier from a checkpoint's saved args and load
    its weights. Returns (model, class_to_idx, idx_to_class)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is None:
        # fall back to a label map shipped next to the checkpoint
        for name in ("coarse_to_id.json", "class_to_idx.json"):
            p = Path(ckpt_path).parent / name
            if p.exists():
                class_to_idx = json.loads(p.read_text())
                break
    if class_to_idx is None:
        raise ValueError(f"No class_to_idx for {ckpt_path}")

    model = MultiViewClassifier(
        model_name=a.get("model", "vit_base_patch16_dinov3.lvd1689m"),
        num_classes=len(class_to_idx),
        img_size=a.get("img_size", 224),
        pretrained=False,
        fuse=a.get("fuse", "gate"),
        use_cosine_head=a.get("use_cosine_head", False),
        verbose=False,
    ).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    if a.get("use_lora", True):
        apply_lora_to_timm_vit(
            model.backbone,
            last_n_blocks=a.get("lora_blocks", "0,2,4,6,8,10,11"),
            r=a.get("lora_rank", 16),
            alpha=a.get("lora_alpha", 32),
            dropout=0.0,
            targets=tuple(t.strip() for t in a.get(
                "lora_targets", "qkv,proj,mlp_fc1,mlp_fc2").split(",") if t.strip()),
            verbose=False,
        )
        model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class


class MorphPTPredictor:
    """Load MorphPT and predict cell types from paired DAPI crops.

    Args:
        weights_dir: directory containing router/ and expert_<Group>/ subfolders
                     (e.g. a Hugging Face snapshot of jilab/MorphPT).
        device:      torch device; defaults to cuda if available else cpu.
    """

    def __init__(self, weights_dir: PathLike, device: Optional[str] = None):
        self.root = Path(weights_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        router_ckpt = self.root / "router" / "best.pt"
        if not router_ckpt.exists():
            raise FileNotFoundError(f"router/best.pt not found under {self.root}")
        self.router, self.coarse_to_id, self.coarse_idx_to_name = _load_one_model(
            router_ckpt, self.device)

        # discover experts; coarse groups without an expert dir are passthrough
        self.experts, self.expert_fine = {}, {}
        for group in self.coarse_to_id:
            eck = self.root / f"expert_{group}" / "best.pt"
            if eck.exists():
                model, c2i, i2c = _load_one_model(eck, self.device)
                self.experts[group] = model
                self.expert_fine[group] = i2c  # local_idx -> fine label
        self.passthrough = [g for g in self.coarse_to_id if g not in self.experts]

        # passthrough fine labels from splits/fine_to_coarse.json if available
        self._passthrough_label = {}
        f2c_path = self.root / "splits" / "fine_to_coarse.json"
        if f2c_path.exists():
            f2c = json.loads(f2c_path.read_text())
            for fine, coarse in f2c.items():
                if coarse in self.passthrough:
                    self._passthrough_label[coarse] = fine
        for g in self.passthrough:
            self._passthrough_label.setdefault(g, g)  # fallback to group name

        # global fine label space
        fines = set()
        for i2c in self.expert_fine.values():
            fines.update(i2c.values())
        fines.update(self._passthrough_label[g] for g in self.passthrough)
        self.fine_classes = sorted(fines)
        self._fine_to_idx = {c: i for i, c in enumerate(self.fine_classes)}
        self._tf = build_post_transform(self.router_img_size, mean=None, std=None)

    @property
    def router_img_size(self) -> int:
        return 224

    @classmethod
    def from_pretrained(cls, repo_id: str = "jilab/MorphPT",
                        device: Optional[str] = None, **kwargs):
        """Download weights from the Hugging Face Hub and load them."""
        from huggingface_hub import snapshot_download
        weights_dir = snapshot_download(repo_id=repo_id, repo_type="model", **kwargs)
        return cls(weights_dir, device=device)

    def _to_tensor(self, img: ImageLike) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        return self._tf(img.convert("L"))

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> dict:
        """x: [1, 2, 3, H, W] -> prediction dict."""
        q = F.softmax(self.router(x)[0][0], dim=0)            # [n_coarse]
        sp, _ = q.sort(descending=True)
        router_margin = float(sp[0] - sp[1]) if q.numel() > 1 else 1.0
        routed = self.coarse_idx_to_name[int(q.argmax())]

        scores = torch.zeros(len(self.fine_classes))
        max_pe = 1.0
        for group, model in self.experts.items():
            gi = self.coarse_to_id[group]
            ep = F.softmax(model(x)[0][0], dim=0)             # [n_fine_g]
            for local_idx, fine in self.expert_fine[group].items():
                scores[self._fine_to_idx[fine]] += float(q[gi]) * float(ep[local_idx])
            if group == routed:
                max_pe = float(ep.max())
        for group in self.passthrough:
            gi = self.coarse_to_id[group]
            scores[self._fine_to_idx[self._passthrough_label[group]]] += float(q[gi])
        if routed in self.passthrough:
            max_pe = 1.0

        k = min(3, len(self.fine_classes))
        top = torch.topk(scores, k)
        return {
            "pred": self.fine_classes[int(top.indices[0])],
            "coarse_group": routed,
            "confidence": router_margin * max_pe,
            "top3": [(self.fine_classes[int(i)], float(v))
                     for i, v in zip(top.indices, top.values)],
        }

    def predict_one(self, img_2p5x: ImageLike, img_10x: ImageLike) -> dict:
        """Predict the fine cell type for one cell from its two crops.

        Returns a dict: pred, coarse_group, confidence (M_r * max expert prob),
        and top3 [(label, score), ...].
        """
        x = torch.stack([self._to_tensor(img_2p5x), self._to_tensor(img_10x)],
                        dim=0).unsqueeze(0).to(self.device)
        return self._forward(x)

    def predict_batch(self, imgs_2p5x, imgs_10x) -> list:
        """Predict for many cells (lists of equal length). Returns list of dicts."""
        if len(imgs_2p5x) != len(imgs_10x):
            raise ValueError("imgs_2p5x and imgs_10x must have the same length")
        return [self.predict_one(a, b) for a, b in zip(imgs_2p5x, imgs_10x)]
