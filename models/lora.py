# model/lora.py
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear with LoRA adapters.

    y = base(x) + (alpha/r) * (Up(Dropout(Down(x))))

    Notes:
    - The original Linear is stored as a submodule `self.base` and frozen.
    - Only LoRA params are trainable.
    """
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(original_linear, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear")

        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0. Got r={r}")

        self.base = original_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = self.base.in_features
        self.out_features = self.base.out_features

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)

        self.lora_down = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_up = nn.Linear(self.r, self.out_features, bias=False)

        # Init: down ~ Kaiming, up = 0 so start identical to base
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.dropout = nn.Dropout(dropout) if (dropout is not None and dropout > 0) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        lora = self.lora_up(self.dropout(self.lora_down(x)))
        return base + self.scaling * lora

    def lora_parameters(self) -> List[nn.Parameter]:
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())


def apply_lora_to_timm_vit(
    model: nn.Module,
    last_n_blocks: int = 4,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    targets: Sequence[str] = ("qkv",),
    verbose: bool = True,
) -> List[nn.Parameter]:
    """
    Inject LoRA into a timm ViT-style model (incl. DINOv3 ViT).

    Default: only attn.qkv in last N blocks. This is the safest setting for
    embedding quality and avoiding overfitting.

    Args:
      model: timm ViT with `model.blocks`
      last_n_blocks: apply LoRA to the last N transformer blocks
      r, alpha, dropout: LoRA hyperparams
      targets: subset of {"qkv","proj","mlp_fc1","mlp_fc2"}
      verbose: print injected layer names

    Returns:
      list of trainable LoRA parameters
    """
    if not hasattr(model, "blocks"):
        raise RuntimeError("apply_lora_to_timm_vit: model has no 'blocks'; expected a timm ViT-like model.")

    allowed = {"qkv", "proj", "mlp_fc1", "mlp_fc2"}
    bad = [t for t in targets if t not in allowed]
    if bad:
        raise ValueError(f"Unknown targets {bad}. Allowed: {sorted(allowed)}")

    blocks = model.blocks
    total = len(blocks)

    if isinstance(last_n_blocks, str) and "," in last_n_blocks:
        target_indices = [int(x.strip()) for x in last_n_blocks.split(",") if x.strip()]
    else:
        n_blocks = int(last_n_blocks)
        target_indices = list(range(max(0, total - n_blocks), total))

    lora_params: List[nn.Parameter] = []
    injected_names: List[str] = []

    for i in target_indices:
        if i < 0 or i >= total:
            continue
        blk = blocks[i]

        # ---- Attention ----
        if hasattr(blk, "attn"):
            if "qkv" in targets and hasattr(blk.attn, "qkv") and isinstance(blk.attn.qkv, nn.Linear):
                blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.attn.qkv.lora_parameters()
                injected_names.append(f"blocks.{i}.attn.qkv")

            if "proj" in targets and hasattr(blk.attn, "proj") and isinstance(blk.attn.proj, nn.Linear):
                blk.attn.proj = LoRALinear(blk.attn.proj, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.attn.proj.lora_parameters()
                injected_names.append(f"blocks.{i}.attn.proj")

        # ---- MLP ----
        if hasattr(blk, "mlp"):
            if "mlp_fc1" in targets and hasattr(blk.mlp, "fc1") and isinstance(blk.mlp.fc1, nn.Linear):
                blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.mlp.fc1.lora_parameters()
                injected_names.append(f"blocks.{i}.mlp.fc1")

            if "mlp_fc2" in targets and hasattr(blk.mlp, "fc2") and isinstance(blk.mlp.fc2, nn.Linear):
                blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.mlp.fc2.lora_parameters()
                injected_names.append(f"blocks.{i}.mlp.fc2")

    if len(lora_params) == 0:
        raise RuntimeError(
            "No LoRA parameters injected. Check that your timm model uses blocks[i].attn.qkv naming."
        )

    # Ensure LoRA params trainable
    for p in lora_params:
        p.requires_grad = True

    if verbose:
        print(f"[LoRA] Injected {len(injected_names)} modules:")
        for n in injected_names[:50]:
            print("  ", n)
        if len(injected_names) > 50:
            print(f"  ... (+{len(injected_names)-50} more)")

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[LoRA] Trainable params: {n_train} / {n_total} ({100.0*n_train/max(1,n_total):.4f}%)")

    return lora_params


def apply_lora_to_timm_swin(
    model: nn.Module,
    last_n_blocks: int = 8,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    targets: Sequence[str] = ("qkv",),
    verbose: bool = True,
) -> List[nn.Parameter]:
    """
    Optional: Inject LoRA into timm Swin models.
    Swin structure: model.layers[*].blocks[*].attn.qkv / attn.proj, mlp.fc1/fc2.

    Args:
      last_n_blocks: last N Swin blocks across all layers in traversal order
      targets: subset of {"qkv","proj","mlp_fc1","mlp_fc2"}
    """
    if not hasattr(model, "layers"):
        raise RuntimeError("apply_lora_to_timm_swin: model has no 'layers'; expected timm Swin model.")

    allowed = {"qkv", "proj", "mlp_fc1", "mlp_fc2"}
    bad = [t for t in targets if t not in allowed]
    if bad:
        raise ValueError(f"Unknown targets {bad}. Allowed: {sorted(allowed)}")

    # Collect blocks in order
    all_blocks: List[Tuple[int, int, nn.Module]] = []  # (layer_idx, block_idx, block)
    for li, layer in enumerate(model.layers):
        if hasattr(layer, "blocks"):
            for bi, blk in enumerate(layer.blocks):
                all_blocks.append((li, bi, blk))

    total = len(all_blocks)
    
    if isinstance(last_n_blocks, str) and "," in last_n_blocks:
        target_indices = [int(x.strip()) for x in last_n_blocks.split(",") if x.strip()]
        target_blocks = [all_blocks[i] for i in target_indices if 0 <= i < total]
    else:
        n_blocks = int(last_n_blocks)
        start = max(0, total - n_blocks)
        target_blocks = all_blocks[start:]

    lora_params: List[nn.Parameter] = []
    injected_names: List[str] = []

    for li, bi, blk in target_blocks:
        if hasattr(blk, "attn"):
            if "qkv" in targets and hasattr(blk.attn, "qkv") and isinstance(blk.attn.qkv, nn.Linear):
                blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.attn.qkv.lora_parameters()
                injected_names.append(f"layers.{li}.blocks.{bi}.attn.qkv")

            if "proj" in targets and hasattr(blk.attn, "proj") and isinstance(blk.attn.proj, nn.Linear):
                blk.attn.proj = LoRALinear(blk.attn.proj, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.attn.proj.lora_parameters()
                injected_names.append(f"layers.{li}.blocks.{bi}.attn.proj")

        if hasattr(blk, "mlp"):
            if "mlp_fc1" in targets and hasattr(blk.mlp, "fc1") and isinstance(blk.mlp.fc1, nn.Linear):
                blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.mlp.fc1.lora_parameters()
                injected_names.append(f"layers.{li}.blocks.{bi}.mlp.fc1")

            if "mlp_fc2" in targets and hasattr(blk.mlp, "fc2") and isinstance(blk.mlp.fc2, nn.Linear):
                blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.mlp.fc2.lora_parameters()
                injected_names.append(f"layers.{li}.blocks.{bi}.mlp.fc2")

    if len(lora_params) == 0:
        raise RuntimeError("No LoRA parameters injected into Swin. Check naming in timm model.")

    for p in lora_params:
        p.requires_grad = True

    if verbose:
        print(f"[LoRA] Injected {len(injected_names)} Swin modules:")
        for n in injected_names[:50]:
            print("  ", n)
        if len(injected_names) > 50:
            print(f"  ... (+{len(injected_names)-50} more)")

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[LoRA] Trainable params: {n_train} / {n_total} ({100.0*n_train/max(1,n_total):.4f}%)")

    return lora_params


def apply_lora_to_timm_convnext(
    model: nn.Module,
    last_n_blocks: int = 8,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    targets: Sequence[str] = ("mlp_fc1", "mlp_fc2"),
    verbose: bool = True,
) -> List[nn.Parameter]:
    """
    Inject LoRA into timm ConvNeXt models.
    ConvNeXt structure: model.stages[*].blocks[*].mlp.fc1/fc2.

    Args:
      last_n_blocks: last N blocks across all stages in traversal order
      targets: subset of {"mlp_fc1","mlp_fc2"}
    """
    if not hasattr(model, "stages"):
        raise RuntimeError("apply_lora_to_timm_convnext: model has no 'stages'; expected timm ConvNeXt model.")

    allowed = {"mlp_fc1", "mlp_fc2"}
    bad = [t for t in targets if t not in allowed]
    if bad:
        raise ValueError(f"Unknown targets {bad}. Allowed: {sorted(allowed)}")

    # Collect blocks in order
    all_blocks: List[Tuple[int, int, nn.Module]] = []  # (stage_idx, block_idx, block)
    for si, stage in enumerate(model.stages):
        if hasattr(stage, "blocks"):
            for bi, blk in enumerate(stage.blocks):
                all_blocks.append((si, bi, blk))

    total = len(all_blocks)
    
    if isinstance(last_n_blocks, str) and "," in last_n_blocks:
        target_indices = [int(x.strip()) for x in last_n_blocks.split(",") if x.strip()]
        target_blocks = [all_blocks[i] for i in target_indices if 0 <= i < total]
    else:
        n_blocks = int(last_n_blocks)
        start = max(0, total - n_blocks)
        target_blocks = all_blocks[start:]

    lora_params: List[nn.Parameter] = []
    injected_names: List[str] = []

    for si, bi, blk in target_blocks:
        if hasattr(blk, "mlp"):
            if "mlp_fc1" in targets and hasattr(blk.mlp, "fc1") and isinstance(blk.mlp.fc1, nn.Linear):
                blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.mlp.fc1.lora_parameters()
                injected_names.append(f"stages.{si}.blocks.{bi}.mlp.fc1")

            if "mlp_fc2" in targets and hasattr(blk.mlp, "fc2") and isinstance(blk.mlp.fc2, nn.Linear):
                blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r=r, alpha=alpha, dropout=dropout)
                lora_params += blk.mlp.fc2.lora_parameters()
                injected_names.append(f"stages.{si}.blocks.{bi}.mlp.fc2")

    if len(lora_params) == 0:
        raise RuntimeError("No LoRA parameters injected into ConvNeXt. Check naming in timm model.")

    for p in lora_params:
        p.requires_grad = True

    if verbose:
        print(f"[LoRA] Injected {len(injected_names)} ConvNeXt modules:")
        for n in injected_names[:50]:
            print("  ", n)
        if len(injected_names) > 50:
            print(f"  ... (+{len(injected_names)-50} more)")

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[LoRA] Trainable params: {n_train} / {n_total} ({100.0*n_train/max(1,n_total):.4f}%)")

    return lora_params
