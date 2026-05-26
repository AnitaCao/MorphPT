"""
visium_regression.py
─────────────────────
VisiumRegressor: DINOv3 backbone + optional Scale Gate + regression head.

Fuse modes:
  'identity'  single scale  input: (B, 3, H, W)
  'gate'      dual scale    input: (B, 2, 3, H, W)

For 'gate', views are sliced with x[:,0] and x[:,1], then concatenated
as [all_v0 | all_v1] before the single backbone forward pass.
This gives correct e1=emb[:B] and e2=emb[B:] alignment.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import timm

from models.lora import apply_lora_to_timm_vit


# ── Scale Gate ─────────────────────────────────────────────────────────────
class ScaleGate(nn.Module):
    """
    Learned weighted average of two scale embeddings.
    Input : e1 (B, d), e2 (B, d)
    Output: fused (B, d), gate_weights (B, 2)
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        # 1. Per-view Normalization: 2.5x and 10x have different feature distributions
        self.emb_norm_a = nn.LayerNorm(d)
        self.emb_norm_b = nn.LayerNorm(d)
        
        # 2. Bottleneck dimension (d // 2) prevents overfitting the megabatch
        h = max(128, d // 2)
        self.gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, h),
            nn.GELU(),
            nn.Linear(h, 2),
        )
        # 3. Zero init → gate initially acts as an Average Pooler [0.5, 0.5]
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
        # Apply independent architectural normalizations
        e1 = self.emb_norm_a(e1)
        e2 = self.emb_norm_b(e2)
        
        w     = torch.softmax(self.gate(torch.cat([e1, e2], dim=-1)), dim=-1)
        fused = w[:, 0:1] * e1 + w[:, 1:2] * e2
        return fused, w


# ── Regressor ─────────────────────────────────────────────────────────────
class VisiumRegressor(nn.Module):
    def __init__(
        self,
        model_name:      str   = "vit_base_patch16_dinov3.lvd1689m",
        img_size:        int   = 224,
        out_dim:         int   = 1000,
        pretrained:      bool  = True,
        fuse:            str   = "identity",
        freeze_backbone: bool  = True,
        lora_blocks:     int | str = 0,
        lora_rank:       int   = 8,
        lora_alpha:      int   = 16,
        lora_dropout:    float = 0.05,
        lora_targets:    str   = "qkv",
        unfreeze_lora:   bool  = False,
        ckpt_path:       Optional[str] = None,
        head_type:       str   = "linear",
        gate_dropout:    float = 0.1,     # ← accepted here
    ):
        super().__init__()
        assert fuse in ("identity", "gate"), f"Unknown fuse: {fuse}"
        self.fuse = fuse

        # ── 1. Backbone ────────────────────────────────────────────────────
        try:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained,
                img_size=img_size, num_classes=0
            )
        except TypeError:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )

        with torch.no_grad():
            d_embed = self.backbone(torch.randn(1, 3, img_size, img_size)).shape[1]

        # ── 2. LoRA injection (BEFORE checkpoint load) ─────────────────────
        self.lora_params = []
        _apply_lora = (
            (isinstance(lora_blocks, int) and lora_blocks > 0) or
            (isinstance(lora_blocks, str) and lora_blocks.strip().lower() not in ("0", "", "none"))
        )
        if _apply_lora:
            if "vit" in model_name.lower() or "dino" in model_name.lower():
                self.lora_params = apply_lora_to_timm_vit(
                    self.backbone,
                    last_n_blocks = lora_blocks,
                    r             = lora_rank,
                    alpha         = lora_alpha,
                    dropout       = lora_dropout,
                    targets       = tuple(t.strip() for t in lora_targets.split(",")),
                    verbose       = True,
                )
            else:
                print(f"[VisiumRegressor] LoRA not implemented for {model_name}")

        # ── 3. Load checkpoint (backbone only, strict=False) ──────────────
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                print(f"[VisiumRegressor] Loading checkpoint: {ckpt_path}")
                ckpt       = torch.load(ckpt_path, map_location="cpu",
                                        weights_only=False)
                state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
                backbone_sd = {
                    k.replace("backbone.", "").replace("module.backbone.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("backbone.") or k.startswith("module.backbone.")
                }
                missing, unexpected = self.backbone.load_state_dict(
                    backbone_sd, strict=False
                )
                print(f"  Keys loaded : {len(backbone_sd)}")
                print(f"  Missing     : {len(missing)}")
                print(f"  Unexpected  : {len(unexpected)}")
            else:
                print(f"[VisiumRegressor][ERROR] Checkpoint not found: {ckpt_path}")

        # ── 4. Freeze backbone ─────────────────────────────────────────────
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[VisiumRegressor] Backbone frozen.")

        # ── 4b. Unfreeze LoRA params only ──────────────────────────────────
        if unfreeze_lora and self.lora_params:
            for p in self.lora_params:
                p.requires_grad = True
            print(f"[VisiumRegressor] LoRA unfrozen ({len(self.lora_params)} params).")

        # ── 5. Scale Gate ──────────────────────────────────────────────────
        self.gate = ScaleGate(d_embed, dropout=gate_dropout) \
                    if fuse == "gate" else None

        # ── 6. Regression head ─────────────────────────────────────────────
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(d_embed, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, out_dim),
            )
        else:
            self.head = nn.Linear(d_embed, out_dim)
            nn.init.zeros_(self.head.bias)

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VisiumRegressor] fuse={fuse}  "
              f"trainable={n_train:,} / total={n_total:,}")

    def get_param_groups(self, lr: float, weight_decay: float,
                         gate_wd: float, lora_lr_scale: float = 0.1):
        """
        Build optimizer param groups that correctly capture ALL trainable params.

        Groups:
          head        — lr, weight_decay
          gate        — lr, gate_wd (stronger regularization)
          lora        — lr * lora_lr_scale, weight_decay
          backbone    — lr * lora_lr_scale, weight_decay
                        (only populated when freeze_backbone=False and no LoRA)

        Using id-based deduplication so no param appears in two groups.
        """
        lora_ids  = set(id(p) for p in self.lora_params)
        head_ids  = set(id(p) for p in self.head.parameters())
        gate_ids  = set(id(p) for p in self.gate.parameters()) \
                    if self.gate is not None else set()

        groups = []

        # Head
        head_p = [p for p in self.head.parameters() if p.requires_grad]
        if head_p:
            groups.append({"params": head_p, "lr": lr,
                           "weight_decay": weight_decay, "name": "head"})

        # Gate
        if self.gate is not None:
            gate_p = [p for p in self.gate.parameters() if p.requires_grad]
            if gate_p:
                groups.append({"params": gate_p, "lr": lr,
                               "weight_decay": gate_wd, "name": "gate"})

        # LoRA
        lora_p = [p for p in self.lora_params if p.requires_grad]
        if lora_p:
            groups.append({"params": lora_p, "lr": lr * lora_lr_scale,
                           "weight_decay": weight_decay, "name": "lora"})

        # Backbone (non-LoRA) — only present when freeze_backbone=False
        already_seen = head_ids | gate_ids | lora_ids
        backbone_p = [
            p for p in self.backbone.parameters()
            if p.requires_grad and id(p) not in already_seen
        ]
        if backbone_p:
            groups.append({"params": backbone_p, "lr": lr * lora_lr_scale,
                           "weight_decay": weight_decay, "name": "backbone"})

        # Safety check — no trainable param left behind
        all_assigned = sum(len(g["params"]) for g in groups)
        all_trainable = sum(1 for p in self.parameters() if p.requires_grad)
        assert all_assigned == all_trainable, (
            f"Param group mismatch: {all_assigned} assigned vs "
            f"{all_trainable} trainable. Check get_param_groups()."
        )

        for g in groups:
            print(f"  Param group '{g['name']}': "
                  f"{len(g['params'])} tensors, lr={g['lr']:.2e}, "
                  f"wd={g['weight_decay']}")
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)      for fuse='identity'
           (B, 2, 3, H, W)   for fuse='gate'
        Returns: (B, out_dim)
        """
        if self.fuse == "identity":
            feat = self.backbone(x)

        else:  # gate
            B = x.shape[0]
            # Slice views before cat — gives [all_v0 | all_v1] ordering
            x_cat   = torch.cat([x[:, 0], x[:, 1]], dim=0)   # (2B, 3, H, W)
            emb_all = self.backbone(x_cat)                    # (2B, d)
            e1, e2  = emb_all[:B], emb_all[B:]
            feat, _ = self.gate(e1, e2)

        return self.head(feat)

    @torch.no_grad()
    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return gate weights (B, 2) for a dual-scale batch.
        Use this for monitoring / analysis — not called during training.
        """
        assert self.fuse == "gate", "get_gate_weights only valid for fuse='gate'"
        B = x.shape[0]
        x_cat   = torch.cat([x[:, 0], x[:, 1]], dim=0)
        emb_all = self.backbone(x_cat)
        e1, e2  = emb_all[:B], emb_all[B:]
        _, w    = self.gate(e1, e2)
        return w   # (B, 2)  columns: [w_scale0, w_scale1]