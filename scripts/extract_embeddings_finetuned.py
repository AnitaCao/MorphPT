#!/usr/bin/env python3
"""
Extract embeddings from a fine-tuned checkpoint.

Usage:
    python scripts/extract_embeddings_finetuned.py \
        --parquet prepared/small_balanced.parquet \
        --checkpoint results/ft_sweep/lora_r16_b4/best.pt \
        --out prepared/embeddings_small_balanced_lora_r16.parquet
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from models.lora import apply_lora_to_timm_vit
from data.dataset import CellParquetMultiView, AugCfg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model_from_checkpoint(ckpt_path: str):
    """Rebuild model architecture from saved args, then load weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ckpt["args"])
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)

    # Lazy import to avoid circular dependency
    from trainer.train_finetune import DinoV3Classifier

    model = DinoV3Classifier(
        model_name=args.model,
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained=False,  # we'll load from checkpoint
        multi_view=bool(args.multi_view),
        fuse=args.fuse,
        use_gate=(args.fuse == "gate"),
    )

    # If LoRA was used, inject LoRA modules before loading state dict
    if args.use_lora:
        for p in model.backbone.parameters():
            p.requires_grad = False
        apply_lora_to_timm_vit(
            model.backbone,
            last_n_blocks=args.lora_blocks,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            targets=("qkv",),
            verbose=True,
        )

    # Load weights
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  model={args.model}  use_lora={args.use_lora}  fuse={args.fuse}")
    print(f"  best_macro_f1={ckpt.get('best_macro_f1', 'N/A')}")

    return model, class_to_idx, args


@torch.no_grad()
def extract_embeddings(model, parquet_path, class_to_idx, img_size=224, batch_size=64, num_workers=8):
    
    ds = CellParquetMultiView(
        parquet_path=parquet_path,
        class_to_idx=class_to_idx,
        size=img_size,
        aug=AugCfg(enable=False),
        return_meta=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE)
        probe = model.forward_features(dummy)
        print(f"forward_features output: shape={probe.shape}, norm={probe.norm(dim=-1).item():.4f}")

    emb2_list, emb10_list, embf_list = [], [], []
    y_list, tissue_list, cellid_list, x_list, ycoord_list = [], [], [], [], []

    use_cuda = (DEVICE == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    for x, y, meta in tqdm(loader, desc="extract finetuned"):
        x = x.to(DEVICE, non_blocking=True)
        y_list.append(y.numpy())

        tissue_list.extend(meta["tissue"])
        cellid_list.extend(meta["cell_id"])
        x_list.extend(meta["x"].tolist())
        ycoord_list.extend(meta["y"].tolist())

        B, S, C, H, W = x.shape
        xs = x.reshape(B * S, C, H, W)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
            emb_all = model.forward_features(xs)  # [B*S, D]

        P = emb_all.reshape(B, S, -1)
        e2 = P[:, 0].float()
        e10 = P[:, 1].float()

        e2n = F.normalize(e2, dim=1)
        e10n = F.normalize(e10, dim=1)
        ef = F.normalize(0.5 * (e2n + e10n), dim=1)

        emb2_list.append(e2n.cpu().numpy().astype(np.float32))
        emb10_list.append(e10n.cpu().numpy().astype(np.float32))
        embf_list.append(ef.cpu().numpy().astype(np.float32))

    emb2 = np.concatenate(emb2_list, axis=0)
    emb10 = np.concatenate(emb10_list, axis=0)
    embf = np.concatenate(embf_list, axis=0)
    labels = np.concatenate(y_list, axis=0)

    out = pd.DataFrame({
        "cell_id": cellid_list,
        "tissue": tissue_list,
        "x_centroid": x_list,
        "y_centroid": ycoord_list,
        "label_id": labels.astype(int),
    })
    out["emb_2p5x"] = list(emb2)
    out["emb_10x"] = list(emb10)
    out["emb_fused"] = list(embf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    model, class_to_idx, train_args = build_model_from_checkpoint(args.checkpoint)

    df = extract_embeddings(
        model=model,
        parquet_path=args.parquet,
        class_to_idx=class_to_idx,
        img_size=train_args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    df.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()