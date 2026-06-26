#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT = Path(os.environ.get("MORPHPT_ROOT", Path(__file__).resolve().parents[1]))  # repo root; override with MORPHPT_ROOT to point at your data/cache
# Append project root to sys path so we can import model and data classes
sys.path.append(str(PROJECT))

from data.visium_dataset import VisiumHDPredictionDataset
from models.visium_regression import VisiumRegressor

def get_args():
    p = argparse.ArgumentParser(description="Extract MorphPT embeddings for caching and fast MLP probing.")
    p.add_argument('--dataset', type=str, default='crc')
    p.add_argument('--split_type', type=str, default='spatial', choices=['spatial', 'random'])
    p.add_argument('--split_layout', type=str, default='default', help='Custom layout folder name under splits/')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    return p.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for extracting embeddings...")

    cache_dir = PROJECT / f"cache_{args.dataset}"
    ckpt_path = PROJECT / "experiments/router_nobreast_vitb_gate_r16_mlp_cw/best.pt"
    
    layout_dir = cache_dir / "splits" / args.split_layout
    if layout_dir.exists():
        out_dir = layout_dir / "embeddings_morphpt"
        print(f"Routing extraction to decoupled split layout: '{args.split_layout}' -> {out_dir}")
    else:
        out_dir = cache_dir / "embeddings_morphpt"
        print(f"Layout folder not found. Routing extraction to base cache -> {out_dir}")

    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load Model
    print(f"Loading MorphPT model from {ckpt_path} ...")
    model = VisiumRegressor(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        img_size=224,
        out_dim=2, # Dummy output, we intercept features before the head
        pretrained=False,
        fuse="gate",
        freeze_backbone=True,
        lora_blocks="0,2,4,6,8,10,11",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_targets="qkv,proj,mlp_fc1,mlp_fc2",
        unfreeze_lora=False,
        ckpt_path=str(ckpt_path),
        head_type="mlp",
        gate_dropout=0.1
    ).to(device)
    model.eval()

    def get_features(batch_imgs):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        # We use autocast to match training precision and speed up inference
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            if model.fuse == "identity":
                feat = model.backbone(batch_imgs)
            else:
                B = batch_imgs.shape[0]
                x_cat = torch.cat([batch_imgs[:, 0], batch_imgs[:, 1]], dim=0)
                emb_all = model.backbone(x_cat)
                # Pass through the gate fusion layer
                feat, _ = model.gate(emb_all[:B], emb_all[B:])
                
            return feat.float() # convert back to float32 for stable saving/MLP training

    # 2. Extract for each split (train, val, test)
    for split in ["train", "val", "test"]:
        print(f"\n────────────────────────────────────────────────────────────")
        print(f" Extracting Split: {split.upper()}")
        print(f"────────────────────────────────────────────────────────────")
        
        ds = VisiumHDPredictionDataset(
            cache_dir=str(cache_dir),
            split=split,
            scales=["2.5x", "10.0x"],
            fuse="gate",
            augment=False, # STRICTLY FALSE FOR EXTRACTION
            split_type=args.split_type,
            split_layout=args.split_layout
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)
        
        all_feats = []
        all_exprs = []
        
        for imgs, exprs, _ in tqdm(loader, desc=f"{split}"):
            feats = get_features(imgs)
            all_feats.append(feats.cpu())
            all_exprs.append(exprs.cpu())
            
        all_feats = torch.cat(all_feats, dim=0)
        all_exprs = torch.cat(all_exprs, dim=0)
        
        print(f"Shape extracted: Features={all_feats.shape}, Expr={all_exprs.shape}")
        
        feat_path = out_dir / f"{split}_features.pt"
        expr_path = out_dir / f"{split}_expr.pt"
        
        torch.save(all_feats, feat_path)
        torch.save(all_exprs, expr_path)
        print(f"Saved: {feat_path}")
        print(f"Saved: {expr_path}")

    print("\nExtraction finished! You are ready to train your 400 MLP Models!")

if __name__ == "__main__":
    main()
