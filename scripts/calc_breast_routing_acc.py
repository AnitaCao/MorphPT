import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Import same as eval_moe_e2e
from models.lora import apply_lora_to_timm_vit
from models.model import MultiViewClassifier
from data.dataset import CellParquetMultiView, AugCfg
from scripts.eval_moe_e2e import load_model, run_inference, FINE_TO_COARSE, PASSTHROUGH_GROUPS

SLIDES = [
    "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon",
    "Xenium_V1_FFPE_Human_Breast_ILC",
    "Xenium_V1_FFPE_Human_Breast_IDC",
    "Xenium_V1_FFPE_Human_Breast_ILC_With_Addon",
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_1",
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_2",
]

def eval_slide(parquet_path, model, c2i, i2c, device, args):
    df = pd.read_parquet(parquet_path)
    if "coarse_label" not in df.columns:
        df["coarse_label"] = df["label"].map(FINE_TO_COARSE)
    
    true_coarse = df["coarse_label"].values

    ds = CellParquetMultiView(
        parquet_path=str(parquet_path),
        class_to_idx=c2i,
        label_col="coarse_label",
        size=args.img_size,
        aug=AugCfg(enable=False),
        return_meta=True,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    logits, _ = run_inference(model, loader, device)
    probs = F.softmax(logits, dim=1)
    
    coarse_pred_idx = probs.argmax(dim=1)
    coarse_pred_names = np.array([i2c[int(c)] for c in coarse_pred_idx])

    # Assign back to df to align with patches
    df["router_pred"] = coarse_pred_names
    
    # Tissue level acc
    correct = (df["router_pred"] == df["coarse_label"]).sum()
    total = len(df)
    tissue_acc = correct / total if total > 0 else 0.0

    # Patch level acc
    patch_accs = {}
    if "patch_id" in df.columns:
        for patch_id, group in df.groupby("patch_id"):
            p_correct = (group["router_pred"] == group["coarse_label"]).sum()
            p_total = len(group)
            patch_accs[patch_id] = float(p_correct / p_total) if p_total > 0 else 0.0
            
    return float(tissue_acc), total, patch_accs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router_ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading router from {args.router_ckpt}")
    model, c2i, i2c = load_model(args.router_ckpt, device)

    base_dir = Path("/hpc/group/jilab/rz179/MorphPT_MOE/prepared/splits_v3_seed1337")
    
    results = {}
    
    for shard_type in ["router_shards", "test_shards"]:
        print(f"\nEvaluating {shard_type}...")
        results[shard_type] = {}
        for slide in SLIDES:
            parquet_path = base_dir / shard_type / f"{slide}.parquet"
            if not parquet_path.exists():
                print(f"Skipping {slide} (not found in {shard_type})")
                continue
            
            print(f"  {slide}")
            tissue_acc, total, patch_accs = eval_slide(parquet_path, model, c2i, i2c, device, args)
            
            mean_patch_acc = float(np.mean(list(patch_accs.values()))) if patch_accs else 0.0
            
            results[shard_type][slide] = {
                "tissue_acc": tissue_acc,
                "total_cells": total,
                "mean_patch_acc": mean_patch_acc,
                "patch_accs": patch_accs
            }
            print(f"    Tissue Acc: {tissue_acc:.4f} ({total} cells)")
            print(f"    Patch Accs: mean={mean_patch_acc:.4f} across {len(patch_accs)} patches")

    out_file = "/hpc/group/jilab/rz179/MorphPT_MOE/breast_routing_acc_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {out_file}")

if __name__ == "__main__":
    main()
