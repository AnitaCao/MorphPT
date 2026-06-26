#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "trainer"))

import argparse
import json
import torch
from torch.utils.data import DataLoader

from trainer.train_single_view import SingleViewClassifier
from data.dataset import CellParquetSingleView, AugCfg
from trainer.trainer_base import per_class_metrics, confusion_matrix_np

@torch.no_grad()
def eval_epoch_single(model, loader, device, amp_dtype):
    model.eval()
    all_pred = []
    all_y = []
    all_margins = []
    all_probs = []

    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for batch in loader:
            if len(batch) == 3 and isinstance(batch[2], dict) and "index" in batch[2]:
                x, y, _meta = batch
            else:
                x, y, _tissue = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            sorted_p, _ = probs.sort(dim=1, descending=True)
            margins = sorted_p[:, 0] - sorted_p[:, 1]
            
            all_pred.append(logits.argmax(1).detach().cpu())
            all_y.append(y.detach().cpu())
            all_margins.append(margins.detach().cpu())
            all_probs.append(sorted_p[:, 0].detach().cpu())

    pred = torch.cat(all_pred) if all_pred else torch.empty(0, dtype=torch.long)
    yy = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)
    margins = torch.cat(all_margins) if all_margins else torch.empty(0, dtype=torch.float)
    max_probs = torch.cat(all_probs) if all_probs else torch.empty(0, dtype=torch.float)
    return pred, yy, margins, max_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading checkpoint {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_args = ckpt["args"]
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model_name = model_args.get("model", "resnet50")
    img_size = model_args.get("img_size", 224)
    view = model_args.get("view", "10x")

    print(f"Model: {model_name}, View: {view}, Classes: {num_classes}")
    
    model = SingleViewClassifier(model_name, num_classes, img_size)

    # Apply LoRA if the model was trained with it
    if model_args.get("use_lora", False) and ("vit" in model_name or "swin" in model_name) and not model_args.get("freeze_backbone", False):
        from models.lora import apply_lora_to_timm_vit, apply_lora_to_timm_swin
        lora_blocks = model_args.get("lora_blocks", "0,2,4,6,8,10,11")
        lora_targets = model_args.get("lora_targets", "qkv,proj,mlp_fc1,mlp_fc2")
        r = model_args.get("lora_rank", 16)
        alpha = model_args.get("lora_alpha", 32)
        dropout = model_args.get("lora_dropout", 0.0)
        targets_tuple = tuple(t.strip() for t in lora_targets.split(",") if t.strip())

        if "swin" in model_name:
            apply_lora_to_timm_swin(
                model.backbone, last_n_blocks=lora_blocks, r=r, 
                alpha=alpha, dropout=dropout, targets=targets_tuple, verbose=False
            )
        else:
            apply_lora_to_timm_vit(
                model.backbone, last_n_blocks=lora_blocks, r=r, 
                alpha=alpha, dropout=dropout, targets=targets_tuple, verbose=False
            )

    model.load_state_dict(ckpt["model"])
    model.to(device)

    print(f"Loading test set: {args.test_dir}")
    ds_test = CellParquetSingleView(
        parquet_path=args.test_dir,
        class_to_idx=class_to_idx,
        img_col=f"img_path_{view}",
        label_col=args.label_col,
        size=img_size,
        aug=AugCfg(enable=False)
    )
    print(f"Test Set Size: {len(ds_test)}")
    loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("Running evaluation...")
    pred, yy, margins, max_probs = eval_epoch_single(model, loader, device, amp_dtype)

    rows = per_class_metrics(pred, yy, num_classes, idx_to_class)
    print(f"\n{'Class':<28} {'N':>8} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("─" * 60)
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(f"{r['class']:<28} {r['n']:>8,} {r['prec']:>7.3f} {r['rec']:>7.3f} {r['f1']:>7.3f}")

    if args.out_dir:
        import pandas as pd
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_parquet(args.test_dir)
        pred_names = [idx_to_class[int(p)] for p in pred]
        df["pred_top1"] = pred_names
        df["expert_margin"] = margins.numpy()  # Align with MoE column styles
        df["expert_top1_prob"] = max_probs.numpy()
        
        out_path = out_dir / "predictions.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\nSaved predictions to {out_path}")
        
        # Save summary
        macro_f1 = sum(r["f1"] for r in rows if r["n"] > 0) / max(1, sum(1 for r in rows if r["n"] > 0))
        summary = {
            "macro_f1": macro_f1,
            "per_class": rows
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        
        # Save Confusion matrix
        cm = confusion_matrix_np(pred, yy, num_classes)
        class_names = [idx_to_class[i] for i in range(num_classes)]
        (out_dir / "confusion_matrix.json").write_text(
            json.dumps({"class_names": class_names, "matrix": cm.tolist()}, indent=2)
        )

if __name__ == "__main__":
    main()
