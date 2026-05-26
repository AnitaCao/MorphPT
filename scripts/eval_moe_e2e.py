#!/usr/bin/env python3
"""
CellPT MoE: End-to-End Two-Stage Evaluation
=============================================
Router -> Expert pipeline on core_test.parquet.

Three routing strategies:
  Top-1:  Route to argmax coarse class -> expert predicts fine class
  Top-K:  Route to top-K experts -> pick highest joint confidence
  Soft:   Weighted ensemble of all expert probs by router softmax

Single-class groups (Stromal, Stem_Progenitor) bypass experts.

Usage:
  python scripts/eval_moe_e2e.py \
    --test_parquet prepared/splits_v3_seed1337/core_test/core_benchmark_test.parquet \
    --router_ckpt experiments/router_best/best.pt \
    --expert_dir experiments/ \
    --out_dir results/moe_e2e \
    --top_k 2
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.lora import apply_lora_to_timm_vit
from models.model import MultiViewClassifier
from data.dataset import CellParquetMultiView, AugCfg


# =============================================
# Coarse <-> Fine mappings (v2)
# =============================================
'''
FINE_TO_COARSE = {
    "Breast cancer cells": "Cancer", "Colon cancer cells": "Cancer",
    "Liver cancer cells": "Cancer", "Lung cancer cells": "Cancer",
    "Ovary cancer cells": "Cancer", "Pancreas cancer cells": "Cancer",
    "Skin cancer cells": "Cancer",
    "T cells": "Lymphoid", "B cells": "Lymphoid", "NK cells": "Lymphoid",
    "Astrocytes": "Neuroglial", "Microglia": "Neuroglial",
    "Neurons": "Neuroglial", "Oligodendrocytes": "Neuroglial",
    "Epithelial cells": "Tissue_Structural", "Fibroblasts": "Tissue_Structural",
    "Pericytes": "Tissue_Structural", "Adipocytes": "Tissue_Structural",
    "Endothelial cells": "Vascular", "Myeloid cells": "Vascular",
    "Smooth muscle cells": "Vascular",
    "Stromal cells": "Stromal", "Stem and progenitor cells": "Stem_Progenitor",
}
# Groups that bypass experts (single fine class)
PASSTHROUGH_GROUPS = {
    "Stromal": "Stromal cells",
    "Stem_Progenitor": "Stem and progenitor cells",
}

EXPERT_GROUPS = ["Cancer", "Lymphoid", "Neuroglial", "Tissue_Structural", "Vascular"]
'''

FINE_TO_COARSE = {
    "Colon cancer cells": "Cancer", "Liver cancer cells": "Cancer",
    "Lung cancer cells": "Cancer", "Ovary cancer cells": "Cancer",
    "Pancreas cancer cells": "Cancer", "Skin cancer cells": "Cancer",
    "T cells": "Lymphoid", "B cells": "Lymphoid", "NK cells": "Lymphoid",
    "Astrocytes": "Neuroglial", "Microglia": "Neuroglial",
    "Neurons": "Neuroglial", "Oligodendrocytes": "Neuroglial",
    "Epithelial cells": "Tissue_Vascular", "Fibroblasts": "Tissue_Vascular",
    "Pericytes": "Tissue_Vascular", "Endothelial cells": "Tissue_Vascular",
    "Myeloid cells": "Tissue_Vascular", "Smooth muscle cells": "Tissue_Vascular",
    "Stromal cells": "Stromal", "Stem and progenitor cells": "Stem_Progenitor",
}

PASSTHROUGH_GROUPS = {
    "Stromal": "Stromal cells",
    "Stem_Progenitor": "Stem and progenitor cells",
}

EXPERT_GROUPS = ["Cancer", "Lymphoid", "Neuroglial", "Tissue_Vascular"]


# =============================================
# Model loading
# =============================================

def load_model(ckpt_path, device):
    """Load a trained model from checkpoint (router or expert).
    Reconstructs architecture from saved args, applies LoRA, loads weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    class_to_idx = ckpt.get("class_to_idx", None)

    if class_to_idx is None:
        cm_path = saved_args.get("class_map", None)
        if cm_path and Path(cm_path).exists():
            with open(cm_path) as f:
                class_to_idx = json.load(f)
        else:
            raise ValueError(f"No class_to_idx in checkpoint {ckpt_path}")

    num_classes = len(class_to_idx)

    model = MultiViewClassifier(
        model_name=saved_args.get("model", "vit_base_patch16_dinov3.lvd1689m"),
        num_classes=num_classes,
        img_size=saved_args.get("img_size", 224),
        pretrained=False,
        fuse=saved_args.get("fuse", "gate"),
        use_cosine_head=saved_args.get("use_cosine_head", False),
        verbose=False,
    ).to(device)

    # Freeze backbone then apply LoRA (must match training config)
    for p in model.backbone.parameters():
        p.requires_grad = False

    if saved_args.get("use_lora", True):
        lora_blocks = saved_args.get("lora_blocks", "0,2,4,6,8,10,11")
        lora_targets = saved_args.get("lora_targets", "qkv,proj,mlp_fc1,mlp_fc2")

        apply_lora_to_timm_vit(
            model.backbone,
            last_n_blocks=lora_blocks,
            r=saved_args.get("lora_rank", 16),
            alpha=saved_args.get("lora_alpha", 32),
            dropout=0.0,
            targets=tuple(t.strip() for t in lora_targets.split(",") if t.strip()),
            verbose=False,
        )
        model.to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"  Loaded {ckpt_path}")
    print(f"    Classes ({num_classes}): {list(class_to_idx.keys())}")

    return model, class_to_idx, idx_to_class


@torch.no_grad()
def run_inference(model, loader, device):
    """Run model on dataloader, return logits and gate weights."""
    all_logits = []
    all_gates = []

    amp_dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)

    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=torch.cuda.is_available()):
            logits, _emb, aux = model(x)

        all_logits.append(logits.float().cpu())
        if "gate_weights" in aux:
            all_gates.append(aux["gate_weights"].float().cpu())

    logits = torch.cat(all_logits, 0)
    gates = torch.cat(all_gates, 0) if all_gates else None
    return logits, gates


# =============================================
# Metrics
# =============================================

def compute_metrics(y_true, y_pred, class_names):
    """Per-class precision, recall, F1 and macro F1."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    true_idx = np.array([name_to_idx.get(y, -1) for y in y_true])
    pred_idx = np.array([name_to_idx.get(y, -1) for y in y_pred])

    rows = []
    for i, cname in enumerate(class_names):
        tp = int(((true_idx == i) & (pred_idx == i)).sum())
        fn = int(((true_idx == i) & (pred_idx != i)).sum())
        fp = int(((true_idx != i) & (pred_idx == i)).sum())
        n = tp + fn
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        rows.append({"class": cname, "n": n, "prec": round(prec, 4),
                      "rec": round(rec, 4), "f1": round(f1, 4)})

    macro_f1 = np.mean([r["f1"] for r in rows if r["n"] > 0])
    acc = float((true_idx == pred_idx).sum()) / max(len(true_idx), 1)
    balanced_acc = float(np.mean([r["rec"] for r in rows if r["n"] > 0]))
    return rows, round(float(macro_f1), 4), round(acc, 4), round(balanced_acc, 4)


def build_confusion_matrix(y_true, y_pred, class_names):
    """Build confusion matrix as numpy array."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti, pi = name_to_idx.get(t, -1), name_to_idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti, pi] += 1
    return cm


def print_metrics(rows, macro_f1, acc, balanced_acc, label=""):
    """Pretty-print per-class metrics."""
    print(f"\n{'=' * 65}")
    print(f"  {label}  |  Macro F1: {macro_f1:.4f}  |  Acc: {acc:.4f}  |  Balanced Acc: {balanced_acc:.4f}")
    print(f"{'=' * 65}")
    print(f"  {'Class':<35} {'N':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-' * 64}")
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(f"  {r['class']:<35} {r['n']:>6,} {r['prec']:>7.3f} "
              f"{r['rec']:>7.3f} {r['f1']:>7.3f}")


# =============================================
# Main
# =============================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_parquet", required=True)
    ap.add_argument("--router_ckpt", required=True)
    ap.add_argument("--expert_dir", default=None,
                    help="Base dir containing expert_Cancer/, expert_Lymphoid/, etc.")
    ap.add_argument("--router_only", action="store_true",
                    help="Only run router inference, skip experts")
    ap.add_argument("--out_dir", default="results/moe_e2e")
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--skip_slides", nargs="*", default=[],
                    help="List of slide names to skip during evaluation.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Global fine class list --
    all_fine_classes = sorted(FINE_TO_COARSE.keys())
    global_fine_to_idx = {c: i for i, c in enumerate(all_fine_classes)}
    n_fine = len(all_fine_classes)

    # -- Load test data --
    df_test = pd.read_parquet(args.test_parquet)
    needs_save = False
    
    if "coarse_label" not in df_test.columns:
        df_test["coarse_label"] = df_test["label"].map(FINE_TO_COARSE)
        needs_save = True

    if args.skip_slides:
        initial_len = len(df_test)
        # Filter out cells belonging to the skipped slides
        mask = ~df_test['cell_id'].apply(lambda x: any(str(x).startswith(s) for s in args.skip_slides))
        df_test = df_test[mask].copy()
        print(f"Skipping {len(args.skip_slides)} slides. Filtered {initial_len - len(df_test):,} cells.")
        needs_save = True
        
    if needs_save:
        import uuid
        eval_parquet = out_dir / f"test_eval_cache_{uuid.uuid4().hex[:8]}.parquet"
        df_test.to_parquet(eval_parquet, index=False)
        args.test_parquet = str(eval_parquet)
        print(f"Saved evaluation parquet to {eval_parquet}")

    print(f"Test set: {len(df_test):,} cells, {df_test['label'].nunique()} fine classes\n")

    true_fine = df_test["label"].values
    true_coarse = df_test["coarse_label"].values
    N = len(df_test)

    # =========================================================
    # STAGE 1: Router
    # =========================================================
    print("=" * 70)
    print("STAGE 1: Router Inference")
    print("=" * 70)

    router_model, router_c2i, router_i2c = load_model(args.router_ckpt, device)

    ds_router = CellParquetMultiView(
        parquet_path=args.test_parquet,
        class_to_idx=router_c2i,
        label_col="coarse_label",
        size=args.img_size,
        aug=AugCfg(enable=False),
        return_meta=True,
    )
    router_loader = DataLoader(
        ds_router, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    t0 = time.time()
    router_logits, router_gates = run_inference(router_model, router_loader, device)
    router_probs = F.softmax(router_logits, dim=1)
    print(f"  Done: {N:,} cells, {time.time()-t0:.1f}s")

    del router_model
    torch.cuda.empty_cache()

    # Top-1 routing
    coarse_pred_idx = router_probs.argmax(dim=1)
    coarse_pred_names = np.array([router_i2c[int(c)] for c in coarse_pred_idx])

    router_acc = float((coarse_pred_names == true_coarse).mean())
    router_rows, router_macro_f1, _, _ = compute_metrics(true_coarse, coarse_pred_names, list(router_c2i.keys()))
    
    # Calculate balanced accuracy (macro-recall)
    valid_recalls = [r["rec"] for r in router_rows if r["n"] > 0]
    router_balanced_acc = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0
    
    print(f"  Router coarse accuracy (micro): {router_acc:.4f}")
    print(f"  Router coarse accuracy (balanced): {router_balanced_acc:.4f}")
    print(f"  Router coarse macro F1: {router_macro_f1:.4f}")

    # Gate analysis
    if router_gates is not None:
        g_mean = router_gates.mean(dim=0).numpy()
        print(f"  Router gate (2.5x/10x): {g_mean[0]:.3f} / {g_mean[1]:.3f}")

    # =========================================================
    # STAGE 2: Expert Inference (all cells through all experts)
    # =========================================================
    experts = {}
    expert_logits = {}
    expert_gates = {}
    expert_to_global = {}

    if args.router_only:
        print("\n--router_only: skipping expert inference")
    elif args.expert_dir is None:
        print("\nNo --expert_dir provided, skipping expert inference")
    else:
        print("\n" + "=" * 70)
        print("STAGE 2: Expert Inference")
        print("=" * 70)

        for group in EXPERT_GROUPS:
            ckpt_path = Path(args.expert_dir) / f"expert_{group}" / "best.pt"
            if not ckpt_path.exists():
                print(f"\n  WARNING: {ckpt_path} not found, skipping {group}")
                continue

            print(f"\n  Expert: {group}")
            model, c2i, i2c = load_model(ckpt_path, device)

            # Pass all possible classes to prevent the dataset from filtering out any cells.
            # We need predictions for ALL cells (N) for every expert to handle incorrect routing.
            dummy_c2i = {lbl: 0 for lbl in df_test["label"].unique()}
            
            ds_expert = CellParquetMultiView(
                parquet_path=args.test_parquet,
                class_to_idx=dummy_c2i,
                label_col="label",
                size=args.img_size,
                aug=AugCfg(enable=False),
                return_meta=True,
            )
            expert_loader = DataLoader(
                ds_expert, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )

            t1 = time.time()
            logits, gates = run_inference(model, expert_loader, device)
            expert_logits[group] = logits
            expert_gates[group] = gates
            experts[group] = {"c2i": c2i, "i2c": i2c}
            print(f"    {logits.shape[1]} classes, {time.time()-t1:.1f}s")

            if gates is not None:
                g_mean = gates.mean(dim=0).numpy()
                print(f"    Gate (2.5x/10x): {g_mean[0]:.3f} / {g_mean[1]:.3f}")

            del model
            torch.cuda.empty_cache()

        # Build expert -> global index mapping
        for group, einfo in experts.items():
            mapping = []
            for local_idx in range(len(einfo["c2i"])):
                fine_name = einfo["i2c"][local_idx]
                mapping.append(global_fine_to_idx[fine_name])
            expert_to_global[group] = mapping

    # =========================================================
    # ASSEMBLE PREDICTIONS (only if experts loaded)
    # =========================================================
    summary = {}
    pred_top1 = None
    pred_topk = None
    pred_soft = None

    if experts:
        print("\n" + "=" * 70)
        print("ASSEMBLING PREDICTIONS")
        print("=" * 70)

        # -- Strategy 0: Oracle (Perfect Routing) --
        pred_oracle = []
        for i in range(N):
            true_c = true_coarse[i]
            if true_c in PASSTHROUGH_GROUPS:
                pred_oracle.append(PASSTHROUGH_GROUPS[true_c])
            elif true_c in experts:
                einfo = experts[true_c]
                fine_idx = int(expert_logits[true_c][i].argmax())
                pred_oracle.append(einfo["i2c"][fine_idx])
            else:
                pred_oracle.append("UNKNOWN")

        # -- Strategy 1: Top-1 --
        pred_top1 = []
        expert_margins_top1 = []
        expert_top1_probs_list = []
        expert_gate_2_5x_list = []
        for i in range(N):
            coarse = coarse_pred_names[i]
            if coarse in PASSTHROUGH_GROUPS:
                pred_top1.append(PASSTHROUGH_GROUPS[coarse])
                expert_margins_top1.append(float("nan"))
                expert_top1_probs_list.append(float("nan"))
                expert_gate_2_5x_list.append(float("nan"))
            elif coarse in experts:
                einfo = experts[coarse]
                probs = F.softmax(expert_logits[coarse][i], dim=0)
                sorted_p, _ = probs.sort(descending=True)
                fine_idx = int(probs.argmax())
                pred_top1.append(einfo["i2c"][fine_idx])
                margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float("nan")
                expert_margins_top1.append(margin)
                expert_top1_probs_list.append(float(sorted_p[0]))
                if expert_gates[coarse] is not None:
                    expert_gate_2_5x_list.append(float(expert_gates[coarse][i, 0]))
                else:
                    expert_gate_2_5x_list.append(float("nan"))
            else:
                pred_top1.append("UNKNOWN")
                expert_margins_top1.append(float("nan"))
                expert_top1_probs_list.append(float("nan"))
                expert_gate_2_5x_list.append(float("nan"))

        # -- Strategy 2: Top-K --
        topk_vals, topk_idxs = router_probs.topk(args.top_k, dim=1)
        pred_topk = []
        for i in range(N):
            best_conf = -1.0
            best_pred = "UNKNOWN"
            for j in range(args.top_k):
                coarse = router_i2c[int(topk_idxs[i, j])]
                router_w = float(topk_vals[i, j])

                if coarse in PASSTHROUGH_GROUPS:
                    conf = router_w
                    fine_name = PASSTHROUGH_GROUPS[coarse]
                elif coarse in experts:
                    einfo = experts[coarse]
                    probs = F.softmax(expert_logits[coarse][i], dim=0)
                    fine_idx = int(probs.argmax())
                    fine_name = einfo["i2c"][fine_idx]
                    conf = router_w * float(probs[fine_idx])
                else:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_pred = fine_name
            pred_topk.append(best_pred)

        # -- Strategy 3: Soft --
        pred_soft = []
        for i in range(N):
            global_scores = torch.zeros(n_fine)

            for group in EXPERT_GROUPS:
                coarse_idx = router_c2i.get(group, -1)
                if coarse_idx < 0:
                    continue
                w = float(router_probs[i, coarse_idx])

                if group in PASSTHROUGH_GROUPS:
                    gidx = global_fine_to_idx[PASSTHROUGH_GROUPS[group]]
                    global_scores[gidx] += w
                elif group in experts:
                    eprobs = F.softmax(expert_logits[group][i], dim=0)
                    for local_idx, gidx in enumerate(expert_to_global[group]):
                        global_scores[gidx] += w * float(eprobs[local_idx])

            # Passthrough groups (they are not in EXPERT_GROUPS)
            for group, fine_name in PASSTHROUGH_GROUPS.items():
                coarse_idx = router_c2i.get(group, -1)
                if coarse_idx < 0:
                    continue
                w = float(router_probs[i, coarse_idx])
                gidx = global_fine_to_idx[fine_name]
                global_scores[gidx] += w

            pred_soft.append(all_fine_classes[int(global_scores.argmax())])

        # =========================================================
        # RESULTS
        # =========================================================
        strategies = {
            "oracle": pred_oracle,
            "top1": pred_top1,
            f"top{args.top_k}": pred_topk,
            "soft": pred_soft,
        }

        for name, preds in strategies.items():
            rows, mf1, acc, bal_acc = compute_metrics(true_fine, preds, all_fine_classes)
            print_metrics(rows, mf1, acc, bal_acc, label=f"Strategy: {name.upper()}")

            cm = build_confusion_matrix(true_fine, preds, all_fine_classes)
            summary[name] = {"macro_f1": mf1, "accuracy": acc, "balanced_acc": bal_acc, "per_class": rows}

            (out_dir / f"per_class_{name}.json").write_text(json.dumps(rows, indent=2))
            (out_dir / f"confusion_matrix_{name}.json").write_text(
                json.dumps({"class_names": all_fine_classes, "matrix": cm.tolist()}, indent=2))

    # =========================================================
    # ROUTING ANALYSIS
    # =========================================================
    print(f"\n{'=' * 70}")
    print("ROUTING ANALYSIS")
    print(f"{'=' * 70}")

    all_coarse_classes = sorted(router_c2i.keys())
    coarse_cm = build_confusion_matrix(true_coarse, coarse_pred_names, all_coarse_classes)
    (out_dir / "confusion_matrix_coarse.json").write_text(
        json.dumps({"class_names": all_coarse_classes, "matrix": coarse_cm.tolist()}, indent=2))

    # Per-coarse
    print(f"\n  {'Coarse':<25} {'N':>6} {'Route Acc':>10}")
    print(f"  {'-' * 43}")
    for cname in sorted(router_c2i.keys()):
        mask = true_coarse == cname
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int((coarse_pred_names[mask] == cname).sum())
        print(f"  {cname:<25} {n:>6,} {correct/n:>10.4f}")

    # Per-fine
    print(f"\n  {'Fine Class':<35} {'Coarse':>20} {'N':>6} {'Route Acc':>10}")
    print(f"  {'-' * 75}")
    fine_route_rows = []
    for fname in sorted(all_fine_classes):
        coarse = FINE_TO_COARSE[fname]
        mask = true_fine == fname
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int((coarse_pred_names[mask] == coarse).sum())
        acc_val = correct / n
        fine_route_rows.append({"fine": fname, "coarse": coarse,
                                "n": n, "route_acc": round(acc_val, 4)})
        print(f"  {fname:<35} {coarse:>20} {n:>6,} {acc_val:>10.4f}")

    # =========================================================
    # CONFIDENCE ANALYSIS
    # =========================================================
    print(f"\n{'=' * 70}")
    print("CONFIDENCE ANALYSIS (Router margin buckets)")
    print(f"{'=' * 70}")

    sorted_p, _ = router_probs.sort(dim=1, descending=True)
    margins = (sorted_p[:, 0] - sorted_p[:, 1]).numpy()

    buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    has_fine = pred_top1 is not None
    header = f"  {'Margin':<15} {'N':>8} {'Route Acc':>10}"
    if has_fine:
        header += f" {'Top1 Fine Acc':>14}"
    print(f"\n{header}")
    print(f"  {'-' * (49 if has_fine else 35)}")
    for lo, hi in buckets:
        mask = (margins >= lo) & (margins < hi)
        nb = int(mask.sum())
        if nb == 0:
            continue
        r_acc = float((coarse_pred_names[mask] == np.array(true_coarse)[mask]).mean())
        line = f"  [{lo:.1f}, {hi:.1f})     {nb:>8,} {r_acc:>10.4f}"
        if has_fine:
            f_acc = float((np.array(pred_top1)[mask] == np.array(true_fine)[mask]).mean())
            line += f" {f_acc:>14.4f}"
        print(line)

    # =========================================================
    # SAVE
    # =========================================================
    summary_json = {
        "router_coarse_acc": router_acc,
        "router_coarse_balanced_acc": router_balanced_acc,
        "router_coarse_macro_f1": router_macro_f1,
        "strategies": {k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "balanced_acc": v["balanced_acc"]}
                       for k, v in summary.items()},
        "fine_routing": fine_route_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    keep_cols = ["cell_id", "label", "coarse_label", "tissue",
                 "x_centroid", "y_centroid", "patch_id",
                 "img_path_2p5x", "img_path_10x"]
    df_out = df_test[[c for c in keep_cols if c in df_test.columns]].copy()
        
    df_out["router_pred"] = coarse_pred_names
    df_out["router_margin"] = margins
    if router_gates is not None:
        df_out["gate_2_5x"] = router_gates[:, 0].numpy()
        df_out["gate_10x"] = router_gates[:, 1].numpy()
    if pred_top1 is not None:
        df_out["pred_top1"] = pred_top1
        df_out[f"pred_top{args.top_k}"] = pred_topk
        df_out["pred_soft"] = pred_soft
        df_out["expert_margin"]    = expert_margins_top1      # expert top1 - top2 (NaN for passthrough)
        df_out["expert_top1_prob"] = expert_top1_probs_list   # expert max(P_e)    (NaN for passthrough)
        df_out["expert_gate_2_5x"] = expert_gate_2_5x_list
        # Passthrough cells have no expert → fill with router signal
        df_out["expert_margin"]    = df_out["expert_margin"].fillna(df_out["router_margin"])  # M_r as fallback
        df_out["expert_top1_prob"] = df_out["expert_top1_prob"].fillna(1.0)                   # single class → P_e = 1
        df_out["router_x_expert"]  = (df_out["router_margin"]
                                      * df_out["expert_margin"])   # M_r × M_e
        df_out["router_x_maxpe"]   = (df_out["router_margin"]
                                      * df_out["expert_top1_prob"])  # M_r × max(P_e)
    df_out.to_parquet(out_dir / "predictions.parquet", index=False)

    if needs_save:
        try:
            eval_parquet.unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: could not delete temporary parquet {eval_parquet}: {e}")


    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Router coarse acc (micro): {router_acc:.4f}")
    print(f"  Router coarse acc (balanced): {router_balanced_acc:.4f}")
    print(f"  Router coarse macro F1: {router_macro_f1:.4f}")
    if summary:
        print(f"\n  {'Strategy':<15} {'Macro F1':>10} {'Accuracy':>10} {'Balanced Acc':>13}")
        print(f"  {'-' * 50}")
        for name, s in summary.items():
            print(f"  {name:<15} {s['macro_f1']:>10.4f} {s['accuracy']:>10.4f} {s['balanced_acc']:>13.4f}")
    print(f"\n  Results: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()