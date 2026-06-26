#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm

from models.lora import apply_lora_to_timm_vit
from data.dataset import CellParquetMultiView, AugCfg


class CosineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, s_init: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self._s_unconstrained = nn.Parameter(torch.tensor(float(s_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        s = F.softplus(self._s_unconstrained) + 1e-6
        return s * (x @ w.t())


class ScaleGate(nn.Module):
    def __init__(self, d_embed: int, n_scales: int = 2, hidden: int | None = None):
        super().__init__()
        h = hidden or max(128, d_embed // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(n_scales * d_embed),
            nn.Linear(n_scales * d_embed, h),
            nn.GELU(),
            nn.Linear(h, n_scales),
        )

    def forward(self, P: torch.Tensor):
        B, S, D = P.shape
        g = F.softmax(self.net(P.reshape(B, S * D)), dim=-1)
        fused = torch.sum(P * g.unsqueeze(-1), dim=1)
        return fused, g


class DinoV3Classifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, img_size: int, pretrained: bool,
                 multi_view: bool, fuse: str, use_gate: bool):
        super().__init__()
        create_kwargs = dict(pretrained=pretrained, num_classes=0)
        try:
            create_kwargs["img_size"] = img_size
        except Exception:
            pass

        self.backbone = timm.create_model(model_name, **create_kwargs)
        self.num_features = getattr(self.backbone, "num_features", None)

        if self.num_features is None:
            with torch.no_grad():
                x = torch.randn(2, 3, img_size, img_size)
                emb = self.forward_features(x)
                self.num_features = int(emb.shape[-1])

        self.head = CosineHead(self.num_features, num_classes, s_init=30.0)

        self.multi_view = bool(multi_view)
        self.fuse = str(fuse)
        self.gate = ScaleGate(self.num_features, n_scales=2) if (self.multi_view and self.fuse == "gate") else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.ndim == 3:
            if getattr(self.backbone, "num_prefix_tokens", 0) > 0:
                return feats[:, 0]
            return feats.mean(dim=1)
        if feats.ndim == 4:
            return feats.mean(dim=(2, 3))
        return feats

    def forward(self, x: torch.Tensor):
        aux = {}
        if x.ndim == 5:
            B, S, C, H, W = x.shape
            xs = x.reshape(B * S, C, H, W)
            emb_all = self.forward_features(xs)             # [B*S, D]
            P = emb_all.reshape(B, S, -1)                   # [B, S, D]
            Pn = F.normalize(P, dim=-1)                     # normalize once
            aux["emb_views"] = Pn                           # store normalized

            if self.fuse == "avg":
                emb = F.normalize(Pn.mean(dim=1), dim=-1)
                aux["g"] = torch.full((B, S), 1.0 / S, device=x.device, dtype=emb.dtype)
            elif self.fuse == "gate":
                emb, g = self.gate(Pn)                      # gate on normalized
                emb = F.normalize(emb, dim=-1)
                aux["g"] = g
            else:
                raise ValueError(f"Unknown fuse={self.fuse}")

        else:
            emb = self.forward_features(x)
            emb = F.normalize(emb, dim=-1)

        logits = self.head(emb)
        return logits, emb, aux


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def macro_f1(pred, y, num_classes: int):
    f1s = []
    pred = pred.cpu()
    y = y.cpu()
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        if tp == 0 and fp == 0 and fn == 0:
            continue
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1s.append(2 * p * r / (p + r + 1e-12))
    return float(np.mean(f1s)) if f1s else 0.0


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, opt, scaler, device, amp_dtype, num_classes, cons_w=0.0, epoch=1):
    model.train()
    loss_sum = 0.0
    cons_sum = 0.0
    n = 0

    # Warmup: ramp cons_w from 0 over epochs 3-7
    ramp = min(1.0, max(0, epoch - 2) / 5.0)
    effective_cons_w = cons_w * ramp

    for x, y, _tissue in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            logits, _emb, aux = model(x)
            loss_ce = F.cross_entropy(logits, y)

            # ── Consistency loss ──
            loss_cons = torch.tensor(0.0, device=device)
            if effective_cons_w > 0 and "emb_views" in aux:
                Pn = aux["emb_views"]                             # already normalized
                cos = (Pn[:, 0] * Pn[:, 1]).sum(dim=1)           # [B]
                loss_cons = (1.0 - cos).mean()

            loss = loss_ce + effective_cons_w * loss_cons

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        loss_sum += float(loss_ce.item()) * y.size(0)
        cons_sum += float(loss_cons.item()) * y.size(0)
        n += y.size(0)

    return loss_sum / max(1, n), cons_sum / max(1, n)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, device, amp_dtype, num_classes):
    model.eval()
    loss_sum = 0.0
    n = 0
    all_pred = []
    all_y = []

    # Cross-view alignment accumulators
    cv_cos_sum = 0.0
    cv_n = 0

    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for x, y, _tissue in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _emb, aux = model(x)
            loss = F.cross_entropy(logits, y)

            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            all_pred.append(logits.argmax(1).detach().cpu())
            all_y.append(y.detach().cpu())

            # Track cross-view cosine
            if "emb_views" in aux:
                Pn = aux["emb_views"]                       # already normalized
                cos = (Pn[:, 0] * Pn[:, 1]).sum(dim=1)
                cv_cos_sum += cos.sum().item()
                cv_n += cos.size(0)

    pred = torch.cat(all_pred) if all_pred else torch.empty(0, dtype=torch.long)
    yy = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)
    acc = (pred == yy).float().mean().item() if yy.numel() else 0.0
    mf1 = macro_f1(pred, yy, num_classes)
    cv_cos = cv_cos_sum / max(1, cv_n)
    return loss_sum / max(1, n), acc, mf1, cv_cos


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--class_map", required=True)
    ap.add_argument("--model", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--multi_view", type=int, default=1)
    ap.add_argument("--fuse", type=str, default="avg", choices=["avg", "gate"])

    ap.add_argument("--use_lora", type=int, default=0)
    ap.add_argument("--lora_blocks", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--cons_weight", type=float, default=0.05,
                    help="Cross-view consistency loss weight (0=off)")

    ap.add_argument("--out_dir", type=str, default="runs/finetune")
    ap.add_argument("--split_npy", type=str, default=None,
                help="Path to save/load split indices for reproducibility")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    # Dataset
    aug = AugCfg(enable=False)
    ds_full = CellParquetMultiView(
        parquet_path=args.train_parquet,
        class_to_idx=class_to_idx,
        size=args.img_size,
        aug=aug,
        return_meta=False
    )
    ds_all = ds_full.df

    N = len(ds_all)
    val_n = int(N * args.val_frac)

    if args.split_npy and Path(args.split_npy).exists():
        idx = np.load(args.split_npy)
        print(f"Loaded split from {args.split_npy}")
    else:
        idx = np.arange(N)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(idx)
        if args.split_npy:
            Path(args.split_npy).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.split_npy, idx)
        print(f"Saved split to {args.split_npy}")

    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    df_train = ds_all.iloc[train_idx].reset_index(drop=True)
    df_val = ds_all.iloc[val_idx].reset_index(drop=True)

    ds_train = CellParquetMultiView(
        args.train_parquet, class_to_idx,
        size=args.img_size,
        aug=AugCfg(enable=True),
        return_meta=False
    )
    ds_train.df = df_train

    ds_val = CellParquetMultiView(
        args.train_parquet, class_to_idx,
        size=args.img_size,
        aug=AugCfg(enable=False),
        return_meta=False
    )
    ds_val.df = df_val

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    model = DinoV3Classifier(
        model_name=args.model,
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained=True,
        multi_view=bool(args.multi_view),
        fuse=args.fuse,
        use_gate=(args.fuse == "gate"),
    ).to(device)

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    lora_params = []
    if args.use_lora:
        lora_params = apply_lora_to_timm_vit(
            model.backbone,
            last_n_blocks=args.lora_blocks,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            targets=("qkv",),
            verbose=True,
        )
        model.to(device)
        n_lora = sum(p.numel() for p in lora_params if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA: {n_lora:,} params | Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

    # Optimizer
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    gate_params = [p for p in model.gate.parameters() if p.requires_grad] if model.gate is not None else []
    params = []
    if head_params:
        params.append({"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay})
    if gate_params:
        params.append({"params": gate_params, "lr": args.lr_head, "weight_decay": args.weight_decay})
    if lora_params:
        params.append({"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.0})

    opt = torch.optim.AdamW(params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best = -1.0
    best_path = out_dir / "best.pt"

    print(f"cons_weight={args.cons_weight}  fuse={args.fuse}  epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_cons = train_one_epoch(
            model, train_loader, opt, scaler, device, amp_dtype, num_classes,
            cons_w=args.cons_weight, epoch=epoch,
        )
        va_loss, va_acc, va_f1, va_cv_cos = eval_epoch(model, val_loader, device, amp_dtype, num_classes)
        dt = time.time() - t0

        cur_lr = scheduler.get_last_lr()[0]
        eff_cw = args.cons_weight * min(1.0, max(0, epoch - 2) / 5.0)
        print(f"epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f}  cons_loss={tr_cons:.4f}  cons_w_eff={eff_cw:.3f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_macro_f1={va_f1:.4f}  "
              f"cv_cos={va_cv_cos:.4f}  lr={cur_lr:.2e}  secs={dt:.1f}")

        scheduler.step()

        score = va_f1
        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "class_to_idx": class_to_idx,
                "best_macro_f1": best,
            }, best_path)
            print(f"saved {best_path}  macro_f1={best:.4f}")

    print(f"done. best macro_f1={best:.4f}  ckpt={best_path}")


if __name__ == "__main__":
    main()