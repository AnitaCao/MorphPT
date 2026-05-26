import os
import torch
import torch.distributed as dist
import numpy as np

# ─────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────

def setup_ddp():
    """Returns (rank, local_rank, world_size, is_ddp).
    Supports both torchrun (RANK/LOCAL_RANK) and srun (SLURM_PROCID/SLURM_LOCALID)."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", -1)))
    if rank < 0:
        return 0, 0, 1, False

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    # When SLURM uses --gpus-per-task, each task sees only its own GPU as device 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" not in os.environ:
        n_visible = torch.cuda.device_count()
        local_rank = int(os.environ.get("SLURM_LOCALID", 0)) % max(n_visible, 1)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Dynamic port from job ID to avoid collisions
    job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    default_port = str(20000 + (job_id % 20000))
    os.environ.setdefault("MASTER_PORT", default_port)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size, True


def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


# ─────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────

def compute_class_weights(ds, label_col: str, num_classes: int, class_to_idx: dict):
    from data.dataset import MemmapDataset  # local import to avoid circular dependency
    counts = torch.zeros(num_classes)
    if isinstance(ds, MemmapDataset):
        # _y is already int64 class indices
        y = ds._y
        for idx in range(num_classes):
            counts[idx] = max(int((y == idx).sum()), 1)
    else:
        for label, idx in class_to_idx.items():
            n = (ds.df[label_col] == label).sum()
            counts[idx] = max(n, 1)
    weights = 1.0 / counts.sqrt()
    weights = weights / weights.sum() * num_classes
    return weights


# ─────────────────────────────────────────────
# General Metrics
# ─────────────────────────────────────────────

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


def per_class_metrics(pred, y, num_classes: int, idx_to_class: dict):
    pred = pred.cpu()
    y = y.cpu()
    rows = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        n = (y == c).sum().item()
        p = tp / (tp + fp + 1e-12) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn + 1e-12) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r + 1e-12) if (p + r) > 0 else 0.0
        rows.append({"class": idx_to_class.get(c, f"cls_{c}"),
                      "n": n, "prec": p, "rec": r, "f1": f1})
    return rows


def confusion_matrix_np(pred, y, num_classes: int):
    """Returns [num_classes, num_classes] confusion matrix as numpy array.
    cm[true, pred] = count."""
    pred = pred.cpu().numpy()
    y = y.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def gather_eval(pred, yy, ii, loss_sum, n_local, gate_sum, gate_n,
                cv_cos_sum, cv_n, num_classes, device, is_ddp, rank):
    """All-gather local results across ranks; compute global metrics on rank 0."""
    if is_ddp:
        # Gather variable-length pred/yy tensors
        def _gather_tensor(t):
            t = t.to(device)
            sizes = [torch.zeros(1, dtype=torch.long, device=device)
                     for _ in range(dist.get_world_size())]
            dist.all_gather(sizes, torch.tensor([t.shape[0]], device=device))
            max_sz = max(s.item() for s in sizes)
            padded = torch.zeros(max_sz, dtype=t.dtype, device=device)
            padded[:t.shape[0]] = t
            bufs = [torch.zeros(max_sz, dtype=t.dtype, device=device)
                    for _ in range(dist.get_world_size())]
            dist.all_gather(bufs, padded)
            return torch.cat([b[:s.item()] for b, s in zip(bufs, sizes)]).cpu()

        pred_all = _gather_tensor(pred)
        yy_all = _gather_tensor(yy)
        ii_all = _gather_tensor(ii)

        # Gather scalars via all_reduce
        t_loss = torch.tensor([loss_sum], device=device)
        t_n    = torch.tensor([n_local],  device=device, dtype=torch.float)
        t_cvc  = torch.tensor([cv_cos_sum], device=device)
        t_cvn  = torch.tensor([cv_n],     device=device, dtype=torch.float)
        for t in (t_loss, t_n, t_cvc, t_cvn):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum_g = t_loss.item()
        n_g        = int(t_n.item())
        cv_cos_g   = t_cvc.item() / max(1, t_cvn.item())

        # Gate weights (optional)
        avg_gate = None
        if gate_sum is not None:
            t_gate  = gate_sum.to(device)
            t_gaten = torch.tensor([gate_n], device=device, dtype=torch.float)
            dist.all_reduce(t_gate,  op=dist.ReduceOp.SUM)
            dist.all_reduce(t_gaten, op=dist.ReduceOp.SUM)
            if rank == 0:
                avg_gate = (t_gate / t_gaten).cpu().tolist()
    else:
        pred_all   = pred
        yy_all     = yy
        ii_all     = ii
        loss_sum_g = loss_sum
        n_g        = n_local
        cv_cos_g   = cv_cos_sum / max(1, cv_n) if cv_n > 0 else 0.0
        avg_gate   = (gate_sum / gate_n).tolist() if gate_n > 0 else None

    # Metrics only meaningful on rank 0
    if rank == 0:
        # Drop duplicates added by DistributedSampler's padding, then sort by original index
        # to perfectly align with static metadata arrays like ds._fine_labels
        if ii_all.numel() > 0:
            sort_idx = torch.argsort(ii_all)
            ii_sorted = ii_all[sort_idx]
            pred_sorted = pred_all[sort_idx]
            yy_sorted = yy_all[sort_idx]
            
            # Then remove identical adjacent indices
            mask = torch.ones(ii_sorted.size(0), dtype=torch.bool)
            mask[1:] = (ii_sorted[1:] != ii_sorted[:-1])
            
            pred_all = pred_sorted[mask]
            yy_all = yy_sorted[mask]
            ii_all = ii_sorted[mask]

        acc  = (pred_all == yy_all).float().mean().item() if yy_all.numel() else 0.0
        mf1  = macro_f1(pred_all, yy_all, num_classes)
        va_loss = loss_sum_g / max(1, n_g)
        return va_loss, acc, mf1, pred_all, yy_all, ii_all, avg_gate, cv_cos_g
    return None, None, None, None, None, None, None, None


@torch.no_grad()
def eval_epoch_local(model, loader, device, amp_dtype, class_weight):
    """Run eval on this rank's shard. Returns local tensors (not yet gathered)."""
    import torch.nn.functional as F
    model.eval()
    loss_sum = 0.0
    n = 0
    all_pred = []
    all_y = []
    all_idx = []
    gate_sum = None
    gate_n = 0
    cv_cos_sum = 0.0
    cv_n = 0

    with torch.amp.autocast("cuda", dtype=amp_dtype,
                            enabled=torch.cuda.is_available()):
        for batch in loader:
            # Handle return_meta=True which yields (x, y, meta) instead of (x, y, tissue)
            if len(batch) == 3 and isinstance(batch[2], dict) and "index" in batch[2]:
                x, y, meta = batch
                idx = meta["index"]
            else:
                x, y, _tissue = batch
                # Fallback if return_meta=False, though we'll force it to True
                idx = torch.arange(n, n + y.size(0), dtype=torch.long)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, _emb, aux = model(x)
            if aux.get("is_log_prob"):
                loss = F.nll_loss(logits, y, weight=class_weight)
            else:
                loss = F.cross_entropy(logits, y, weight=class_weight)

            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            all_pred.append(logits.argmax(1).detach().cpu())
            all_y.append(y.detach().cpu())
            all_idx.append(idx.detach().cpu())

            if "gate_weights" in aux:
                g = aux["gate_weights"]
                if gate_sum is None:
                    gate_sum = g.sum(0).detach().cpu()
                else:
                    gate_sum += g.sum(0).detach().cpu()
                gate_n += g.size(0)

            if "emb_views" in aux:
                P = aux["emb_views"]
                Pn = F.normalize(P, dim=-1)
                cos = (Pn[:, 0] * Pn[:, 1]).sum(dim=1)
                cv_cos_sum += cos.sum().item()
                cv_n += cos.size(0)

    pred = torch.cat(all_pred) if all_pred else torch.empty(0, dtype=torch.long)
    yy = torch.cat(all_y) if all_y else torch.empty(0, dtype=torch.long)
    ii = torch.cat(all_idx) if all_idx else torch.empty(0, dtype=torch.long)
    return pred, yy, ii, loss_sum, n, gate_sum, gate_n, cv_cos_sum, cv_n


# ─────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────

def get_base_parser():
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    # Data
    ap.add_argument("--train_dir", default="",
                    help="Parquet train dir (ignored if --cache_dir is set)")
    ap.add_argument("--val_dir", default="",
                    help="Parquet val dir (ignored if --cache_dir is set)")
    ap.add_argument("--test_dir", type=str, default="",
                    help="Parquet test dir (ignored if --cache_dir is set)")
    ap.add_argument("--cache_dir", type=str, default="",
                    help="Memmap cache dir (from preprocess_memmap.py). "
                         "If set, uses MemmapDataset instead of parquet. ")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split",   type=str, default="val")
    ap.add_argument("--test_split",  type=str, default="test")
    ap.add_argument("--class_map", required=True)
    ap.add_argument("--label_col", default="coarse_label")

    # Model
    ap.add_argument("--model", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--fuse", type=str, required=True,
                    choices=["avg", "concat", "gate", "late"])

    # LoRA
    ap.add_argument("--use_lora", type=int, default=1)
    ap.add_argument("--lora_blocks", type=str, default="4",
                    help="Number of blocks from the end, or comma-separated block indices")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="qkv",
                    help="Comma-separated LoRA targets, e.g. 'qkv' or 'qkv,proj'")

    # Training
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Per-GPU batch size")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--cons_weight", type=float, default=0.0,
                    help="Cross-view consistency loss weight. Ramps over epochs 3-7.")
    ap.add_argument("--cons_T", type=float, default=3.0,
                    help="Temperature for cross-view consistency KL Divergence.")
    ap.add_argument("--ckpt_score_w", type=float, default=0.5,
                    help="Weight for macro_f1 in checkpoint score")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=2,
                    help="Run validation every N epochs (always runs on last epoch)")
    ap.add_argument("--use_cosine_head", action="store_true",
                    help="Use a learnable temperature cosine logic for classification head")
    ap.add_argument("--resume", type=str, default="",
                    help="Path to periodic checkpoint to resume from")

    ap.add_argument("--no_class_weight", action="store_true",
                help="Disable class weights (use uniform weights)")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--wandb_project", type=str, default="cellpt",
                    help="W&B project name. Set to '' to disable.")
    ap.add_argument("--wandb_entity", type=str, default=None)
    
    return ap

