"""
spatial_maps.py
───────────────
Visualize true vs predicted gene expression on slide spatial coordinates.

Fixes vs previous version:
  1. Inference caching — saves predictions_test_job{N}.npz on first run,
     reloads on subsequent runs. Saves significant GPU time.
  2. select_genes() now takes per_gene_r as input and selects one strong,
     one median, and one hard gene per coverage tier — not just highest coverage.
  3. Axis labels say "expression value" not "log1p" — the targets are
     already transformed (continuous floats, not raw counts).
  4. safe_pearsonr() handles constant arrays gracefully.
  5. gate_dropout passed correctly when rebuilding model from checkpoint.

Usage:
  python spatial_maps.py --job 10
  python spatial_maps.py --job 10 --genes PHGR1,VIM,COL3A1,CD74,VEGFA,IGKC
  python spatial_maps.py --job 10 --force_rerun   # ignore cached predictions
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT   = Path("/hpc/group/jilab/tc459/MorphPT")
CACHE     = PROJECT / "cache_visium"
EXPR_ROOT = Path("/hpc/group/jilab/boxuan/visiumHD/human_crc")
MORPH_DIR = Path("/hpc/group/jilab/hz/MorphPT/data/visiumHD/human_crc")
OUT_DIR   = PROJECT / "analysis/spatial_maps"

JOBS = {
    1:  "visium_dinov3_25x",
    2:  "visium_morphpt_25x",
    3:  "visium_morphpt_lora_25x",
    4:  "visium_morphpt_10x",
    5:  "visium_dinov3_10x",
    6:  "visium_morphpt_gate",
    7:  "visium_morphpt_25x_mlp",
    8:  "visium_morphpt_10x_mlp",
    9:  "visium_morphpt_gate_mlp",
    10: "visium_morphpt_lora_10x",
    11: "visium_morphpt_lora_gate",
    12: "visium_morphpt_lora_10x_mlp",
    13: "visium_morphpt_lora_gate_mlp",
    14: "visium_dinov3_lora_25x",
    15: "visium_dinov3_lora_10x",
}

STYLE = {
    "figure.dpi": 150,
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)


# ── Safe Pearson r ─────────────────────────────────────────────────────────
def safe_pearsonr(a, b):
    """
    Pearson r between two 1D arrays.
    Returns 0.0 if either side is constant (std < 1e-8),
    avoiding division by zero and scipy warnings for sparse genes.
    """
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    from scipy.stats import pearsonr
    r, _ = pearsonr(a, b)
    return float(r)


# ── Inference with caching ─────────────────────────────────────────────────
def get_predictions(job_id, split="test", force_rerun=False):
    """
    Load or compute predictions for the given job and split.
    On first run: runs inference, saves to predictions_{split}_job{N}.npz.
    On subsequent runs: loads from cache (fast).

    Returns:
        cell_ids : list of str
        y_true   : (N, 1000) float32  — denormalized expression values
        y_pred   : (N, 1000) float32  — denormalized predicted values
    """
    import sys
    sys.path.insert(0, str(PROJECT))

    cache_path = PROJECT / "experiments" / JOBS[job_id] / \
                 f"predictions_{split}.npz"

    if cache_path.exists() and not force_rerun:
        print(f"  Loading cached predictions: {cache_path}")
        npz      = np.load(str(cache_path), allow_pickle=True)
        cell_ids = npz["cell_ids"].tolist()
        y_true   = npz["y_true"]
        y_pred   = npz["y_pred"]
        print(f"  Loaded: {len(cell_ids):,} cells")
        return cell_ids, y_true, y_pred

    # ── Run inference ──────────────────────────────────────────────────────
    from data.visium_dataset import VisiumHDPredictionDataset
    from models.visium_regression import VisiumRegressor

    exp_dir  = PROJECT / "experiments" / JOBS[job_id]
    ckpt     = torch.load(exp_dir / "best.pth", map_location="cpu",
                          weights_only=False)
    args     = ckpt["args"]
    scales   = [s.strip() for s in args["scales"].split(",")]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Running inference: Job {job_id} ({JOBS[job_id]})")
    print(f"  Scales={scales}  Fuse={args['fuse']}  Device={device}")

    ds = VisiumHDPredictionDataset(
        cache_dir = CACHE,
        split     = split,
        scales    = scales,
        fuse      = args["fuse"],
        augment   = False,
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = VisiumRegressor(
        model_name      = args["model"],
        img_size        = args["img_size"],
        out_dim         = 1000,
        pretrained      = False,
        fuse            = args["fuse"],
        freeze_backbone = True,
        lora_blocks     = args["lora_blocks"],
        lora_rank       = args["lora_rank"],
        lora_alpha      = args["lora_alpha"],
        lora_dropout    = args["lora_dropout"],
        lora_targets    = args["lora_targets"],
        gate_dropout    = args.get("gate_dropout", 0.1),  # match training config
        ckpt_path       = None,
        head_type       = args["head_type"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    gene_mean = ds.gene_mean.numpy()
    gene_std  = ds.gene_std.numpy()

    all_pred, all_true, all_ids = [], [], []
    with torch.no_grad():
        for imgs, expr, meta in loader:
            imgs = imgs.to(device)
            pred = model(imgs).cpu().numpy()
            expr = expr.numpy()
            # Denormalize to original transformed expression space
            pred_denorm = pred * gene_std + gene_mean
            true_denorm = expr * gene_std + gene_mean
            all_pred.append(pred_denorm)
            all_true.append(true_denorm)
            all_ids.extend(meta["cell_id"])

    y_pred   = np.concatenate(all_pred, axis=0).astype(np.float32)
    y_true   = np.concatenate(all_true, axis=0).astype(np.float32)
    cell_ids = all_ids

    # Save for future runs
    np.savez_compressed(str(cache_path),
                        cell_ids=np.array(cell_ids),
                        y_true=y_true,
                        y_pred=y_pred)
    print(f"  Saved predictions → {cache_path}")

    return cell_ids, y_true, y_pred


# ── Gene selection ────────────────────────────────────────────────────────
def select_genes(y_true, genes, per_gene_r, n_per_tier=1):
    """
    Select representative genes across coverage × performance tiers.
    For each coverage tier (high/medium/low), picks:
      - one strong gene  (top Pearson r within tier)
      - one median gene  (median Pearson r within tier)
      - one hard gene    (bottom Pearson r within tier, r > 0)

    Args:
        y_true      : (N_test, G) expression values
        genes       : list of G gene names
        per_gene_r  : (G,) per-gene Pearson r on test set
        n_per_tier  : how many genes to pick per performance level per tier
    """
    coverage = (y_true > 0).mean(axis=0)   # (G,)

    tiers = [
        ("high",   coverage > 0.20),
        ("medium", (coverage >= 0.05) & (coverage <= 0.20)),
        ("low",    coverage < 0.05),
    ]
    levels = ["strong", "median", "hard"]
    selected = []

    for tier_name, tier_mask in tiers:
        idx = np.where(tier_mask)[0]
        if len(idx) < 3:
            continue
        r_tier = per_gene_r[idx]

        # Strong: top Pearson within tier
        strong = idx[np.argsort(r_tier)[::-1][:n_per_tier]]
        # Median: middle Pearson
        mid    = len(idx) // 2
        median = idx[np.argsort(r_tier)[mid:mid+n_per_tier]]
        # Hard: lowest Pearson but still r > 0 (not completely unpredictable)
        pos_mask = r_tier > 0
        if pos_mask.sum() > 0:
            hard = idx[np.where(pos_mask)[0][np.argsort(r_tier[pos_mask])[:n_per_tier]]]
        else:
            hard = idx[np.argsort(r_tier)[:n_per_tier]]

        for gene_idx in strong:
            selected.append((int(gene_idx), genes[gene_idx],
                             tier_name, "strong"))
        for gene_idx in median:
            selected.append((int(gene_idx), genes[gene_idx],
                             tier_name, "median"))
        for gene_idx in hard:
            selected.append((int(gene_idx), genes[gene_idx],
                             tier_name, "hard"))

    return selected


# ── Spatial panel for one gene ─────────────────────────────────────────────
def plot_gene_panel(axes, x, y, true_vals, pred_vals, gene_name, r, coverage):
    """
    4-panel figure for one gene:
      axes[0]: true spatial map
      axes[1]: predicted spatial map
      axes[2]: true vs predicted scatter
      axes[3]: absolute error spatial map
    """
    vmin = min(true_vals.min(), pred_vals.min())
    vmax = max(true_vals.max(), pred_vals.max())
    s    = 2

    sc1 = axes[0].scatter(x, y, c=true_vals, cmap="viridis",
                          vmin=vmin, vmax=vmax, s=s, rasterized=True)
    axes[0].set_title(f"{gene_name} — True\n(coverage={coverage:.1%})", fontsize=8)
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    plt.colorbar(sc1, ax=axes[0], shrink=0.6, label="Expression value")

    sc2 = axes[1].scatter(x, y, c=pred_vals, cmap="viridis",
                          vmin=vmin, vmax=vmax, s=s, rasterized=True)
    axes[1].set_title(f"{gene_name} — Predicted (r={r:.3f})", fontsize=8)
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    plt.colorbar(sc2, ax=axes[1], shrink=0.6, label="Expression value")

    # Scatter: true vs predicted
    axes[2].scatter(true_vals, pred_vals, alpha=0.3, s=4, color="#4C72B0",
                    rasterized=True)
    lim = [vmin, vmax]
    axes[2].plot(lim, lim, "r--", lw=0.8)
    axes[2].set_xlabel("True expression value")
    axes[2].set_ylabel("Predicted expression value")
    axes[2].set_title(f"{gene_name} — Scatter (r={r:.3f})", fontsize=8)

    # Error map
    err = np.abs(pred_vals - true_vals)
    sc3 = axes[3].scatter(x, y, c=err, cmap="Reds", s=s, rasterized=True)
    axes[3].set_title(f"{gene_name} — |Error|", fontsize=8)
    axes[3].set_aspect("equal")
    axes[3].axis("off")
    plt.colorbar(sc3, ax=axes[3], shrink=0.6, label="|Error|")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job",          type=int,  required=True)
    ap.add_argument("--genes",        type=str,  default=None,
                    help="Comma-separated gene names (default: auto-select)")
    ap.add_argument("--split",        type=str,  default="test")
    ap.add_argument("--n_per_tier",   type=int,  default=1,
                    help="Genes per (tier × performance level) in auto-selection")
    ap.add_argument("--force_rerun",  action="store_true",
                    help="Ignore cached predictions and rerun inference")
    ap.add_argument("--save_predictions", action="store_true",
                    help="Save predictions_test.npz (also done automatically)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    genes = (EXPR_ROOT / "expr/genes.txt").read_text().splitlines()

    # Load or compute predictions
    print(f"Getting predictions for Job {args.job}...")
    cell_ids, y_true, y_pred = get_predictions(
        args.job, split=args.split, force_rerun=args.force_rerun
    )

    # Per-gene Pearson on this split
    print("Computing per-gene Pearson r...")
    per_gene_r = np.array([safe_pearsonr(y_true[:, g], y_pred[:, g])
                           for g in range(y_true.shape[1])])

    # Load spatial coordinates
    print("Loading spatial coordinates...")
    spatial    = pd.read_csv(MORPH_DIR / "spatial.csv")
    id_to_xy   = dict(zip(spatial["cell_id"].astype(str),
                          zip(spatial["x_centroid"], spatial["y_centroid"])))
    coords     = np.array([id_to_xy.get(cid, (np.nan, np.nan))
                           for cid in cell_ids])
    x, y_coord = coords[:, 0], -coords[:, 1]   # flip y for anatomical orientation

    # Select genes
    if args.genes:
        gene_list = args.genes.split(",")
        selected  = [(genes.index(g), g, "manual", "manual")
                     for g in gene_list if g in genes]
        missing   = [g for g in gene_list if g not in genes]
        if missing:
            print(f"  Genes not found: {missing}")
    else:
        selected = select_genes(y_true, genes, per_gene_r,
                                n_per_tier=args.n_per_tier)

    print(f"\nPlotting {len(selected)} genes...")

    # Per-gene figures (4-panel each)
    for gene_idx, gene_name, tier, level in selected:
        true_vals = y_true[:, gene_idx]
        pred_vals = y_pred[:, gene_idx]
        r         = per_gene_r[gene_idx]
        cov       = float((y_true[:, gene_idx] > 0).mean())

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        plot_gene_panel(axes, x, y_coord, true_vals, pred_vals,
                        gene_name, r, cov)
        fig.suptitle(
            f"Job {args.job} | {JOBS[args.job]} | "
            f"{gene_name} (tier={tier}, level={level}, r={r:.3f})",
            fontweight="bold", fontsize=9
        )
        fig.tight_layout()
        out = OUT_DIR / f"job{args.job}_{gene_name}_{tier}_{level}.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  {gene_name:<16} tier={tier:<8} level={level:<8} "
              f"r={r:.3f}  cov={cov:.1%}  → {out.name}")

    # Summary: all selected genes, 2 columns (true | pred)
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (gene_idx, gene_name, tier, level) in enumerate(selected):
        true_vals = y_true[:, gene_idx]
        pred_vals = y_pred[:, gene_idx]
        r         = per_gene_r[gene_idx]
        vmin = min(true_vals.min(), pred_vals.min())
        vmax = max(true_vals.max(), pred_vals.max())

        axes[row, 0].scatter(x, y_coord, c=true_vals, cmap="viridis",
                             vmin=vmin, vmax=vmax, s=1.5, rasterized=True)
        axes[row, 0].set_title(f"{gene_name} — True ({tier}, {level})",
                               fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 0].set_aspect("equal")

        axes[row, 1].scatter(x, y_coord, c=pred_vals, cmap="viridis",
                             vmin=vmin, vmax=vmax, s=1.5, rasterized=True)
        axes[row, 1].set_title(f"{gene_name} — Pred r={r:.3f}", fontsize=8)
        axes[row, 1].axis("off")
        axes[row, 1].set_aspect("equal")

    fig.suptitle(f"Spatial maps: Job {args.job} | {JOBS[args.job]}",
                 fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"job{args.job}_summary.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    main()