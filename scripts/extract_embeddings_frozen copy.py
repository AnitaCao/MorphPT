import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from data.dataset import CellParquetMultiView, AugCfg


"""
python extract_embeddings_frozen.py \
  --parquet prepared/small_balanced.parquet \
  --class_map prepared/class_to_idx.json \
  --model vit_base_patch16_dinov3.lvd1689m \
  --out prepared/embeddings_small_balanced_frozen.parquet

"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def forward_features(backbone, x):
    feats = backbone.forward_features(x)
    if feats.ndim == 3:
        emb = feats[:, 0]              # CLS
    elif feats.ndim == 4:
        emb = feats.mean(dim=(2, 3))   # GAP
    else:
        emb = feats
    return emb

@torch.no_grad()
def extract_frozen_embeddings(parquet_path, class_to_idx, model_name, img_size=224, batch_size=64, num_workers=8):
    ds = CellParquetMultiView(
        parquet_path=parquet_path,
        class_to_idx=class_to_idx,
        size=img_size,
        aug=AugCfg(enable=False),
        return_meta=True,   # 返回 meta: tissue,x,y,cell_id,path_2p5x,path_10x
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size).to(DEVICE)
    backbone.eval()

    emb2_list, emb10_list, embf_list = [], [], []
    y_list, tissue_list, cellid_list, x_list, ycoord_list = [], [], [], [], []

    use_cuda = (DEVICE == "cuda")
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    import time
    pbar = tqdm(loader, desc="extract frozen")
    t0 = time.time()
    for x, y, meta in pbar:
        t_data = time.time() - t0
        # x: [B,2,3,H,W]
        x = x.to(DEVICE, non_blocking=True)
        y_list.append(y.numpy())

        # meta is a dict of lists/tensors after collation
        tissue_list.extend(meta["tissue"])
        cellid_list.extend(meta["cell_id"])
        x_list.extend(meta["x"].tolist())
        ycoord_list.extend(meta["y"].tolist())

        B, S, C, H, W = x.shape
        xs = x.reshape(B * S, C, H, W)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
            emb_all = forward_features(backbone, xs)      # [B*S, D]

        P = emb_all.reshape(B, S, -1)                     # [B,2,D]
        e2 = P[:, 0].float()
        e10 = P[:, 1].float()

        # normalize per view then average then normalize
        e2n = F.normalize(e2, dim=1)
        e10n = F.normalize(e10, dim=1)
        ef = F.normalize(0.5 * (e2n + e10n), dim=1)

        emb2_list.append(e2n.cpu().numpy().astype(np.float32))
        emb10_list.append(e10n.cpu().numpy().astype(np.float32))
        embf_list.append(ef.cpu().numpy().astype(np.float32))

        if use_cuda:
            torch.cuda.synchronize()
        t_batch = time.time() - t0
        t_compute = t_batch - t_data
        pbar.set_postfix(data=f"{t_data:.4f}", compute=f"{t_compute:.4f}")
        t0 = time.time()

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
    # 把 embedding 存成 numpy array 列（parquet 支持 list 类型）
    out["emb_2p5x"] = list(emb2)
    out["emb_10x"] = list(emb10)
    out["emb_fused"] = list(embf)
    return out

def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--class_map", required=True)
    ap.add_argument("--model", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.class_map, "r") as f:
        class_to_idx = json.load(f)

    df = extract_frozen_embeddings(
        parquet_path=args.parquet,
        class_to_idx=class_to_idx,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    df.to_parquet(args.out, index=False)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
