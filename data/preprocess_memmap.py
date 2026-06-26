#!/usr/bin/env python3
"""
preprocess_memmap.py
--------------------
Resize cell images to 224×224 and store as contiguous uint8 memmap arrays.
Training then does np.memmap → tensor — zero PNG decode, zero disk IO after warm-up.

Output layout:
    cache_224/
        router_shards_x.dat         # uint8 [N, 2, 224, 224]
        router_shards_y.npy         # int64 [N]
        router_shards_tissue.npy    # object [N]
        router_shards_coarse_label.npy
        router_shards_fine_label.npy   (if 'label' col exists)
        router_shards_meta.json     # {n, shape, class_to_idx, ...}
        val_balanced_x.dat
        ...

Usage:
    python preprocess_memmap.py \\
        --parquet_dir <prepared/splits_dir> \\
        --class_map   <prepared/splits_dir/coarse_to_id.json> \\
        --out_dir     <cache_224> \\
        --splits      router_shards val_balanced test_shards \\
        --size        224 \\
        --workers     16
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode as IM


# ── Worker functions (run in subprocess) ─────────────────────────────────────

def resize_to_uint8(src_path: str, size: int) -> np.ndarray:
    img = Image.open(src_path).convert("L")
    w, h = img.size
    
    # --- 核心优化逻辑 ---
    if min(w, h) > size:
        # 下采样 (331 -> 224): 推荐使用 LANCZOS 或 BILINEAR 
        # PIL 的 LANCZOS 在处理 DAPI 这种高频信号时抗锯齿效果最好
        interp = IM.LANCZOS 
    else:
        # 上采样 (81 -> 224): BICUBIC 是平衡平滑度和锐度的最佳选择
        interp = IM.BICUBIC
        
    img = TF.resize(img, (size, size), interpolation=interp, antialias=True)
    return np.array(img, dtype=np.uint8)

def process_batch(
    x_path: str,
    shape: tuple,
    batch: list[tuple[int, str, str, int]],
) -> list[tuple[int, bool]]:
    """
    Process a batch of (global_index, path_a, path_b, size) tuples.
    Opens the memmap directly in the worker to avoid large IPC packet sizes.
    Returns list of (global_index, had_error).
    """
    x_mm = np.memmap(x_path, dtype=np.uint8, mode="r+", shape=shape)
    results = []
    for idx, pa, pb, size in batch:
        had_error = False
        try:
            arr_a = resize_to_uint8(pa, size)
            arr_b = resize_to_uint8(pb, size)
            x_mm[idx, 0] = arr_a
            x_mm[idx, 1] = arr_b
        except Exception as e:
            print(f"  ERROR idx={idx} pa={pa}: {e}", flush=True)
            zero = np.zeros((size, size), dtype=np.uint8)
            x_mm[idx, 0] = zero
            x_mm[idx, 1] = zero
            had_error = True
        results.append((idx, had_error))
    
    del x_mm
    return results


# ── Per-split processing ──────────────────────────────────────────────────────

def process_split(
    split_name: str,
    parquet_dir: Path,
    out_dir: Path,
    class_to_idx: dict,
    label_col: str,
    fine_col: str,
    tissue_col: str,
    img_col_a: str,
    img_col_b: str,
    size: int,
    workers: int,
    batch_size: int,
    force: bool,
):
    src = parquet_dir / split_name
    if not src.exists():
        print(f"SKIP {split_name}: {src} not found")
        return

    x_path = out_dir / f"{split_name}_x.dat"
    meta_path = out_dir / f"{split_name}_meta.json"

    if x_path.exists() and meta_path.exists():
        if not force:
            print(f"SKIP {split_name}: already exists (use --force to rerun)")
            return
        else:
            print(f"[{split_name}] --force is set, deleting old memmap file {x_path}")
            x_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    # ── 1. First pass: count rows & preallocate memmap ────────────────────
    if src.is_dir():
        shards = sorted(src.glob("*.parquet"))
        if not shards:
            print(f"SKIP {split_name}: no .parquet files in {src}")
            return
        print(f"\n[{split_name}] Analyzing {len(shards)} shards ...", flush=True)
        # Just scan to find total rows of valid classes
        total_n = 0
        allowed = set(class_to_idx.keys())
        for s in shards:
            # Only read the label column to save memory
            df_lbl = pd.read_parquet(s, columns=[label_col])
            total_n += int(df_lbl[label_col].isin(allowed).sum())
    else:
        print(f"\n[{split_name}] Analyzing {src} ...", flush=True)
        shards = [src]
        allowed = set(class_to_idx.keys())
        df_lbl = pd.read_parquet(src, columns=[label_col])
        total_n = int(df_lbl[label_col].isin(allowed).sum())

    print(f"  {total_n:,} rows after class filtering", flush=True)
    if total_n == 0:
        print("  No valid rows — skipping.")
        return

    # ── 2. Create memmap: [N, 2, H, W] uint8 ──────────────────────────────
    shape = (total_n, 2, size, size)
    gb = total_n * 2 * size * size / 1e9
    print(f"  Creating {x_path} | shape={shape} | {gb:.2f} GB", flush=True)
    x_mm = np.memmap(str(x_path), dtype=np.uint8, mode="w+", shape=shape)

    # Pre-allocate metadata arrays
    y_arr = np.empty(total_n, dtype=np.int64)
    tissue_arr = np.empty(total_n, dtype=object)
    coarse_label_arr = np.empty(total_n, dtype=object)
    
    # We don't know yet if fine_col exists in all shards, so we'll 
    # check during the first dataframe load.
    fine_label_arr = None

    t0 = time.time()
    done = 0
    errors = 0
    report_every_n = max(1, total_n // 20)   # ~5% increments
    last_reported = 0

    import concurrent.futures

    # Bounded queue logic
    in_flight_max = workers * 2

    # Global index tracker across all shards
    global_idx = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = set()

        # ── 3. Streaming read & execution ─────────────────────────────────────
        for shard_idx, shard_path in enumerate(shards):
            # Load ONE shard
            df = pd.read_parquet(shard_path)
            
            # Filter
            df = df[df[label_col].isin(allowed)].reset_index(drop=True)
            n_shard = len(df)
            if n_shard == 0:
                continue

            # Initialize fine_label_arr if needed and not yet done
            if fine_label_arr is None and fine_col in df.columns:
                fine_label_arr = np.empty(total_n, dtype=object)

            # Store metadata for this shard
            end_idx = global_idx + n_shard
            
            y_arr[global_idx:end_idx] = [class_to_idx[lbl] for lbl in df[label_col].values]
            tissue_arr[global_idx:end_idx] = df[tissue_col].values
            coarse_label_arr[global_idx:end_idx] = df[label_col].values
            if fine_label_arr is not None and fine_col in df.columns:
                fine_label_arr[global_idx:end_idx] = df[fine_col].values

            paths_a = df[img_col_a].values
            paths_b = df[img_col_b].values

            # Build batches for THIS shard
            for start in range(0, n_shard, batch_size):
                end = min(start + batch_size, n_shard)
                batch = [(global_idx + i, paths_a[i], paths_b[i], size) for i in range(start, end)]

                # Submit to bounded queue
                while len(futures) >= in_flight_max:
                    done_futures, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for f in done_futures:
                        for idx, had_error in f.result():
                            if had_error:
                                errors += 1
                            done += 1
                        
                        # Flush periodically based on completed images
                        if done - last_reported >= report_every_n:
                            x_mm.flush()
                            elapsed = time.time() - t0
                            rate = done / elapsed if elapsed > 0 else 1
                            eta = (total_n - done) / rate / 60
                            print(f"  {done:>9,}/{total_n:,}  ({100*done/total_n:5.1f}%)  "
                                  f"{rate:,.0f} img/s  ETA {eta:.1f} min", flush=True)
                            last_reported = done

                futures.add(pool.submit(process_batch, str(x_path), shape, batch))
            
            # Update global index for the next shard
            global_idx = end_idx
            
            # Free dataframe memory for this shard
            del df

        # Wait for remaining futures
        for f in concurrent.futures.as_completed(futures):
            for idx, had_error in f.result():
                if had_error:
                    errors += 1
                done += 1
            
            if done - last_reported >= report_every_n:
                x_mm.flush()
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 1
                eta = (total_n - done) / rate / 60
                print(f"  {done:>9,}/{total_n:,}  ({100*done/total_n:5.1f}%)  "
                      f"{rate:,.0f} img/s  ETA {eta:.1f} min", flush=True)
                last_reported = done

    x_mm.flush()
    del x_mm

    elapsed = time.time() - t0
    print(f"  Done: {elapsed/60:.1f} min | {total_n/elapsed:.0f} img/s | {errors} errors",
          flush=True)

    # ── Save metadata arrays ──────────────────────────────────────────────
    np.save(str(out_dir / f"{split_name}_y.npy"), y_arr)
    np.save(str(out_dir / f"{split_name}_tissue.npy"), tissue_arr)
    np.save(str(out_dir / f"{split_name}_coarse_label.npy"), coarse_label_arr)

    if fine_label_arr is None:
        pass
    else:
        np.save(str(out_dir / f"{split_name}_fine_label.npy"), fine_label_arr)

    # ── Save meta JSON ────────────────────────────────────────────────────
    meta = {
        "n": total_n,
        "shape": list(shape),
        "dtype": "uint8",
        "size": size,
        "label_col": label_col,
        "fine_col": fine_col,
        "tissue_col": tissue_col,
        "img_col_a": img_col_a,
        "img_col_b": img_col_b,
        "class_to_idx": class_to_idx,
        "errors": errors,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    x_gb = x_path.stat().st_size / 1e9
    print(f"  ✓ {split_name}: {x_gb:.2f} GB | {total_n:,} samples | {errors} errors")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess tiles → memmap uint8 arrays")
    parser.add_argument("--parquet_dir", required=True,
                        help="Root dir containing split sub-dirs with .parquet files")
    parser.add_argument("--class_map", required=True,
                        help="JSON: class_name → int index")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for .dat / .npy / .json files")
    parser.add_argument("--splits", nargs="+",
                        default=["router_shards", "val_balanced", "test_shards"])
    parser.add_argument("--label_col", default="coarse_label",
                        help="Column used for class_to_idx lookup (default: coarse_label)")
    parser.add_argument("--fine_col", default="label",
                        help="Fine-grained label column to preserve (default: label)")
    parser.add_argument("--tissue_col", default="tissue")
    parser.add_argument("--img_col_a", default="img_path_2p5x")
    parser.add_argument("--img_col_b", default="img_path_10x")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Images per worker batch (keep small to limit IPC payload)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if output already exists")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.class_map) as f:
        class_to_idx = json.load(f)

    print(f"\n{'='*64}")
    print(f"  preprocess_memmap.py")
    print(f"  parquet_dir : {args.parquet_dir}")
    print(f"  out_dir     : {args.out_dir}")
    print(f"  splits      : {args.splits}")
    print(f"  class_map   : {len(class_to_idx)} classes")
    print(f"  size        : {args.size}×{args.size}")
    print(f"  workers     : {args.workers}  batch_size: {args.batch_size}")
    print(f"{'='*64}\n")

    t_all = time.time()
    for split in args.splits:
        process_split(
            split_name=split,
            parquet_dir=Path(args.parquet_dir),
            out_dir=out_dir,
            class_to_idx=class_to_idx,
            label_col=args.label_col,
            fine_col=args.fine_col,
            tissue_col=args.tissue_col,
            img_col_a=args.img_col_a,
            img_col_b=args.img_col_b,
            size=args.size,
            workers=args.workers,
            batch_size=args.batch_size,
            force=args.force,
        )

    print(f"\nAll splits done in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
