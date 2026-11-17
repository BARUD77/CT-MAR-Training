"""Collect metal-centred training regions for the near-metal refiner (run locally, D: mounted).

For every slice: the frozen global Swin's prediction ("base"), plus MA, LI, GT, metal mask and distance-to-metal,
cropped to the bounding box of the gate region (d <= 20 mm) plus a 64-px margin (at least 128 x 128), saved as one
.npz per slice (float16). The refiner samples its 128 x 128 training crops from these regions.

Modes (which network produces "base"):
  oof       out-of-fold (cross-fitting): training slices of half A are predicted by the model trained on half B and
            vice versa (--ckpt_A = model trained on half A, --ckpt_B = model trained on half B;
            halves from splits/grouped_split_v1_cfA.csv). This is the main training set.
  insample  every training slice predicted by the full-data model (--ckpt_full) that was trained on it
            (ablation C1: the refiner without cross-fitting).
  val       validation slices predicted by the full-data model (--ckpt_full), for monitoring only.

Data are read with baselines/common/data.build_datasets (grouped split v1); the test set is not used.

Usage:
  python refiner/collect_crops.py --mode oof --ckpt_A <crossfit_A last.pt> --ckpt_B <crossfit_B last.pt>
  python refiner/collect_crops.py --mode val --ckpt_full <best_loss_base_e96_s1.pt>
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from baselines.common.data import build_datasets, unpack_batch  # noqa: E402
from models.near_metal_refiner import PX_MM, distance_mm  # noqa: E402
from models.swin_unet_dual.load import load_swin  # noqa: E402

CFG = os.path.join(REPO, "models", "swin_unet_v2", "config.yaml")
R_ZERO, MARGIN, MIN_SIZE = 20.0, 64, 128


def region_box(d, shape):
    ys, xs = np.nonzero(d <= R_ZERO)
    y0, y1 = ys.min() - MARGIN, ys.max() + MARGIN + 1
    x0, x1 = xs.min() - MARGIN, xs.max() + MARGIN + 1
    H, W = shape

    def fit(a, b, n):
        if b - a < MIN_SIZE:
            c = (a + b) // 2
            a, b = c - MIN_SIZE // 2, c + MIN_SIZE // 2
        a, b = max(a, 0), min(b, n)
        if b - a < MIN_SIZE:                     # at the image border: shift inward
            a, b = (0, MIN_SIZE) if a == 0 else (n - MIN_SIZE, n)
        return int(a), int(b)
    (y0, y1), (x0, x1) = fit(y0, y1, H), fit(x0, x1, W)
    return y0, y1, x0, x1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oof", "insample", "val"], required=True)
    ap.add_argument("--ckpt_A")
    ap.add_argument("--ckpt_B")
    ap.add_argument("--ckpt_full")
    ap.add_argument("--ma_dir", default=r"D:\data_npy\input\MA_image")
    ap.add_argument("--gt_dir", default=r"D:\data_npy\GT")
    ap.add_argument("--li_dir", default=r"D:\data_npy\input\LI")
    ap.add_argument("--mask_dir", default=r"D:\data_npy\input\metal_mask")
    ap.add_argument("--split_file", default=os.path.join(REPO, "splits", "grouped_split_v1.csv"))
    ap.add_argument("--fold_file", default=os.path.join(REPO, "splits", "grouped_split_v1_cfA.csv"))
    ap.add_argument("--out_root", default=r"D:\data_npy\refiner_crops")
    ap.add_argument("--max_slices", type=int, default=0, help="for testing")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train_ds, val_ds = build_datasets(args.ma_dir, args.gt_dir, li_dir=args.li_dir, mask_dir=args.mask_dir,
                                      split_file=args.split_file)
    ds = val_ds if args.mode == "val" else train_ds
    rx = re.compile(r"_(body|head)_metalart_img(\d+)_", re.I)
    keys = [(m.group(1).lower(), int(m.group(2))) for m in (rx.search(os.path.basename(p[0])) for p in ds.pairs)]

    if args.mode == "oof":
        half = pd.read_csv(args.fold_file).set_index(["region", "id"])["half"]
        models = {"A": load_swin("swinunet_v2", CFG, args.ckpt_A, device=dev)[0],
                  "B": load_swin("swinunet_v2", CFG, args.ckpt_B, device=dev)[0]}
        # a slice of half A is predicted by the model trained on half B, and vice versa
        pick = [("B" if half[k] == "A" else "A") for k in keys]
        if not set(half[k] for k in keys) <= {"A", "B"}:
            raise ValueError("fold file does not assign every training slice to half A or B")
    else:
        models = {"full": load_swin("swinunet_v2", CFG, args.ckpt_full, device=dev)[0]}
        pick = ["full"] * len(keys)

    out_dir = os.path.join(args.out_root, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    n = len(keys) if not args.max_slices else min(args.max_slices, len(keys))
    sel = np.arange(len(keys)) if n == len(keys) else np.linspace(0, len(keys) - 1, n).astype(int)
    print(f"mode {args.mode}: {n} slices -> {out_dir} | models {sorted(models)} | {dev}", flush=True)
    t0, sizes, skipped = time.time(), [], 0
    with torch.no_grad():
        for c, i in enumerate(sel):
            region, idx = keys[i]
            it = unpack_batch(ds[int(i)], has_mask=True, has_li=True)
            metal = it["mask"][0].numpy() > 0.5
            if not metal.any():
                skipped += 1
                continue
            ma, li = it["ma"][None].to(dev), it["li"][None].to(dev)
            base = models[pick[i]](torch.cat([ma, li], 1)).float()[0, 0].cpu().numpy()
            d = distance_mm(metal, PX_MM[region])
            y0, y1, x0, x1 = region_box(d, metal.shape)
            crop = lambda a: np.ascontiguousarray(a[y0:y1, x0:x1])
            np.savez(os.path.join(out_dir, f"{region}_{idx}.npz"),
                     ma=crop(it["ma"][0].numpy()).astype(np.float16), li=crop(it["li"][0].numpy()).astype(np.float16),
                     base=crop(base).astype(np.float16), gt=crop(it["gt"][0].numpy()).astype(np.float16),
                     metal=crop(metal).astype(np.uint8), dist=crop(np.minimum(d, 1000.0)).astype(np.float16),
                     box=np.array([y0, y1, x0, x1]), region=region, id=idx, model=pick[i])
            sizes.append((y1 - y0) * (x1 - x0))
            if c % 500 == 0:
                print(f"  {c + 1}/{n} ({time.time() - t0:.0f} s)", flush=True)
    print(f"done: {len(sizes)} regions saved, {skipped} slices without metal skipped, median region "
          f"{np.sqrt(np.median(sizes)):.0f} px side ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
