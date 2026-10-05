"""Colab check for Stage 2 (run on REAL data; no GPU needed, no model needed).

Prints:
  1. split sizes + first/last 5 MA file names of train and val from baselines/common/data.py AND from
     the original trainer_wandb_unclipped.py code path, with MATCH / MISMATCH lines;
  2. validation metrics (masked SSIM, masked HU RMSE, PSNR) computed with the ORIGINAL metric code and
     with baselines/common/eval_utils.py, with MATCH / MISMATCH lines. Since training is from scratch
     (no checkpoint yet), the LI image and the MA image are used as stand-in "predictions".

Usage (from the repo root):
  python colab/verify_common.py --ma_dir /content/data/MA --gt_dir /content/data/GT \
      --li_dir /content/data/LI --mask_dir /content/data/mask [--max_val_images 0]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.data import build_datasets, unpack_batch, DATASET_KWARGS  # noqa: E402
from baselines.common.eval_utils import evaluate_predictions, summarize_metrics  # noqa: E402
from baselines.common.tests.original_metric_code import original_val_metrics  # noqa: E402
from baselines.common.tests.original_split_code import original_trainer_datasets  # noqa: E402

LINE = "=" * 78


def names(ds):
    return [p[0] for p in ds.pairs]


def show_split(tag, train_ds, val_ds):
    tr, va = names(train_ds), names(val_ds)
    print(f"  [{tag}] train={len(tr)} val={len(va)}")
    print(f"    train first 5: {tr[:5]}")
    print(f"    train last 5 : {tr[-5:]}")
    print(f"    val   first 5: {va[:5]}")
    print(f"    val   last 5 : {va[-5:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ma_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--li_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--max_val_images", type=int, default=0, help="0 = all validation images")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    results = []
    print(LINE)
    print("VERIFY_COMMON REPORT")
    print(f"torch {torch.__version__} | numpy {np.__version__} | dataset kwargs: {DATASET_KWARGS}")
    print(LINE)

    # ---------------------------------------------------------------- 1. split
    print("1) TRAIN/VAL SPLIT")
    tr_new, va_new = build_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir)
    tr_old, va_old = original_trainer_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir,
                                               uses_li=True)
    show_split("common/data.py", tr_new, va_new)
    show_split("original trainer (LI + mask dirs)", tr_old, va_old)
    split_match = tr_new.pairs == tr_old.pairs and va_new.pairs == va_old.pairs
    results.append(("split common/data.py vs original trainer (LI+mask)", split_match))
    print(f"  SPLIT (full file lists, same order): {'MATCH' if split_match else 'MISMATCH'}")

    # Reference models that did NOT load LI (MA-only input) intersect keys without the LI dir.
    tr_ma, va_ma = original_trainer_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir,
                                             uses_li=False)
    ma_only_match = names(tr_ma) == names(tr_new) and names(va_ma) == names(va_new)
    print(f"  (info) split of MA-only reference runs (no LI dir) vs common/data.py: "
          f"{'MATCH' if ma_only_match else 'MISMATCH'} (train={len(tr_ma)} val={len(va_ma)})")
    overlap = set(names(tr_new)) & set(names(va_new))
    results.append(("train/val disjoint", not overlap))
    print(f"  train/val overlap: {len(overlap)} -> {'PASS' if not overlap else 'FAIL'}")

    # ---------------------------------------------------------------- 2. metrics
    print(LINE)
    print("2) VALIDATION METRICS: original code vs eval_utils.py (stand-in predictions: LI and MA)")
    loader = DataLoader(va_new, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    hu_min, hu_max = DATASET_KWARGS["hu_min"], DATASET_KWARGS["hu_max"]
    per = {src: {"new": {"ssim": [], "rmse": [], "psnr": []}, "old": {"ssim": [], "rmse": [], "psnr": []}}
           for src in ["LI", "MA"]}
    n_done, t0 = 0, time.time()
    for batch in loader:
        b = unpack_batch(batch, has_mask=True, has_li=True)
        for src, pred in [("LI", b["li"]), ("MA", b["ma"])]:
            new = evaluate_predictions(pred, b["gt"], b["mask"], hu_min, hu_max)
            o_ssim, o_psnr, o_rmse, _ = original_val_metrics(pred, b["gt"], b["mask"], hu_min, hu_max)
            for k, v in new.items():
                per[src]["new"][k] += v
            per[src]["old"]["ssim"] += o_ssim
            per[src]["old"]["rmse"] += o_rmse
            per[src]["old"]["psnr"] += o_psnr
        n_done += b["gt"].shape[0]
        if args.max_val_images and n_done >= args.max_val_images:
            break
    print(f"  evaluated {n_done} val images in {time.time() - t0:.1f}s")
    for src in ["LI", "MA"]:
        new, old = per[src]["new"], per[src]["old"]
        max_diff = max(max(abs(a - c) for a, c in zip(new[k], old[k])) for k in new)
        exact = all(new[k] == old[k] for k in new)
        s_new, s_old = summarize_metrics(new), summarize_metrics(old)
        print(f"  [{src} as prediction]")
        for k in ["ssim", "rmse", "psnr"]:
            print(f"    {k:5s} original: {s_old[k]:.6f} +/- {s_old[k + '_std']:.6f} | "
                  f"eval_utils: {s_new[k]:.6f} +/- {s_new[k + '_std']:.6f}")
        print(f"    per-image max |diff| = {max_diff:.3e} -> {'MATCH' if exact else 'MISMATCH'}")
        results.append((f"metrics {src}", exact))

    print(LINE)
    n_fail = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"OVERALL: {'ALL MATCH' if n_fail == 0 else f'{n_fail} MISMATCH/FAIL'}")
    print(LINE)


if __name__ == "__main__":
    main()
