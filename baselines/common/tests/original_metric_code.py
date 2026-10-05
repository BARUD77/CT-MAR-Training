"""FROZEN reference copy of the ORIGINAL validation-metric code, used only to verify that
baselines/common/eval_utils.py reproduces it exactly.

Copied verbatim from trainer_wandb_unclipped.py at git commit cd22fab (validation loop,
lines 803-808 and 921-983 before the eval_utils refactor). Only the enclosing function and the
`args.hu_min/args.hu_max` -> `hu_min/hu_max` argument plumbing were added. Do not edit.
"""
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metrics import compute_SSIM, compute_masked_SSIM, compute_masked_RMSE_HU  # noqa: E402


def original_val_metrics(pred, y_batch, mask_batch, hu_min, hu_max):
    """Returns (ssim_list, psnr_list, rmse_list, summary_dict) exactly as the original loop."""
    from types import SimpleNamespace
    args = SimpleNamespace(hu_min=hu_min, hu_max=hu_max)

    ssim_list = []
    psnr_list = []
    rmse_list = []
    # normalized metal fill (100 HU) used to fill the dilated metal region for SSIM
    metal_fill_norm = (100.0 - float(args.hu_min)) / (float(args.hu_max) - float(args.hu_min))

    # Cast back to fp32 so downstream numpy-based metrics work with bf16/fp16 autocast.
    pred = pred.float()

    # Normalized metrics are computed on clamped [0,1] tensors.
    pred_eval = torch.clamp(pred, 0, 1)
    gt_eval   = torch.clamp(y_batch, 0, 1)

    # compute per-image metrics
    for i in range(pred_eval.size(0)):
        img_pred = pred_eval[i, 0]
        img_gt = gt_eval[i, 0]

        # Extract per-image metal mask first (needed by all metrics below).
        metalmask_arg = None
        non_metal_mask = None
        if mask_batch is not None:
            m = mask_batch[i]
            if m.dim() == 3 and m.size(0) == 1:
                m = m.squeeze(0)
            if m.dim() == 3 and m.size(0) == 1:
                m = m.squeeze(0)
            metalmask_arg = m
            non_metal_mask = (m == 0).float()

        # ---- SSIM: dilate metal by 2 px, fill dilated region to 100 HU,
        #      average full SSIM map (data_range=1) over the non-dilated pixels.
        if metalmask_arg is not None:
            _, ssim_val = compute_masked_SSIM(
                img_pred,
                img_gt,
                data_range=1.0,
                mask=None,
                metalmask=metalmask_arg,
                metal_fill=metal_fill_norm,
                dilate_iters=2,
                hu_min=args.hu_min,
                hu_max=args.hu_max,
            )
        else:
            ssim_val = compute_SSIM(img_pred, img_gt, data_range=1.0)
        ssim_list.append(float(ssim_val))

        # ---- RMSE (HU) and PSNR: denormalize to HU, dilate metal by 1 px and
        #      exclude it; RMSE in HU, PSNR = 20*log10(MAX/RMSE) with MAX=HU range.
        rmse_hu = compute_masked_RMSE_HU(
            img_pred,
            img_gt,
            metalmask=metalmask_arg,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            dilate_iters=1,
        )
        rmse_list.append(float(rmse_hu))
        psnr_max = float(args.hu_max) - float(args.hu_min)  # 4096 HU range
        rmse_hu_safe = rmse_hu if rmse_hu > 0 else 1e-10
        psnr_list.append(float(20.0 * np.log10(psnr_max / rmse_hu_safe)))

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
    std_psnr = float(np.std(psnr_list, ddof=1)) if len(psnr_list) > 1 else 0.0
    std_ssim = float(np.std(ssim_list, ddof=1)) if len(ssim_list) > 1 else 0.0
    std_rmse = float(np.std(rmse_list, ddof=1)) if len(rmse_list) > 1 else 0.0
    summary = {"psnr": avg_psnr, "psnr_std": std_psnr, "ssim": avg_ssim, "ssim_std": std_ssim,
               "rmse": avg_rmse, "rmse_std": std_rmse}
    return ssim_list, psnr_list, rmse_list, summary
