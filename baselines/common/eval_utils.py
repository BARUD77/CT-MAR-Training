"""Shared validation metrics for ALL baselines.

Moved verbatim (numerically) from the validation loop of trainer_wandb_unclipped.py:
  - masked SSIM: metal mask dilated 2 px, dilated region filled with 100 HU (normalized),
    data_range=1, SSIM map averaged over pixels outside the dilated metal;
    plain SSIM if no metal mask is available;
  - masked RMSE in HU: metal mask dilated 1 px and excluded;
  - PSNR = 20*log10((hu_max - hu_min) / RMSE_HU);
  - mean and sample std (ddof=1) over images.

Inputs are in the dataset's normalized [0, 1] space (output_space="norm").
"""
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metrics import compute_SSIM, compute_masked_SSIM, compute_masked_RMSE_HU  # noqa: E402


def evaluate_predictions(pred, gt, metal_mask, hu_min, hu_max):
    """Per-image validation metrics.

    Args:
        pred: (B,1,H,W) prediction in normalized [0,1] space (clamped to [0,1] here).
        gt: (B,1,H,W) target in normalized [0,1] space (clamped to [0,1] here).
        metal_mask: (B,1,H,W) binary METAL mask (1 = metal), or None.
        hu_min, hu_max: HU window used for normalization.

    Returns:
        dict with lists of python floats: {"ssim": [...], "rmse": [...], "psnr": [...]}.
    """
    ssim_list, rmse_list, psnr_list = [], [], []
    # normalized metal fill (100 HU) used to fill the dilated metal region for SSIM
    metal_fill_norm = (100.0 - float(hu_min)) / (float(hu_max) - float(hu_min))

    pred = pred.float()
    # Normalized metrics are computed on clamped [0,1] tensors.
    pred_eval = torch.clamp(pred, 0, 1)
    gt_eval = torch.clamp(gt, 0, 1)

    for i in range(pred_eval.size(0)):
        img_pred = pred_eval[i, 0]
        img_gt = gt_eval[i, 0]

        metalmask_arg = None
        if metal_mask is not None:
            m = metal_mask[i]
            if m.dim() == 3 and m.size(0) == 1:
                m = m.squeeze(0)
            if m.dim() == 3 and m.size(0) == 1:
                m = m.squeeze(0)
            metalmask_arg = m

        # SSIM: dilate metal by 2 px, fill dilated region to 100 HU,
        # average full SSIM map (data_range=1) over the non-dilated pixels.
        if metalmask_arg is not None:
            _, ssim_val = compute_masked_SSIM(
                img_pred,
                img_gt,
                data_range=1.0,
                mask=None,
                metalmask=metalmask_arg,
                metal_fill=metal_fill_norm,
                dilate_iters=2,
                hu_min=hu_min,
                hu_max=hu_max,
            )
        else:
            ssim_val = compute_SSIM(img_pred, img_gt, data_range=1.0)
        ssim_list.append(float(ssim_val))

        # RMSE (HU) and PSNR: denormalize to HU, dilate metal by 1 px and exclude it.
        rmse_hu = compute_masked_RMSE_HU(
            img_pred,
            img_gt,
            metalmask=metalmask_arg,
            hu_min=hu_min,
            hu_max=hu_max,
            dilate_iters=1,
        )
        rmse_list.append(float(rmse_hu))
        psnr_max = float(hu_max) - float(hu_min)
        rmse_hu_safe = rmse_hu if rmse_hu > 0 else 1e-10
        psnr_list.append(float(20.0 * np.log10(psnr_max / rmse_hu_safe)))

    return {"ssim": ssim_list, "rmse": rmse_list, "psnr": psnr_list}


def summarize_metrics(per_image):
    """Mean and sample std (ddof=1) per metric. Empty -> 0.0; single image -> std 0.0.

    Returns dict like {"ssim": mean, "ssim_std": std, "rmse": ..., "psnr": ...}.
    """
    out = {}
    for name, vals in per_image.items():
        out[name] = float(np.mean(vals)) if vals else 0.0
        out[f"{name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out
