"""Converts samples from baselines/common/data.py into FIND-Net inputs, and predictions back.

My data (common/data.py, fixed args): HU clipped to [hu_min, hu_max] = [-1024, 3072] and mapped to
[0, 1] ("norm"). FIND-Net's original pipeline feeds images in [0, 255]
(HU -> (HU + 1500) / 5000 -> clip [0, 1] -> x255). The conversion is selected by `normalization`:

  "scale255"      x_findnet = 255 * x_norm                       (my window, FIND-Net's 0-255 scale)
  "native_window" x_findnet = 255 * clip((HU + 1500) / 5000, 0, 1) with HU = x_norm*(hu_max-hu_min)+hu_min
                  (FIND-Net's own HU window; my [-1024, 3072] clip still applies first)

`to_norm` is the exact inverse back to my [0, 1] normalization (used for validation metrics).

Masks: the non-metal mask is 1 - metal mask, with the metal mask taken from the AAPM metal_mask files
(via the dataset); masks are never derived by thresholding intensities.
Augmentation (train only, original FIND-Net dataset): independent random horizontal flip (p=0.5)
and vertical flip (p=0.5), applied identically to MA, LI, GT and mask.
"""
import os
import random
import sys

import torch
from torch.utils.data import Dataset, Subset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.data import build_datasets, unpack_batch, HU_MIN, HU_MAX  # noqa: E402

NORMALIZATIONS = ("scale255", "native_window")
FINDNET_HU_OFFSET = 1500.0   # original Dataset/dataset.py: (HU + 1500) / 5000
FINDNET_HU_SCALE = 5000.0
FINDNET_MAX = 255.0


class NormConverter:
    def __init__(self, normalization="scale255", hu_min=HU_MIN, hu_max=HU_MAX):
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}")
        self.normalization = normalization
        self.hu_min, self.hu_max = float(hu_min), float(hu_max)

    def to_findnet(self, x_norm):
        if self.normalization == "scale255":
            return x_norm * FINDNET_MAX
        hu = x_norm * (self.hu_max - self.hu_min) + self.hu_min
        return torch.clamp((hu + FINDNET_HU_OFFSET) / FINDNET_HU_SCALE, 0.0, 1.0) * FINDNET_MAX

    def to_norm(self, x_findnet):
        """FIND-Net units -> my [0, 1] normalization (not clamped; eval_utils clamps)."""
        if self.normalization == "scale255":
            return x_findnet / FINDNET_MAX
        hu = x_findnet / FINDNET_MAX * FINDNET_HU_SCALE - FINDNET_HU_OFFSET
        return (hu - self.hu_min) / (self.hu_max - self.hu_min)


class FindNetAdapter(Dataset):
    """Wraps a CTMetalArtifactDataset built with MA, GT, LI and mask dirs.

    Each item is a dict of (1,H,W) float tensors:
      ma, li, gt     -- FIND-Net units
      nonmetal       -- 1 - metal mask (FIND-Net's `Mask` argument)
      metal_mask     -- metal mask, for eval_utils
      gt_norm        -- GT in my [0,1] normalization, for eval_utils (avoids round-trip error)
    """

    def __init__(self, base_ds, converter, augment=False):
        self.base = base_ds
        self.conv = converter
        self.augment = bool(augment)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = unpack_batch(self.base[idx], has_mask=True, has_li=True)
        ma, gt, li, metal = s["ma"], s["gt"], s["li"], s["mask"]
        if self.augment:
            # original: hflip = random() < 0.5 ; vflip = random() < 0.5 ; same flips for all images
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dims = ([2] if hflip else []) + ([1] if vflip else [])
            if dims:
                ma, gt, li, metal = (torch.flip(t, dims) for t in (ma, gt, li, metal))
        return {
            "ma": self.conv.to_findnet(ma),
            "li": self.conv.to_findnet(li),
            "gt": self.conv.to_findnet(gt),
            "nonmetal": 1.0 - metal,
            "metal_mask": metal,
            "gt_norm": gt,
        }


def build_findnet_datasets(ma_dir, gt_dir, li_dir, mask_dir, normalization="scale255",
                           augment_train=True, subset=None, val_subset=None):
    """(train, val, converter). `subset` / `val_subset` keep only the first N slices of each split."""
    if li_dir is None or mask_dir is None:
        raise ValueError("FIND-Net needs both li_dir and mask_dir.")
    train_base, val_base = build_datasets(ma_dir, gt_dir, li_dir=li_dir, mask_dir=mask_dir)
    conv = NormConverter(normalization)
    train = FindNetAdapter(train_base, conv, augment=augment_train)
    val = FindNetAdapter(val_base, conv, augment=False)
    if subset:
        train = Subset(train, list(range(min(int(subset), len(train)))))
    if val_subset:
        val = Subset(val, list(range(min(int(val_subset), len(val)))))
    return train, val, conv
