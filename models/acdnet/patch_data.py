"""Random-patch training data for ACDNet on top of CTMetalArtifactDataset (train split only).

Each sampled item is one slice plus a global draw id; it yields `patches_per_slice` random
`patch_size` windows with identical coordinates/flips for MA, LI, GT and the metal mask.

* Files are opened with np.load(mmap_mode='r') and only the window is read.
* The per-patch normalization calls the dataset's own methods (astype(float32) -> _li_to_hu for LI
  -> _clip_and_norm01), all element-wise, so a patch is bit-identical to the same window of the
  dataset's full normalized slice. The metal mask follows the dataset rule, which depends on the
  max of the WHOLE slice (`if mask.max() > 1: mask = (mask > 0)`); that per-file decision is
  computed once per worker and cached.
* Randomness is counter-based: crop positions/flips come from default_rng([seed, 2, draw_id]) and
  the slice order from per-pass permutations default_rng([seed, 1, k]). The stream is therefore
  independent of worker count/scheduling, and resuming at epoch e reproduces exactly the draws an
  uninterrupted run would have used.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _mmap_2d(path):
    arr = np.load(path, mmap_mode="r")
    # Same shape handling as CTMetalArtifactDataset._load_npy.
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape} in {path}")
    return arr


class ACDNetPatchDataset(Dataset):
    """Indexed by (slice_index, draw_id) tuples produced by `DrawStreamSampler`.

    Returns a dict of float32 tensors, each (patches_per_slice, 1, patch_size, patch_size):
    'ma', 'li', 'gt' in normalized [0,1] space and 'mask' (1 = metal), exactly as the dataset.
    """

    def __init__(self, base_ds, patch_size=64, patches_per_slice=4, seed=42, augment=True):
        if base_ds.split != "train":
            raise ValueError("ACDNet patches must be drawn from the train split only.")
        if base_ds.output_space != "norm" or base_ds.transform is not None:
            raise ValueError("Expected output_space='norm' and no transform on the base dataset.")
        if base_ds.li_dir is None or base_ds.mask_dir is None:
            raise ValueError("ACDNet needs LI and metal masks (pass li_dir and mask_dir).")
        for entry in base_ds.pairs:
            if len(entry) != 4:
                raise ValueError(f"Expected (ma, gt, mask, li) file tuples, got {entry}")
        self.base = base_ds
        self.patch_size = int(patch_size)
        self.patches_per_slice = int(patches_per_slice)
        self.seed = int(seed)
        self.augment = bool(augment)
        self._mask_binarize = {}  # mask filename -> bool (dataset rule on the full slice)

    def __len__(self):
        return len(self.base.pairs)

    def crop_params(self, draw_id, h, w):
        """[(row, col, hflip, vflip)] * patches_per_slice for a given draw (uniform over all windows)."""
        p = self.patch_size
        if h < p or w < p:
            raise ValueError(f"Slice {h}x{w} smaller than patch {p}")
        rng = np.random.default_rng([self.seed, 2, int(draw_id)])
        out = []
        for _ in range(self.patches_per_slice):
            r = int(rng.integers(0, h - p + 1))
            c = int(rng.integers(0, w - p + 1))
            hflip = bool(rng.random() < 0.5)
            vflip = bool(rng.random() < 0.5)
            if not self.augment:
                hflip = vflip = False
            out.append((r, c, hflip, vflip))
        return out

    def _binarize_mask(self, mask_file, mask_mm):
        flag = self._mask_binarize.get(mask_file)
        if flag is None:
            flag = bool(mask_mm.max() > 1)
            self._mask_binarize[mask_file] = flag
        return flag

    @staticmethod
    def _flip(a, hflip, vflip):
        # Original ACDNet augment(): hflip = img[:, ::-1], vflip = img[::-1, :].
        if hflip:
            a = a[:, ::-1]
        if vflip:
            a = a[::-1, :]
        return np.ascontiguousarray(a)

    def __getitem__(self, key):
        idx, draw_id = key
        ds = self.base
        ma_file, gt_file, mask_file, li_file = ds.pairs[idx]
        ma_mm = _mmap_2d(os.path.join(ds.ma_dir, ma_file))
        gt_mm = _mmap_2d(os.path.join(ds.gt_dir, gt_file))
        li_mm = _mmap_2d(os.path.join(ds.li_dir, li_file))
        mask_mm = _mmap_2d(os.path.join(ds.mask_dir, mask_file))
        h, w = ma_mm.shape
        if gt_mm.shape != (h, w) or li_mm.shape != (h, w) or mask_mm.shape != (h, w):
            raise ValueError(f"Shape mismatch for {ma_file}: MA {ma_mm.shape}, GT {gt_mm.shape}, "
                             f"LI {li_mm.shape}, mask {mask_mm.shape}")
        binarize = self._binarize_mask(mask_file, mask_mm)
        p = self.patch_size

        out = {"ma": [], "li": [], "gt": [], "mask": []}
        for r, c, hflip, vflip in self.crop_params(draw_id, h, w):
            win = (slice(r, r + p), slice(c, c + p))
            # Identical element-wise ops to CTMetalArtifactDataset.__getitem__ (output_space='norm').
            ma = ds._clip_and_norm01(np.array(ma_mm[win]).astype(np.float32))
            gt = ds._clip_and_norm01(np.array(gt_mm[win]).astype(np.float32))
            li = ds._clip_and_norm01(ds._li_to_hu(np.array(li_mm[win]).astype(np.float32)))
            mask = np.array(mask_mm[win])
            if binarize:
                mask = (mask > 0).astype(np.float32)
            for name, a in (("ma", ma), ("li", li), ("gt", gt), ("mask", mask)):
                out[name].append(torch.from_numpy(self._flip(a, hflip, vflip)).unsqueeze(0).float())
        return {k: torch.stack(v, dim=0) for k, v in out.items()}


class DrawStreamSampler(Sampler):
    """Yields (slice_index, draw_id) for draw_id in [start_draw, start_draw + num_draws).

    Draw g uses slice perm_k[g % n] with k = g // n, perm_k = default_rng([seed, 1, k]).permutation(n):
    every slice is seen once per pass, and the stream is a pure function of (seed, g).
    """

    def __init__(self, n_slices, seed, start_draw, num_draws):
        self.n = int(n_slices)
        self.seed = int(seed)
        self.start = int(start_draw)
        self.num = int(num_draws)

    def __len__(self):
        return self.num

    def __iter__(self):
        perm_k, perm = None, None
        for g in range(self.start, self.start + self.num):
            k = g // self.n
            if k != perm_k:
                perm_k, perm = k, np.random.default_rng([self.seed, 1, k]).permutation(self.n)
            yield int(perm[g % self.n]), g


def worker_init_fn(worker_id):
    # Per-worker seeding of the global numpy/random RNGs (DataLoader already seeds torch per worker).
    # The patch sampling itself does not use them; it is counter-based (see module docstring).
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)
