"""Unpaired training data for ADN on top of CTMetalArtifactDataset (train split only).

Each draw yields one artifact (MA) image and one clean (GT) image that are NOT paired:
  * --unpaired_mode disjoint (default): the train slices are split once into two fixed seeded halves;
    MA images come only from half A and clean images only from half B (the paper's SYN protocol and
    the official `partial_holdout: 0.5`). An epoch is one pass over half A in a random order; each
    MA draw is matched with a clean image drawn uniformly at random from half B (official:
    `choice(self.invisible_files)`).
  * --unpaired_mode shuffle: MA and clean images come from two independent per-epoch shuffles of
    the full train split (an epoch is one pass over the full split).
Only MA and GT files are read for training; the metal mask is read only to choose the MA crop
location (never returned, never used by a loss). LI is not used.

* Files are opened with np.load(mmap_mode='r') and only the needed window is read. The
  normalization calls the dataset's own `_clip_and_norm01` on `astype(float32)` data (all
  element-wise), so a crop is bit-identical to the same window of the dataset's normalized slice.
* Optional `crop_size`: the MA and clean crops are drawn independently. With probability
  `metal_crop_prob` the MA crop is centred on a random metal pixel plus a uniform jitter of up to
  +-`jitter_frac` * crop_size per axis (then shifted to lie inside the slice); otherwise (and if the
  slice has no metal) the crop position is uniform.
* Optional random vertical flip (official DeepLesion `random_flip: True`: `img[::-1, :]` with p=0.5,
  one coin per sample applied to both images).
* Randomness is counter-based on (seed, epoch, draw index): the stream does not depend on the
  number of workers, and resuming at epoch e reproduces exactly what an uninterrupted run would draw.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

UNPAIRED_MODES = ("disjoint", "shuffle")


def _mmap_2d(path):
    arr = np.load(path, mmap_mode="r")
    # Same shape handling as CTMetalArtifactDataset._load_npy.
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape} in {path}")
    return arr


def split_pools(n, mode, seed):
    """(ma_pool, clean_pool) index arrays into the train pairs."""
    if mode == "disjoint":
        perm = np.random.default_rng([seed, 0]).permutation(n)
        return np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if mode == "shuffle":
        return np.arange(n), np.arange(n)
    raise ValueError(f"unpaired_mode must be one of {UNPAIRED_MODES}, got {mode!r}")


class UnpairedADNDataset(Dataset):
    """Indexed by (epoch, k) keys from `EpochDrawSampler`.

    Returns {'ma': (1,h,w), 'clean': (1,h,w)} float32 tensors in normalized [0,1] space and the
    train-pair indices 'ma_idx', 'clean_idx'.
    """

    def __init__(self, base_ds, mode="disjoint", crop_size=None, metal_crop_prob=0.7, random_flip=True,
                 seed=42, jitter_frac=0.25):
        if base_ds.split != "train":
            raise ValueError("ADN training images must come from the train split only.")
        if base_ds.output_space != "norm" or base_ds.transform is not None:
            raise ValueError("Expected output_space='norm' and no transform on the base dataset.")
        self.base = base_ds
        self.mode = mode
        self.crop_size = int(crop_size) if crop_size else None
        self.metal_crop_prob = float(metal_crop_prob) if self.crop_size else 0.0
        self.random_flip = bool(random_flip)
        self.seed = int(seed)
        self.jitter_frac = float(jitter_frac)
        if self.metal_crop_prob > 0 and base_ds.mask_dir is None:
            raise ValueError("metal_crop_prob > 0 needs the metal masks (mask_dir).")
        self.ma_pool, self.clean_pool = split_pools(len(base_ds.pairs), mode, self.seed)
        self._perm_cache = {}

    @property
    def n_ma(self):
        return len(self.ma_pool)

    def __len__(self):
        return self.n_ma

    # ---------------- files ----------------
    def _files(self, idx):
        entry = self.base.pairs[idx]
        mask_file = None
        if len(entry) == 3 and "metalonlymask" in entry[2].lower():
            mask_file = entry[2]
        elif len(entry) == 4:
            mask_file = entry[2]
        return entry[0], entry[1], mask_file

    def _norm(self, mm, win):
        # Identical element-wise ops to CTMetalArtifactDataset.__getitem__ (output_space='norm').
        return self.base._clip_and_norm01(np.array(mm[win]).astype(np.float32))

    # ---------------- randomness ----------------
    def _perm(self, tag, epoch, n):
        key = (tag, epoch)
        p = self._perm_cache.get(key)
        if p is None:
            if len(self._perm_cache) > 8:
                self._perm_cache.clear()
            p = np.random.default_rng([self.seed, tag, int(epoch)]).permutation(n)
            self._perm_cache[key] = p
        return p

    def plan(self, epoch, k):
        """(ma_idx, clean_idx) of draw k in `epoch` (pure function of seed, epoch, k)."""
        ma_idx = int(self.ma_pool[self._perm(1, epoch, self.n_ma)[k]])
        if self.mode == "disjoint":
            j = int(np.random.default_rng([self.seed, 2, int(epoch), int(k)]).integers(len(self.clean_pool)))
        else:
            j = int(self._perm(3, epoch, len(self.clean_pool))[k])
        return ma_idx, int(self.clean_pool[j])

    def _uniform_origin(self, rng, h, w):
        c = self.crop_size
        return int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))

    def _metal_origin(self, rng, mask_mm, h, w):
        """Top-left of a crop centred near a random metal pixel, or None if the slice has no metal."""
        mask = np.asarray(mask_mm)
        # Same binarization rule as the dataset (whole-slice max); 0/1 masks are thresholded at 0.5.
        metal = np.flatnonzero(mask > 0) if mask.max() > 1 else np.flatnonzero(mask > 0.5)
        if metal.size == 0:
            return None
        cy, cx = divmod(int(metal[int(rng.integers(metal.size))]), w)
        c = self.crop_size
        j = int(round(self.jitter_frac * c))
        cy += int(rng.integers(-j, j + 1))
        cx += int(rng.integers(-j, j + 1))
        r = min(max(cy - c // 2, 0), h - c)
        col = min(max(cx - c // 2, 0), w - c)
        return r, col

    def aug_params(self, epoch, k, ma_mm, clean_mm, mask_mm):
        """(ma_window, clean_window, flip) of draw k in `epoch`; windows are (slice, slice)."""
        rng = np.random.default_rng([self.seed, 4, int(epoch), int(k)])
        flip = bool(rng.random() < 0.5) and self.random_flip
        use_metal = bool(rng.random() < self.metal_crop_prob)
        if self.crop_size is None:
            full = (slice(None), slice(None))
            return full, full, flip
        c = self.crop_size
        for name, mm in (("MA", ma_mm), ("clean", clean_mm)):
            if mm.shape[0] < c or mm.shape[1] < c:
                raise ValueError(f"{name} slice {mm.shape} smaller than crop {c}")
        h, w = ma_mm.shape
        o = self._metal_origin(rng, mask_mm, h, w) if (use_metal and mask_mm is not None) else None
        if o is None:
            o = self._uniform_origin(rng, h, w)
        o2 = self._uniform_origin(rng, *clean_mm.shape)
        win = lambda r, col: (slice(r, r + c), slice(col, col + c))  # noqa: E731
        return win(*o), win(*o2), flip

    # ---------------- item ----------------
    def __getitem__(self, key):
        epoch, k = key
        ma_idx, clean_idx = self.plan(epoch, k)
        ds = self.base
        ma_file, _, mask_file = self._files(ma_idx)
        _, clean_file, _ = self._files(clean_idx)
        ma_mm = _mmap_2d(os.path.join(ds.ma_dir, ma_file))
        clean_mm = _mmap_2d(os.path.join(ds.gt_dir, clean_file))
        mask_mm = None
        if self.metal_crop_prob > 0 and mask_file is not None:
            mask_mm = _mmap_2d(os.path.join(ds.mask_dir, mask_file))
            if mask_mm.shape != ma_mm.shape:
                raise ValueError(f"Shape mismatch for {ma_file}: MA {ma_mm.shape}, mask {mask_mm.shape}")
        ma_win, clean_win, flip = self.aug_params(epoch, k, ma_mm, clean_mm, mask_mm)
        ma = self._norm(ma_mm, ma_win)
        clean = self._norm(clean_mm, clean_win)
        if flip:
            ma, clean = ma[::-1, :], clean[::-1, :]
        return {"ma": torch.from_numpy(np.ascontiguousarray(ma)).unsqueeze(0),
                "clean": torch.from_numpy(np.ascontiguousarray(clean)).unsqueeze(0),
                "ma_idx": ma_idx, "clean_idx": clean_idx}


class EpochDrawSampler(Sampler):
    """Yields (epoch, k) for epoch in [start_epoch, end_epoch] and k in [0, draws_per_epoch)."""

    def __init__(self, start_epoch, end_epoch, draws_per_epoch):
        self.start, self.end, self.dpe = int(start_epoch), int(end_epoch), int(draws_per_epoch)

    def __len__(self):
        return max(0, self.end - self.start + 1) * self.dpe

    def __iter__(self):
        for e in range(self.start, self.end + 1):
            for k in range(self.dpe):
                yield e, k


def worker_init_fn(worker_id):
    # Per-worker seeding of the global numpy/random RNGs (DataLoader already seeds torch per worker).
    # The sampling itself does not use them; it is counter-based (see module docstring).
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)
