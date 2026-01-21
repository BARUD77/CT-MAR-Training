# ct_mar_dataset.py
# import os, re, numpy as np, torch
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split

# class CTMetalArtifactDataset(Dataset):
#     """
#     Loads MA/GT (+ optional mask, LI) from .npy files named like:
#       training_{region}_{kind}_img{ID}_{HxWxZ}.npy
#     where:
#       region ∈ {body, head}
#       kind   ∈ {metalart, nometal, metalonlymask, li}
#       ID     ∈ integers (non-contiguous is fine)

#     Returns (depending on what exists):
#       (x, y) or (x, y, m) or (x, y, m, li)
#       x  = MA  in [0,1]  shape (1,H,W)
#       y  = GT  in [0,1]  shape (1,H,W)
#       m  = mask in {0,1} shape (1,H,W)
#       li = LI  in [0,1]  shape (1,H,W)
#     """

#     def __init__(
#         self,
#         ma_dir: str,
#         gt_dir: str,
#         mask_dir: str | None = None,   # optional
#         li_dir: str | None = None,     # optional
#         split: str = "train",          # {"train","val"}
#         val_size: float = 0.1,
#         seed: int = 42,
#         image_shape: tuple[int, int] = (512, 512),
#         hu_min: float = -1024.0,
#         hu_max: float = 3072.0,
#         transform=None,
#         # region filter: "all" (default), "body" (only body), "head" (only head)
#         region_policy: str = "all",
#     ):
#         assert split in {"train", "val"}
#         assert 0.0 <= val_size < 1.0
#         assert hu_max > hu_min
#         assert region_policy in {"all", "body", "head"}

#         self.ma_dir, self.gt_dir = ma_dir, gt_dir
#         self.mask_dir, self.li_dir = mask_dir, li_dir
#         self.split, self.transform = split, transform
#         self.image_shape = tuple(image_shape)
#         self.hu_min, self.hu_max = float(hu_min), float(hu_max)
#         self._dr = self.hu_max - self.hu_min

#         # Regex for your filenames
#         # training_{region}_{kind}_img{ID}_{HxWxZ}.npy
#         rx = re.compile(
#             r'^training_(body|head)_(metalart|nometal|metalonlymask|li)_img(\d+)_\d+x\d+x\d+(?:\.npy)?$',
#             re.IGNORECASE
#         )

#         def list_npy(d):
#             if not os.path.isdir(d):
#                 raise FileNotFoundError(f"Directory not found: {d}")
#             return [f for f in os.listdir(d) if f.lower().endswith(".npy") or rx.match(f)]

#         # Build maps keyed by (region, id)
#         ma_map, gt_map, mask_map, li_map = {}, {}, {}, {}

#         for f in list_npy(ma_dir):
#             m = rx.match(f)
#             if m and m.group(2).lower() == "metalart":
#                 key = (m.group(1).lower(), int(m.group(3)))
#                 ma_map[key] = f

#         for f in list_npy(gt_dir):
#             m = rx.match(f)
#             if m and m.group(2).lower() == "nometal":
#                 key = (m.group(1).lower(), int(m.group(3)))
#                 gt_map[key] = f

#         if mask_dir:
#             for f in list_npy(mask_dir):
#                 m = rx.match(f)
#                 if m and m.group(2).lower() == "metalonlymask":
#                     key = (m.group(1).lower(), int(m.group(3)))
#                     mask_map[key] = f

#         if li_dir:
#             for f in list_npy(li_dir):
#                 m = rx.match(f)
#                 if m and m.group(2).lower() == "li":
#                     key = (m.group(1).lower(), int(m.group(3)))
#                     li_map[key] = f

#         # Intersect keys present in required modalities
#         keys = sorted(set(ma_map) & set(gt_map))
#         if mask_dir:
#             keys = sorted(set(keys) & set(mask_map))
#         if li_dir:
#             keys = sorted(set(keys) & set(li_map))

#         if not keys:
#             raise RuntimeError("No matched MA/GT pairs (and mask/LI if requested). Check dirs/filenames.")

#         # Region filter
#         if region_policy != "all":
#             keys = [k for k in keys if k[0] == region_policy.lower()]
#             if not keys:
#                 raise RuntimeError(f"No pairs after region_policy='{region_policy}' filter.")

#         # Build file tuples
#         pairs_all = []
#         for key in keys:
#             items = [ma_map[key], gt_map[key]]
#             if mask_dir:
#                 items.append(mask_map[key])
#             if li_dir:
#                 items.append(li_map[key])
#             pairs_all.append(tuple(items))

#         # Train/Val split
#         if val_size > 0:
#             train, val = train_test_split(pairs_all, test_size=val_size, random_state=seed, shuffle=True)
#         else:
#             train, val = pairs_all, []

#         self.pairs = {"train": train, "val": val}[split]
#         self._ma_map, self._gt_map = ma_map, gt_map
#         self._mask_map, self._li_map = mask_map, li_map
#         self._keys = keys
#         self.region_policy = region_policy

#         print(f"[CTDataset] total={len(pairs_all)} | train={len(train)} | val={len(val)} "
#               f"| region={region_policy} | window=[{self.hu_min},{self.hu_max}] | shape={self.image_shape}")

#     # ----------------- helpers -----------------
#     def _load_npy(self, path: str) -> np.ndarray:
#         arr = np.load(path)
#         if arr.ndim == 3 and arr.shape[-1] == 1:
#             arr = arr[..., 0]
#         if arr.ndim != 2:
#             raise ValueError(f"Expected 2D array, got {arr.shape} in {path}")
#         return arr

#     def _clip_and_norm01(self, hu: np.ndarray) -> np.ndarray:
#         hu = np.clip(hu, self.hu_min, self.hu_max)
#         return (hu - self.hu_min) / self._dr

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx: int):
#         item = self.pairs[idx]
#         # item order: MA, GT, [mask], [li]
#         offs = 0
#         ma_file, gt_file = item[0], item[1]; offs = 2
#         mask_file = item[offs] if (self.mask_dir and len(item) > offs) else None
#         li_file   = item[offs+1] if (self.li_dir and len(item) > offs + (1 if mask_file else 0)) else None

#         ma = self._clip_and_norm01(self._load_npy(os.path.join(self.ma_dir, ma_file)))
#         gt = self._clip_and_norm01(self._load_npy(os.path.join(self.gt_dir, gt_file)))

#         x = torch.from_numpy(ma).unsqueeze(0).float()
#         y = torch.from_numpy(gt).unsqueeze(0).float()

#         m = None
#         if mask_file:
#             mask_np = self._load_npy(os.path.join(self.mask_dir, mask_file))
#             if mask_np.max() > 1:
#                 mask_np = (mask_np > 0).astype(np.float32)
#             m = torch.from_numpy(mask_np).unsqueeze(0).float()

#         li = None
#         if li_file:
#             li_np = self._clip_and_norm01(self._load_npy(os.path.join(self.li_dir, li_file)))
#             li = torch.from_numpy(li_np).unsqueeze(0).float()

#         if self.transform:
#             x = self.transform(x); y = self.transform(y)
#             if m is not None: m = self.transform(m)
#             if li is not None: li = self.transform(li)

#         # Return tuple with whatever is available
#         if m is not None and li is not None:   return x, y, m, li
#         if m is not None:                      return x, y, m
#         if li is not None:                     return x, y, li
#         return x, y


# ct_mar_dataset.py
import os
import re
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CTMetalArtifactDataset(Dataset):
    """
    Loads MA/GT/LI from .npy files named like:
      training_{region}_{kind}_img{ID}_{HxWxZ}.npy
    where:
      region ∈ {body, head}
      kind   ∈ {metalart, nometal, li}
      ID     ∈ integers

    Returns:
      (x, y, li)
      x  = MA  in [0,1]  shape (1,H,W)
      y  = GT  in [0,1]  shape (1,H,W)
      li = LI  in [0,1]  shape (1,H,W)   <-- provided and used as input (artifact guidance)
    """

    def __init__(
        self,
        ma_dir: str,
        gt_dir: str,
        li_dir: str,
        mask_dir: str | None = None,
        split: str = "train",          # {"train","val"}
        val_size: float = 0.1,
        seed: int = 42,
        image_shape: Tuple[int, int] = (512, 512),
        hu_min: float = -1024.0,
        hu_max: float = 3072.0,
        transform=None,
        # region filter: "all" (default), "body" (only body), "head" (only head)
        region_policy: str = "all",
    ):
        assert split in {"train", "val", "test"}
        assert 0.0 <= val_size < 1.0
        assert hu_max > hu_min
        assert region_policy in {"all", "body", "head"}
        if li_dir is None:
            raise ValueError("li_dir must be provided for artifact-guided training (LI is required).")

        self.ma_dir, self.gt_dir, self.li_dir = ma_dir, gt_dir, li_dir
        self.split, self.transform = split, transform
        self.image_shape = tuple(image_shape)
        self.hu_min, self.hu_max = float(hu_min), float(hu_max)
        self._dr = self.hu_max - self.hu_min

        # Regex for your filenames
        # training_{region}_{kind}_img{ID}_{HxWxZ}.npy
        # Accept either training_ or test_ prefixes (e.g. training_body_... or test_body_...)
        rx = re.compile(
            r'^(?:training|test)_(body|head)_(metalart|nometal|li|metalonlymask)_img(\d+)_\d+x\d+x\d+(?:\.npy)?$',
            re.IGNORECASE
        )

        def list_npy(d):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Directory not found: {d}")
            # return only .npy files (and/or those matching the expected pattern)
            return [f for f in os.listdir(d) if f.lower().endswith(".npy") or rx.match(f)]

        # Build maps keyed by (region, id)
        ma_map, gt_map, li_map, mask_map = {}, {}, {}, {}

        for f in list_npy(ma_dir):
            m = rx.match(f)
            if m and m.group(2).lower() == "metalart":
                key = (m.group(1).lower(), int(m.group(3)))
                ma_map[key] = f

        for f in list_npy(gt_dir):
            m = rx.match(f)
            if m and m.group(2).lower() == "nometal":
                key = (m.group(1).lower(), int(m.group(3)))
                gt_map[key] = f

        for f in list_npy(li_dir):
            m = rx.match(f)
            if m and m.group(2).lower() == "li":
                key = (m.group(1).lower(), int(m.group(3)))
                li_map[key] = f

        # optional metal-only masks
        if mask_dir is not None:
            for f in list_npy(mask_dir):
                m = rx.match(f)
                if m and m.group(2).lower() == "metalonlymask":
                    key = (m.group(1).lower(), int(m.group(3)))
                    mask_map[key] = f

        # Intersect keys present in required modalities (MA, GT, LI)
        keys = sorted(set(ma_map) & set(gt_map) & set(li_map))
        # If mask_dir provided, require mask to be present as well
        if mask_dir is not None:
            keys = sorted(set(keys) & set(mask_map))

        if not keys:
            raise RuntimeError("No matched MA/GT/LI triplets. Check dirs/filenames.")

        # Region filter
        if region_policy != "all":
            keys = [k for k in keys if k[0] == region_policy.lower()]
            if not keys:
                raise RuntimeError(f"No pairs after region_policy='{region_policy}' filter.")

        # Build file tuples. If mask present we insert mask before LI so
        # tuples become (MA, GT, MASK, LI) which keeps LI as last element.
        triplets_all = []
        for key in keys:
            if mask_dir is not None:
                triplets_all.append((ma_map[key], gt_map[key], mask_map[key], li_map[key]))
            else:
                triplets_all.append((ma_map[key], gt_map[key], li_map[key]))

        # Train/Val split (test uses the full set)
        if split == "test":
            train, val = [], []
            self.pairs = triplets_all
        else:
            if val_size > 0:
                train, val = train_test_split(triplets_all, test_size=val_size, random_state=seed, shuffle=True)
            else:
                train, val = triplets_all, []
            self.pairs = {"train": train, "val": val}[split]
        self._ma_map, self._gt_map, self._li_map = ma_map, gt_map, li_map
        self._keys = keys
        self.region_policy = region_policy

        print(f"[CTDataset] total={len(triplets_all)} | train={len(train)} | val={len(val)} "
              f"| region={region_policy} | window=[{self.hu_min},{self.hu_max}] | shape={self.image_shape}")

    # ----------------- helpers -----------------
    def _load_npy(self, path: str) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.shape} in {path}")
        return arr

    def _clip_and_norm01(self, hu: np.ndarray) -> np.ndarray:
        hu = np.clip(hu, self.hu_min, self.hu_max)
        return (hu - self.hu_min) / self._dr

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        # Support pairs of length 3: (ma, gt, li) and length 4: (ma, gt, mask, li)
        entry = self.pairs[idx]
        if len(entry) == 3:
            ma_file, gt_file, li_file = entry
            mask_file = None
        else:
            ma_file, gt_file, mask_file, li_file = entry

        ma = self._clip_and_norm01(self._load_npy(os.path.join(self.ma_dir, ma_file)))
        gt = self._clip_and_norm01(self._load_npy(os.path.join(self.gt_dir, gt_file)))
        li = self._clip_and_norm01(self._load_npy(os.path.join(self.li_dir, li_file)))

        x = torch.from_numpy(ma).unsqueeze(0).float()   # MA input
        y = torch.from_numpy(gt).unsqueeze(0).float()   # GT target
        li_t = torch.from_numpy(li).unsqueeze(0).float()# LI guidance map

        m_t = None
        if mask_file is not None:
            mask_np = self._load_npy(os.path.join(self.mask_dir, mask_file))
            # normalize to 0/1
            if mask_np.max() > 1:
                mask_np = (mask_np > 0).astype(np.float32)
            m_t = torch.from_numpy(mask_np).unsqueeze(0).float()

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            li_t = self.transform(li_t)
            if m_t is not None:
                m_t = self.transform(m_t)

        # Return (x, y, li) or (x, y, mask, li)
        if m_t is not None:
            return x, y, m_t, li_t
        return x, y, li_t

