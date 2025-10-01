# ct_mar_dataset.py
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CTMetalArtifactDataset(Dataset):
    """
    MA→GT pairs from .npy float32 images of size HxW (default 512x512).

    Pipeline:
      1) load raw HU from .npy
      2) clip to [hu_min, hu_max]
      3) normalize to [0,1]

    Returns:
      x: (1, H, W)  normalized MA in [0,1]
      y: (1, H, W)  normalized GT in [0,1]
      m: (1, H, W)  mask (0/1 float) if mask_dir is given, else omitted

    Notes:
      - Splits into TRAIN and VAL only. You said you have a separate TEST set.
      - Head/body filtering is numeric: IDs <= id_max_body = body, > id_max_body = head.
    """

    def __init__(
        self,
        ma_dir: str,
        gt_dir: str,
        mask_dir: str | None = None,   # optional
        split: str = 'train',           # 'train' or 'val'
        val_size: float = 0.1,          # fraction (0..1)
        seed: int = 42,
        image_shape: tuple[int, int] = (512, 512),
        hu_min: float = -1024.0,
        hu_max: float = 3072.0,
        transform=None,
        # --- head/body control ---
        #head_policy: str = "exclude",       # {"all","exclude","only"} -> all, body-only, head-only
        #id_max_body: int = 12374,       # IDs <= this are BODY; IDs > this are HEAD
    ):
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        assert 0.0 <= val_size < 1.0, "val_size must be in [0,1)"
        assert hu_max > hu_min, "hu_max must be > hu_min"
        #assert head_policy in {"all", "exclude", "only"}, "head_policy must be one of {'all','exclude','only'}"

        self.ma_dir   = ma_dir
        self.gt_dir   = gt_dir
        self.mask_dir = mask_dir
        self.split    = split
        self.transform = transform

        self.image_shape = tuple(image_shape)
        self._numel = self.image_shape[0] * self.image_shape[1]
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self._dr = self.hu_max - self.hu_min

        # Filename patterns (adjust if yours differ)
        # Example: Input_Image_123_512x512.npy  /  Simulated_Image_123_512x512.npy
        ma_pat   = re.compile(r'^test_body_metalart_img(\d+)_\d+x\d+(?:\.npy)?$',      re.IGNORECASE)
        gt_pat   = re.compile(r'^test_body_nometal_img(\d+)_\d+x\d+(?:\.npy)?$',  re.IGNORECASE)
        mask_pat = re.compile(r'^.*mask.*_(\d+)_\d+x\d+(?:\.npy)?$',         re.IGNORECASE)

        def list_npy(d):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Directory not found: {d}")
            return [f for f in os.listdir(d) if f.lower().endswith('.npy')]

        # Build ID→filename maps
        ma_map, gt_map, mask_map = {}, {}, {}
        for f in list_npy(ma_dir):
            m = ma_pat.match(f)
            if m:
                ma_map[int(m.group(1))] = f
        for f in list_npy(gt_dir):
            m = gt_pat.match(f)
            if m:
                gt_map[int(m.group(1))] = f
        if mask_dir:
            for f in list_npy(mask_dir):
                m = mask_pat.match(f)
                if m:
                    mask_map[int(m.group(1))] = f

        # Intersect IDs present in required modalities
        common_ids = sorted(set(ma_map) & set(gt_map))
        if mask_dir:
            common_ids = sorted(set(common_ids) & set(mask_map))

        if not common_ids:
            raise RuntimeError("No matched MA/GT (and mask) pairs found. Check names/dirs/patterns.")

    
        # else 'all': keep all

        if not common_ids:
            raise RuntimeError("No pairs after applying head_policy filter.")

        # Build pair tuples from the filtered IDs
        if mask_dir:
            pairs_all = [(ma_map[i], gt_map[i], mask_map[i]) for i in common_ids]
        else:
            pairs_all = [(ma_map[i], gt_map[i]) for i in common_ids]

        # Train/Val split only (reproducible)
        if val_size > 0:
            train, val = train_test_split(
                pairs_all,
                test_size=val_size,
                random_state=seed,
                shuffle=True,
            )
        else:
            train, val = pairs_all, []

        # Bookkeeping
        self.pairs_train = train
        self.pairs_val   = val
        self.pairs       = {'train': train, 'val': val}[split]

        # Summary
        min_id, max_id = (min(common_ids), max(common_ids)) if common_ids else (None, None)
        print(
            f"[CTDataset] kept={len(pairs_all)} | train={len(train)} | val={len(val)} "
            f"| ids=[{min_id}..{max_id}]"
            f"| window=[{self.hu_min}, {self.hu_max}] | shape={self.image_shape}"
        )

    # --- I/O helpers ---
    def _load_npy(self, path: str) -> np.ndarray:
        """Load (H,W) float32 array from .npy."""
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        return arr

    def _clip_and_norm01(self, hu: np.ndarray) -> np.ndarray:
        """Clip to [hu_min, hu_max] and map to [0,1]."""
        hu = np.clip(hu, self.hu_min, self.hu_max)
        return (hu - self.hu_min) / self._dr

    # --- Dataset API ---
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]

        if self.mask_dir and len(item) == 3:
            ma_file, gt_file, mask_file = item
            mask_path = os.path.join(self.mask_dir, mask_file)
        else:
            ma_file, gt_file = item[:2]
            mask_path = None

        ma_path = os.path.join(self.ma_dir, ma_file)
        gt_path = os.path.join(self.gt_dir, gt_file)

        # 1) Load raw HU
        ma = self._load_npy(ma_path)
        gt = self._load_npy(gt_path)

        # 2) Clip and 3) normalize to [0,1]
        ma01 = self._clip_and_norm01(ma)
        gt01 = self._clip_and_norm01(gt)

        # To tensors (1,H,W)
        x = torch.from_numpy(ma01).unsqueeze(0).float()
        y = torch.from_numpy(gt01).unsqueeze(0).float()

        # Optional mask (returned but NOT applied here)
        if mask_path is not None:
            mask_np = self._load_npy(mask_path)
            # force binary {0,1} if needed
            if mask_np.max() > 1.0:
                mask_np = (mask_np > 0).astype(np.float32)
            m = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            m = None

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            if m is not None:
                m = self.transform(m)

        return (x, y, m) if m is not None else (x, y)




