# import os
# import re
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split

# class CTMetalArtifactDataset(Dataset):
#     def __init__(self, ma_dir, li_dir, gt_dir, split='train', val_size=0.1, test_size=0.1, seed=42, transform=None):
#         self.ma_dir = ma_dir
#         self.li_dir = li_dir
#         self.gt_dir = gt_dir
#         self.transform = transform
#         self.image_shape = (512, 512)
#         self.global_min = -7387.15771484375
#         self.global_max = 65204.90625

#         # Helper: extract image ID
#         def extract_id(filename, pattern):
#             match = re.search(pattern, filename)
#             return match.group(1) if match else None

#         # Patterns for file matching
#         ma_pattern = r'metalart_img(\d+)_'
#         li_pattern = r'li_img(\d+)_'
#         gt_pattern = r'Simulated_Image_(\d+)_'

#         # Index all files
#         # Only include 'training_body_' files
#         # Only include 'training_body_' files
#         ma_files = {
#             extract_id(f, ma_pattern): f
#             for f in os.listdir(ma_dir)
#             if f.startswith('training_body_') and f.endswith('.npy')
#         }
#         li_files = {
#             extract_id(f, li_pattern): f
#             for f in os.listdir(li_dir)
#             if f.startswith('training_body_') and f.endswith('.npy')
#         }
#         gt_files = {
#             extract_id(f, gt_pattern): f
#             for f in os.listdir(gt_dir) if f.endswith('.npy')
#         }

#         # Match all three modalities
#         matched_ids = list(set(ma_files) & set(li_files) & set(gt_files))
#         matched_ids.sort(key=lambda x: int(x))  # sort by integer ID
#         matched_triplets = [(ma_files[i], li_files[i], gt_files[i]) for i in matched_ids]

#         # Split reproducibly
#         train_val_ids, test_ids = train_test_split(matched_triplets, test_size=test_size, random_state=seed)
#         train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size / (1 - test_size), random_state=seed)

#         if split == 'train':
#             self.pairs = train_ids
#         elif split == 'val':
#             self.pairs = val_ids
#         elif split == 'test':
#             self.pairs = test_ids
#         else:
#             raise ValueError("split must be one of ['train', 'val', 'test']")

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         ma_file, li_file, gt_file = self.pairs[idx]

#         ma_path = os.path.join(self.ma_dir, ma_file)
#         li_path = os.path.join(self.li_dir, li_file)
#         gt_path = os.path.join(self.gt_dir, gt_file)

#         # Load .npy files
#         ma_image = np.load(ma_path).reshape(self.image_shape)
#         li_image = np.load(li_path).reshape(self.image_shape)
#         gt_image = np.load(gt_path).reshape(self.image_shape)

#         # Normalize
#         ma_image = (ma_image - self.global_min) / (self.global_max - self.global_min)
#         li_image = (li_image - self.global_min) / (self.global_max - self.global_min)
#         gt_image = (gt_image - self.global_min) / (self.global_max - self.global_min)

#         ma_image = np.clip(ma_image, 0, 1)
#         li_image = np.clip(li_image, 0, 1)
#         gt_image = np.clip(gt_image, 0, 1)

#         # Stack MA and LI into 2 channels
#         input_tensor = np.stack([ma_image, li_image], axis=0)
#         target_tensor = np.expand_dims(gt_image, axis=0)

#         input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
#         target_tensor = torch.tensor(target_tensor, dtype=torch.float32)

#         if self.transform:
#             input_tensor = self.transform(input_tensor)
#             target_tensor = self.transform(target_tensor)

#         return input_tensor, target_tensor


# import os
# import re
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split

# class CTMetalArtifactDataset(Dataset):
#     def __init__(
#         self,
#         ma_dir,
#         li_dir,
#         gt_dir,
#         split='train',
#         val_size=0.1,
#         test_size=0.1,
#         seed=42,
#         transform=None,
#         body_offset=12373,      # last BODY global id; HEAD global ids start at body_offset+1
#         include_body=True,
#         include_head=True,
#         match_mode='id'         # 'id' (exact id match) or 'serial' (order-based fallback)
#     ):
#         """
#         Filenames:
#           MA: training_{body|head}_metalart_img{LOCAL_ID}_*.npy
#           LI: training_{body|head}_li_img{LOCAL_ID}_*.npy
#           GT: Simulated_Image_{GLOBAL_ID}_*.npy  (GLOBAL_ID: body: LOCAL_ID, head: body_offset+LOCAL_ID)
#         """
#         assert include_body or include_head, "At least one of include_body/include_head must be True."
#         assert match_mode in ('id', 'serial'), "match_mode must be 'id' or 'serial'."

#         self.ma_dir = ma_dir
#         self.li_dir = li_dir
#         self.gt_dir = gt_dir
#         self.transform = transform
#         self.image_shape = (512, 512)
#         self.global_min = -7387.15771484375
#         self.global_max = 65204.90625
#         self.body_offset = int(body_offset)
#         self.match_mode = match_mode

#         # ---- Regex ----
#         ma_pat_local = re.compile(r'^training_(body|head)_metalart_img(\d+)_')
#         li_pat_local = re.compile(r'^training_(body|head)_li_img(\d+)_')
#         gt_pat_global = re.compile(r'^Simulated_Image_(\d+)(?:_|\.npy)')

#         def list_npy(d): return [f for f in os.listdir(d) if f.endswith('.npy')]

#         # ---- Collect per-subset LOCAL maps for MA/LI: subset -> {local_id: filename}
#         def collect_local_maps(dir_path, pattern):
#             maps = {'body': {}, 'head': {}}
#             for f in list_npy(dir_path):
#                 m = pattern.match(f)
#                 if not m:
#                     continue
#                 subset, local_id = m.group(1), int(m.group(2))
#                 if subset == 'body' and include_body:
#                     maps['body'][local_id] = f
#                 elif subset == 'head' and include_head:
#                     maps['head'][local_id] = f
#             return maps

#         ma_local = collect_local_maps(self.ma_dir, ma_pat_local)
#         li_local = collect_local_maps(self.li_dir, li_pat_local)

#         # ---- Build GLOBAL maps (by computed global_id) if using id-mode
#         def to_global_id(subset, local_id):
#             return local_id if subset == 'body' else self.body_offset + local_id

#         if match_mode == 'id':
#             ma_map = {}
#             li_map = {}
#             for subset in ('body', 'head'):
#                 for lid, f in ma_local[subset].items():
#                     gid = to_global_id(subset, lid)
#                     ma_map[gid] = f
#                 for lid, f in li_local[subset].items():
#                     gid = to_global_id(subset, lid)
#                     li_map[gid] = f

#             # GT global map
#             gt_map = {}
#             for f in list_npy(self.gt_dir):
#                 m = gt_pat_global.match(f)
#                 if not m:
#                     continue
#                 gid = int(m.group(1))
#                 # keep only gids we could possibly match to (optional, but cleaner)
#                 if (include_body and 1 <= gid <= self.body_offset) or (include_head and gid > self.body_offset):
#                     gt_map[gid] = f

#             # Intersect by exact global ID
#             common_ids = sorted(set(ma_map).intersection(li_map).intersection(gt_map))
#             matched_triplets = [(ma_map[i], li_map[i], gt_map[i]) for i in common_ids]

#             body_kept = sum(1 for i in common_ids if 1 <= i <= self.body_offset)
#             head_kept = sum(1 for i in common_ids if i > self.body_offset)

#         else:  # match_mode == 'serial'
#             # 1) For each subset, intersect MA and LI by LOCAL id (ensures MA–LI alignment)
#             # 2) Sort by LOCAL id (stable order)
#             # 3) Build GT list for the subset sorted by GLOBAL id
#             # 4) Zip/truncate MA–LI pairs with GT list
#             # 5) Concatenate body+head results
#             matched_triplets = []
#             body_kept = head_kept = 0

#             # Build GT global dict once
#             gt_global = {}
#             for f in list_npy(self.gt_dir):
#                 m = gt_pat_global.match(f)
#                 if not m:
#                     continue
#                 gid = int(m.group(1))
#                 gt_global[gid] = f

#             for subset in ('body', 'head'):
#                 if subset == 'body' and not include_body:
#                     continue
#                 if subset == 'head' and not include_head:
#                     continue

#                 common_local = sorted(set(ma_local[subset]).intersection(li_local[subset]))
#                 ma_li_pairs_sorted = [(ma_local[subset][i], li_local[subset][i]) for i in common_local]

#                 if subset == 'body':
#                     g_candidates = sorted([gid for gid in gt_global if 1 <= gid <= self.body_offset])
#                 else:
#                     g_candidates = sorted([gid for gid in gt_global if gid > self.body_offset])

#                 gt_list_sorted = [gt_global[gid] for gid in g_candidates]
#                 m = min(len(ma_li_pairs_sorted), len(gt_list_sorted))

#                 for k in range(m):
#                     ma_f, li_f = ma_li_pairs_sorted[k]
#                     gt_f = gt_list_sorted[k]
#                     matched_triplets.append((ma_f, li_f, gt_f))

#                 if subset == 'body':
#                     body_kept = m
#                 else:
#                     head_kept = m

#         if not matched_triplets:
#             raise RuntimeError(
#                 "No matched MA/LI/GT triplets found. Check directories, patterns, body_offset, and match_mode."
#             )

#         # ---- Train/Val/Test split (reproducible)
#         train_val, test = train_test_split(
#             matched_triplets, test_size=test_size, random_state=seed, shuffle=True
#         )
#         eff_val = 0 if val_size == 0 else val_size / (1 - test_size)
#         if eff_val > 0:
#             train, val = train_test_split(train_val, test_size=eff_val, random_state=seed, shuffle=True)
#         else:
#             train, val = train_val, []

#         if split == 'train':
#             self.pairs = train
#         elif split == 'val':
#             self.pairs = val
#         elif split == 'test':
#             self.pairs = test
#         else:
#             raise ValueError("split must be one of ['train','val','test']")

#         # Diagnostics
#         total_kept = len(matched_triplets)
#         print(f"[CTDataset] match_mode={match_mode} | kept: total={total_kept}, body={body_kept}, head={head_kept} "
#               f"| splits -> train={len(train)}, val={len(val)}, test={len(test)}")

#     def __len__(self):
#         return len(self.pairs)

#     def _load_npy_img(self, path):
#         arr = np.load(path)
#         if arr.ndim == 3 and arr.shape[-1] == 1:
#             arr = arr[..., 0]
#         arr = np.squeeze(arr)
#         assert tuple(arr.shape) == self.image_shape, f"Unexpected shape {arr.shape} for {path}"
#         return arr

#     def __getitem__(self, idx):
#         ma_file, li_file, gt_file = self.pairs[idx]

#         ma_path = os.path.join(self.ma_dir, ma_file)
#         li_path = os.path.join(self.li_dir, li_file)
#         gt_path = os.path.join(self.gt_dir, gt_file)

#         ma_image = self._load_npy_img(ma_path)
#         li_image = self._load_npy_img(li_path)
#         gt_image = self._load_npy_img(gt_path)

#         # Normalize to [0,1]
#         denom = (self.global_max - self.global_min)
#         ma_image = np.clip((ma_image - self.global_min) / denom, 0, 1)
#         li_image = np.clip((li_image - self.global_min) / denom, 0, 1)
#         gt_image = np.clip((gt_image - self.global_min) / denom, 0, 1)

#         x = torch.tensor(np.stack([ma_image, li_image], axis=0), dtype=torch.float32)  # (2,H,W)
#         y = torch.tensor(np.expand_dims(gt_image, axis=0), dtype=torch.float32)        # (1,H,W)

#         if self.transform:
#             x = self.transform(x)
#             y = self.transform(y)
#         return x, y


# dataset.py for .npy mode

# import os
# import re
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split

# class CTMetalArtifactDataset(Dataset):
#     def __init__(
#         self,
#         ma_dir,
#         li_dir,
#         gt_dir,
#         split='train',
#         val_size=0.1,
#         test_size=0.1,
#         seed=42,
#         transform=None,
#         body_offset=12373,      # last BODY global id; HEAD global ids start at body_offset+1
#         include_body=True,
#         include_head=True,
#         match_mode='id',        # 'id' (exact id match) or 'serial' (order-based)
#         input_mode='ma_li'      # 'ma_li' (2-ch input) or 'ma' (MA-only, 1-ch input)
#     ):
#         """
#         Filenames:
#           MA: training_{body|head}_metalart_img{LOCAL_ID}_*.npy
#           LI: training_{body|head}_li_img{LOCAL_ID}_*.npy
#           GT: Simulated_Image_{GLOBAL_ID}_*.npy
#               (GLOBAL_ID: body: LOCAL_ID, head: body_offset + LOCAL_ID)

#         input_mode:
#           - 'ma_li': model input = [MA, LI]  (2 channels)
#           - 'ma'   : model input = [MA]      (1 channel)  <-- LI not required
#         """
#         assert include_body or include_head, "At least one of include_body/include_head must be True."
#         assert match_mode in ('id', 'serial')
#         assert input_mode in ('ma_li', 'ma')

#         self.ma_dir = ma_dir
#         self.li_dir = li_dir
#         self.gt_dir = gt_dir
#         self.transform = transform
#         self.image_shape = (512, 512)
#         self.global_min = -7387.15771484375
#         self.global_max = 65204.90625
#         self.body_offset = int(body_offset)
#         self.match_mode = match_mode
#         self.input_mode = input_mode
#         self.require_li = (input_mode == 'ma_li')

#         # ---- Regex ----
#         ma_pat_local = re.compile(r'^training_(body|head)_metalart_img(\d+)_')
#         li_pat_local = re.compile(r'^training_(body|head)_li_img(\d+)_')
#         gt_pat_global = re.compile(r'^Simulated_Image_(\d+)(?:_|\.npy)')

#         def list_npy(d): 
#             return [f for f in os.listdir(d) if f.endswith('.npy')]

#         # ---- Collect per-subset LOCAL maps for MA/LI: subset -> {local_id: filename}
#         def collect_local_maps(dir_path, pattern, want_li=True):
#             maps = {'body': {}, 'head': {}}
#             if (not want_li) and (pattern is li_pat_local):
#                 return maps  # don't fill if LI not required
#             for f in list_npy(dir_path):
#                 m = pattern.match(f)
#                 if not m:
#                     continue
#                 subset, local_id = m.group(1), int(m.group(2))
#                 if subset == 'body' and include_body:
#                     maps['body'][local_id] = f
#                 elif subset == 'head' and include_head:
#                     maps['head'][local_id] = f
#             return maps

#         ma_local = collect_local_maps(self.ma_dir, ma_pat_local, want_li=True)
#         li_local = collect_local_maps(self.li_dir, li_pat_local, want_li=self.require_li)

#         # ---- Helper to get global id
#         def to_global_id(subset, local_id):
#             return local_id if subset == 'body' else self.body_offset + local_id

#         # ---- Build matches
#         if match_mode == 'id':
#             # Build MA global map
#             ma_map = {}
#             for subset in ('body', 'head'):
#                 for lid, f in ma_local[subset].items():
#                     ma_map[to_global_id(subset, lid)] = f

#             # Build LI global map only if required
#             li_map = {}
#             if self.require_li:
#                 for subset in ('body', 'head'):
#                     for lid, f in li_local[subset].items():
#                         li_map[to_global_id(subset, lid)] = f

#             # GT global map (filter to chosen subsets)
#             gt_map = {}
#             for f in list_npy(self.gt_dir):
#                 m = gt_pat_global.match(f)
#                 if not m:
#                     continue
#                 gid = int(m.group(1))
#                 if (include_body and 1 <= gid <= self.body_offset) or (include_head and gid > self.body_offset):
#                     gt_map[gid] = f

#             if self.require_li:
#                 common_ids = sorted(set(ma_map).intersection(li_map).intersection(gt_map))
#                 matched = [(ma_map[i], li_map[i], gt_map[i]) for i in common_ids]
#             else:
#                 common_ids = sorted(set(ma_map).intersection(gt_map))
#                 # store a placeholder for LI (None) so __getitem__ knows to skip loading it
#                 matched = [(ma_map[i], None, gt_map[i]) for i in common_ids]

#             matched_triplets = matched
#             body_kept = sum(1 for i in common_ids if 1 <= i <= self.body_offset)
#             head_kept = sum(1 for i in common_ids if i > self.body_offset)

#         else:  # match_mode == 'serial'
#             matched_triplets = []
#             body_kept = head_kept = 0

#             # Build GT global dict once
#             gt_global = {}
#             for f in list_npy(self.gt_dir):
#                 m = gt_pat_global.match(f)
#                 if not m:
#                     continue
#                 gt_global[int(m.group(1))] = f

#             for subset in ('body', 'head'):
#                 if subset == 'body' and not include_body:
#                     continue
#                 if subset == 'head' and not include_head:
#                     continue

#                 # MA ids
#                 ma_ids = sorted(ma_local[subset].keys())
#                 if self.require_li:
#                     # Only pairs that exist in both MA and LI
#                     common_local = sorted(set(ma_local[subset]).intersection(li_local[subset]))
#                     ma_li_pairs_sorted = [(ma_local[subset][i], li_local[subset][i]) for i in common_local]
#                 else:
#                     # MA only
#                     ma_li_pairs_sorted = [(ma_local[subset][i], None) for i in ma_ids]

#                 # GT list for this subset, sorted by GLOBAL id
#                 if subset == 'body':
#                     g_candidates = sorted([gid for gid in gt_global if 1 <= gid <= self.body_offset])
#                 else:
#                     g_candidates = sorted([gid for gid in gt_global if gid > self.body_offset])

#                 gt_list_sorted = [gt_global[gid] for gid in g_candidates]

#                 mlen = min(len(ma_li_pairs_sorted), len(gt_list_sorted))
#                 for k in range(mlen):
#                     ma_f, li_f = ma_li_pairs_sorted[k]
#                     gt_f = gt_list_sorted[k]
#                     matched_triplets.append((ma_f, li_f, gt_f))

#                 if subset == 'body':
#                     body_kept = mlen
#                 else:
#                     head_kept = mlen

#         if not matched_triplets:
#             raise RuntimeError("No matched samples. Check dirs/patterns/body_offset/match_mode/input_mode.")

#         # ---- Train/Val/Test split (reproducible)
#         train_val, test = train_test_split(matched_triplets, test_size=test_size, random_state=seed, shuffle=True)
#         eff_val = 0 if val_size == 0 else val_size / (1 - test_size)
#         if eff_val > 0:
#             train, val = train_test_split(train_val, test_size=eff_val, random_state=seed, shuffle=True)
#         else:
#             train, val = train_val, []

#         if split == 'train':
#             self.pairs = train
#         elif split == 'val':
#             self.pairs = val
#         elif split == 'test':
#             self.pairs = test
#         else:
#             raise ValueError("split must be one of ['train','val','test']")

#         total_kept = len(matched_triplets)
#         print(f"[CTDataset] match_mode={match_mode} input_mode={input_mode} | kept: total={total_kept}, "
#               f"body={body_kept}, head={head_kept} | splits -> train={len(train)}, val={len(val)}, test={len(test)}")

#     def __len__(self):
#         return len(self.pairs)

#     def _load_npy_img(self, path):
#         arr = np.load(path)
#         if arr.ndim == 3 and arr.shape[-1] == 1:
#             arr = arr[..., 0]
#         arr = np.squeeze(arr)
#         assert tuple(arr.shape) == self.image_shape, f"Unexpected shape {arr.shape} for {path}"
#         return arr

#     def __getitem__(self, idx):
#         ma_file, li_file, gt_file = self.pairs[idx]

#         ma_path = os.path.join(self.ma_dir, ma_file)
#         gt_path = os.path.join(self.gt_dir, gt_file)

#         ma_image = self._load_npy_img(ma_path)
#         gt_image = self._load_npy_img(gt_path)

#         # Normalize to [0,1]
#         denom = (self.global_max - self.global_min)
#         ma_image = np.clip((ma_image - self.global_min) / denom, 0, 1)
#         gt_image = np.clip((gt_image - self.global_min) / denom, 0, 1)

#         if self.input_mode == 'ma_li':
#             assert li_file is not None, "input_mode='ma_li' requires LI files."
#             li_path = os.path.join(self.li_dir, li_file)
#             li_image = self._load_npy_img(li_path)
#             li_image = np.clip((li_image - self.global_min) / denom, 0, 1)
#             x_np = np.stack([ma_image, li_image], axis=0)   # (2,H,W)
#         else:
#             x_np = np.expand_dims(ma_image, axis=0)         # (1,H,W)

#         y_np = np.expand_dims(gt_image, axis=0)             # (1,H,W)

#         x = torch.tensor(x_np, dtype=torch.float32)
#         y = torch.tensor(y_np, dtype=torch.float32)
#         if self.transform:
#             x = self.transform(x)
#             y = self.transform(y)
#         return x, y


import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CTMetalArtifactDataset(Dataset):
    """
    MA→GT pairs from .raw (float32) images of size 512x512.
    Pipeline:
      1) load raw HU
      2) clip to [hu_min, hu_max]
      3) normalize to [0,1]
    Returns:
      x: (1, H, W)  normalized MA  in [0,1]
      y: (1, H, W)  normalized GT  in [0,1]
      m: (1, H, W)  mask (0/1 float) if mask_dir is given, else None
    NOTE: mask is returned but NOT applied/multiplied here.
    """

    def __init__(
        self,
        ma_dir: str,
        gt_dir: str,
        mask_dir: str | None = None,   # optional
        split: str = 'train',
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
        image_shape: tuple[int, int] = (512, 512),
        hu_min: float = -1024.0,
        hu_max: float = 3072.0,        # global window for clipping/normalization
        transform=None,
    ):
        assert split in ('train', 'val', 'test')
        assert hu_max > hu_min

        self.ma_dir   = ma_dir
        self.gt_dir   = gt_dir
        self.mask_dir = mask_dir
        self.split    = split
        self.transform = transform

        self.image_shape = tuple(image_shape)
        self._numel = self.image_shape[0] * self.image_shape[1]

        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self._dr = self.hu_max - self.hu_min  # data_range for [0,1] mapping

        # filename patterns (adjust if yours differ)
        ma_pat   = re.compile(r'^Input_Image_(\d+)_512x512(?:\.npy)?$',      re.IGNORECASE)
        gt_pat   = re.compile(r'^Simulated_Image_(\d+)_512x512(?:\.npy)?$',  re.IGNORECASE)
        mask_pat = re.compile(r'^.*mask.*_(\d+)_512x512(?:\.raw)?$',         re.IGNORECASE)

        def list_npy(d):
            return [f for f in os.listdir(d) if f.lower().endswith('.npy')]

        # Build ID→filename maps
        ma_map, gt_map, mask_map = {}, {}, {}
        for f in list_npy(ma_dir):
            m = ma_pat.match(f)
            if m: ma_map[int(m.group(1))] = f
        for f in list_npy(gt_dir):
            m = gt_pat.match(f)
            if m: gt_map[int(m.group(1))] = f
        if mask_dir:
            for f in list_npy(mask_dir):
                m = mask_pat.match(f)
                if m: mask_map[int(m.group(1))] = f

        # Intersect IDs
        common_ids = sorted(set(ma_map) & set(gt_map))
        if mask_dir:
            common_ids = sorted(set(common_ids) & set(mask_map))

        if not common_ids:
            raise RuntimeError("No matched MA/GT (and mask) pairs found. Check names/dirs.")

        # Build pair tuples
        if mask_dir:
            pairs = [(ma_map[i], gt_map[i], mask_map[i]) for i in common_ids]
        else:
            pairs = [(ma_map[i], gt_map[i]) for i in common_ids]

        # Split (reproducible)
        train_val, test = train_test_split(pairs, test_size=test_size,
                                           random_state=seed, shuffle=True)
        eff_val = 0 if val_size == 0 else val_size / (1 - test_size)
        if eff_val > 0:
            train, val = train_test_split(train_val, test_size=eff_val,
                                          random_state=seed, shuffle=True)
        else:
            train, val = train_val, []

        self.pairs_train = train
        self.pairs_val   = val
        self.pairs_test  = test

        self.pairs = {'train': train, 'val': val, 'test': test}[split]

        print(f"[CTDataset] total={len(pairs)} | train={len(train)} "
              f"| val={len(val)} | test={len(test)} | window=[{self.hu_min}, {self.hu_max}]")

    def __len__(self):
        return len(self.pairs)

    def _load_npy(self, path: str) -> np.ndarray:
        """Read .raw float32 file and reshape to (H,W)."""
        arr = np.load(path)
        return arr

    def _clip_and_norm01(self, hu: np.ndarray) -> np.ndarray:
        """Clip to [hu_min, hu_max] and map to [0,1]."""
        hu = np.clip(hu, self.hu_min, self.hu_max)
        return (hu - self.hu_min) / self._dr

    def __getitem__(self, idx: int):
        item = self.pairs[idx]

        if self.mask_dir:
            ma_file, gt_file, mask_file = item
            mask_path = os.path.join(self.mask_dir, mask_file)
        else:
            ma_file, gt_file = item
            mask_path = None

        ma_path = os.path.join(self.ma_dir, ma_file)
        gt_path = os.path.join(self.gt_dir, gt_file)

        # 1) Load raw HU
        ma = self._load_npy(ma_path)
        gt = self._load_npy(gt_path)

        # 2) Clip to global HU window
        # 3) Normalize to [0,1]
        ma01 = self._clip_and_norm01(ma)
        gt01 = self._clip_and_norm01(gt)

        # To tensors (1,H,W)
        x = torch.from_numpy(ma01).unsqueeze(0)  # (1,H,W), float32
        y = torch.from_numpy(gt01).unsqueeze(0)  # (1,H,W), float32

        # Optional mask (returned but NOT applied)
        m = None
        if mask_path:
            mask_np = self._load_npy(mask_path)
            # Ensure binary {0,1} if mask stored as 0/255 or HU-like:
            if mask_np.max() > 1.0:
                mask_np = (mask_np > 0).astype(np.float32)
            m = torch.from_numpy(mask_np).unsqueeze(0)  # (1,H,W)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            if m is not None:
                m = self.transform(m)

        # Return without multiplying by mask (you’ll do it in metrics)
        return (x, y, m) if m is not None else (x, y)



