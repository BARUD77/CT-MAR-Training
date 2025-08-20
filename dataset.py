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


import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CTMetalArtifactDataset(Dataset):
    def __init__(
        self,
        ma_dir,
        li_dir,
        gt_dir,
        split='train',
        val_size=0.1,
        test_size=0.1,
        seed=42,
        transform=None,
        body_offset=12373,      # last BODY global id; HEAD global ids start at body_offset+1
        include_body=True,
        include_head=True,
        match_mode='id'         # 'id' (exact id match) or 'serial' (order-based fallback)
    ):
        """
        Filenames:
          MA: training_{body|head}_metalart_img{LOCAL_ID}_*.npy
          LI: training_{body|head}_li_img{LOCAL_ID}_*.npy
          GT: Simulated_Image_{GLOBAL_ID}_*.npy  (GLOBAL_ID: body: LOCAL_ID, head: body_offset+LOCAL_ID)
        """
        assert include_body or include_head, "At least one of include_body/include_head must be True."
        assert match_mode in ('id', 'serial'), "match_mode must be 'id' or 'serial'."

        self.ma_dir = ma_dir
        self.li_dir = li_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.image_shape = (512, 512)
        self.global_min = -7387.15771484375
        self.global_max = 65204.90625
        self.body_offset = int(body_offset)
        self.match_mode = match_mode

        # ---- Regex ----
        ma_pat_local = re.compile(r'^training_(body|head)_metalart_img(\d+)_')
        li_pat_local = re.compile(r'^training_(body|head)_li_img(\d+)_')
        gt_pat_global = re.compile(r'^Simulated_Image_(\d+)(?:_|\.npy)')

        def list_npy(d): return [f for f in os.listdir(d) if f.endswith('.npy')]

        # ---- Collect per-subset LOCAL maps for MA/LI: subset -> {local_id: filename}
        def collect_local_maps(dir_path, pattern):
            maps = {'body': {}, 'head': {}}
            for f in list_npy(dir_path):
                m = pattern.match(f)
                if not m:
                    continue
                subset, local_id = m.group(1), int(m.group(2))
                if subset == 'body' and include_body:
                    maps['body'][local_id] = f
                elif subset == 'head' and include_head:
                    maps['head'][local_id] = f
            return maps

        ma_local = collect_local_maps(self.ma_dir, ma_pat_local)
        li_local = collect_local_maps(self.li_dir, li_pat_local)

        # ---- Build GLOBAL maps (by computed global_id) if using id-mode
        def to_global_id(subset, local_id):
            return local_id if subset == 'body' else self.body_offset + local_id

        if match_mode == 'id':
            ma_map = {}
            li_map = {}
            for subset in ('body', 'head'):
                for lid, f in ma_local[subset].items():
                    gid = to_global_id(subset, lid)
                    ma_map[gid] = f
                for lid, f in li_local[subset].items():
                    gid = to_global_id(subset, lid)
                    li_map[gid] = f

            # GT global map
            gt_map = {}
            for f in list_npy(self.gt_dir):
                m = gt_pat_global.match(f)
                if not m:
                    continue
                gid = int(m.group(1))
                # keep only gids we could possibly match to (optional, but cleaner)
                if (include_body and 1 <= gid <= self.body_offset) or (include_head and gid > self.body_offset):
                    gt_map[gid] = f

            # Intersect by exact global ID
            common_ids = sorted(set(ma_map).intersection(li_map).intersection(gt_map))
            matched_triplets = [(ma_map[i], li_map[i], gt_map[i]) for i in common_ids]

            body_kept = sum(1 for i in common_ids if 1 <= i <= self.body_offset)
            head_kept = sum(1 for i in common_ids if i > self.body_offset)

        else:  # match_mode == 'serial'
            # 1) For each subset, intersect MA and LI by LOCAL id (ensures MA–LI alignment)
            # 2) Sort by LOCAL id (stable order)
            # 3) Build GT list for the subset sorted by GLOBAL id
            # 4) Zip/truncate MA–LI pairs with GT list
            # 5) Concatenate body+head results
            matched_triplets = []
            body_kept = head_kept = 0

            # Build GT global dict once
            gt_global = {}
            for f in list_npy(self.gt_dir):
                m = gt_pat_global.match(f)
                if not m:
                    continue
                gid = int(m.group(1))
                gt_global[gid] = f

            for subset in ('body', 'head'):
                if subset == 'body' and not include_body:
                    continue
                if subset == 'head' and not include_head:
                    continue

                common_local = sorted(set(ma_local[subset]).intersection(li_local[subset]))
                ma_li_pairs_sorted = [(ma_local[subset][i], li_local[subset][i]) for i in common_local]

                if subset == 'body':
                    g_candidates = sorted([gid for gid in gt_global if 1 <= gid <= self.body_offset])
                else:
                    g_candidates = sorted([gid for gid in gt_global if gid > self.body_offset])

                gt_list_sorted = [gt_global[gid] for gid in g_candidates]
                m = min(len(ma_li_pairs_sorted), len(gt_list_sorted))

                for k in range(m):
                    ma_f, li_f = ma_li_pairs_sorted[k]
                    gt_f = gt_list_sorted[k]
                    matched_triplets.append((ma_f, li_f, gt_f))

                if subset == 'body':
                    body_kept = m
                else:
                    head_kept = m

        if not matched_triplets:
            raise RuntimeError(
                "No matched MA/LI/GT triplets found. Check directories, patterns, body_offset, and match_mode."
            )

        # ---- Train/Val/Test split (reproducible)
        train_val, test = train_test_split(
            matched_triplets, test_size=test_size, random_state=seed, shuffle=True
        )
        eff_val = 0 if val_size == 0 else val_size / (1 - test_size)
        if eff_val > 0:
            train, val = train_test_split(train_val, test_size=eff_val, random_state=seed, shuffle=True)
        else:
            train, val = train_val, []

        if split == 'train':
            self.pairs = train
        elif split == 'val':
            self.pairs = val
        elif split == 'test':
            self.pairs = test
        else:
            raise ValueError("split must be one of ['train','val','test']")

        # Diagnostics
        total_kept = len(matched_triplets)
        print(f"[CTDataset] match_mode={match_mode} | kept: total={total_kept}, body={body_kept}, head={head_kept} "
              f"| splits -> train={len(train)}, val={len(val)}, test={len(test)}")

    def __len__(self):
        return len(self.pairs)

    def _load_npy_img(self, path):
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = np.squeeze(arr)
        assert tuple(arr.shape) == self.image_shape, f"Unexpected shape {arr.shape} for {path}"
        return arr

    def __getitem__(self, idx):
        ma_file, li_file, gt_file = self.pairs[idx]

        ma_path = os.path.join(self.ma_dir, ma_file)
        li_path = os.path.join(self.li_dir, li_file)
        gt_path = os.path.join(self.gt_dir, gt_file)

        ma_image = self._load_npy_img(ma_path)
        li_image = self._load_npy_img(li_path)
        gt_image = self._load_npy_img(gt_path)

        # Normalize to [0,1]
        denom = (self.global_max - self.global_min)
        ma_image = np.clip((ma_image - self.global_min) / denom, 0, 1)
        li_image = np.clip((li_image - self.global_min) / denom, 0, 1)
        gt_image = np.clip((gt_image - self.global_min) / denom, 0, 1)

        x = torch.tensor(np.stack([ma_image, li_image], axis=0), dtype=torch.float32)  # (2,H,W)
        y = torch.tensor(np.expand_dims(gt_image, axis=0), dtype=torch.float32)        # (1,H,W)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
