import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CTMetalArtifactDataset(Dataset):
    def __init__(self, ma_dir, li_dir, gt_dir, split='train', val_size=0.1, test_size=0.1, seed=42, transform=None):
        self.ma_dir = ma_dir
        self.li_dir = li_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.image_shape = (512, 512)
        self.global_min = -7387.15771484375
        self.global_max = 65204.90625

        # Helper: extract image ID
        def extract_id(filename, pattern):
            match = re.search(pattern, filename)
            return match.group(1) if match else None

        # Patterns for file matching
        ma_pattern = r'metalart_img(\d+)_'
        li_pattern = r'li_img(\d+)_'
        gt_pattern = r'Simulated_Image_(\d+)_'

        # Index all files
        # Only include 'training_body_' files
        # Only include 'training_body_' files
        ma_files = {
            extract_id(f, ma_pattern): f
            for f in os.listdir(ma_dir)
            if f.startswith('training_body_') and f.endswith('.npy')
        }
        li_files = {
            extract_id(f, li_pattern): f
            for f in os.listdir(li_dir)
            if f.startswith('training_body_') and f.endswith('.npy')
        }
        gt_files = {
            extract_id(f, gt_pattern): f
            for f in os.listdir(gt_dir) if f.endswith('.npy')
        }

        # Match all three modalities
        matched_ids = list(set(ma_files) & set(li_files) & set(gt_files))
        matched_ids.sort(key=lambda x: int(x))  # sort by integer ID
        matched_triplets = [(ma_files[i], li_files[i], gt_files[i]) for i in matched_ids]

        # Split reproducibly
        train_val_ids, test_ids = train_test_split(matched_triplets, test_size=test_size, random_state=seed)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size / (1 - test_size), random_state=seed)

        if split == 'train':
            self.pairs = train_ids
        elif split == 'val':
            self.pairs = val_ids
        elif split == 'test':
            self.pairs = test_ids
        else:
            raise ValueError("split must be one of ['train', 'val', 'test']")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ma_file, li_file, gt_file = self.pairs[idx]

        ma_path = os.path.join(self.ma_dir, ma_file)
        li_path = os.path.join(self.li_dir, li_file)
        gt_path = os.path.join(self.gt_dir, gt_file)

        # Load .npy files
        ma_image = np.load(ma_path).reshape(self.image_shape)
        li_image = np.load(li_path).reshape(self.image_shape)
        gt_image = np.load(gt_path).reshape(self.image_shape)

        # Normalize
        ma_image = (ma_image - self.global_min) / (self.global_max - self.global_min)
        li_image = (li_image - self.global_min) / (self.global_max - self.global_min)
        gt_image = (gt_image - self.global_min) / (self.global_max - self.global_min)

        ma_image = np.clip(ma_image, 0, 1)
        li_image = np.clip(li_image, 0, 1)
        gt_image = np.clip(gt_image, 0, 1)

        # Stack MA and LI into 2 channels
        input_tensor = np.stack([ma_image, li_image], axis=0)
        target_tensor = np.expand_dims(gt_image, axis=0)

        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float32)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor
