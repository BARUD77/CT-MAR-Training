"""Shared dataset construction for ALL baselines.

Builds the train/val CTMetalArtifactDataset with exactly the arguments used by the reported models
(see CLAUDE.md). Only directory paths are configurable. The split is produced by aapm_dataset.py
(sklearn train_test_split, stratified by region, random_state=seed) over the (region, id) keys
present in ALL given directories, so pass the same set of directories (MA, GT, LI, mask) as the
reference runs to get the identical split.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aapm_dataset import CTMetalArtifactDataset  # noqa: E402

# Fixed dataset arguments (CLAUDE.md). Do not change per baseline.
HU_MIN = -1024.0
HU_MAX = 3072.0
CLIP_HU = True
OUTPUT_SPACE = "norm"
REGION_POLICY = "all"
SEED = 42
VAL_SIZE = 0.1

DATASET_KWARGS = dict(
    hu_min=HU_MIN,
    hu_max=HU_MAX,
    clip_hu=CLIP_HU,
    output_space=OUTPUT_SPACE,
    region_policy=REGION_POLICY,
    seed=SEED,
    val_size=VAL_SIZE,
)


def build_datasets(ma_dir, gt_dir, li_dir=None, mask_dir=None):
    """Return (train_ds, val_ds) CTMetalArtifactDataset instances.

    Samples are tuples in the dataset's order: (ma, gt[, mask][, li]), all (1,H,W) float tensors in
    [0,1] (mask is binary, 1 = metal). Use `unpack_batch` to get named fields.
    """
    common = dict(ma_dir=ma_dir, gt_dir=gt_dir, li_dir=li_dir, mask_dir=mask_dir, **DATASET_KWARGS)
    train_ds = CTMetalArtifactDataset(split="train", **common)
    val_ds = CTMetalArtifactDataset(split="val", **common)
    return train_ds, val_ds


def unpack_batch(batch, has_mask, has_li):
    """Map a dataset sample/batch tuple to a dict with keys ma, gt, mask, li (None if absent).

    Unlike the trainer's binary-value heuristic, this uses the known dataset configuration.
    """
    batch = list(batch)
    out = {"ma": batch[0], "gt": batch[1], "mask": None, "li": None}
    idx = 2
    if has_mask:
        out["mask"] = batch[idx]
        idx += 1
    if has_li:
        out["li"] = batch[idx]
        idx += 1
    if idx != len(batch):
        raise ValueError(f"Batch has {len(batch)} items, expected {idx} "
                         f"(has_mask={has_mask}, has_li={has_li}).")
    return out
