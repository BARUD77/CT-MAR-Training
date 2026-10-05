"""Reference copy of how trainer_wandb_unclipped.py builds its train/val datasets (git commit
cd22fab, main(), "Dataset / loaders" section), with argparse defaults filled in:
hu_min=-1024, hu_max=3072, no_clip=False, region_policy='all'. Used only to verify that
baselines/common/data.py produces the identical split. Do not edit.
"""
import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aapm_dataset import CTMetalArtifactDataset  # noqa: E402


def original_trainer_datasets(ma_dir, gt_dir, li_dir, mask_dir, uses_li=True):
    """uses_li mirrors `(args.input_mode == 'ma_li' or is_swin_fg or is_swin_3ch or is_swin_v2_spade)`."""
    args = SimpleNamespace(ma_dir=ma_dir, gt_dir=gt_dir, li_dir=li_dir, mask_dir=mask_dir,
                           hu_min=-1024.0, hu_max=3072.0, no_clip=False, region_policy="all")
    dataset_kwargs = dict(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        li_dir=args.li_dir if uses_li else None,
        mask_dir=args.mask_dir,
        split='train',
        hu_min=float(args.hu_min),
        hu_max=float(args.hu_max),
        clip_hu=(not args.no_clip),
        output_space="norm",
        region_policy=args.region_policy,
        seed=42,
        val_size=0.1
    )
    train_ds = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "train"})
    val_ds = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "val"})
    return train_ds, val_ds
