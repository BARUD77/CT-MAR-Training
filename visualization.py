import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Optional W&B
import wandb

# Your modules
from test_dataset import CTMetalArtifactDataset as CTDatasetLegacy
from models.swin_unet_mask_guided.vision_transformer import SwinUnet
from models.unet import UnetGenerator
from types import SimpleNamespace
import yaml

# ------------------------
# Helpers
# ------------------------
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)

def to_numpy_image(t: torch.Tensor):
    """
    t: (1,H,W) or (H,W) torch float tensor in [0,1]
    return: (H,W) numpy float in [0,1]
    """
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    return t.detach().cpu().float().clamp(0,1).numpy()

def save_triptych(ma, pr, gt, out_path, suptitle=None, cmap='gray'):
    """
    ma, pr, gt: numpy arrays (H,W) in [0,1]
    """
    H, W = ma.shape
    fig = plt.figure(figsize=(12, 4), dpi=150)

    ax1 = plt.subplot(1,3,1); ax1.imshow(ma, cmap=cmap, vmin=0, vmax=1); ax1.set_title("MA (artifact-affected)")
    ax2 = plt.subplot(1,3,2); ax2.imshow(pr, cmap=cmap, vmin=0, vmax=1); ax2.set_title("Pred (artifact-reduced)")
    ax3 = plt.subplot(1,3,3); ax3.imshow(gt, cmap=cmap, vmin=0, vmax=1); ax3.set_title("GT (ground truth)")

    for ax in (ax1, ax2, ax3):
        ax.axis('off')

    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def load_model(model_name, device, in_ch, config_path=None):
    if model_name == 'swinunet':
        if not config_path:
            raise ValueError("SwinUnet requires --config path to YAML file.")
        cfg = load_config(config_path)
        try:
            cfg.MODEL.SWIN.IN_CHANS = in_ch
        except AttributeError:
            pass
        # Ensure the network head produces a single-channel output for CT
        try:
            cfg.MODEL.SWIN.NUM_CLASSES = 1
        except AttributeError:
            # If cfg structure differs, fall back to passing num_classes explicitly
            pass
        model = SwinUnet(config=cfg, num_classes=1).to(device)
    elif model_name == 'unet':
        try:
            model = UnetGenerator(in_ch=in_ch).to(device)
        except TypeError:
            model = UnetGenerator().to(device)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    Robust checkpoint loader: accepts checkpoints saved as raw state_dict, or dicts
    containing keys like 'model_state', 'state_dict', or 'model'. Removes common
    'module.' prefixes, and falls back to strict=False or partial key copying.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # extract candidate state dict
    state = None
    if isinstance(ckpt, dict):
        for k in ("model_state", "state_dict", "model", "state"):
            if k in ckpt:
                state = ckpt[k]
                break
    if state is None:
        state = ckpt

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(state)}")

    # strip 'module.' prefix if present
    def _strip_module(d):
        if any(k.startswith("module.") for k in d.keys()):
            return {k.replace("module.", ""): v for k, v in d.items()}
        return d

    sd = _strip_module(state)

    # Try direct load
    try:
        model.load_state_dict(sd)
        print("Checkpoint loaded (exact match).")
        return
    except Exception as e:
        print(f"Exact load failed: {e}")

    # Try strict=False
    try:
        model.load_state_dict(sd, strict=False)
        print("Checkpoint loaded with strict=False (partial match).")
        return
    except Exception as e:
        print(f"Load with strict=False failed: {e}")

    # If model contains a wrapped attribute (e.g., model.swin_unet), try loading into that
    if hasattr(model, "swin_unet") and isinstance(sd, dict):
        try:
            model.swin_unet.load_state_dict(sd, strict=False)
            print("Loaded checkpoint into model.swin_unet with strict=False.")
            return
        except Exception as e:
            print(f"Loading into model.swin_unet failed: {e}")

    # As last resort, copy all compatible keys
    model_dict = model.state_dict()
    compatible = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
    if not compatible:
        raise RuntimeError("No compatible parameter shapes found in checkpoint to partially load.")
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    print(f"Partially loaded checkpoint ({len(compatible)} keys matched by shape).")

# ------------------------
# Main
# ------------------------
def main():
    p = argparse.ArgumentParser(description="Visualize MA vs Pred vs GT triptychs")
    # Data / eval choices
    p.add_argument('--mode', choices=['from_model', 'from_npy'], default='from_model',
                   help="from_model: run the model; from_npy: compose images from pre-saved .npy files")
    p.add_argument('--split', choices=['train','val','test'], default='val')
    p.add_argument('--ma_dir', type=str, help='Required if mode=from_model', default=None)
    p.add_argument('--gt_dir', type=str, help='Required if mode=from_model', default=None)
    p.add_argument('--li_dir', type=str, default=None, help='LI directory (optional). If provided, uses aapm_dataset with LI support')
    p.add_argument('--mask_dir', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_samples', type=int, default=15, help='Number of random samples to visualize')
    p.add_argument('--seed', type=int, default=123, help='Random seed for selecting samples')

    # Model / ckpt (for mode=from_model)
    p.add_argument('--model', choices=['unet','swinunet'], default='unet')
    p.add_argument('--in_ch', type=int, default=1)
    p.add_argument('--config', type=str, help='SwinUnet config path (if using swinunet)')
    p.add_argument('--ckpt', type=str, help='Path to .pt checkpoint')

    # From .npy folders (for mode=from_npy)
    p.add_argument('--npy_ma_glob', type=str, default=None, help='Glob for MA npy files, e.g. ./eval_outputs/*_ma.npy')
    p.add_argument('--npy_pred_glob', type=str, default=None, help='Glob for Pred npy files, e.g. ./eval_outputs/*_pred.npy')
    p.add_argument('--npy_gt_glob', type=str, default=None, help='Glob for GT npy files, e.g. ./eval_outputs/*_gt.npy')

    # HU window (dataset assumed to output [0,1] already; HU range here only for metadata/logging)
    p.add_argument('--hu_min', type=float, default=-1024.0)
    p.add_argument('--hu_max', type=float, default=3072.0)

    # Output / W&B
    p.add_argument('--out_dir', type=str, default='./viz_triptychs')
    p.add_argument('--project', type=str, default=None, help='W&B project (optional)')
    p.add_argument('--entity', type=str, default=None)
    p.add_argument('--run_name', type=str, default='viz')
    p.add_argument('--log_to_wandb', action='store_true')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional W&B
    if args.log_to_wandb and args.project:
        wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                   config=vars(args), dir=str(out_dir), reinit=True, save_code=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    if args.mode == 'from_model':
        if args.ma_dir is None or args.gt_dir is None:
            raise ValueError("--ma_dir and --gt_dir are required for mode=from_model")

        # dataset returns normalized [0,1] tensors (1,H,W)
        # If LI guidance is provided, prefer the aapm_dataset that returns (x,y,li)
        if args.li_dir is not None:
            from aapm_dataset import CTMetalArtifactDataset
            ds_full = CTMetalArtifactDataset(
                ma_dir=args.ma_dir, gt_dir=args.gt_dir, li_dir=args.li_dir,
                split=args.split, hu_min=args.hu_min, hu_max=args.hu_max
            )
        else:
            # legacy dataset returning (x,y) or (x,y,mask)
            CTMetalArtifactDataset = CTDatasetLegacy
            ds_full = CTMetalArtifactDataset(
                ma_dir=args.ma_dir, gt_dir=args.gt_dir, mask_dir=args.mask_dir,
                split=args.split, hu_min=args.hu_min, hu_max=args.hu_max
            )

        # --- Random, reproducible subset of size K ---
        k = min(args.num_samples, len(ds_full))
        sel_indices = rng.choice(len(ds_full), size=k, replace=False).tolist()
        print(f"[viz] Selected indices ({k} of {len(ds_full)}): {sorted(sel_indices)}")

        ds = Subset(ds_full, sel_indices)

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                            pin_memory=(device.type == 'cuda'))

        # load model
        model = load_model(args.model, device, in_ch=args.in_ch, config_path=args.config)
        if args.ckpt:
            try:
                load_checkpoint(model, args.ckpt, device)
            except Exception as e:
                print(f"Failed to load checkpoint {args.ckpt}: {e}")
                raise
        model.eval()

        saved = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # dataset may return (x,y), (x,y,mask) or (x,y,li)
                li = None
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, third = batch
                    # If user provided --li_dir we expect the third element to be LI
                    if args.li_dir is not None:
                        x, y, li = x, y, third
                    else:
                        x, y = x, y
                else:
                    x, y = batch

                x = x.to(device, non_blocking=True)   # (B,1,H,W)
                y = y.to(device, non_blocking=True)

                # If using swinunet + LI guidance, compute artifact_map from (li - ma)
                if args.model == 'swinunet' and li is not None:
                    li = li.to(device, non_blocking=True)
                    artifact_map = torch.relu(li - x)
                    # normalize per-sample to [0,1] (avoid division by zero)
                    B = artifact_map.size(0)
                    for ii in range(B):
                        mmax = float(artifact_map[ii].max())
                        if mmax > 0:
                            artifact_map[ii] = artifact_map[ii] / mmax
                    pred = model(x, artifact_map=artifact_map)
                else:
                    pred = model(x)
                pred_eval = torch.clamp(pred, 0, 1)   # ensure [0,1]
                gt_eval   = torch.clamp(y,   0, 1)

                B = x.size(0)
                for i in range(B):
                    if saved >= k:
                        break

                    ma_np  = to_numpy_image(x[i])          # (H,W)
                    pr_np  = to_numpy_image(pred_eval[i])
                    gt_np  = to_numpy_image(gt_eval[i])

                    # Use saved as the rank; optionally include source index
                    png_path = out_dir / f"{args.split}_rand_idx_{saved:05d}.png"
                    save_triptych(ma_np, pr_np, gt_np, png_path,
                                  suptitle=f"{args.split} rand#{saved}")

                    if args.log_to_wandb and args.project:
                        wandb.log({
                            "triptych": wandb.Image(png_path, caption=f"{args.split} rand#{saved}")
                        }, step=saved)

                    saved += 1

                if saved >= k:
                    break

        print(f"Saved {saved} triptychs to: {out_dir}")

    else:  # mode == 'from_npy'
        # Expect three matching globs for *_ma.npy, *_pred.npy, *_gt.npy
        if not (args.npy_ma_glob and args.npy_pred_glob and args.npy_gt_glob):
            raise ValueError("Please provide --npy_ma_glob, --npy_pred_glob, --npy_gt_glob for mode=from_npy")

        ma_files   = sorted(map(str, Path().glob(args.npy_ma_glob)))
        pred_files = sorted(map(str, Path().glob(args.npy_pred_glob)))
        gt_files   = sorted(map(str, Path().glob(args.npy_gt_glob)))

        n_all = min(len(ma_files), len(pred_files), len(gt_files))
        if n_all == 0:
            raise RuntimeError("No files found for the given globs.")

        k = min(args.num_samples, n_all)
        sel = sorted(rng.choice(n_all, size=k, replace=False).tolist())
        print(f"[viz] Selected file indices ({k} of {n_all}): {sel}")

        saved = 0
        for rank, idx in enumerate(sel):
            ma_np  = np.load(ma_files[idx]).astype(np.float32)
            pr_np  = np.load(pred_files[idx]).astype(np.float32)
            gt_np  = np.load(gt_files[idx]).astype(np.float32)

            # Ensure [0,1] clip if needed
            ma_np = np.clip(ma_np, 0, 1)
            pr_np = np.clip(pr_np, 0, 1)
            gt_np = np.clip(gt_np, 0, 1)

            png_path = out_dir / f"npy_rand_{rank:02d}_src_{idx:05d}.png"
            save_triptych(ma_np, pr_np, gt_np, png_path, suptitle=f"sample {idx}")
            if args.log_to_wandb and args.project:
                wandb.log({
                    "triptych": wandb.Image(png_path, caption=f"npy sample {idx}")
                }, step=rank)
            saved += 1

        print(f"Saved {saved} triptychs to: {out_dir}")

    if args.log_to_wandb and args.project:
        wandb.finish()

if __name__ == "__main__":
    main()
