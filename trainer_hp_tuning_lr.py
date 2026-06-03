# trainer_hp_tuning_lr.py
# Hyperparameter tuning script: learning rate sweep for Swin-UNet V2
# Tests LRs: 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 1e-3
# Each LR runs for 40 epochs. Logs all runs to W&B for comparison.

import argparse
import os
import json
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader

# NOTE: import path must match your repo layout
from aapm_dataset import CTMetalArtifactDataset
from models.swin_unet_v2.swin_unet_v2 import SwinTransformerSys as SwinUnetV2

import numpy as np
from metrics import compute_SSIM, compute_PSNR, compute_RMSE, compute_masked_SSIM, compute_masked_SSIM_per_image
import wandb


def _to_hu(x: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    return x * (float(hu_max) - float(hu_min)) + float(hu_min)

# ----------------------------- Utilities -----------------------------
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config_namespace(path):
    return dict_to_namespace(load_yaml(path))

def param_groups(model, wd, nowd_names=('bias', 'bn', 'norm')):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n.lower() for k in nowd_names):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

# ----------------------------- Training Function ----------------
def train_single_lr(learning_rate, args, device, train_loader, val_loader, val_loader_hu, run_name_suffix=""):
    """
    Train Swin-UNet V2 for 40 epochs with a single learning rate.
    Returns final best SSIM achieved.
    """
    
    # Initialize W&B run for this LR
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=f"swinunet_v2_lr_{learning_rate:.0e}{run_name_suffix}",
        config={
            "model": "swinunet_v2",
            "learning_rate": learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hu_min": args.hu_min,
            "hu_max": args.hu_max,
            "input_mode": "ma",
        },
        dir=args.log_dir,
        save_code=False,
        tags=["lr_sweep"],
    )
    
    # Build model
    cfg_ns = load_config_namespace(args.model_cfg)
    try:
        cfg_ns.MODEL.SWIN.IN_CHANS = 1  # Single-channel MA input
    except Exception:
        pass
    try:
        cfg_ns.MODEL.SWIN.NUM_CLASSES = 1
    except Exception:
        pass

    model = SwinUnetV2(
        img_size=int(cfg_ns.DATA.IMG_SIZE),
        patch_size=int(cfg_ns.MODEL.SWIN.PATCH_SIZE),
        in_chans=int(cfg_ns.MODEL.SWIN.IN_CHANS),
        num_classes=int(cfg_ns.MODEL.SWIN.NUM_CLASSES),
        embed_dim=int(cfg_ns.MODEL.SWIN.EMBED_DIM),
        depths=list(cfg_ns.MODEL.SWIN.DEPTHS),
        depths_decoder=list(cfg_ns.MODEL.SWIN.DECODER_DEPTHS),
        num_heads=list(cfg_ns.MODEL.SWIN.NUM_HEADS),
        window_size=int(cfg_ns.MODEL.SWIN.WINDOW_SIZE),
        drop_path_rate=float(cfg_ns.MODEL.DROP_PATH_RATE),
        final_upsample=str(cfg_ns.MODEL.SWIN.FINAL_UPSAMPLE),
    ).to(device)

    wandb.watch(model, log='gradients', log_freq=100)

    # Optimizer + loss
    optimizer = torch.optim.AdamW(
        param_groups(model, wd=1e-4),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    loss_fn = torch.nn.L1Loss()

    best_ssim = -1.0
    best_ckpt_path = os.path.join(args.log_dir, f"best_lr_{learning_rate:.0e}.pt")
    epochs_since_improve = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x_batch, y_batch = batch
                else:
                    x_batch = batch[0]
                    y_batch = batch[1]
            else:
                x_batch, y_batch = batch

            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass (MA only)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        ssim_list = []
        psnr_list = []
        rmse_list = []
        total_masked_ssim = 0.0
        total_body_masked_ssim = 0.0
        n_body_masked = 0
        total_score_hu_ssim = 0.0
        
        metal_fill_norm = (100.0 - float(args.hu_min)) / (float(args.hu_max) - float(args.hu_min))
        metal_fill_hu = 100.0
        data_range_hu = 5000.0

        with torch.no_grad():
            for batch, batch_hu in zip(val_loader, val_loader_hu):
                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x_batch, y_batch = batch
                        mask_batch = None
                    elif len(batch) == 3:
                        third = batch[2]
                        try:
                            uniq = torch.unique(third)
                            is_binary = (uniq.numel() <= 2) and torch.all((uniq == 0) | (uniq == 1))
                        except Exception:
                            is_binary = False
                        if is_binary:
                            x_batch, y_batch, mask_batch = batch
                        else:
                            x_batch, y_batch = batch[:2]
                            mask_batch = None
                    else:
                        x_batch, y_batch, mask_batch = batch[0], batch[1], batch[2]
                else:
                    x_batch, y_batch = batch
                    mask_batch = None

                # Unpack HU batch
                if isinstance(batch_hu, (list, tuple)):
                    if len(batch_hu) == 2:
                        x_hu, y_hu = batch_hu
                        mask_hu = None
                    else:
                        x_hu, y_hu, mask_hu = batch_hu[0], batch_hu[1], batch_hu[2] if len(batch_hu) > 2 else None
                else:
                    x_hu, y_hu = batch_hu
                    mask_hu = None

                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                y_hu = y_hu.to(device, non_blocking=True)
                if mask_batch is not None:
                    mask_batch = mask_batch.to(device, non_blocking=True)

                # Forward pass (MA only)
                pred = model(x_batch)
                pred_eval = torch.clamp(pred, 0, 1)
                gt_eval = torch.clamp(y_batch, 0, 1)

                # Compute metrics per image
                for i in range(pred_eval.size(0)):
                    img_pred = pred_eval[i, 0]
                    img_gt = gt_eval[i, 0]
                    ssim_list.append(float(compute_SSIM(img_pred, img_gt, data_range=1.0)))
                    psnr_list.append(float(compute_PSNR(img_pred, img_gt, data_range=1.0)))
                    rmse_list.append(float(compute_RMSE(img_pred, img_gt)))

                    # Masked SSIM
                    metalmask_arg = None
                    non_metal_mask = None
                    if mask_batch is not None:
                        m = mask_batch[i]
                        if m.dim() == 3 and m.size(0) == 1:
                            m = m.squeeze(0)
                        metalmask_arg = m
                        non_metal_mask = (m == 0).float()

                        _, masked_val = compute_masked_SSIM(
                            img_pred,
                            img_gt,
                            data_range=1.0,
                            mask=non_metal_mask,
                            metalmask=metalmask_arg,
                            metal_fill=metal_fill_norm,
                            hu_min=args.hu_min,
                            hu_max=args.hu_max,
                        )
                        total_masked_ssim += masked_val

                        # AAPM-style body+metal SSIM
                        if y_hu is not None:
                            img_gt_hu_i = y_hu[i, 0]
                            body_mask = (img_gt_hu_i > float(args.body_hu_thresh)).float()
                            body_avg_mask = body_mask * non_metal_mask
                            _, body_masked_val = compute_masked_SSIM(
                                img_pred,
                                img_gt,
                                data_range=1.0,
                                mask=body_avg_mask,
                                metalmask=metalmask_arg,
                                metal_fill=metal_fill_norm,
                                hu_min=args.hu_min,
                                hu_max=args.hu_max,
                            )
                            total_body_masked_ssim += body_masked_val
                            n_body_masked += 1

                        # HU-domain masked SSIM
                        img_pred_hu = _to_hu(pred[i, 0], args.hu_min, args.hu_max)
                        img_gt_hu = y_hu[i, 0]
                        _, hu_val = compute_masked_SSIM(
                            img_pred_hu,
                            img_gt_hu,
                            data_range=data_range_hu,
                            mask=non_metal_mask,
                            metalmask=metalmask_arg,
                            metal_fill=metal_fill_hu,
                        )
                        total_score_hu_ssim += hu_val

        # Aggregate metrics
        avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        avg_rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
        std_psnr = float(np.std(psnr_list, ddof=1)) if len(psnr_list) > 1 else 0.0
        std_ssim = float(np.std(ssim_list, ddof=1)) if len(ssim_list) > 1 else 0.0
        std_rmse = float(np.std(rmse_list, ddof=1)) if len(rmse_list) > 1 else 0.0
        avg_masked_ssim = total_masked_ssim / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
        avg_body_masked_ssim = (total_body_masked_ssim / n_body_masked) if n_body_masked > 0 else 0.0
        avg_score_hu_ssim = total_score_hu_ssim / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0

        # Log to W&B
        wandb.log({
            "loss": avg_loss,
            "psnr": avg_psnr,
            "psnr_std": std_psnr,
            "ssim": avg_ssim,
            "ssim_std": std_ssim,
            "masked_ssim": avg_masked_ssim,
            "body_masked_ssim": avg_body_masked_ssim,
            "rmse": avg_rmse,
            "rmse_std": std_rmse,
            "score_hu_ssim": avg_score_hu_ssim,
            "epoch": epoch,
        }, step=epoch)

        print(
            f"[LR={learning_rate:.0e}] [Epoch {epoch:2d}/40] Loss: {avg_loss:.4f} "
            f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f} "
            f"PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} "
            f"masked_SSIM: {avg_masked_ssim:.4f}"
        )

        # Save best checkpoint
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save({
                "epoch": epoch,
                "learning_rate": learning_rate,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_ssim": best_ssim,
            }, best_ckpt_path)
            print(f"  ✓ Best SSIM updated: {best_ssim:.4f}, checkpoint saved")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

    # Finish this W&B run
    wandb.summary["final_best_ssim"] = best_ssim
    wandb.summary["final_best_ckpt"] = best_ckpt_path
    wandb.finish()

    print(f"✅ LR {learning_rate:.0e} complete. Best SSIM: {best_ssim:.4f}\n")
    return best_ssim


# ----------------------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="LR hyperparameter tuning for Swin-UNet V2")
    
    # Data setup
    parser.add_argument('--ma_dir', type=str, required=True, help='Directory with MA files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with GT files')
    parser.add_argument('--mask_dir', type=str, required=False, help='Directory with metal masks')
    parser.add_argument('--li_dir', type=str, required=False, help='(Not used for V2, included for compatibility)')
    
    # Model config
    parser.add_argument('--model_cfg', type=str, 
                        default=r"models/swin_unet_v2/config.yaml",
                        help='Path to Swin-UNet V2 config YAML')
    
    # Training setup
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40, help='Epochs per LR trial')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--body_hu_thresh', type=float, default=-500.0)
    
    # HU normalization
    parser.add_argument('--hu_min', type=float, default=-1024.0)
    parser.add_argument('--hu_max', type=float, default=3072.0)
    parser.add_argument('--no_clip', action='store_true', help='Disable HU clipping')
    
    # W&B setup
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory for checkpoints')
    parser.add_argument('--project', type=str, default='ct-mar', help='W&B project')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--region_policy', type=str, default='all', choices=['all', 'body', 'head'])
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Learning rates to sweep
    learning_rates = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 1e-3]

    # Build datasets once (shared across all LR trials)
    print("Loading datasets...")
    dataset_kwargs = dict(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        li_dir=None,  # V2 doesn't use LI
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
    val_ds_hu = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "val", "output_space": "hu"})
    val_ds_hu.pairs = val_ds.pairs

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=pin)
    val_loader_hu = DataLoader(val_ds_hu, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=pin)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples\n")

    # Run LR sweep
    print("=" * 70)
    print("LEARNING RATE HYPERPARAMETER SWEEP (Swin-UNet V2)")
    print("=" * 70)
    print(f"Testing LRs: {learning_rates}")
    print(f"Epochs per LR: {args.epochs}")
    print("=" * 70 + "\n")

    results = {}
    for lr in learning_rates:
        best_ssim = train_single_lr(lr, args, device, train_loader, val_loader, val_loader_hu)
        results[f"lr_{lr:.0e}"] = best_ssim

    # Summary
    print("\n" + "=" * 70)
    print("LEARNING RATE SWEEP SUMMARY")
    print("=" * 70)
    for lr_key, ssim in sorted(results.items(), key=lambda x: x[1], reverse=True):
        lr = float(lr_key.split('_')[1])
        print(f"  LR {lr:.0e}: best SSIM = {ssim:.4f}")
    
    best_lr = max(results.items(), key=lambda x: x[1])
    print(f"\n✅ Best LR: {best_lr[0]} with SSIM = {best_lr[1]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
