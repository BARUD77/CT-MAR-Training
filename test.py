import argparse
import os
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader
from test_dataset import CTMetalArtifactDataset
from models.swin_unet.vision_transformer import SwinUnet
from unet import UnetGenerator
from metrics import compute_SSIM, compute_PSNR
import numpy as np
import wandb

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)

def load_model(args, device, in_ch):
    if args.model == 'swinunet':
        if not args.config:
            raise ValueError("SwinUnet requires --config path to YAML file.")
        config = load_config(args.config)
        # ensure input chans match data
        try:
            config.MODEL.SWIN.IN_CHANS = in_ch
        except AttributeError:
            pass
        model = SwinUnet(config=config).to(device)
    else:
        try:
            model = UnetGenerator(in_ch=in_ch).to(device)
        except TypeError:
            model = UnetGenerator().to(device)
    return model

def main():
    p = argparse.ArgumentParser(description="Eval script (W&B optional)")
    p.add_argument('--model', type=str, choices=['unet','swinunet'], required=True)
    p.add_argument('--config', type=str, help='Path to config.yaml (required for swinunet)')
    p.add_argument('--ma_dir', type=str, required=True)
    p.add_argument('--gt_dir', type=str, required=True)
    p.add_argument('--split', type=str, choices=['val','test'], default='val')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--ckpt', type=str, required=True, help='Path to .pt checkpoint saved during training')
    p.add_argument('--project', type=str, default=None, help='W&B project (optional)')
    p.add_argument('--entity', type=str, default=None, help='W&B entity (optional)')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--log_dir', type=str, default='./runs_eval')
    p.add_argument('--save_preds', action='store_true', help='Save MA/GT/Pred as .npy')
    p.add_argument('--save_dir', type=str, default='./eval_outputs')
    p.add_argument('--in_ch', type=int, default=1, help='Model input channels (1 for MA)')
    p.add_argument('--hu_min', type=float, default=-1024.0)
    p.add_argument('--hu_max', type=float, default=3072.0)
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    if args.save_preds:
        os.makedirs(args.save_dir, exist_ok=True)

    # Optional W&B
    use_wandb = args.project is not None
    if use_wandb:
        wandb.init(
            project=args.project, entity=args.entity, name=args.run_name,
            config=vars(args), dir=args.log_dir, save_code=True, reinit=True
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device, in_ch=args.in_ch)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get('model_state', ckpt)  # support raw state_dict
    model.load_state_dict(state)
    model.eval()

    # Dataset & loader
    ds = CTMetalArtifactDataset(
        ma_dir=args.ma_dir, gt_dir=args.gt_dir, mask_dir=None,
        split=args.split, val_size=0.0, hu_min=args.hu_min, hu_max=args.hu_max
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    print(f"Evaluating on {args.split} set: {len(ds)} samples")

    # Metrics
    ssim_list, psnr_list = [], []

    with torch.no_grad():
        idx_base = 0
        for batch_idx, batch in enumerate(loader):
            # dataset returns (x, y) or (x, y, m)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to(device, non_blocking=True)  # (B,1,H,W)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            pred_eval = torch.clamp(pred, 0, 1)
            gt_eval   = torch.clamp(y,   0, 1)

            B = pred_eval.size(0)
            for i in range(B):
                ssim_val = compute_SSIM(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)
                psnr_val = compute_PSNR(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)
                ssim_list.append(float(ssim_val))
                psnr_list.append(float(psnr_val))

                if args.save_preds:
                    # Save as numpy for inspection
                    np.save(os.path.join(args.save_dir, f"idx_{idx_base + i:06d}_ma.npy"),
                            x[i,0].detach().cpu().float().numpy())
                    np.save(os.path.join(args.save_dir, f"idx_{idx_base + i:06d}_gt.npy"),
                            gt_eval[i,0].detach().cpu().float().numpy())
                    np.save(os.path.join(args.save_dir, f"idx_{idx_base + i:06d}_pred.npy"),
                            pred_eval[i,0].detach().cpu().float().numpy())

            idx_base += B

    avg_ssim = float(np.mean(ssim_list)) if len(ssim_list) else 0.0
    avg_psnr = float(np.mean(psnr_list)) if len(psnr_list) else 0.0

    print(f"[EVAL] {args.split} | SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f} dB")

    if use_wandb:
        wandb.log({
            "eval/ssim_mean": avg_ssim,
            "eval/psnr_mean": avg_psnr,
            "eval/ssim_hist": wandb.Histogram(ssim_list),
            "eval/psnr_hist": wandb.Histogram(psnr_list),
        })
        wandb.finish()

if __name__ == "__main__":
    main()
