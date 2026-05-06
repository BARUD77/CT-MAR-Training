import argparse
import os
import json
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader

from aapm_dataset import CTMetalArtifactDataset
from models.swin_unet.vision_transformer import SwinUnet
from models.swin_unet_feature_gating.vision_transformer import SwinUnet as SwinUnetFG
from models.swin_unet_three_channel.vision_transformer import SwinUnet as SwinUnet3Ch
from models.unet import UnetGenerator
from metrics import compute_SSIM, compute_PSNR, compute_RMSE, compute_masked_SSIM
import numpy as np
import wandb


def _to_hu(x: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    return x * (float(hu_max) - float(hu_min)) + float(hu_min)


def _unpack_batch(batch):
    """Unpack dataset batch to (x, y, mask, li) where mask/li may be None.

    Supports tuples of length 3: (x,y,li) and length 4: (x,y,mask,li).
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    if len(batch) == 3:
        x, y, third = batch
        # In our aapm_dataset, third is LI (float). If someone passed masks it might be binary.
        try:
            uniq = torch.unique(third)
            is_binary = (uniq.numel() <= 2) and torch.all((uniq == 0) | (uniq == 1))
        except Exception:
            is_binary = False
        if is_binary:
            return x, y, third, None
        return x, y, None, third
    if len(batch) >= 4:
        x, y, mask, li = batch[:4]
        return x, y, mask, li
    if len(batch) == 2:
        x, y = batch
        return x, y, None, None
    raise ValueError(f"Unsupported batch length: {len(batch)}")

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)

def parse_json_or_none(s):
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"--model_kwargs must be valid JSON. Error: {e}\nGot: {s}")

def load_model(args, device, in_ch):
    if args.model in ('swinunet', 'swinunet_fg', 'swinunet_3ch'):
        if not args.model_cfg:
            raise ValueError("SwinUnet requires --model_cfg path to YAML file.")
        config = load_config(args.model_cfg)
        # ensure input chans match data
        try:
            config.MODEL.SWIN.IN_CHANS = in_ch
        except AttributeError:
            pass
        # ensure output channels match target
        try:
            config.MODEL.SWIN.NUM_CLASSES = int(args.num_classes)
        except AttributeError:
            pass

        model_kwargs = parse_json_or_none(args.model_kwargs) or {}
        # SwinUnet wrappers read channels/classes from config.
        model_kwargs.pop("num_classes", None)
        if args.model == 'swinunet_fg':
            cls = SwinUnetFG
        elif args.model == 'swinunet_3ch':
            cls = SwinUnet3Ch
        else:
            cls = SwinUnet
        model = cls(config=config, **model_kwargs).to(device)
    else:
        model_kwargs = parse_json_or_none(args.model_kwargs) or {}
        model = UnetGenerator(in_channels=in_ch, **model_kwargs).to(device)
    return model

def main():
    p = argparse.ArgumentParser(description="Eval script (W&B optional)")
    p.add_argument('--model', type=str, choices=['unet','swinunet','swinunet_fg','swinunet_3ch'], required=False,
                   help='Required unless --baseline is set.')
    p.add_argument('--baseline', type=str, default='none',
                   choices=['none','li','ma','nmar'],
                   help="Skip model and score a baseline image directly. "
                        "'li' uses LI from --li_dir, 'ma' uses MA from --ma_dir, "
                        "'nmar' uses images from --nmar_dir.")
    p.add_argument('--nmar_dir', type=str, default=None,
                   help='Directory of NMAR-corrected images (required for --baseline nmar).')
    p.add_argument('--model_cfg', type=str, help='Path to YAML config (required for swinunet)')
    p.add_argument('--model_kwargs', type=str, default=None,
                   help='Optional JSON dict of kwargs for the chosen model. Example: "{\\"num_classes\\":1}"')
    p.add_argument('--ma_dir', type=str, required=True)
    p.add_argument('--li_dir', type=str, required=False, help='Required for input_mode=ma_li (2-channel input)')
    p.add_argument('--gt_dir', type=str, required=True)
    p.add_argument('--mask_dir', type=str, default=None, help='Optional metal-only mask dir (enables masked SSIM + score HU SSIM)')
    p.add_argument('--split', type=str, choices=['train','val','test'], default='train')
    p.add_argument('--input_mode', type=str, choices=['ma','ma_li','ma_amg','ma_li_amg'], default='ma',
                   help="'ma' uses MA only. 'ma_li' concatenates [MA,LI] (2ch). "
                        "'ma_amg' feeds MA (1ch) + artifact_map=A_MG kwarg (for swinunet_fg). "
                        "'ma_li_amg' concatenates [MA,LI,A_MG] (3ch, for swinunet_3ch).")
    p.add_argument('--num_classes', type=int, default=1, help='Output channels/classes')
    p.add_argument('--region_policy', type=str, default='all', choices=['all','body','head'])
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--ckpt', type=str, required=False, help='Path to .pt checkpoint (required unless --baseline is set)')
    p.add_argument('--project', type=str, default=None, help='W&B project (optional)')
    p.add_argument('--entity', type=str, default=None, help='W&B entity (optional)')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--log_dir', type=str, default='./runs_eval')
    p.add_argument('--save_preds', action='store_true', help='Save MA/GT/Pred as .npy')
    p.add_argument('--save_dir', type=str, default='./eval_outputs')
    p.add_argument('--hu_min', type=float, default=-1024.0)
    p.add_argument('--hu_max', type=float, default=3072.0)
    args = p.parse_args()

    is_baseline = (args.baseline != 'none')
    if not is_baseline:
        if not args.model:
            raise ValueError("--model is required unless --baseline is set")
        if not args.ckpt:
            raise ValueError("--ckpt is required unless --baseline is set")
    if args.baseline == 'li' and not args.li_dir:
        raise ValueError("--baseline li requires --li_dir")
    if args.baseline == 'nmar' and not args.nmar_dir:
        raise ValueError("--baseline nmar requires --nmar_dir")

    if args.input_mode == 'ma_li' and not args.li_dir:
        raise ValueError("input_mode='ma_li' requires --li_dir")
    if args.input_mode == 'ma_amg' and not args.li_dir:
        raise ValueError("input_mode='ma_amg' requires --li_dir (used to compute A_MG = relu(LI - MA))")
    if args.input_mode == 'ma_li_amg' and not args.li_dir:
        raise ValueError("input_mode='ma_li_amg' requires --li_dir")
    if args.model == 'swinunet_fg' and args.input_mode != 'ma_amg':
        raise ValueError("--model swinunet_fg requires --input_mode ma_amg")
    if args.model == 'swinunet_3ch' and args.input_mode != 'ma_li_amg':
        raise ValueError("--model swinunet_3ch requires --input_mode ma_li_amg")

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

    # Match training behavior:
    # - input_mode='ma'        -> MA-only input (1 channel)
    # - input_mode='ma_li'     -> concatenate [MA, LI] (2 channels)
    # - input_mode='ma_amg'    -> MA (1 channel) + artifact_map kwarg (FG model)
    # - input_mode='ma_li_amg' -> concatenate [MA, LI, A_MG] (3 channels, for 3ch swin)
    if args.input_mode == 'ma_li_amg':
        in_ch = 3
    elif args.input_mode == 'ma_li':
        in_ch = 2
    else:
        in_ch = 1

    if is_baseline:
        model = None
        print(f"Baseline mode: '{args.baseline}' (no model, no checkpoint)")
    else:
        model = load_model(args, device, in_ch=in_ch)
        # Load checkpoint
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt.get('model_state', ckpt)  # support raw state_dict
        model.load_state_dict(state)
        model.eval()

    # Dataset & loaders
    needs_li = args.input_mode in ('ma_li', 'ma_amg', 'ma_li_amg') or args.baseline == 'li'
    ds = CTMetalArtifactDataset(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        li_dir=args.li_dir if needs_li else None,
        mask_dir=args.mask_dir,
        split=args.split,
        val_size=0.0,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        region_policy=args.region_policy,
        output_space="norm",
    )
    ds_hu = CTMetalArtifactDataset(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        li_dir=args.li_dir if needs_li else None,
        mask_dir=args.mask_dir,
        split=args.split,
        val_size=0.0,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        region_policy=args.region_policy,
        output_space="hu",
    )
    # Ensure identical ordering even if dataset internals change later.
    ds_hu.pairs = ds.pairs

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    loader_hu = DataLoader(
        ds_hu, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    print(f"Evaluating on {args.split} set: {len(ds)} samples")

    # Metrics
    ssim_list, psnr_list, rmse_list = [], [], []
    masked_ssim_list, score_hu_ssim_list = [], []
    metal_fill_norm = (100.0 - float(args.hu_min)) / (float(args.hu_max) - float(args.hu_min))
    metal_fill_hu = 100.0
    data_range_hu = 5000.0

    with torch.no_grad():
        idx_base = 0
        for batch_idx, (batch, batch_hu) in enumerate(zip(loader, loader_hu)):
            x, y, mask, li = _unpack_batch(batch)
            _x_hu, y_hu, _mask_hu, _li_hu = _unpack_batch(batch_hu)

            x = x.to(device, non_blocking=True)  # (B,1,H,W)
            y = y.to(device, non_blocking=True)
            y_hu = y_hu.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            # Forward: build prediction based on baseline / input_mode
            if is_baseline:
                if args.baseline == 'li':
                    if li is None:
                        raise RuntimeError("--baseline li requires LI in the batch.")
                    pred = li.to(device, non_blocking=True)
                elif args.baseline == 'ma':
                    pred = x
                elif args.baseline == 'nmar':
                    # Load NMAR images on-the-fly using the dataset's filename ordering.
                    nmar_imgs = []
                    for j in range(x.size(0)):
                        entry = ds.pairs[idx_base + j]
                        ma_file = entry[0]
                        nmar_path = os.path.join(args.nmar_dir, ma_file)
                        arr = np.load(nmar_path).astype(np.float32)
                        if arr.ndim == 3 and arr.shape[0] == 1:
                            arr = arr[0]
                        # Normalize to [0,1] using the same hu_min/hu_max as the dataset.
                        arr01 = (arr - float(args.hu_min)) / (float(args.hu_max) - float(args.hu_min))
                        nmar_imgs.append(torch.from_numpy(arr01))
                    pred = torch.stack(nmar_imgs, dim=0).unsqueeze(1).to(device, non_blocking=True)
                else:
                    raise ValueError(f"Unknown baseline: {args.baseline}")
            elif args.input_mode == 'ma_li':
                if li is None:
                    raise RuntimeError("input_mode='ma_li' requires LI in the batch, but li is None.")
                li = li.to(device, non_blocking=True)
                x_in = torch.cat([x, li], dim=1)
                pred = model(x_in)
            elif args.input_mode == 'ma_amg':
                if li is None:
                    raise RuntimeError("input_mode='ma_amg' requires LI in the batch, but li is None.")
                li = li.to(device, non_blocking=True)
                am = torch.relu(li - x)
                am = am / (am.amax(dim=(2, 3), keepdim=True) + 1e-6)
                pred = model(x, artifact_map=am)
            elif args.input_mode == 'ma_li_amg':
                if li is None:
                    raise RuntimeError("input_mode='ma_li_amg' requires LI in the batch, but li is None.")
                li = li.to(device, non_blocking=True)
                am = torch.relu(li - x)
                am = am / (am.amax(dim=(2, 3), keepdim=True) + 1e-6)
                x_in = torch.cat([x, li, am], dim=1)
                pred = model(x_in)
            else:
                x_in = x
                pred = model(x_in)

            pred_eval = torch.clamp(pred, 0, 1)
            gt_eval   = torch.clamp(y,   0, 1)

            B = pred_eval.size(0)
            for i in range(B):
                ssim_val = compute_SSIM(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)
                psnr_val = compute_PSNR(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)
                rmse_val = compute_RMSE(pred_eval[i, 0], gt_eval[i, 0])
                ssim_list.append(float(ssim_val))
                psnr_list.append(float(psnr_val))
                rmse_list.append(float(rmse_val))

                # masked SSIM + score HU SSIM (only meaningful if mask is available)
                metalmask_arg = None
                non_metal_mask = None
                if mask is not None:
                    m = mask[i]
                    if m.dim() == 3 and m.size(0) == 1:
                        m = m.squeeze(0)
                    metalmask_arg = m
                    non_metal_mask = (m == 0).float()

                _, masked_val = compute_masked_SSIM(
                    pred_eval[i, 0],
                    gt_eval[i, 0],
                    data_range=1.0,
                    mask=non_metal_mask,
                    metalmask=metalmask_arg,
                    metal_fill=metal_fill_norm,
                    hu_min=args.hu_min,
                    hu_max=args.hu_max,
                )
                masked_ssim_list.append(float(masked_val))

                if metalmask_arg is not None:
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
                    score_hu_ssim_list.append(float(hu_val))

                if args.save_preds:
                    # Save as numpy for inspection
                    np.save(os.path.join(args.save_dir, f"idx_{idx_base + i:06d}_ma.npy"),
                            x[i,0].detach().cpu().float().numpy())
                    np.save(os.path.join(args.save_dir, f"idx_{idx_base + i:06d}_gt.npy"),
                            gt_eval[i,0].detach().cpu().float().numpy())
                    np.save(os.path.join(args.save_dir, f"idx_{idx_base + i:06d}_pred.npy"),
                            pred_eval[i,0].detach().cpu().float().numpy())

            idx_base += B

    def _mean_std(vals):
        if not len(vals):
            return 0.0, 0.0
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return m, s

    avg_ssim, std_ssim = _mean_std(ssim_list)
    avg_psnr, std_psnr = _mean_std(psnr_list)
    avg_rmse, std_rmse = _mean_std(rmse_list)
    avg_masked_ssim, std_masked_ssim = _mean_std(masked_ssim_list)
    avg_score_hu_ssim, std_score_hu_ssim = _mean_std(score_hu_ssim_list)

    print(
        f"[EVAL] {args.split} "
        f"| SSIM: {avg_ssim:.4f} ± {std_ssim:.4f} "
        f"| PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB "
        f"| RMSE: {avg_rmse:.6f} ± {std_rmse:.6f} "
        f"| masked_SSIM: {avg_masked_ssim:.4f} ± {std_masked_ssim:.4f} "
        f"| score_hu_ssim: {avg_score_hu_ssim:.4f} ± {std_score_hu_ssim:.4f}"
    )

    if use_wandb:
        wandb.log({
            "eval/ssim_mean": avg_ssim,
            "eval/ssim_std": std_ssim,
            "eval/psnr_mean": avg_psnr,
            "eval/psnr_std": std_psnr,
            "eval/rmse_mean": avg_rmse,
            "eval/rmse_std": std_rmse,
            "eval/masked_ssim_mean": avg_masked_ssim,
            "eval/masked_ssim_std": std_masked_ssim,
            "eval/score_hu_ssim_mean": avg_score_hu_ssim,
            "eval/score_hu_ssim_std": std_score_hu_ssim,
            "eval/ssim_hist": wandb.Histogram(ssim_list),
            "eval/psnr_hist": wandb.Histogram(psnr_list),
            "eval/rmse_hist": wandb.Histogram(rmse_list),
            "eval/masked_ssim_hist": wandb.Histogram(masked_ssim_list) if masked_ssim_list else None,
            "eval/score_hu_ssim_hist": wandb.Histogram(score_hu_ssim_list) if score_hu_ssim_list else None,
        })
        wandb.finish()

if __name__ == "__main__":
    main()
