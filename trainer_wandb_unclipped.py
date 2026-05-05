# # trainer_wandb.py
# import argparse
# import os
# import json
# import torch
# import yaml
# from types import SimpleNamespace
# from torch.utils.data import DataLoader

# from dataset import CTMetalArtifactDataset
# from models.swin_unet.vision_transformer import SwinUnet
# from models.unet import UnetGenerator                               # <- the vanilla UNet I shared earlier

# import numpy as np
# from metrics import compute_SSIM, compute_PSNR
# import wandb


# # ----------------------------- Utilities -----------------------------
# def dict_to_namespace(d):
#     if isinstance(d, dict):
#         return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
#     return d

# def load_yaml(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def load_config_namespace(path):
#     return dict_to_namespace(load_yaml(path))

# def parse_json_or_none(s):
#     if s is None:
#         return None
#     try:
#         return json.loads(s)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"--model_kwargs must be valid JSON. Error: {e}\nGot: {s}")


# # ----------------------------- Model Factory -----------------------------
# def build_model(name: str,
#                 in_ch: int,
#                 num_classes: int,
#                 device: torch.device,
#                 model_cfg_path: str | None = None,
#                 model_kwargs: dict | None = None):
#     """
#     Centralized place to instantiate any model without changing the training loop.
#     Extend this with new entries as you add architectures.
#     """
#     model_kwargs = model_kwargs or {}

#     name = name.lower()
#     if name in ("unet", "vanilla_unet", "u-net"):
#         # UNet uses explicit keyword args. We allow overrides from --model_kwargs.
#         base_channels = int(model_kwargs.pop("base_channels", 64))
#         bilinear = bool(model_kwargs.pop("bilinear", False))
#         m = UnetGenerator(
#             in_channels=in_ch,
#             **model_kwargs
#         )
#         return m.to(device)

#     # elif name in ("unetgen", "pix2pix_unet"):
#     #     # Your existing UnetGenerator (some repos call it with only in_ch, some without).
#     #     try:
#     #         m = UnetGenerator(in_ch=in_ch, **model_kwargs)
#     #     except TypeError:
#     #         # some versions ignore in_ch
#     #         m = UnetGenerator(**model_kwargs)
#     #     return m.to(device)

#     elif name in ("swinunet", "swin_unet", "swin-u"):
#         if not model_cfg_path:
#             raise ValueError("SwinUnet requires --model_cfg <path to YAML>.")
#         cfg_ns = load_config_namespace(model_cfg_path)
#         # many SwinUnet repos store this here; guard if layout differs
#         try:
#             cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
#         except AttributeError:
#             pass
#         # pass through arbitrary kwargs if your impl supports them
#         m = SwinUnet(config=cfg_ns, **model_kwargs)
#         return m.to(device)

#     else:
#         raise ValueError(f"Unknown model: {name}. Supported: unet, unetgen, swinunet")

# def param_groups(model, wd, nowd_names=('bias', 'bn', 'norm')):
#     decay, no_decay = [], []
#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if any(k in n.lower() for k in nowd_names):
#             no_decay.append(p)
#         else:
#             decay.append(p)
#     return [
#         {'params': decay, 'weight_decay': wd},
#         {'params': no_decay, 'weight_decay': 0.0},
#     ]
# # ----------------------------- Main -----------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Training script (W&B only, model-agnostic)")
#     # Model selection / config
#     parser.add_argument('--model', type=str,
#                         choices=['unet', 'unetgen', 'swinunet'],
#                         required=True,
#                         help="Pick an architecture from the registry.")
#     parser.add_argument('--model_cfg', type=str, default=None,
#                         help="Optional: path to a YAML config (required for swinunet).")
#     parser.add_argument('--model_kwargs', type=str, default=None,
#                         help='Optional: JSON dict of kwargs for the chosen model. '
#                              'Example: \'{"base_channels":48,"bilinear":true}\'')

#     # Data & run setup
#     parser.add_argument('--ma_dir', type=str, required=True)
#     parser.add_argument('--li_dir', type=str, required=False, help='Required if input_mode=ma_li')
#     parser.add_argument('--gt_dir', type=str, required=True)
#     parser.add_argument('--input_mode', type=str, choices=['ma_li', 'ma'], default='ma',
#                         help="Use 'ma_li' for 2-ch input [MA,LI] or 'ma' for MA-only")
#     parser.add_argument('--num_classes', type=int, default=1, help='Output channels/classes')
#     parser.add_argument('--batch_size', type=int, default=4)
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--learning_rate', type=float, default=1e-4)
#     parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to save logs and weights')
#     parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
#     parser.add_argument('--project', type=str, default='ct-mar', help='W&B project name')
#     parser.add_argument('--entity', type=str, default=None, help='W&B entity (team/user) if needed')

#     args = parser.parse_args()

#     # Validate LI path only when needed
#     if args.input_mode == 'ma_li' and not args.li_dir:
#         raise ValueError("input_mode='ma_li' requires --li_dir")

#     os.makedirs(args.log_dir, exist_ok=True)

#     # ---------------- W&B init ----------------
#     run = wandb.init(
#         project=args.project,
#         entity=args.entity,
#         name=args.run_name,
#         config=vars(args),   # logs CLI as config
#         dir=args.log_dir,
#         save_code=True,
#         reinit=True,
#     )

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     in_ch = 2 if args.input_mode == 'ma_li' else 1

#     # ---------------- Build model (agnostic) ----------------
#     model = build_model(
#         name=args.model,
#         in_ch=in_ch,
#         num_classes=args.num_classes,
#         device=device,
#         model_cfg_path=args.model_cfg,
#         model_kwargs=parse_json_or_none(args.model_kwargs)
#     )

#     # (Optional) track gradients/weights
#     wandb.watch(model, log='gradients', log_freq=1)

#     # ---------------- Dataset / loaders ----------------
#     HU_MIN, HU_MAX = -1024.0, 3072.0
#     train_ds = CTMetalArtifactDataset(
#         ma_dir=args.ma_dir, gt_dir=args.gt_dir, mask_dir=None,
#         split='train', hu_min=HU_MIN, hu_max=HU_MAX, 
#         head_policy="exclude",  # hardcoding to body-only
#         seed=42, val_size=0.1
#     )
#     val_ds = CTMetalArtifactDataset(
#         ma_dir=args.ma_dir, gt_dir=args.gt_dir, mask_dir=None,
#         split='val', hu_min=HU_MIN, hu_max=HU_MAX, 
#         head_policy="exclude",  # hardcoding to body-only
#         seed=42, val_size=0.1
#     )

#     pin = (device.type == 'cuda')
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

#     print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")

#     # ---------------- Optimizer / loss ----------------
#     import torch, itertools as it



#     optimizer = torch.optim.AdamW(param_groups(model, wd=1e-4),
#                               lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

#     loss_fn = torch.nn.L1Loss()

#     best_ssim = -1.0
#     best_ckpt_path = None

#     for epoch in range(1, args.epochs + 1):
#         # ---------------- Train ----------------
#         model.train()
#         train_loss = 0.0

#         for x, y in train_loader:
#             x = x.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             pred = model(x)
#             loss = loss_fn(pred, y)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#         avg_loss = train_loss / max(1, len(train_loader))

#         # ---------------- Val ----------------
#         model.eval()
#         total_ssim = 0.0
#         total_psnr = 0.0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x = x.to(device, non_blocking=True)
#                 y = y.to(device, non_blocking=True)
#                 pred = model(x)

#                 # Clamp to [0,1] for metrics
#                 pred_eval = torch.clamp(pred, 0, 1)
#                 gt_eval   = torch.clamp(y,   0, 1)

#                 # loop over batch
#                 for i in range(pred_eval.size(0)):
#                     total_ssim += compute_SSIM(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)
#                     total_psnr += compute_PSNR(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)

#         avg_psnr = total_psnr / len(val_ds)
#         avg_ssim = total_ssim / len(val_ds)

#         # ---------------- Log to W&B ----------------
#         wandb.log(
#             {"loss": avg_loss, "psnr": avg_psnr, "ssim": avg_ssim, "epoch": epoch},
#             step=epoch
#         )
#         print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.4f}")

#         # ---------------- Save checkpoints ----------------
#         ckpt_name = f"{args.model}_epoch{epoch:03d}.pt"
#         ckpt_path = os.path.join(args.log_dir, ckpt_name)
#         torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)

#         # upload "last" each epoch (keeps only latest)
#         last_art = wandb.Artifact(f"{args.model}-last", type="model")
#         last_art.add_file(ckpt_path)
#         wandb.log_artifact(last_art, aliases=["latest"])

#         # save best-by-SSIM
#         if avg_ssim > best_ssim:
#             best_ssim = avg_ssim
#             best_ckpt_path = os.path.join(args.log_dir, f"{args.model}_best.pt")
#             torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_ckpt_path)

#             best_art = wandb.Artifact(f"{args.model}-best", type="model")
#             best_art.add_file(best_ckpt_path)
#             wandb.log_artifact(best_art, aliases=["best"])

#     wandb.finish()
#     print("✅ Training complete!")


# if __name__ == "__main__":
#     main()



# trainer_wandb.py (patched)
import argparse
import os
import json
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader

# NOTE: import path must match your repo layout
from aapm_dataset import CTMetalArtifactDataset   # patched dataset that returns (x, y, li)
from models.swin_unet.vision_transformer import SwinUnet
from models.swin_unet_feature_gating.vision_transformer import SwinUnet as SwinUnetFG
from models.swin_unet_three_channel.vision_transformer import SwinUnet as SwinUnet3Ch
from models.unet import UnetGenerator

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

def parse_json_or_none(s):
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"--model_kwargs must be valid JSON. Error: {e}\nGot: {s}")

# ----------------------------- Model Factory -----------------------------
def build_model(name: str,
                in_ch: int,
                num_classes: int,
                device: torch.device,
                model_cfg_path: str | None = None,
                model_kwargs: dict | None = None):
    """
    Instantiate model. For SwinUnet we expect a config YAML path (model_cfg_path).
    """
    model_kwargs = model_kwargs or {}
    name = name.lower()
    if name in ("unet", "vanilla_unet", "u-net"):
        m = UnetGenerator(
            in_channels=in_ch,
            **model_kwargs
        )
        return m.to(device)
    elif name in ("swinunet", "swin_unet", "swin-u"):
        if not model_cfg_path:
            raise ValueError("SwinUnet requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        # try to inject in_ch / num_classes to the config if present
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except Exception:
            # ignore if config layout differs
            pass
        try:
            cfg_ns.MODEL.SWIN.NUM_CLASSES = int(num_classes)
        except Exception:
            # ignore if config layout differs
            pass

        # The non-guided SwinUnet wrapper does not accept num_classes as a kwarg.
        # It reads NUM_CLASSES from the config instead.
        model_kwargs.pop("num_classes", None)
        m = SwinUnet(config=cfg_ns, **model_kwargs)
        return m.to(device)
    elif name in ("swinunet_fg", "swin_unet_fg", "swinunet-fg", "feature_gating"):
        if not model_cfg_path:
            raise ValueError("SwinUnet (feature gating) requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except Exception:
            pass
        try:
            cfg_ns.MODEL.SWIN.NUM_CLASSES = int(num_classes)
        except Exception:
            pass
        # Wrapper reads NUM_CLASSES from the config; same shape as plain swinunet.
        model_kwargs.pop("num_classes", None)
        m = SwinUnetFG(config=cfg_ns, **model_kwargs)
        return m.to(device)
    elif name in ("swinunet_3ch", "swin_unet_3ch", "swinunet-3ch", "three_channel"):
        if not model_cfg_path:
            raise ValueError("SwinUnet (3-channel) requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch  # expected to be 3 for [MA, LI, A_MG]
        except Exception:
            pass
        try:
            cfg_ns.MODEL.SWIN.NUM_CLASSES = int(num_classes)
        except Exception:
            pass
        model_kwargs.pop("num_classes", None)
        m = SwinUnet3Ch(config=cfg_ns, **model_kwargs)
        return m.to(device)
    else:
        raise ValueError(f"Unknown model: {name}. Supported: unet, swinunet, swinunet_fg, swinunet_3ch")

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

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Training script (W&B only, model-agnostic)")
    # Model selection / config
    parser.add_argument('--model', type=str,
                        choices=['unet', 'swinunet', 'swinunet_fg', 'swinunet_3ch'],
                        required=True,
                        help="Pick an architecture from the registry.")
    parser.add_argument('--model_cfg', type=str, default=None,
                        help="Optional: path to a YAML config (required for swinunet).")
    parser.add_argument('--model_kwargs', type=str, default=None,
                        help='Optional: JSON dict of kwargs for the chosen model. '
                             'Example: \'{"base_channels":48,"bilinear":true}\'')

    # Data & run setup
    parser.add_argument('--ma_dir', type=str, required=True)
    parser.add_argument('--li_dir', type=str, required=False, help='Directory with LI files (required by the dataset)')
    parser.add_argument('--mask_dir', type=str, required=False, help='Optional: directory with metal masks (metalonlymask files)')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--input_mode', type=str, choices=['ma_li', 'ma'], default='ma',
                        help="Use 'ma_li' to concatenate MA+LI as 2-channel input, or 'ma' for MA-only input.")
    parser.add_argument('--num_classes', type=int, default=1, help='Output channels/classes')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers. In Colab, set --num_workers 0 if you hit multiprocessing issues.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to save logs and weights')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--project', type=str, default='ct-mar', help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (team/user) if needed')
    parser.add_argument('--region_policy', type=str, default='all', choices=['all','body','head'],
                        help='Region filter passed to dataset (defaults to "all").')

    # Intensity window (HU) used for clipping/normalization inside the dataset
    parser.add_argument('--hu_min', type=float, default=-1024.0, help='Minimum HU for clipping before normalization')
    parser.add_argument('--hu_max', type=float, default=3072.0, help='Maximum HU for clipping before normalization')
    parser.add_argument('--no_clip', action='store_true', help='Disable HU clipping in the dataset (values may go outside [0,1] after normalization).')

    args = parser.parse_args()

    # LI is required for ma_li mode and for the feature-gating Swin-UNet
    # (which uses LI to build the soft artifact map A_MG).
    is_swin_fg = args.model.lower() in ("swinunet_fg", "swin_unet_fg", "swinunet-fg", "feature_gating")
    is_swin_3ch = args.model.lower() in ("swinunet_3ch", "swin_unet_3ch", "swinunet-3ch", "three_channel")
    if args.input_mode == 'ma_li' and not args.li_dir:
        raise ValueError("input_mode='ma_li' requires --li_dir (LI files).")
    if is_swin_fg and not args.li_dir:
        raise ValueError("--model swinunet_fg requires --li_dir (LI files) to build the artifact map.")
    if is_swin_3ch and not args.li_dir:
        raise ValueError("--model swinunet_3ch requires --li_dir (LI files) to build the [MA, LI, A_MG] input.")

    os.makedirs(args.log_dir, exist_ok=True)

    # ---------------- W&B init ----------------
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=vars(args),
        dir=args.log_dir,
        save_code=True,
        reinit=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Decide input channels to pass to model builder
    # - swinunet_fg          -> single-stream MA input (1ch); LI is used as artifact_map
    # - swinunet_3ch         -> 3-channel input [MA, LI, A_MG]
    # - input_mode='ma_li'   -> 2-channel input [MA, LI] (vanilla UNet / plain SwinUnet)
    # - input_mode='ma'      -> 1-channel input [MA]
    if is_swin_fg:
        in_ch = 1
    elif is_swin_3ch:
        in_ch = 3
    else:
        in_ch = 2 if args.input_mode == 'ma_li' else 1

    # ---------------- Build model ----------------
    model = build_model(
        name=args.model,
        in_ch=in_ch,
        num_classes=args.num_classes,
        device=device,
        model_cfg_path=args.model_cfg,
        model_kwargs=parse_json_or_none(args.model_kwargs)
    )

    # track gradients/weights (optional)
    wandb.watch(model, log='gradients', log_freq=100)

    # ---------------- Dataset / loaders ----------------

    # Our CTMetalArtifactDataset (patched) requires li_dir when we want LI returned.
    # If input_mode == 'ma_li' we will request li_dir to be used.
    dataset_kwargs = dict(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        # LI is loaded whenever input_mode is ma_li OR we're running a model that needs LI.
        li_dir=args.li_dir if (args.input_mode == 'ma_li' or is_swin_fg or is_swin_3ch) else None,
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

    # Build train/val datasets
    train_ds = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "train"})
    # Normalized validation dataset for model I/O + normalized metrics
    val_ds   = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "val"})
    # HU validation dataset for HU-domain metrics (no extra np.load inside the loop)
    val_ds_hu = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "val", "output_space": "hu"})
    # Ensure perfect alignment between the two val loaders
    val_ds_hu.pairs = val_ds.pairs

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    val_loader_hu = DataLoader(val_ds_hu, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")

    # ---------------- Optimizer / loss ----------------
    optimizer = torch.optim.AdamW(param_groups(model, wd=1e-4),
                              lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = torch.nn.L1Loss()

    best_ssim = -1.0
    best_ckpt_path = None

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # unpack batch robustly: support (x,y), (x,y,li), (x,y,mask), (x,y,mask,li)
            x_batch = y_batch = li_batch = mask_batch = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x_batch, y_batch = batch
                elif len(batch) == 3:
                    third = batch[2]
                    # lightweight heuristic: if third is binary -> mask, else LI
                    is_binary = False
                    try:
                        uniq = torch.unique(third)
                        is_binary = (uniq.numel() <= 2) and torch.all((uniq == 0) | (uniq == 1))
                    except Exception:
                        is_binary = False
                    if is_binary:
                        x_batch, y_batch, mask_batch = batch
                    else:
                        x_batch, y_batch, li_batch = batch
                else:
                    # 4+ items -> assume (x,y,mask,li)
                    x_batch, y_batch, mask_batch, li_batch = batch[:4]
            else:
                x_batch, y_batch = batch

            # Move to device
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if li_batch is not None:
                li_batch = li_batch.to(device, non_blocking=True)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Build input tensor based on model + input_mode
            if is_swin_fg:
                # Single-stream MA input; pass A_MG = normalize(relu(LI - MA)) as artifact_map.
                if li_batch is None:
                    raise RuntimeError("swinunet_fg requires LI in the batch, but li_batch is None.")
                artifact_map = torch.relu(li_batch - x_batch)
                am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                artifact_map = artifact_map / (am_max + 1e-6)
                pred = model(x_batch, artifact_map=artifact_map)
            elif is_swin_3ch:
                # 3-channel input: [MA, LI, A_MG]
                if li_batch is None:
                    raise RuntimeError("swinunet_3ch requires LI in the batch, but li_batch is None.")
                artifact_map = torch.relu(li_batch - x_batch)
                am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                artifact_map = artifact_map / (am_max + 1e-6)
                x_in = torch.cat([x_batch, li_batch, artifact_map], dim=1)
                pred = model(x_in)
            else:
                if args.input_mode == 'ma_li':
                    if li_batch is None:
                        raise RuntimeError("input_mode='ma_li' requires LI in the batch, but li_batch is None.")
                    x_in = torch.cat([x_batch, li_batch], dim=1)
                else:
                    x_in = x_batch
                pred = model(x_in)

            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / max(1, len(train_loader))

        # ---------------- Validation ----------------
        model.eval()
        ssim_list = []
        psnr_list = []
        rmse_list = []
        total_masked_ssim = 0.0
        total_score_hu_ssim = 0.0
        total_li_gt_hu_ssim = 0.0
        # compute normalized metal fill for normalized images
        metal_fill_norm = (100.0 - float(args.hu_min)) / (float(args.hu_max) - float(args.hu_min))
        metal_fill_hu = 100.0
        data_range_hu = 5000.0
        with torch.no_grad():
            for batch, batch_hu in zip(val_loader, val_loader_hu):
                # unpack batch robustly: support (x,y), (x,y,li), (x,y,mask), (x,y,mask,li)
                x_batch = y_batch = li_batch = mask_batch = None
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x_batch, y_batch = batch
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
                            x_batch, y_batch, li_batch = batch
                    else:
                        x_batch, y_batch, mask_batch, li_batch = batch[:4]
                else:
                    x_batch, y_batch = batch

                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if li_batch is not None:
                    li_batch = li_batch.to(device, non_blocking=True)
                if mask_batch is not None:
                    mask_batch = mask_batch.to(device, non_blocking=True)

                # Unpack HU batch (same split/order; only used for HU-domain metrics)
                x_hu = y_hu = li_hu = mask_hu = None
                if isinstance(batch_hu, (list, tuple)):
                    if len(batch_hu) == 2:
                        x_hu, y_hu = batch_hu
                    elif len(batch_hu) == 3:
                        third_hu = batch_hu[2]
                        try:
                            uniq_hu = torch.unique(third_hu)
                            is_binary_hu = (uniq_hu.numel() <= 2) and torch.all((uniq_hu == 0) | (uniq_hu == 1))
                        except Exception:
                            is_binary_hu = False
                        if is_binary_hu:
                            x_hu, y_hu, mask_hu = batch_hu
                        else:
                            x_hu, y_hu, li_hu = batch_hu
                    else:
                        x_hu, y_hu, mask_hu, li_hu = batch_hu[:4]
                else:
                    x_hu, y_hu = batch_hu

                y_hu = y_hu.to(device, non_blocking=True)
                if li_hu is not None:
                    li_hu = li_hu.to(device, non_blocking=True)

                if args.input_mode == 'ma_li':
                    if li_batch is None:
                        raise RuntimeError("input_mode='ma_li' requires LI in the batch, but li_batch is None.")
                    x_in = torch.cat([x_batch, li_batch], dim=1)
                else:
                    x_in = x_batch
                if is_swin_fg:
                    if li_batch is None:
                        raise RuntimeError("swinunet_fg requires LI in the batch, but li_batch is None.")
                    artifact_map = torch.relu(li_batch - x_batch)
                    am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                    artifact_map = artifact_map / (am_max + 1e-6)
                    pred = model(x_batch, artifact_map=artifact_map)
                elif is_swin_3ch:
                    if li_batch is None:
                        raise RuntimeError("swinunet_3ch requires LI in the batch, but li_batch is None.")
                    artifact_map = torch.relu(li_batch - x_batch)
                    am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                    artifact_map = artifact_map / (am_max + 1e-6)
                    x_in = torch.cat([x_batch, li_batch, artifact_map], dim=1)
                    pred = model(x_in)
                else:
                    pred = model(x_in)

                # Normalized metrics are computed on clamped [0,1] tensors.
                pred_eval = torch.clamp(pred, 0, 1)
                gt_eval   = torch.clamp(y_batch, 0, 1)

                # compute per-image metrics
                for i in range(pred_eval.size(0)):
                    img_pred = pred_eval[i, 0]
                    img_gt = gt_eval[i, 0]
                    ssim_list.append(float(compute_SSIM(img_pred, img_gt, data_range=1.0)))
                    psnr_list.append(float(compute_PSNR(img_pred, img_gt, data_range=1.0)))
                    rmse_list.append(float(compute_RMSE(img_pred, img_gt)))

                    # compute masked SSIM: treat dataset mask as metal mask (set metals to normalized 100HU)
                    metalmask_arg = None
                    non_metal_mask = None
                    if mask_batch is not None:
                        m = mask_batch[i]
                        if m.dim() == 3 and m.size(0) == 1:
                            m = m.squeeze(0)
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

                    # HU-domain masked SSIM computed on GPU tensors (no skimage, no np.load)
                    if metalmask_arg is not None:
                        # Use the model output (normalized space) mapped back to HU, but compare against
                        # the HU-domain ground truth from val_ds_hu (optionally unclipped if --no_clip).
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

                        if li_hu is not None:
                            li_img_hu = li_hu[i, 0]
                            _, li_gt_hu_val = compute_masked_SSIM(
                                li_img_hu,
                                img_gt_hu,
                                data_range=data_range_hu,
                                mask=non_metal_mask,
                                metalmask=metalmask_arg,
                                metal_fill=metal_fill_hu,
                            )
                            total_li_gt_hu_ssim += li_gt_hu_val

        avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        avg_rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
        std_psnr = float(np.std(psnr_list, ddof=1)) if len(psnr_list) > 1 else 0.0
        std_ssim = float(np.std(ssim_list, ddof=1)) if len(ssim_list) > 1 else 0.0
        std_rmse = float(np.std(rmse_list, ddof=1)) if len(rmse_list) > 1 else 0.0
        avg_masked_ssim = total_masked_ssim / len(val_ds)
        avg_score_hu_ssim = total_score_hu_ssim / len(val_ds)
        avg_li_gt_hu_ssim = total_li_gt_hu_ssim / len(val_ds)

        # ---------------- Log to W&B ----------------
        wandb.log(
            {
                "loss": avg_loss,
                "psnr": avg_psnr,
                "psnr_std": std_psnr,
                "ssim": avg_ssim,
                "ssim_std": std_ssim,
                "masked_ssim": avg_masked_ssim,
                "rmse": avg_rmse,
                "rmse_std": std_rmse,
                "score_hu_ssim": avg_score_hu_ssim,
                "li_gt_score_hu_ssim": avg_li_gt_hu_ssim,
                "epoch": epoch,
            },
            step=epoch
        )
        print(
            f"[Epoch {epoch}] Loss: {avg_loss:.4f} "
            f"PSNR: {avg_psnr:.2f} \u00b1 {std_psnr:.2f} "
            f"SSIM: {avg_ssim:.4f} \u00b1 {std_ssim:.4f} "
            f"RMSE: {avg_rmse:.4f} \u00b1 {std_rmse:.4f} "
            f"masked_SSIM: {avg_masked_ssim:.4f} "
            f"score_hu_ssim: {avg_score_hu_ssim:.4f} li_gt_score_hu_ssim: {avg_li_gt_hu_ssim:.4f}"
        )

        # ---------------- Save checkpoints ----------------
        # Save only a rolling "last.pt" (overwritten every epoch)
        last_ckpt_path = os.path.join(args.log_dir, "last.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, last_ckpt_path)

        # Upload/update "last" artifact each epoch (creates versions in W&B)
        last_art = wandb.Artifact(f"{args.model}-last", type="model")
        last_art.add_file(last_ckpt_path)
        wandb.log_artifact(last_art, aliases=["latest"])

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_ckpt_path = os.path.join(args.log_dir, "best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_ckpt_path)
            best_art = wandb.Artifact(f"{args.model}-best", type="model")
            best_art.add_file(best_ckpt_path)
            wandb.log_artifact(best_art, aliases=["best"])

    wandb.finish()
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
