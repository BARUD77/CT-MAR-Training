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
from ct_mar_dataset import CTMetalArtifactDataset   # patched dataset that returns (x, y, li)
from models.swin_unet.vision_transformer import SwinUnet
from models.unet import UnetGenerator

import numpy as np
from metrics import compute_SSIM, compute_PSNR
import wandb

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
        # try to inject in_ch to the config if present
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except Exception:
            # ignore if config layout differs
            pass
        m = SwinUnet(config=cfg_ns, **model_kwargs)
        return m.to(device)
    else:
        raise ValueError(f"Unknown model: {name}. Supported: unet, swinunet")

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
                        choices=['unet', 'swinunet'],
                        required=True,
                        help="Pick an architecture from the registry.")
    parser.add_argument('--model_cfg', type=str, default=None,
                        help="Optional: path to a YAML config (required for swinunet).")
    parser.add_argument('--model_kwargs', type=str, default=None,
                        help='Optional: JSON dict of kwargs for the chosen model. '
                             'Example: \'{"base_channels":48,"bilinear":true}\'')

    # Data & run setup
    parser.add_argument('--ma_dir', type=str, required=True)
    parser.add_argument('--li_dir', type=str, required=False, help='Required for artifact guidance / ma_li mode')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--input_mode', type=str, choices=['ma_li', 'ma'], default='ma',
                        help="Use 'ma_li' for using LI guidance. For Swin->single-stream keep MA-only input but pass LI as artifact_map.")
    parser.add_argument('--num_classes', type=int, default=1, help='Output channels/classes')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to save logs and weights')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--project', type=str, default='ct-mar', help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (team/user) if needed')
    parser.add_argument('--region_policy', type=str, default='all', choices=['all','body','head'],
                        help='Region filter passed to dataset (defaults to "all").')

    args = parser.parse_args()

    # Validate LI path when input_mode=ma_li
    if args.input_mode == 'ma_li' and not args.li_dir:
        raise ValueError("input_mode='ma_li' requires --li_dir (LI guidance files).")

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

    # Decide input channels to pass to model builder:
    # - For SwinUnet + artifact guidance we will use single-stream input (MA only) and pass LI separately
    # - For vanilla UNet if input_mode=='ma_li' we will concatenate MA and LI -> in_ch = 2
    if args.model.lower() in ("swinunet",):
        # We use single-stream MA input. Model will receive artifact_map via forward call.
        in_ch = 1
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
    HU_MIN, HU_MAX = -1024.0, 3072.0

    # Our CTMetalArtifactDataset (patched) requires li_dir when we want LI returned.
    # If input_mode == 'ma_li' we will request li_dir to be used.
    dataset_kwargs = dict(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        li_dir=args.li_dir if args.input_mode == 'ma_li' else None,
        split='train',
        hu_min=HU_MIN,
        hu_max=HU_MAX,
        region_policy=args.region_policy,
        seed=42,
        val_size=0.1
    )

    # Build train/val datasets
    train_ds = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "train"})
    val_ds   = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "val"})

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

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
            # dataset returns either (x,y,li) or (x,y) depending on dataset construction.
            if args.input_mode == 'ma_li':
                # dataset returns x,y,li
                x_batch, y_batch, li_batch = batch
            else:
                # dataset returns x,y
                x_batch, y_batch = batch
                li_batch = None

            # Move to device
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if li_batch is not None:
                li_batch = li_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Model-specific forward:
            if args.model.lower() in ("swinunet",):
                # SwinUnet accepts (x, artifact_map=...) in our patched implementation.
                # Ensure we pass MA-only as input (x_batch should be single channel).
                if args.input_mode == 'ma_li':
                    # x_batch is MA (1,ch), li_batch is the guidance A_MG
                    pred = model(x_batch, artifact_map=li_batch)
                else:
                    pred = model(x_batch, artifact_map=None)
            else:
                # vanilla UNet: if li present and input_mode 'ma_li', concatenate channels
                if args.input_mode == 'ma_li' and li_batch is not None:
                    x_in = torch.cat([x_batch, li_batch], dim=1)  # concat along channel dim
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
        total_ssim = 0.0
        total_psnr = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if args.input_mode == 'ma_li':
                    x_batch, y_batch, li_batch = batch
                else:
                    x_batch, y_batch = batch
                    li_batch = None

                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if li_batch is not None:
                    li_batch = li_batch.to(device, non_blocking=True)

                if args.model.lower() in ("swinunet",):
                    if li_batch is not None:
                        pred = model(x_batch, artifact_map=li_batch)
                    else:
                        pred = model(x_batch, artifact_map=None)
                else:
                    if args.input_mode == 'ma_li' and li_batch is not None:
                        x_in = torch.cat([x_batch, li_batch], dim=1)
                    else:
                        x_in = x_batch
                    pred = model(x_in)

                pred_eval = torch.clamp(pred, 0, 1)
                gt_eval   = torch.clamp(y_batch, 0, 1)

                # compute per-image metrics
                for i in range(pred_eval.size(0)):
                    total_ssim += compute_SSIM(pred_eval[i, 0].cpu(), gt_eval[i, 0].cpu(), data_range=1.0)
                    total_psnr += compute_PSNR(pred_eval[i, 0].cpu(), gt_eval[i, 0].cpu(), data_range=1.0)

        avg_psnr = total_psnr / len(val_ds)
        avg_ssim = total_ssim / len(val_ds)

        # ---------------- Log to W&B ----------------
        wandb.log(
            {"loss": avg_loss, "psnr": avg_psnr, "ssim": avg_ssim, "epoch": epoch},
            step=epoch
        )
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.4f}")

        # ---------------- Save checkpoints ----------------
        ckpt_name = f"{args.model}_epoch{epoch:03d}.pt"
        ckpt_path = os.path.join(args.log_dir, ckpt_name)
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)

        last_art = wandb.Artifact(f"{args.model}-last", type="model")
        last_art.add_file(ckpt_path)
        wandb.log_artifact(last_art, aliases=["latest"])

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_ckpt_path = os.path.join(args.log_dir, f"{args.model}_best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_ckpt_path)
            best_art = wandb.Artifact(f"{args.model}-best", type="model")
            best_art.add_file(best_ckpt_path)
            wandb.log_artifact(best_art, aliases=["best"])

    wandb.finish()
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
