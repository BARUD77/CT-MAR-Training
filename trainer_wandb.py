# trainer_wandb.py
import argparse
import os
import json
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader

from dataset import CTMetalArtifactDataset
from models.swin_unet.vision_transformer import SwinUnet
from models.unet import UnetGenerator                               # <- the vanilla UNet I shared earlier

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
    Centralized place to instantiate any model without changing the training loop.
    Extend this with new entries as you add architectures.
    """
    model_kwargs = model_kwargs or {}

    name = name.lower()
    if name in ("unet", "vanilla_unet", "u-net"):
        # UNet uses explicit keyword args. We allow overrides from --model_kwargs.
        base_channels = int(model_kwargs.pop("base_channels", 64))
        bilinear = bool(model_kwargs.pop("bilinear", False))
        m = UnetGenerator(
            in_channels=in_ch,
            num_classes=num_classes,
            base_channels=base_channels,
            bilinear=bilinear,
            **model_kwargs
        )
        return m.to(device)

    # elif name in ("unetgen", "pix2pix_unet"):
    #     # Your existing UnetGenerator (some repos call it with only in_ch, some without).
    #     try:
    #         m = UnetGenerator(in_ch=in_ch, **model_kwargs)
    #     except TypeError:
    #         # some versions ignore in_ch
    #         m = UnetGenerator(**model_kwargs)
    #     return m.to(device)

    elif name in ("swinunet", "swin_unet", "swin-u"):
        if not model_cfg_path:
            raise ValueError("SwinUnet requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        # many SwinUnet repos store this here; guard if layout differs
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except AttributeError:
            pass
        # pass through arbitrary kwargs if your impl supports them
        m = SwinUnet(config=cfg_ns, **model_kwargs)
        return m.to(device)

    else:
        raise ValueError(f"Unknown model: {name}. Supported: unet, unetgen, swinunet")


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Training script (W&B only, model-agnostic)")
    # Model selection / config
    parser.add_argument('--model', type=str,
                        choices=['unet', 'unetgen', 'swinunet'],
                        required=True,
                        help="Pick an architecture from the registry.")
    parser.add_argument('--model_cfg', type=str, default=None,
                        help="Optional: path to a YAML config (required for swinunet).")
    parser.add_argument('--model_kwargs', type=str, default=None,
                        help='Optional: JSON dict of kwargs for the chosen model. '
                             'Example: \'{"base_channels":48,"bilinear":true}\'')

    # Data & run setup
    parser.add_argument('--ma_dir', type=str, required=True)
    parser.add_argument('--li_dir', type=str, required=False, help='Required if input_mode=ma_li')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--input_mode', type=str, choices=['ma_li', 'ma'], default='ma',
                        help="Use 'ma_li' for 2-ch input [MA,LI] or 'ma' for MA-only")
    parser.add_argument('--num_classes', type=int, default=1, help='Output channels/classes')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to save logs and weights')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--project', type=str, default='ct-mar', help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (team/user) if needed')

    args = parser.parse_args()

    # Validate LI path only when needed
    if args.input_mode == 'ma_li' and not args.li_dir:
        raise ValueError("input_mode='ma_li' requires --li_dir")

    os.makedirs(args.log_dir, exist_ok=True)

    # ---------------- W&B init ----------------
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=vars(args),   # logs CLI as config
        dir=args.log_dir,
        save_code=True,
        reinit=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_ch = 2 if args.input_mode == 'ma_li' else 1

    # ---------------- Build model (agnostic) ----------------
    model = build_model(
        name=args.model,
        in_ch=in_ch,
        num_classes=args.num_classes,
        device=device,
        model_cfg_path=args.model_cfg,
        model_kwargs=parse_json_or_none(args.model_kwargs)
    )

    # (Optional) track gradients/weights
    wandb.watch(model, log='gradients', log_freq=1)

    # ---------------- Dataset / loaders ----------------
    HU_MIN, HU_MAX = -1024.0, 3072.0
    train_ds = CTMetalArtifactDataset(
        ma_dir=args.ma_dir, gt_dir=args.gt_dir, mask_dir=None,
        split='train', hu_min=HU_MIN, hu_max=HU_MAX
    )
    val_ds = CTMetalArtifactDataset(
        ma_dir=args.ma_dir, gt_dir=args.gt_dir, mask_dir=None,
        split='val', hu_min=HU_MIN, hu_max=HU_MAX
    )

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")

    # ---------------- Optimizer / loss ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.L1Loss()

    best_ssim = -1.0
    best_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / max(1, len(train_loader))

        # ---------------- Val ----------------
        model.eval()
        total_ssim = 0.0
        total_psnr = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)

                # Clamp to [0,1] for metrics
                pred_eval = torch.clamp(pred, 0, 1)
                gt_eval   = torch.clamp(y,   0, 1)

                # loop over batch
                for i in range(pred_eval.size(0)):
                    total_ssim += compute_SSIM(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)
                    total_psnr += compute_PSNR(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)

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

        # upload "last" each epoch (keeps only latest)
        last_art = wandb.Artifact(f"{args.model}-last", type="model")
        last_art.add_file(ckpt_path)
        wandb.log_artifact(last_art, aliases=["latest"])

        # save best-by-SSIM
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
