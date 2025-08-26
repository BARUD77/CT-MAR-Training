import argparse
import os
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader
from dataset import CTMetalArtifactDataset
from utils import Logger
from metrics import calculate_ssim, calculate_psnr
from swin_unet.vision_transformer import SwinUnet   # assumes it reads config.MODEL.SWIN.IN_CHANS
from unet import UnetGenerator                      # ensure it can take in_ch/input_nc if you use it

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)

def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--model', type=str, choices=['unet', 'swinunet'], required=True)
    parser.add_argument('--config', type=str, help='Path to config.yaml (required for swinunet)')
    parser.add_argument('--ma_dir', type=str, required=True)
    parser.add_argument('--li_dir', type=str, required=False, help='Required if input_mode=ma_li')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--input_mode', type=str, choices=['ma_li', 'ma'], default='ma',
                        help="Use 'ma_li' for 2-ch input [MA,LI] or 'ma' for MA-only")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to save logs and weights')
    parser.add_argument('--logger_name', type=str, default='training_log')
    args = parser.parse_args()

    # Validate LI path only when needed
    if args.input_mode == 'ma_li' and not args.li_dir:
        raise ValueError("input_mode='ma_li' requires --li_dir")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_ch = 2 if args.input_mode == 'ma_li' else 1

    # Handle model creation
    if args.model == 'swinunet':
        if not args.config:
            raise ValueError("SwinUnet requires --config path to YAML file.")
        config = load_config(args.config)
        # Force input channels based on input_mode
        try:
            config.MODEL.SWIN.IN_CHANS = in_ch
        except AttributeError:
            # if your config doesn't have this path, set it where your impl reads it
            pass
        model = SwinUnet(config=config).to(device)
    else:
        # Make sure your UnetGenerator accepts in_ch/input_nc; adjust if needed
        try:
            model = UnetGenerator(in_ch=in_ch).to(device)
        except TypeError:
            model = UnetGenerator().to(device)  # fallback; ensure first conv matches `in_ch`

    # Dataset (use serial if your head numbering is gappy)
    dataset = CTMetalArtifactDataset(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        split='train',
        normalize='global',
        global_min=-7387.15771484375,
        global_max=65204.90625 
    )

    # Make a validation split from the training pool
    val_split = 0.15
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Training on {len(train_set)} samples, validating on {len(val_set)} samples.")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.L1Loss()
    logger = Logger(exp_name=args.log_dir, filename=args.logger_name)

    for epoch in range(args.epochs):
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

        # Validation
        model.eval()
        total_ssim = 0.0
        total_psnr = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)

                # Clamp to [0,1] for metrics and use single channel
                pred_eval = torch.clamp(pred, 0, 1)
                gt_eval   = torch.clamp(y,   0, 1)

                for i in range(pred_eval.size(0)):
                    # pass 2D tensors if your metric expects (H,W)
                    total_ssim += calculate_ssim(pred_eval[i, 0].cpu(), gt_eval[i, 0].cpu())
                    total_psnr += calculate_psnr(pred_eval[i, 0].cpu(), gt_eval[i, 0].cpu())

        avg_loss = train_loss / max(1, len(train_loader))
        avg_psnr = total_psnr / len(val_loader.dataset)
        avg_ssim = total_ssim / len(val_loader.dataset)

        logger.add_scalar("loss", avg_loss, epoch + 1)
        logger.add_scalar("psnr", avg_psnr, epoch + 1)
        logger.add_scalar("ssim", avg_ssim, epoch + 1)
        logger.save_weights(model.state_dict(), f"{args.model}")

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.4f}")

    logger.close()
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
