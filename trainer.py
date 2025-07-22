import argparse
import os
import time
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader
from dataset import CTMetalArtifactDataset
from utils import Logger
from metrics import calculate_ssim, calculate_psnr
from swin_unet.vision_transformer import SwinUnet  # Example
from unet import UnetGenerator  # Example

# Convert nested dict to SimpleNamespace
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
    parser.add_argument('--li_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--logger_name', type=str, default='training_log')
    args = parser.parse_args()

    # Handle SwinUnet YAML config
    if args.model == 'swinunet':
        if not args.config:
            raise ValueError("SwinUnet requires --config path to YAML file.")
        config = load_config(args.config)
        model = SwinUnet(config=config).to('cuda')
        img_size = config.DATA.IMG_SIZE
    else:
        model = UnetGenerator().to('cuda')
        img_size = 512  # default for Unet

    # Dataset and loader
    dataset = CTMetalArtifactDataset(ma_dir=args.ma_dir, li_dir=args.li_dir, gt_dir=args.gt_dir)
    val_split = 0.15
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.L1Loss()
    logger = Logger(filename=args.logger_name)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        total_ssim = total_psnr = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                for i in range(x.size(0)):
                    total_ssim += calculate_ssim(pred[i], y[i])
                    total_psnr += calculate_psnr(pred[i], y[i])

        logger.add_scalar("loss", train_loss / len(train_loader), epoch + 1)
        logger.add_scalar("psnr", total_psnr / len(val_loader.dataset), epoch + 1)
        logger.add_scalar("ssim", total_ssim / len(val_loader.dataset), epoch + 1)
        logger.save_weights(model.state_dict(), f"{args.model}")

        print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} PSNR: {total_psnr:.2f} SSIM: {total_ssim:.4f}")

    logger.close()
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
