import argparse
import os
import torch
import time
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from progress.bar import IncrementalBar

from unet import UnetGenerator
from dataset import CTMetalArtifactDataset
from utils import Logger
from metrics import calculate_ssim, calculate_psnr

def main():
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Argument parser
    parser = argparse.ArgumentParser(description="U-Net for Metal Artifact Reduction")
    parser.add_argument('--ma_dir', type=str, required=True, help='Directory for metal artifact .raw images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory for ground truth .raw images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--logger_name', type=str, default='unet_training', help='Logger filename')
    args = parser.parse_args()

    # Image Transform (assuming normalized float32 arrays in [-1, 1])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # From HxW numpy to CxHxW tensor
    # ])

    # Dataset and Dataloader
    dataset = CTMetalArtifactDataset(ma_dir=args.ma_dir, gt_dir=args.gt_dir)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model, optimizer, loss
    model = UnetGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    loss_fn = torch.nn.L1Loss()
    logger = Logger(filename=args.logger_name)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(train_loader))
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            bar.next()
        bar.finish()

        # Validation
        model.eval()
        val_psnr = 0
        val_ssim = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                for i in range(x.size(0)):
                    val_ssim += calculate_ssim(pred[i], y[i])
                    val_psnr += calculate_psnr(pred[i], y[i])

        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = val_psnr / len(val_loader.dataset)
        avg_ssim = val_ssim / len(val_loader.dataset)

        logger.add_scalar("loss", avg_loss, epoch + 1)
        logger.add_scalar("psnr", avg_psnr, epoch + 1)
        logger.add_scalar("ssim", avg_ssim, epoch + 1)
        logger.save_weights(model.state_dict(), "unet")

        print(f"\n[Epoch {epoch+1}] Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | Time: {time.time():.2f}s")

    logger.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
