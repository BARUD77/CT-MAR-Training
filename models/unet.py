import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Encoder block with optional BatchNorm"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        layers = [
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block with optional Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        layers = [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UnetGenerator(nn.Module):
    """UNet-style encoder-decoder with skip connections"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)  # no norm
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 512)
        self.enc6 = EncoderBlock(512, 512)
        self.enc7 = EncoderBlock(512, 512)
        self.enc8 = EncoderBlock(512, 512, norm=False)

        # Decoder
        self.dec8 = DecoderBlock(512, 512, dropout=True)
        self.dec7 = DecoderBlock(1024, 512, dropout=True)
        self.dec6 = DecoderBlock(1024, 512, dropout=True)
        self.dec5 = DecoderBlock(1024, 512)
        self.dec4 = DecoderBlock(1024, 256)
        self.dec3 = DecoderBlock(512, 128)
        self.dec2 = DecoderBlock(256, 64)
        self.dec1 = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoding path with skip connections
        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([d8, e7], dim=1))
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(d1)
