import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv => BN => ReLU) x 2
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    If bilinear=True, uses bilinear upsampling + 1x1 conv to reduce channels.
    Else uses transposed convolution.
    """
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

        self.bilinear = bilinear

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: from previous layer (decoder), x2: skip connection (encoder)
        if self.bilinear:
            x1 = self.up(x1)
        else:
            # for transposed conv, reduce channels before upscaling
            x1 = self.up(x1)

        # pad x1 if needed to match x2 (handles odd sizes gracefully)
        diff_y = x2.size(-2) - x1.size(-2)
        diff_x = x2.size(-1) - x1.size(-1)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for 2D image segmentation.
    Default config expects 1-channel 512x512 CT slices but works for any HxW divisible by 16.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        bilinear: bool = False,
    ):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, (base_channels * 16) // factor)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear=bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.outc = OutConv(base_channels, num_classes)

        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)          # 1/1
        x2 = self.down1(x1)       # 1/2
        x3 = self.down2(x2)       # 1/4
        x4 = self.down3(x3)       # 1/8
        x5 = self.down4(x4)       # 1/16

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# Example:
# model = UNet(in_channels=1, num_classes=1, base_channels=64, bilinear=False)
# x = torch.randn(2, 1, 512, 512)
# y = model(x)  # shape: (2, num_classes, 512, 512)
