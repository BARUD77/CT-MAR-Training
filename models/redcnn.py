"""
RED-CNN: Residual Encoder-Decoder Convolutional Neural Network.

Reference:
    Chen et al., "Low-Dose CT With a Residual Encoder-Decoder Convolutional
    Neural Network," IEEE TMI 2017.

All conv / deconv layers are 5x5, stride 1, 'same' padding, so the spatial
resolution of the input is preserved end-to-end. The network therefore works
for any HxW that is a positive multiple of 1; in particular it works directly
on 512x512 CT slices without any resizing or padding.

Shortcut (residual) connections are element-wise sums (NOT concatenations) and
the final shortcut adds the input image to the output, so the network learns
the residual artifact map.
"""

import torch
import torch.nn as nn


class REDCNN(nn.Module):
    """10-layer RED-CNN (5 conv + 5 deconv) as in Chen et al., TMI 2017.

    Args:
        in_channels:  number of input channels (1 for MA-only, 2 for MA+LI,
                      3 for MA+LI+mask, etc.).
        out_channels: number of output channels (1 for grayscale CT).
        num_features: number of feature maps in every hidden layer (96 in the
                      original paper).
        ksize:        convolution kernel size (5 in the original paper).
        use_input_residual: if True, add the first input channel to the final
                      output (global residual learning). Set False if the
                      number of input channels does not match out_channels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 96,
        ksize: int = 5,
        use_input_residual: bool = True,
    ):
        super().__init__()
        p = ksize // 2  # 'same' padding for stride 1

        # ---- Encoder (5 conv layers, no pooling, no stride) ----
        self.conv1 = nn.Conv2d(in_channels,  num_features, ksize, stride=1, padding=p)
        self.conv2 = nn.Conv2d(num_features, num_features, ksize, stride=1, padding=p)
        self.conv3 = nn.Conv2d(num_features, num_features, ksize, stride=1, padding=p)
        self.conv4 = nn.Conv2d(num_features, num_features, ksize, stride=1, padding=p)
        self.conv5 = nn.Conv2d(num_features, num_features, ksize, stride=1, padding=p)

        # ---- Decoder (5 deconv layers, mirror of encoder) ----
        self.deconv1 = nn.ConvTranspose2d(num_features, num_features, ksize, stride=1, padding=p)
        self.deconv2 = nn.ConvTranspose2d(num_features, num_features, ksize, stride=1, padding=p)
        self.deconv3 = nn.ConvTranspose2d(num_features, num_features, ksize, stride=1, padding=p)
        self.deconv4 = nn.ConvTranspose2d(num_features, num_features, ksize, stride=1, padding=p)
        self.deconv5 = nn.ConvTranspose2d(num_features, out_channels, ksize, stride=1, padding=p)

        self.relu = nn.ReLU(inplace=True)

        # Only add the input back if the channel counts allow it.
        self.use_input_residual = bool(use_input_residual) and (in_channels >= out_channels)
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the first `out_channels` of the input for the global residual.
        # For single-channel CT this is just `x` itself.
        if self.use_input_residual:
            x_res = x[:, : self._out_channels]
        else:
            x_res = None

        # Encoder
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        c3 = self.relu(self.conv3(c2))
        c4 = self.relu(self.conv4(c3))
        c5 = self.relu(self.conv5(c4))

        # Decoder with symmetric, every-other-layer residual shortcuts (sum).
        d1 = self.relu(self.deconv1(c5) + c5)
        d2 = self.relu(self.deconv2(d1) + c3)
        d3 = self.relu(self.deconv3(d2))
        d4 = self.relu(self.deconv4(d3) + c1)
        out = self.deconv5(d4)
        if x_res is not None:
            out = out + x_res
        # Final ReLU as in the original paper (targets are non-negative since
        # training data is normalized to [0, 1]).
        out = self.relu(out)
        return out


if __name__ == "__main__":
    # Quick sanity check on a 512x512 input.
    net = REDCNN(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 512, 512)
    y = net(x)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"input  : {tuple(x.shape)}")
    print(f"output : {tuple(y.shape)}")
    print(f"params : {n_params/1e6:.2f}M")
