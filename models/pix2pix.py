"""pix2pix generator (U-Net) and 70x70 PatchGAN discriminator for CT MAR.

Ported from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix (models/networks.py):
get_norm_layer, init_weights (normal init only), UnetGenerator, UnetSkipConnectionBlock,
NLayerDiscriminator. The layer structure is kept verbatim; the changes are:
  - the outermost Tanh is optional (output_activation='tanh' | 'none');
  - thin wrappers (Pix2PixGenerator / PatchGANDiscriminator) map the dataset's normalized
    [0, 1] space to the [-1, 1] convention the networks were designed for, and back;
  - the discriminator wrapper takes (x_in, image) and concatenates them (conditional GAN).

Original copyright notice (BSD 2-Clause), as required by the upstream LICENSE:

    Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import functools

import torch
import torch.nn as nn
from torch.nn import init


# ----------------------------- upstream helpers -----------------------------
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="batch"):
    """batch: affine + running stats. instance: no affine, no running stats. none: identity."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        def norm_layer(x):
            return Identity()
        return norm_layer
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_weights(net, init_gain=0.02):
    """Upstream 'normal' init: conv/linear weights ~ N(0, gain), biases 0; BN weight ~ N(1, gain)."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def _uses_bias(norm_layer):
    # No conv bias before BatchNorm (it has its own affine shift); InstanceNorm has no affine.
    if type(norm_layer) == functools.partial:
        return norm_layer.func == nn.InstanceNorm2d
    return norm_layer == nn.InstanceNorm2d


class UnetSkipConnectionBlock(nn.Module):
    """U-Net submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outermost_tanh=True):
        super().__init__()
        self.outermost = outermost
        use_bias = _uses_bias(norm_layer)
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv] + ([nn.Tanh()] if outermost_tanh else [])
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Upstream U-Net, built recursively from the innermost block outwards.
    num_downs=8 -> unet_256 (256x256 input -> 1x1 bottleneck; 512x512 -> 2x2)."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, outermost_tanh=True):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer, outermost_tanh=outermost_tanh)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    """Upstream PatchGAN. n_layers=3 -> 70x70 receptive field. Returns raw logits."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = _uses_bias(norm_layer)
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# ----------------------------- project wrappers -----------------------------
class Pix2PixGenerator(nn.Module):
    """pix2pix U-Net generator operating on the dataset's normalized [0, 1] space.

    forward: x01 -> net(x01 * 2 - 1) -> (out + 1) / 2.
    output_activation='tanh' bounds the output to [0, 1]; 'none' (linear) lets it go outside,
    which is needed when targets are not clipped to the HU window (--no_clip).
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=64, num_downs=8, norm="batch",
                 use_dropout=True, output_activation="tanh", init_gain=0.02):
        super().__init__()
        if output_activation not in ("tanh", "none"):
            raise ValueError(f"output_activation must be 'tanh' or 'none', got {output_activation!r}")
        if num_downs < 5:
            raise ValueError(f"num_downs must be >= 5, got {num_downs}")
        self.num_downs = int(num_downs)
        self.output_activation = output_activation
        self.net = UnetGenerator(in_channels, out_channels, int(num_downs), ngf=int(ngf),
                                 norm_layer=get_norm_layer(norm), use_dropout=bool(use_dropout),
                                 outermost_tanh=(output_activation == "tanh"))
        init_weights(self, init_gain)

    def forward(self, x):
        div = 2 ** self.num_downs
        if x.shape[-2] % div or x.shape[-1] % div:
            raise ValueError(f"Pix2PixGenerator(num_downs={self.num_downs}) needs H, W divisible by {div}; "
                             f"got {tuple(x.shape[-2:])}")
        out = self.net(x * 2.0 - 1.0)
        return (out + 1.0) / 2.0


class PatchGANDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN: D(cat([x_in, image])) -> per-patch logits (no sigmoid).

    in_channels = (#input channels of G) + (#output channels of G). Both inputs are given in the
    normalized [0, 1] space and mapped to [-1, 1] here, matching what the generator sees.
    """

    def __init__(self, in_channels, ndf=64, n_layers=3, norm="batch", init_gain=0.02):
        super().__init__()
        self.net = NLayerDiscriminator(in_channels, ndf=int(ndf), n_layers=int(n_layers),
                                       norm_layer=get_norm_layer(norm))
        init_weights(self, init_gain)

    def forward(self, x_in, image):
        return self.net(torch.cat([x_in, image], dim=1) * 2.0 - 1.0)
