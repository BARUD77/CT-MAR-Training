"""ADN generator and discriminator, ported from the official implementation.

Source: https://github.com/liaohaofu/adn (commit 10fd630), files adn/networks/blocks.py and
adn/networks/adn.py. Copyright (c) 2019, Haofu Liao. NLayerDiscriminator is adopted (by ADN) from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, Copyright (c) 2017, Jun-Yan Zhu and
Taesung Park. Both are BSD licensed; see models/adn/LICENSE.

Paper: H. Liao, W.-A. Lin, S. K. Zhou, J. Luo, "ADN: Artifact Disentanglement Network for
Unsupervised Metal Artifact Reduction", IEEE TMI 39(3), 2020.

Port notes (behaviour and parameter names are kept so the official checkpoints load unchanged):
* Layer construction order, default PyTorch initialization and all state_dict keys match the
  original, so a seeded port initializes to the same weights as the original.
* Python-list `layers` and the `fuse` lambda are replaced by equivalent methods (picklable).
* Official quirk, kept on purpose: `decoder_art = deepcopy(decoder)`, but the deep-copied `fuse`
  lambda is still bound to the ORIGINAL decoder, so decoder_art fuses the artifact side features
  with `decoder.fuse{i}`; `decoder_art.fuse{i}` exist (and are in the checkpoint) but are never
  used. The official pretrained deep_lesion_49.pt is consistent with this (decoder_art.fuse* are
  still at their initial-value statistics). Here this is `Decoder._fuse_source`.
* The cached `self.saved` side features of the original forward1/forward2 are passed explicitly.
* `output_act` ('tanh' = official, or 'linear') selects the last decoder activation.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ----------------------------------------------------------------------------------------------
# blocks.py
# ----------------------------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """Per-sample layer norm over (C,H,W) with a per-channel affine (gamma ~ U[0,1), beta = 0)."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.empty(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


PAD = dict(zero=nn.ZeroPad2d, reflect=nn.ReflectionPad2d, replicate=nn.ReplicationPad2d)
NORM = dict(
    none=lambda c: nn.Identity(),
    spectral=lambda c: nn.Identity(),
    batch=nn.BatchNorm2d,
    instance=nn.InstanceNorm2d,  # affine=False, track_running_stats=False (as in the original)
    layer=LayerNorm)
ACTIV = dict(
    none=nn.Identity,
    linear=nn.Identity,
    relu=lambda: nn.ReLU(inplace=True),
    lrelu=lambda: nn.LeakyReLU(0.2, inplace=True),
    tanh=nn.Tanh)


class ConvolutionBlock(nn.Module):
    """pad -> conv2d -> norm -> activation."""

    def __init__(self, norm='instance', activ='relu', pad='reflect', padding=0, **conv_opts):
        super().__init__()
        self.pad = PAD[pad](padding)
        self.conv = nn.Conv2d(**conv_opts)
        self.norm = NORM[norm](conv_opts['out_channels'])
        if norm == "spectral":
            self.conv = spectral_norm(self.conv)
        self.activ = ACTIV[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.conv(self.pad(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm='instance', activ='relu', pad='reflect'):
        super().__init__()
        self.model = nn.Sequential(
            ConvolutionBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=1,
                             padding=1, norm=norm, activ=activ, pad=pad),
            ConvolutionBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=1,
                             padding=1, norm=norm, activ='none', pad=pad))

    def forward(self, x):
        return self.model(x) + x


# ----------------------------------------------------------------------------------------------
# adn.py
# ----------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance'):
        super().__init__()
        self.num_down, self.num_residual = num_down, num_residual
        self.conv0 = ConvolutionBlock(in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
                                      padding=3, pad='reflect', norm=down_norm, activ='relu')
        output_ch = base_ch
        for i in range(1, num_down + 1):
            setattr(self, f"conv{i}", ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4, stride=2, padding=1,
                pad='reflect', norm=down_norm, activ='relu'))
            output_ch *= 2
        for i in range(num_residual):
            setattr(self, f"res{i}", ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

    @property
    def layers(self):
        return [getattr(self, f"conv{i}") for i in range(self.num_down + 1)] + \
               [getattr(self, f"res{i}") for i in range(self.num_residual)]

    def forward(self, x):
        """Returns (code, sides) with sides = all layer outputs, deepest first (original order)."""
        sides = []
        for layer in self.layers:
            x = layer(x)
            sides.append(x)
        return x, sides[::-1]


class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, res_norm='instance',
                 up_norm='layer', fuse=False, output_act='tanh'):
        super().__init__()
        self.num_up, self.num_residual = num_up, num_residual
        input_ch = base_ch * 2 ** num_up
        input_chs = []
        for i in range(num_residual):
            setattr(self, f"res{i}", ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)
        for i in range(num_up):
            setattr(self, f"conv{i}", nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5, stride=1,
                                 padding=2, pad='reflect', norm=up_norm, activ='lrelu')))
            input_chs.append(input_ch)
            input_ch //= 2
        setattr(self, f"conv{num_up}", ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7, stride=1, padding=3, pad='reflect',
            norm='none', activ='tanh' if output_act == 'tanh' else 'linear'))
        input_chs.append(base_ch)

        # fuse=True: concat side + decoder features and 1x1 conv; otherwise add them.
        self.use_fuse = bool(fuse)
        if self.use_fuse:
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, f"fuse{i}", nn.Conv2d(input_chs[i] * 2, input_chs[i], 1))
        # Module whose fuse{i} convs are used (None = self); see the module docstring.
        self._fuse_source = None

    @property
    def layers(self):
        return [getattr(self, f"res{i}") for i in range(self.num_residual)] + \
               [getattr(self, f"conv{i}") for i in range(self.num_up + 1)]

    def _fuse(self, x, y, i):
        if not self.use_fuse:
            return x + y
        src = self if self._fuse_source is None else self._fuse_source
        return getattr(src, f"fuse{i}")(torch.cat((x, y), 1))

    def forward(self, x, sides=()):
        layers = self.layers
        m, n = len(layers), len(sides)
        assert m >= n, "Invalid side inputs"
        for i in range(m - n):
            x = layers[i](x)
        for i, j in enumerate(range(m - n, m)):
            x = self._fuse(x, sides[i], i)
            x = layers[j](x)
        return x


class ADN(nn.Module):
    """Artifact Disentanglement Network generator side ("low" = with artifact, "high" = clean)."""

    def __init__(self, input_ch=1, base_ch=64, num_down=2, num_residual=4, num_sides="all",
                 res_norm='instance', down_norm='instance', up_norm='layer', fuse=True, shared_decoder=False,
                 output_act='tanh'):
        super().__init__()
        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.encoder_low = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_high = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_art = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.decoder = Decoder(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse,
                               output_act)
        if shared_decoder:
            self.decoder_art = self.decoder
        else:
            self.decoder_art = deepcopy(self.decoder)
            # Official behaviour: the deep-copied fuse lambda still refers to the original decoder.
            # object.__setattr__ keeps it out of the module tree (no duplicate parameters/keys).
            object.__setattr__(self.decoder_art, "_fuse_source", self.decoder)

    def encode_art(self, x_low):
        _, sides = self.encoder_art(x_low)
        return sides[-self.n:]

    def forward1(self, x_low):
        """x_low -> (pred_ll, pred_lh, art_sides): reconstruction and artifact-reduced image."""
        art = self.encode_art(x_low)
        code, _ = self.encoder_low(x_low)
        return self.decoder_art(code, art), self.decoder(code), art

    def forward2(self, x_high, art_sides):
        """x_high + artifact of x_low -> (pred_hl, pred_hh): artifact-transferred and reconstruction."""
        code, _ = self.encoder_high(x_high)
        return self.decoder_art(code, art_sides), self.decoder(code)

    def forward_lh(self, x_low):
        code, _ = self.encoder_low(x_low)
        return self.decoder(code)

    def forward_hl(self, x_low, x_high):
        code, _ = self.encoder_high(x_high)
        return self.decoder_art(code, self.encode_art(x_low))

    def forward(self, x_low):
        return self.forward_lh(x_low)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (CycleGAN/pix2pix), as used by ADN. Fully convolutional."""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer="instance"):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = {"layer": nn.LayerNorm, "instance": nn.InstanceNorm2d,
                          "batch": nn.BatchNorm2d, "none": None}[norm_layer]
        use_bias = norm_layer != nn.BatchNorm2d
        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                   bias=use_bias)]
            sequence += ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,
                               bias=use_bias)]
        sequence += ([norm_layer(ndf * nf_mult)] if norm_layer else []) + [nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
