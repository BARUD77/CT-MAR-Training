"""Dual-encoder Swin-UNet (v2 blocks) for CT metal artifact reduction.

Input is the same 2-channel tensor as the MA+LI baseline, [MA, LI]; it is split inside the model.

  * MA encoder + decoder + output head: exactly models/swin_unet_v2 SwinTransformerSys (this class
    subclasses it with in_chans=1), so parameter names match the baseline.
  * LI encoder: own 1-channel patch embedding and light Swin-v2 stages (li_depths blocks per stage),
    same embed dim per stage as the MA encoder. Only the stages needed by fuse_stages are built.
  * Fusion (stage numbering 1..4 = 1/4, 1/8, 1/16, 1/32 resolution of a 512 input):
      s in {1,2,3}: fuses the token map that is BOTH the decoder skip x_downsample[s-1] and the input
                    of encoder stage s (Swin-UNet takes skips at the input of each encoder layer).
      s == 4:       fuses the bottleneck output (after the last encoder layer, before self.norm);
                    there is no skip at 1/32.
    Each fusion block has a scalar alpha initialised to 0 and a residual FFN whose last Linear is
    zero-initialised, so at initialisation the network is exactly an MA-only Swin-UNet.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.swin_unet_v2.swin_unet_v2 import (
    BasicLayer, Mlp, PatchEmbed, PatchMerging, SwinTransformerSys,
)

FUSIONS = ("add", "gate", "xattn")


def sincos_2d(h, w, dim):
    """Fixed 2D sin-cos positional embedding, (h*w, dim), row-major like the token order."""
    assert dim % 4 == 0, f"sin-cos embedding needs dim % 4 == 0, got {dim}"
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    oy = ys.reshape(-1, 1) * omega[None]
    ox = xs.reshape(-1, 1) * omega[None]
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], dim=1)


class CrossAttention(nn.Module):
    """Global multi-head attention, queries from MA tokens, keys/values from LI tokens.

    attn_bf16: on CUDA, run only the softmax(QK^T)V product in bf16 (flash-attention kernel) even when
    the rest of the model trains in fp32; projections stay in the ambient dtype. ~2x faster training for
    fuse_stages=[1,2,3,4] (16k-token attention at 1/4 resolution). No-op under bf16 autocast.
    """

    def __init__(self, dim, num_heads, attn_bf16=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.attn_bf16 = attn_bf16
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q_in, k_in, v_in):
        B, L, C = q_in.shape
        h = self.num_heads

        sdpa_dtype = torch.bfloat16 if (self.attn_bf16 and q_in.is_cuda) else None

        def split(t):
            t = t.view(B, -1, h, C // h).transpose(1, 2)  # B h L d
            return t.to(sdpa_dtype) if sdpa_dtype is not None else t

        q = self.q(q_in)
        out = F.scaled_dot_product_attention(split(q), split(self.k(k_in)), split(self.v(v_in))).to(q.dtype)
        return self.proj(out.transpose(1, 2).reshape(B, L, C))


class FusionBlock(nn.Module):
    """F = F_MA + alpha * mix(F_MA, F_LI);  F = F + MLP(LN(F)).  No BatchNorm."""

    def __init__(self, dim, num_heads, resolution, fusion, mlp_ratio=4.0, attn_bf16=True):
        super().__init__()
        assert fusion in FUSIONS, f"fusion must be one of {FUSIONS}, got {fusion!r}"
        self.fusion = fusion
        self.norm_li = nn.LayerNorm(dim)
        if fusion in ("gate", "xattn"):
            self.norm_ma = nn.LayerNorm(dim)
        if fusion in ("add", "gate"):
            self.li_proj = nn.Linear(dim, dim)
        if fusion == "gate":
            self.gate = nn.Linear(2 * dim, dim)
        if fusion == "xattn":
            self.attn = CrossAttention(dim, num_heads, attn_bf16=attn_bf16)
            self.register_buffer("pos", sincos_2d(resolution[0], resolution[1], dim)[None], persistent=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, f_ma, f_li):
        li = self.norm_li(f_li)
        if self.fusion == "add":
            mix = self.li_proj(li)
        elif self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([self.norm_ma(f_ma), li], dim=-1)))
            mix = g * self.li_proj(li)
        else:  # xattn
            pos = self.pos.to(li.dtype)
            mix = self.attn(self.norm_ma(f_ma) + pos, li + pos, li)
        f = f_ma + self.alpha * mix
        return f + self.ffn(self.norm_ffn(f))


class SwinUnetDual(SwinTransformerSys):
    def __init__(self, fusion="xattn", li_depths=(1, 1, 1, 1), fuse_stages=(3, 4), attn_bf16=True, **kwargs):
        kwargs["in_chans"] = 1  # MA encoder sees the MA channel only
        super().__init__(**kwargs)
        assert fusion in FUSIONS, f"fusion must be one of {FUSIONS}, got {fusion!r}"
        fuse_stages = sorted({int(s) for s in fuse_stages})
        assert fuse_stages and all(1 <= s <= self.num_layers for s in fuse_stages), \
            f"fuse_stages must be in 1..{self.num_layers}, got {fuse_stages}"
        li_depths = [int(d) for d in li_depths]
        assert len(li_depths) == self.num_layers
        self.fusion = fusion
        self.fuse_stages = fuse_stages

        embed_dim = self.embed_dim
        num_heads = kwargs.get("num_heads", [3, 6, 12, 24])
        window_size = kwargs.get("window_size", 7)
        mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        drop_path_rate = kwargs.get("drop_path_rate", 0.1)
        img_size = kwargs.get("img_size", 224)
        patch_size = kwargs.get("patch_size", 4)
        res0 = self.patches_resolution

        # LI encoder: only the stages whose output is consumed by a fusion block.
        last = max(fuse_stages)
        n_li = self.num_layers if last == self.num_layers else last - 1
        self.li_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=1,
                                         embed_dim=embed_dim, norm_layer=nn.LayerNorm if self.patch_norm else None)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(1, sum(li_depths[:n_li])))]
        self.li_layers = nn.ModuleList()
        for i in range(n_li):
            self.li_layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(res0[0] // 2 ** i, res0[1] // 2 ** i),
                depth=li_depths[i], num_heads=num_heads[i], window_size=window_size, mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(li_depths[:i]):sum(li_depths[:i + 1])],
                downsample=PatchMerging if i < self.num_layers - 1 else None))

        # One fusion block per fused stage; stage s works at dim C*2^(s-1), grid res0 / 2^(s-1).
        self.fusions = nn.ModuleDict()
        for s in fuse_stages:
            i = s - 1
            self.fusions[str(s)] = FusionBlock(
                dim=int(embed_dim * 2 ** i), num_heads=num_heads[i],
                resolution=(res0[0] // 2 ** i, res0[1] // 2 ** i), fusion=fusion, mlp_ratio=mlp_ratio,
                attn_bf16=attn_bf16)

        new = [self.li_patch_embed, self.li_layers, self.fusions]
        for m in new:
            m.apply(self._init_weights)
        for fb in self.fusions.values():  # exact MA-only Swin-UNet at init
            nn.init.zeros_(fb.alpha)
            nn.init.zeros_(fb.ffn.fc2.weight)
            nn.init.zeros_(fb.ffn.fc2.bias)

    def forward_li(self, li):
        """LI token maps keyed by stage, at the same positions the MA features are fused."""
        feats = {}
        x = self.li_patch_embed(li)
        for i, layer in enumerate(self.li_layers):
            if (i + 1) in self.fuse_stages and (i + 1) < self.num_layers:
                feats[i + 1] = x
            x = layer(x)
        if len(self.li_layers) == self.num_layers:
            feats[self.num_layers] = x
        elif (len(self.li_layers) + 1) in self.fuse_stages:
            feats[len(self.li_layers) + 1] = x
        return feats

    def forward_features(self, x):
        assert x.shape[1] == 2, f"SwinUnetDual expects [MA, LI] (2 channels), got {x.shape[1]}"
        ma, li = x[:, 0:1], x[:, 1:2]
        li_feats = self.forward_li(li)

        x = self.pos_drop(self.patch_embed(ma))
        x_downsample = []
        for i, layer in enumerate(self.layers):
            s = i + 1
            if s in self.fuse_stages and s < self.num_layers:
                x = self.fusions[str(s)](x, li_feats[s])  # fused map -> skip AND next stage
            x_downsample.append(x)
            x = layer(x)
        if self.num_layers in self.fuse_stages:
            x = self.fusions[str(self.num_layers)](x, li_feats[self.num_layers])  # bottleneck
        x = self.norm(x)
        return x, x_downsample

    @torch.no_grad()
    def fusion_alphas(self):
        """{'fusion/alpha_s3': float, ...} for per-epoch logging."""
        return {f"fusion/alpha_s{s}": float(fb.alpha.item()) for s, fb in self.fusions.items()}
