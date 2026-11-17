"""Decoupled near-metal refiner: a small residual U-Net applied on top of a FROZEN global MAR network, confined to
the metal neighbourhood by a fixed distance gate.

    final = base + g(d) * r,     r = RefinerUNet([MA, LI, base, dist, metal])

    d      distance to the nearest metal pixel in mm (metal mask known at inference)
    g(d)   1 for d <= r_full (10 mm), cosine fade to 0 at r_zero (20 mm), 0 beyond
           -> beyond r_zero the output equals the global network's output exactly (by construction)
    r      residual in the normalized image space ([0, 1] over [-1024, 3072] HU); the last layer is zero-initialized,
           so before training final == base.

Inputs are (B,1,H,W) normalized images; the network is fully convolutional (trained on crops, applied to full images).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIST_SCALE_MM = 20.0      # distance channel = min(d, 2 * DIST_SCALE_MM) / DIST_SCALE_MM  -> [0, 2]
PX_MM = {"body": 400.0 / 512, "head": 221.6 / 512}   # training data pixel size (as in the capacity decomposition)


# ----------------------------------------------------------------------------- distance + gate
def distance_mm(metal, px_mm):
    """metal: (H,W) bool numpy; returns (H,W) float32 distance (mm) to the nearest metal pixel (0 on metal).
    Without metal: a large distance everywhere (gate 0, refiner inactive)."""
    from scipy import ndimage as ndi
    if not metal.any():
        return np.full(metal.shape, 1e4, np.float32)
    return (ndi.distance_transform_edt(~metal) * px_mm).astype(np.float32)


def gate_from_distance(d_mm, r_full=10.0, r_zero=20.0):
    """Works on numpy arrays or torch tensors. 1 inside r_full, cosine fade, exactly 0 at and beyond r_zero."""
    if isinstance(d_mm, torch.Tensor):
        t = ((d_mm - r_full) / (r_zero - r_full)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * t))
    t = np.clip((d_mm - r_full) / (r_zero - r_full), 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)


def dist_channel(d_mm):
    if isinstance(d_mm, torch.Tensor):
        return d_mm.clamp(max=2 * DIST_SCALE_MM) / DIST_SCALE_MM
    return (np.minimum(d_mm, 2 * DIST_SCALE_MM) / DIST_SCALE_MM).astype(np.float32)


# ----------------------------------------------------------------------------- network
class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class RefinerUNet(nn.Module):
    """Small U-Net (4 levels). in_ch = 5 ([MA, LI, base, dist, metal]) or 4 without the distance channel."""

    def __init__(self, in_ch=5, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        c = in_ch
        for co in ch:
            self.enc.append(_Block(c, co))
            c = co
        self.dec = nn.ModuleList([_Block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1))])
        self.out = nn.Conv2d(ch[0], 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** (len(self.enc) - 1)
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for blk, s in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, s], 1))
        return self.out(x)[..., :H, :W]


class GatedRefiner(nn.Module):
    """final = base + g * r. Inputs (B,1,H,W): ma, li, base (normalized), dist (mm), metal (0/1)."""

    def __init__(self, use_dist=True, base=32, r_full=10.0, r_zero=20.0):
        super().__init__()
        self.use_dist, self.r_full, self.r_zero = use_dist, r_full, r_zero
        self.net = RefinerUNet(in_ch=5 if use_dist else 4, base=base)

    def inputs(self, ma, li, base, dist, metal):
        chans = [ma, li, base] + ([dist_channel(dist)] if self.use_dist else []) + [metal]
        return torch.cat(chans, 1)

    def forward(self, ma, li, base, dist, metal, return_parts=False):
        g = gate_from_distance(dist, self.r_full, self.r_zero)
        r = self.net(self.inputs(ma, li, base, dist, metal))
        out = base + g * r
        return (out, r, g) if return_parts else out


def config_dict(model):
    return {"use_dist": model.use_dist, "base": model.net.enc[0].net[0].out_channels,
            "r_full": model.r_full, "r_zero": model.r_zero}


def load_refiner(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GatedRefiner(**ck["config"])
    m.load_state_dict(ck["model_state"], strict=True)
    return m.to(device).eval(), ck


# ----------------------------------------------------------------------------- self-test (synthetic)
def _self_test():
    torch.manual_seed(0)
    ok = True
    H = 160
    metal = np.zeros((H, H), bool)
    metal[70:76, 60:68] = True
    d = distance_mm(metal, 0.78125)
    g = gate_from_distance(d)
    good = g[metal].min() == 1.0 and g[d >= 20].max() == 0.0 and 0 < g[(d > 12) & (d < 18)].mean() < 1
    print(f"  gate: 1 at metal, 0 beyond 20 mm, fading between: {'PASS' if good else 'FAIL'}")
    ok &= bool(good)
    m = GatedRefiner()
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None, None]
    ma, li, base = torch.rand(1, 1, H, H), torch.rand(1, 1, H, H), torch.rand(1, 1, H, H)
    out = m(ma, li, base, t(d), t(metal))
    good = torch.equal(out, base)
    print(f"  zero-initialized: output == base before training: {'PASS' if good else 'FAIL'}")
    ok &= good
    with torch.no_grad():                              # random weights: far field must still be identical
        for p in m.parameters():
            p.normal_(0, 0.1)
    out, r, gg = m(ma, li, base, t(d), t(metal), return_parts=True)
    far = t(d) >= 20
    good = torch.equal(out[far], base[far]) and (out[~far] != base[~far]).any()
    print(f"  far field (>= 20 mm) identical to base for arbitrary weights, near field changed: "
          f"{'PASS' if good else 'FAIL'}")
    ok &= bool(good)
    good = m(ma[..., :150, :137], li[..., :150, :137], base[..., :150, :137], t(d)[..., :150, :137],
             t(metal)[..., :150, :137]).shape == (1, 1, 150, 137)
    print(f"  odd image sizes handled: {'PASS' if good else 'FAIL'}")
    ok &= good
    n = sum(p.numel() for p in GatedRefiner().parameters())
    print(f"  parameters: {n / 1e6:.2f}M")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
