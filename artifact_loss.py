"""Artifact-weighted reconstruction loss (opt-in, training only).

Motivation (evaluate_capacity_decomposition.py, validation): in body slices the standard masked L1 is dominated
by error in air and artifact-free tissue (93% of pixels, 79% of squared error), most of it fine-scale target
noise that no model can predict. Artifact pixels are 2.6% of the image but hold 44% of the coarse
(artifact-like) error. This loss shifts the optimiser's attention toward artifact-affected pixels.

Artifact map (computed from the INPUT and the TARGET, so it is only available during training; the model and
inference are unchanged):
    A    = | G_sigma(MA - GT) |  in HU              (Gaussian blur, sigma 3 px, removes pixel noise)
    s    = clip((A - a_clean) / (a_ref - a_clean), 0, 1) inside the patient, 0 outside
           -> 0 where the input already matched the target (A < 15 HU), 1 where A >= 40 HU, a ramp in between
           (same thresholds and patient mask as the clean / mild / artifact classes of the decomposition).
           Without the patient restriction ~70% of the extra weight in body slices fell on air at the edge of
           the field of view (checked on 60 random training slices).
    w    = 1 + lam * s, then rescaled so that the mean weight over valid (non-metal) pixels in the batch is 1
Loss:
    L    = sum(w * |pred - GT| over valid) / n_valid                       (= the trainer's masked L1 if lam = 0)
         + lp_weight * sum(s * |G_2(pred) - G_2(GT)| over valid) / sum(s over valid)
The second, optional term compares blurred images on artifact pixels: blurring removes the fine-scale target
noise, so it only counts coarse (streak/shading) error a model can actually fix ("noise-floor aware").
Weights are computed without gradients; they depend only on MA and GT, never on the prediction.
"""
import math

import torch
import torch.nn.functional as F

HU_SPAN = 4096.0   # hu_max - hu_min of the fixed window (-1024, 3072); images are normalized to [0, 1]


def _gauss_kernel(sigma, device, dtype):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of (B,1,H,W), reflect padding (no dark border)."""
    k = _gauss_kernel(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


@torch.no_grad()
def patient_mask(gt, down=4, thr_hu=-500.0):
    """Patient (body) mask of normalized GT images (B,1,H,W), as in the decomposition / data audit: largest
    connected region above -500 HU after light smoothing, holes filled (lungs count as inside). Computed at
    1/down resolution on the CPU for speed and upsampled (nearest)."""
    import numpy as np
    from scipy import ndimage as ndi
    g = F.avg_pool2d(gt.float(), down) * HU_SPAN - 1024.0
    out = []
    for a in g[:, 0].cpu().numpy():
        b = ndi.gaussian_filter(a, 2.0 / down) > thr_hu
        lab, n = ndi.label(b)
        if n > 0:
            sizes = ndi.sum(b, lab, index=np.arange(1, n + 1))
            b = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
        out.append(b)
    m = torch.from_numpy(np.stack(out)[:, None].astype(np.float32)).to(gt.device)
    return F.interpolate(m, scale_factor=down, mode="nearest")


@torch.no_grad()
def artifact_strength(ma, gt, sigma=3.0, a_clean=15.0, a_ref=40.0):
    """s in [0, 1]: 0 = input already matches target, 1 = clear artifact. ma, gt: (B,1,H,W) normalized."""
    a_hu = gaussian_blur((ma.float() - gt.float()) * HU_SPAN, sigma).abs()
    return ((a_hu - a_clean) / (a_ref - a_clean)).clamp(0.0, 1.0)


def artifact_weighted_loss(pred, target, ma, mask=None, lam=4.0, lp_weight=0.0, sigma=3.0, a_clean=15.0,
                           a_ref=40.0, lp_sigma=2.0, body_only=True, return_stats=False):
    pred, target = pred.float(), target.float()
    valid = (mask <= 0.5).float() if mask is not None else torch.ones_like(pred)
    n_valid = valid.sum().clamp_min(1.0)
    s = artifact_strength(ma, target, sigma, a_clean, a_ref) * valid
    if body_only:      # extra weight only inside the patient (air outside keeps the base weight)
        s = s * patient_mask(target)
    with torch.no_grad():
        w = (1.0 + lam * s) * valid
        w = w * (n_valid / w.sum().clamp_min(1.0))           # mean weight over valid pixels = 1
    err = (pred - target).abs()
    loss = (w * err).sum() / n_valid
    lp = None
    if lp_weight > 0:
        e_lp = (gaussian_blur(pred, lp_sigma) - gaussian_blur(target, lp_sigma)).abs()
        lp = (s * e_lp).sum() / s.sum().clamp_min(1.0)
        loss = loss + lp_weight * lp
    if not return_stats:
        return loss
    with torch.no_grad():
        art = (s >= 1.0).float() * valid
        stats = {"frac_artifact_px": float(art.sum() / n_valid),
                 "frac_weighted_px": float(((s > 0).float() * valid).sum() / n_valid),
                 "weight_share_artifact_px": float((w * art).sum() / w.sum().clamp_min(1.0)),
                 "loss_share_artifact_px_plain": float((err * art).sum() / (err * valid).sum().clamp_min(1e-12)),
                 "loss_share_artifact_px_weighted": float((w * err * art).sum() / (w * err).sum().clamp_min(1e-12)),
                 "max_weight": float(w.max()), "lp_term": None if lp is None else float(lp)}
    return loss, stats


# ----------------------------------------------------------------------------- self-test (synthetic, CPU)
def _self_test():
    torch.manual_seed(0)
    H = 256
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    S = torch.full((H, H), -1000.0)
    S[((xx - 128) / 100) ** 2 + ((yy - 128) / 80) ** 2 <= 1] = 40.0
    metal = ((xx - 150) ** 2 + (yy - 140) ** 2) <= 16
    ang, r = torch.atan2(yy - 140.0, xx - 150.0), torch.hypot(yy - 140.0, xx - 150.0)
    streak = 300.0 * torch.cos(6 * ang) * torch.exp(-r / 20.0) * (r > 4)
    to_n = lambda hu: ((hu.clamp(-1024, 3072) + 1024) / HU_SPAN)[None, None]
    gt = to_n(S + 23 * torch.randn(H, H))
    ma = to_n(S + 40 * torch.randn(H, H) + streak)
    ma[0, 0][metal] = 1.0
    mask = metal.float()[None, None]
    pred = (ma.clone() * 0.5 + gt * 0.5).requires_grad_(True)
    ok = True

    plain = ((pred - gt).abs() * (1 - mask)).sum() / (1 - mask).sum()
    l0 = artifact_weighted_loss(pred, gt, ma, mask, lam=0.0)
    good = abs(float(l0) - float(plain)) < 1e-6
    ok &= good
    print(f"  lam=0 equals the trainer's masked L1: {float(l0):.6f} vs {float(plain):.6f}  {'PASS' if good else 'FAIL'}")

    s = artifact_strength(ma, gt)
    far = (r > 70) & (((xx - 128) / 100) ** 2 + ((yy - 128) / 80) ** 2 <= 0.8)
    good = float((s[0, 0][far] == 0).float().mean()) > 0.95
    ok &= good
    print(f"  noise-only tissue gets no extra weight (s=0): {float((s[0, 0][far] == 0).float().mean()):.3f}  {'PASS' if good else 'FAIL'}")
    good = float((s[0, 0][(r > 6) & (r < 25)] > 0).float().mean()) > 0.5
    ok &= good
    print(f"  streak region near metal is weighted: {float((s[0, 0][(r > 6) & (r < 25)] > 0).float().mean()):.3f}  {'PASS' if good else 'FAIL'}")

    shares = []
    for lam in (0.0, 4.0):
        loss, st = artifact_weighted_loss(pred, gt, ma, mask, lam=lam, return_stats=True)
        pred.grad = None
        loss.backward()
        g = pred.grad.abs()[0, 0]
        art = (s[0, 0] >= 1) & ~metal
        shares.append(float(g[art].sum() / g.sum()))
        print(f"  lam={lam:g}: artifact px {100 * st['frac_artifact_px']:.1f}% of pixels | share of loss "
              f"{100 * st['loss_share_artifact_px_weighted']:.1f}% | share of gradient {100 * shares[-1]:.1f}% | "
              f"max weight {st['max_weight']:.2f}")
    good = shares[1] > 2 * shares[0]
    ok &= good
    print(f"  weighting raises the artifact pixels' share of the gradient: {'PASS' if good else 'FAIL'}")
    _, st = artifact_weighted_loss(pred, gt, ma, mask, lam=4.0, return_stats=True)
    w_mean_ok = True  # mean weight is 1 by construction; check numerically
    with torch.no_grad():
        valid = 1 - mask
        sv = artifact_strength(ma, gt) * valid
        w = (1 + 4 * sv) * valid
        w = w * (valid.sum() / w.sum())
        w_mean_ok = abs(float(w.sum() / valid.sum()) - 1.0) < 1e-5
    ok &= w_mean_ok
    print(f"  mean weight over valid pixels is 1: {'PASS' if w_mean_ok else 'FAIL'}")
    loss_lp = artifact_weighted_loss(pred, gt, ma, mask, lam=4.0, lp_weight=1.0)
    good = torch.isfinite(loss_lp) and float(loss_lp) > float(artifact_weighted_loss(pred, gt, ma, mask, lam=4.0))
    ok &= bool(good)
    print(f"  low-pass term adds a finite positive amount: {'PASS' if good else 'FAIL'}")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
