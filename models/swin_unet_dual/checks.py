"""Checks for the dual-encoder Swin-UNet models (Step 2 tests 1-6).

Each check takes plain tensors, so the same code runs on synthetic data (this file's __main__, local)
and on real data (colab/swin_dual_checks.py). Models are built through the trainer's build_model so the
registration is exercised too.

  python -m models.swin_unet_dual.checks [--amp] [--perf_bs 2 4 8] [--overfit_steps 200]
"""
import argparse
import contextlib
import io
import math
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
try:
    import wandb  # noqa: F401
except ImportError:  # local machine without wandb: the trainer imports it at module level
    sys.modules["wandb"] = types.ModuleType("wandb")
from trainer_wandb_unclipped import build_model, param_groups, recon_loss, apply_metal_on_gt  # noqa: E402

BASE_CFG = os.path.join(_REPO, "models", "swin_unet_v2", "config.yaml")
WIDE_CFG = os.path.join(_REPO, "models", "swin_unet_dual", "config_wide.yaml")
MODELS = {  # name -> (trainer --model, --model_cfg, --model_kwargs)
    "swinunet_v2 (MA+LI)": ("swinunet_v2", BASE_CFG, None),
    "swin_dual_add": ("swin_dual_add", BASE_CFG, None),
    "swin_dual_gate": ("swin_dual_gate", BASE_CFG, None),
    "swin_dual_xattn": ("swin_dual_xattn", BASE_CFG, {"fuse_stages": [1, 2, 3, 4]}),  # x-attn at every stage
    "swin_wide": ("swin_wide", WIDE_CFG, None),
}
DUAL = ["swin_dual_add", "swin_dual_gate", "swin_dual_xattn"]
RESULTS = []


def verdict(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  -> {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""), flush=True)


def build(key, device):
    model_name, cfg, kwargs = MODELS[key]
    with contextlib.redirect_stdout(io.StringIO()):
        return build_model(model_name, in_ch=2, num_classes=1, device=device, model_cfg_path=cfg,
                           model_kwargs=dict(kwargs) if kwargs else None)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def li_encoder_params(m):
    return [p for n, p in m.named_parameters() if n.startswith(("li_patch_embed.", "li_layers."))]


def make_optimizer(model, lr):
    # identical to the trainer: AdamW, wd 1e-4 with bias/norm excluded
    return torch.optim.AdamW(param_groups(model, wd=1e-4), lr=lr, betas=(0.9, 0.999), eps=1e-8)


def train_step(model, opt, x, y, mask, amp, l1=torch.nn.L1Loss()):
    """One trainer-equivalent step for this run config: metal painted on GT, masked L1, ssim weight 0."""
    y = apply_metal_on_gt(y, x[:, :1], mask) if mask is not None else y
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
        pred = model(x)
        loss = recon_loss(pred, y, mask, l1, ssim_weight=0.0, ssim_dilation=2)
    loss.backward()
    opt.step()
    return float(loss.item())


# ----------------------------------------------------------------------------- tests
def test_forward(device):
    print("\n1) FORWARD on random [2,2,512,512]")
    x = torch.rand(2, 2, 512, 512, device=device)
    for key in MODELS:
        m = build(key, device).eval()
        with torch.no_grad():
            out = m(x)
        ok = tuple(out.shape) == (2, 1, 512, 512) and bool(torch.isfinite(out).all())
        verdict(f"forward {key}", ok, f"shape={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
        del m


def test_li_invariance(device, x):
    print("\n2) LI INVARIANCE AT INIT (alpha=0): perturb LI channel, compare outputs")
    x = x[:2].to(device)
    x2 = x.clone()
    x2[:, 1:2] = torch.rand_like(x2[:, 1:2])
    for key in DUAL:
        m = build(key, device).eval()
        with torch.no_grad():
            d = (m(x) - m(x2)).abs().max().item()
        verdict(f"LI-invariant {key}", d == 0.0, f"max|diff|={d:.3e}")
        # Same weights in a plain 1-channel MA-only SwinTransformerSys -> identical output.
        from models.swin_unet_v2.swin_unet_v2 import SwinTransformerSys
        with contextlib.redirect_stdout(io.StringIO()):
            ref = build_model("swinunet_v2", in_ch=1, num_classes=1, device=device,
                              model_cfg_path=BASE_CFG).eval()
        sd = {k: v for k, v in m.state_dict().items() if not k.startswith(("li_", "fusions."))}
        ref.load_state_dict(sd, strict=True)
        assert isinstance(ref, SwinTransformerSys)
        with torch.no_grad():
            d_ref = (m(x) - ref(x[:, :1])).abs().max().item()
        verdict(f"== MA-only Swin-UNet at init {key}", d_ref == 0.0, f"max|diff|={d_ref:.3e}")
        del m, ref


def test_gradients(device, x, y, mask, lr, amp):
    print(f"\n3) GRADIENTS over 2 optimizer steps on one batch (bs={x.shape[0]}, lr={lr}, amp={amp})")
    x, y = x.to(device), y.to(device)
    mask = mask.to(device) if mask is not None else None
    for key in DUAL:
        torch.manual_seed(0)
        m = build(key, device).train()
        opt = make_optimizer(m, lr)
        info = []
        alpha_ok = li_ok = True
        for step in (1, 2):
            y_ = apply_metal_on_gt(y, x[:, :1], mask) if mask is not None else y
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = recon_loss(m(x), y_, mask, torch.nn.L1Loss(), ssim_weight=0.0)
            loss.backward()
            a_grads = {s: (fb.alpha.grad.abs().item() if fb.alpha.grad is not None else 0.0)
                       for s, fb in m.fusions.items()}
            li_gn = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in li_encoder_params(m)
                                  if p.grad is not None))
            info.append(f"step{step}: |dL/dalpha|={ {k: f'{v:.2e}' for k, v in a_grads.items()} } "
                        f"||grad LI-enc||={li_gn:.2e} alpha={m.fusion_alphas()}")
            if step == 1:
                alpha_ok = all(v > 0 for v in a_grads.values())
            else:
                li_ok = li_gn > 0
            opt.step()
        for line in info:
            print("    " + line)
        verdict(f"alpha grad nonzero at step 1 [{key}]", alpha_ok)
        verdict(f"LI-encoder grad nonzero at step 2 [{key}]", li_ok)
        del m, opt


def test_perf(device, batch_sizes, amp_modes, steps=6):
    print(f"\n4) PARAMS / PEAK MEMORY / SEC PER STEP  (device={torch.cuda.get_device_name(0)})")
    print(f"    {'model':22s} {'params':>12s} {'amp':>5s} {'bs':>4s} {'peak GiB':>9s} {'s/step':>7s}")
    rows = []
    for key in MODELS:
        for amp in amp_modes:
            for bs in batch_sizes:
                torch.cuda.empty_cache()
                try:
                    m = build(key, device).train()
                    opt = make_optimizer(m, 2e-4)
                    x = torch.rand(bs, 2, 512, 512, device=device)
                    y = torch.rand(bs, 1, 512, 512, device=device)
                    mask = torch.zeros(bs, 1, 512, 512, device=device)
                    torch.cuda.reset_peak_memory_stats()
                    ts = []
                    for _ in range(steps):
                        torch.cuda.synchronize(); t0 = time.time()
                        train_step(m, opt, x, y, mask, amp)
                        torch.cuda.synchronize(); ts.append(time.time() - t0)
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    sps = sum(ts[2:]) / len(ts[2:])
                    print(f"    {key:22s} {n_params(m):12,d} {str(amp):>5s} {bs:4d} {peak:9.2f} {sps:7.3f}", flush=True)
                    rows.append((key, n_params(m), amp, bs, peak, sps))
                    del m, opt, x, y, mask
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    print(f"    {key:22s} {'':12s} {str(amp):>5s} {bs:4d} {'OOM':>9s}", flush=True)
                    m = opt = x = y = mask = None
                    torch.cuda.empty_cache()
    return rows


def test_overfit(device, x, y, mask, steps, lr, amp, ratio_thresh=0.5):
    print(f"\n5) OVERFIT {steps} steps on {x.shape[0]} fixed slices (lr={lr} constant, amp={amp})")
    x, y = x.to(device), y.to(device)
    mask = mask.to(device) if mask is not None else None
    for key in MODELS:
        torch.manual_seed(0)
        m = build(key, device).train()
        opt = make_optimizer(m, lr)
        losses = [train_step(m, opt, x, y, mask, amp) for _ in range(steps)]
        first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
        marks = " ".join(f"@{i}={losses[i - 1]:.4f}" for i in (1, steps // 4, steps // 2, steps) if i >= 1)
        extra = f" alphas={ {k: round(v, 4) for k, v in m.fusion_alphas().items()} }" if hasattr(m, "fusion_alphas") else ""
        print(f"    {key:22s} {marks}{extra}")
        verdict(f"overfit {key}", math.isfinite(last) and last < ratio_thresh * first,
                f"mean first5={first:.4f} last5={last:.4f} ratio={last / first:.3f}")
        del m, opt


def test_alpha_logging(device):
    print("\n6) ALPHA LOGGING HOOK (model.fusion_alphas -> wandb payload in the trainer)")
    for key in DUAL:
        m = build(key, device)
        a = m.fusion_alphas()
        want = {f"fusion/alpha_s{s}" for s in m.fuse_stages}
        ok = set(a) == want and all(isinstance(v, float) for v in a.values())
        verdict(f"fusion_alphas {key}", ok, str(a))
        del m


def summary():
    print("\n" + "=" * 70)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    for name, ok in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return n_fail


# ----------------------------------------------------------------------------- synthetic data
def synthetic_slices(n, seed=0, size=512):
    """Smooth 'anatomy' GT, metal disks, streaky MA, smoothed LI-like prior. Values in [0,1]."""
    g = torch.Generator().manual_seed(seed)
    gt = F.avg_pool2d(torch.rand(n, 1, size // 16, size // 16, generator=g), 3, 1, 1)
    gt = 0.2 + 0.3 * F.interpolate(gt, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    mask = torch.zeros(n, 1, size, size)
    streaks = torch.zeros(n, 1, size, size)
    for i in range(n):
        cy, cx = (torch.rand(2, generator=g) * size * 0.5 + size * 0.25).tolist()
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask[i, 0] = (r2 < 12 ** 2).float()
        ang = torch.atan2(yy - cy, xx - cx)
        streaks[i, 0] = 0.15 * torch.cos(8 * ang) * torch.exp(-r2.sqrt() / 150)
    ma = torch.where(mask > 0.5, torch.ones_like(gt), (gt + streaks).clamp(0, 1))
    li = (gt + 0.02 * F.interpolate(torch.randn(n, 1, size // 8, size // 8, generator=g), size=size,
                                    mode="bilinear", align_corners=False)).clamp(0, 1)
    return torch.cat([ma, li], 1), gt, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for tests 3 and 5")
    ap.add_argument("--perf_bs", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--perf_amp", type=str, default="both", choices=["both", "amp", "fp32"])
    ap.add_argument("--overfit_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--skip", type=str, nargs="*", default=[], help="test numbers to skip, e.g. 4 5")
    ap.add_argument("--models", type=str, nargs="*", default=None, help="subset of MODELS keys")
    ap.add_argument("--xattn_stages", type=int, nargs="+", default=None, help="override xattn fuse_stages")
    args = ap.parse_args()
    if args.xattn_stages:
        MODELS["swin_dual_xattn"] = ("swin_dual_xattn", BASE_CFG, {"fuse_stages": args.xattn_stages})
    if args.models:
        for k in list(MODELS):
            if k not in args.models:
                del MODELS[k]
        DUAL[:] = [k for k in DUAL if k in MODELS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"SYNTHETIC checks | torch {torch.__version__} | device {device}")
    x, y, mask = synthetic_slices(4)
    if "1" not in args.skip: test_forward(device)
    if "2" not in args.skip: test_li_invariance(device, x)
    if "3" not in args.skip: test_gradients(device, x[:2], y[:2], mask[:2], args.lr, args.amp)
    if "4" not in args.skip and device.type == "cuda":
        test_perf(device, args.perf_bs, {"both": [True, False], "amp": [True], "fp32": [False]}[args.perf_amp])
    if "5" not in args.skip: test_overfit(device, x, y, mask, args.overfit_steps, args.lr, args.amp)
    if "6" not in args.skip: test_alpha_logging(device)
    sys.exit(1 if summary() else 0)


if __name__ == "__main__":
    main()
