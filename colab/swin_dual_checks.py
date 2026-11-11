"""Dual-encoder Swin-UNet checks on REAL data (run in Colab on the training GPU). Each check ends PASS/FAIL.

Uses the TRAIN split only (built via baselines/common/data.py); the test set is never touched.
  1. forward on random [2,2,512,512] for the baseline, the 3 dual models and swin_wide
  2. at init (alpha=0): dual output independent of LI, and identical to an MA-only Swin-UNet
  3. gradients over 2 optimizer steps on one real batch: alpha grad != 0 at step 1, LI-encoder grad != 0 at step 2
  4. params / peak GPU memory / s per step at --perf_bs (default 16), with bf16 AMP and fp32 (your runs: fp32)
  5. overfit: --overfit_steps (200) on 4 fixed real train slices, loss must fall below half its start
  6. model.fusion_alphas() payload
  7. trainer smoke run: 1 epoch of trainer_wandb_unclipped.py per new model on a debug subset (W&B disabled);
     checks the [Epoch 1] line and, for the dual models, the per-epoch alpha log line.

Usage (repo root):
  python colab/swin_dual_checks.py --ma_dir /content/data/MA/MA_image --li_dir /content/data/LI/LI \
      --mask_dir /content/data/mask/metal_mask --gt_dir /content/data/GT/GT --work_dir /content/swin_dual_checks
"""
import argparse
import json
import os
import re
import subprocess
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.data import build_datasets, unpack_batch  # noqa: E402
from models.swin_unet_dual import checks as C  # noqa: E402

TRAINER = os.path.join(_REPO_ROOT, "trainer_wandb_unclipped.py")


def real_slices(train_ds, n):
    xs, ys, ms = [], [], []
    for i in range(n):  # first n slices of the fixed train split (deterministic)
        s = unpack_batch(train_ds[i], has_mask=True, has_li=True)
        xs.append(torch.cat([s["ma"], s["li"]], 0))
        ys.append(s["gt"])
        ms.append(s["mask"])
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


def trainer_smoke(args):
    print("\n7) TRAINER SMOKE RUN (1 epoch, debug subset 32/8, W&B disabled, same flags as the real runs)")
    for model in ["swin_dual_add", "swin_dual_gate", "swin_dual_xattn", "swin_wide"]:
        cfg = C.WIDE_CFG if model == "swin_wide" else C.BASE_CFG
        cmd = [sys.executable, TRAINER, "--model", model, "--model_cfg", cfg,
               "--ma_dir", args.ma_dir, "--li_dir", args.li_dir, "--mask_dir", args.mask_dir, "--gt_dir", args.gt_dir,
               "--input_mode", "ma_li", "--region_policy", "all", "--hu_min", "-1024", "--hu_max", "3072",
               "--batch_size", "4", "--epochs", "1", "--learning_rate", "2e-4", "--warmup_epochs", "5",
               "--min_lr_ratio", "0.01", "--ssim_loss_weight", "0", "--monitor_metric", "ssim",
               "--metal_mask_on_gt", "--num_workers", "2",
               "--debug_max_train_samples", "32", "--debug_max_val_samples", "8",
               "--project", "swin-dual-checks", "--log_dir", os.path.join(args.work_dir, model)]
        kw = C.MODELS[model][2]
        if kw:
            cmd += ["--model_kwargs", json.dumps(kw)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT,
                           env=dict(os.environ, WANDB_MODE="disabled"))
        out = p.stdout + "\n" + p.stderr
        for line in out.splitlines():
            if any(k in line for k in ("[Epoch", "fusion/alpha", "Training on", "Error", "Traceback")):
                print("   ", line)
        ok = p.returncode == 0 and re.search(r"\[Epoch 1\] Loss: [0-9.]+", out) is not None
        if model != "swin_wide":
            stages = (kw or {}).get("fuse_stages", [3, 4])
            ok = ok and all(f"fusion/alpha_s{s}=" in out for s in stages)
        C.verdict(f"trainer 1-epoch {model}", ok, f"returncode={p.returncode}")


def main():
    ap = argparse.ArgumentParser()
    for d in ("ma_dir", "gt_dir", "li_dir", "mask_dir"):
        ap.add_argument(f"--{d}", required=True)
    ap.add_argument("--work_dir", default="/content/swin_dual_checks", help="smoke-run checkpoints (local disk)")
    ap.add_argument("--grad_bs", type=int, default=4)
    ap.add_argument("--perf_bs", type=int, nargs="+", default=[16])
    ap.add_argument("--perf_amp", default="both", choices=["both", "amp", "fp32"])
    ap.add_argument("--overfit_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--amp", action="store_true", help="bf16 AMP for tests 3/5 (your runs are fp32: leave off)")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"REAL-DATA checks | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    train_ds, _ = build_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir)
    x, y, mask = real_slices(train_ds, max(4, args.grad_bs))

    if "1" not in args.skip: C.test_forward(device)
    if "2" not in args.skip: C.test_li_invariance(device, x)
    if "3" not in args.skip: C.test_gradients(device, x[:args.grad_bs], y[:args.grad_bs], mask[:args.grad_bs],
                                              args.lr, args.amp)
    if "4" not in args.skip: C.test_perf(device, args.perf_bs,
                                         {"both": [True, False], "amp": [True], "fp32": [False]}[args.perf_amp])
    if "5" not in args.skip: C.test_overfit(device, x[:4], y[:4], mask[:4], args.overfit_steps, args.lr, args.amp)
    if "6" not in args.skip: C.test_alpha_logging(device)
    if "7" not in args.skip: trainer_smoke(args)
    sys.exit(1 if C.summary() else 0)


if __name__ == "__main__":
    main()
