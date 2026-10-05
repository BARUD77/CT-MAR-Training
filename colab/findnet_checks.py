"""FIND-Net checks on REAL data (run in Colab on the GPU). Prints one report; each check ends PASS/FAIL.

  1. forward pass on 2 real training slices (shapes, value ranges, no NaNs, conversion back to [0,1])
  2. GPU memory: peak training memory at the real image size for batch 1 and 2 -> per-image cost and the
     largest batch that fits; checks 3-5 use min(config batch, largest fitting batch)
  3. overfit: train_findnet.py --subset 8 for a few hundred iterations, loss at start and end
  4. resume: train_findnet.py to 30 iterations, restart to 50, confirm it resumed at step 30
     (and the same W&B run id when W&B is enabled)
  5. timing: N iterations after W warm-up iterations; s/iter, peak GPU memory, estimated total time

Usage (repo root):
  python colab/findnet_checks.py --ma_dir ... --gt_dir ... --li_dir ... --mask_dir ... \
      --work_dir /content/findnet_checks [--wandb_mode offline]
Checkpoints of checks 2-3 go to --work_dir (local disk), never to the real checkpoint directory.
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.findnet.dataset_adapter import build_findnet_datasets  # noqa: E402
from baselines.findnet.model_wrapper import build_findnet, final_output, findnet_commit  # noqa: E402
from baselines.findnet.train_findnet import DEFAULT_CONFIG, findnet_loss  # noqa: E402

TRAIN_SCRIPT = os.path.join(_REPO_ROOT, "baselines", "findnet", "train_findnet.py")
LINE = "=" * 78
RESULTS = []


def verdict(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  -> {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""), flush=True)


def rng(t):
    return f"[{float(t.min()):.4g}, {float(t.max()):.4g}]"


def run_trainer(args, extra, tag):
    """Run train_findnet.py as a subprocess; return its stdout (also echoed, filtered)."""
    cmd = [sys.executable, TRAIN_SCRIPT, "--config", args.config,
           "--ma_dir", args.ma_dir, "--gt_dir", args.gt_dir, "--li_dir", args.li_dir, "--mask_dir", args.mask_dir,
           "--num_workers", str(args.num_workers), "--wandb_mode", args.wandb_mode,
           "--project", args.project, "--batch_size", str(args.eff_batch)] + extra
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    out = p.stdout + "\n" + p.stderr
    keep = [l for l in out.splitlines() if l.startswith(("[train]", "[resume]", "[stop]", "[val]", "[wandb]",
                                                         "[findnet]", "[epoch]", "Traceback", "  File", "RuntimeError",
                                                         "ValueError", "FloatingPointError", "torch."))]
    for l in keep[-25:]:
        print(f"    | {l}")
    print(f"  ({tag}: exit code {p.returncode}, {time.time() - t0:.0f}s)", flush=True)
    return p.returncode, out


def train_losses(out):
    return [(int(m.group(1)), float(m.group(2)))
            for m in re.finditer(r"\[train\] epoch \d+ it \d+/\d+ step (\d+) loss ([0-9.eE+-]+)", out)]


# --------------------------------------------------------------------------------------------
def check_forward(args, cfg, device):
    print(LINE + "\n1) FORWARD PASS ON 2 REAL TRAINING SLICES")
    train_ds, _, conv = build_findnet_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir,
                                               normalization=cfg["adapter"]["normalization"],
                                               augment_train=False, subset=2)
    b = next(iter(DataLoader(train_ds, batch_size=2, shuffle=False)))
    for k in ("ma", "li", "gt", "nonmetal", "metal_mask", "gt_norm"):
        print(f"  input {k:10s} shape {tuple(b[k].shape)} range {rng(b[k])}")
    nm_vals = sorted(torch.unique(b["nonmetal"]).tolist())
    mask_ok = set(nm_vals) <= {0.0, 1.0} and torch.equal(b["nonmetal"], 1 - b["metal_mask"])
    print(f"  nonmetal unique values {nm_vals}; metal fraction {float(b['metal_mask'].mean()):.4%}")
    model = build_findnet(**cfg["model"]).to(device).eval()
    with torch.no_grad():
        X0, ListX, ListA = model(*(b[k].to(device) for k in ("ma", "li", "nonmetal")))
    final = final_output((X0, ListX, ListA))
    back = conv.to_norm(final)
    finite = all(torch.isfinite(t).all() for t in [X0, *ListX, *ListA])
    print(f"  output X0 {tuple(X0.shape)} | len(ListX)={len(ListX)} len(ListA)={len(ListA)} | "
          f"final {tuple(final.shape)} range {rng(final)} | to_norm range {rng(back)}")
    rt = float((conv.to_norm(b["gt"]) - b["gt_norm"]).abs().max())
    print(f"  conversion round trip max |to_norm(to_findnet(gt)) - gt| = {rt:.2e}")
    print(f"  params {sum(p.numel() for p in model.parameters()):,} | commit {findnet_commit()}")
    ok = (finite and mask_ok and tuple(final.shape) == tuple(b["gt"].shape) and len(ListX) == cfg["model"]["S"] + 1
          and len(ListA) == cfg["model"]["S"] and rt < 1e-5)
    verdict("forward pass", ok, f"finite={finite} mask_binary={mask_ok}")
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None


def check_memory(args, cfg, device):
    print(LINE + "\n2) GPU MEMORY (training step: forward + original loss + backward, fp32)")
    cfg_bs = int(cfg["train"]["batch_size"])
    if device.type != "cuda":
        args.eff_batch = args.batch_size or cfg_bs
        verdict("gpu memory", True, "skipped (no CUDA)")
        return
    train_ds, _, _ = build_findnet_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir,
                                            normalization=cfg["adapter"]["normalization"],
                                            augment_train=False, subset=1)
    H, W = train_ds[0]["ma"].shape[-2:]
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    model = build_findnet(**cfg["model"]).to(device).train()
    loss_kwargs = dict(S=int(cfg["model"]["S"]), **cfg["loss"])
    peaks = {}
    for b in (1, 2):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.rand(b, 1, H, W, device=device) * 255
            nm = torch.ones_like(x)
            out = model(x, x, nm)
            loss, _ = findnet_loss(x, x, nm, *out, **loss_kwargs)
            loss.backward()
            peaks[b] = torch.cuda.max_memory_allocated() / 2**30
            print(f"  {H}x{W}, batch {b}: peak {peaks[b]:.1f} GiB")
        except torch.OutOfMemoryError:
            print(f"  {H}x{W}, batch {b}: OUT OF MEMORY")
            break
        finally:
            out = loss = x = nm = None
    del model
    torch.cuda.empty_cache()
    if 1 not in peaks:
        args.eff_batch = 1
        verdict("gpu memory", False, f"batch 1 at {H}x{W} does not fit in {total:.0f} GiB")
        return
    if 2 in peaks:
        per, fixed = peaks[2] - peaks[1], 2 * peaks[1] - peaks[2]
        max_bs = max(1, int((0.92 * total - fixed) // per))
    else:
        per, fixed, max_bs = float("nan"), float("nan"), 1
    need = fixed + cfg_bs * per
    args.eff_batch = args.batch_size or min(cfg_bs, max_bs)
    print(f"  GPU total {total:.1f} GiB | per image {per:.1f} GiB | fixed {fixed:.1f} GiB")
    print(f"  config batch {cfg_bs} needs ~{need:.0f} GiB -> {'FITS' if cfg_bs <= max_bs else 'DOES NOT FIT'}; "
          f"largest batch that fits ~{max_bs}")
    print(f"  checks below use batch {args.eff_batch}")
    verdict("gpu memory", cfg_bs <= max_bs,
            f"config batch {cfg_bs} {'fits' if cfg_bs <= max_bs else 'does NOT fit'}; max ~{max_bs}")


def check_overfit(args):
    print(LINE + f"\n3) OVERFIT CHECK (--subset 8, {args.overfit_iters} iterations, batch {args.eff_batch}, no validation)")
    d = os.path.join(args.work_dir, "overfit")
    shutil.rmtree(d, ignore_errors=True)
    code, out = run_trainer(args, ["--ckpt_dir", d, "--subset", "8", "--max_iters", str(args.overfit_iters),
                                   "--log_every_iters", "10", "--val_every_epochs", "100000",
                                   "--run_name", "check_overfit", "--no_resume"], "overfit")
    losses = train_losses(out)
    if code != 0 or len(losses) < 2:
        verdict("overfit", False, f"exit code {code}, {len(losses)} loss values parsed")
        return
    start, end = losses[0], losses[-1]
    best = min(l for _, l in losses)
    ratio = end[1] / start[1]
    print(f"  loss at step {start[0]}: {start[1]:.4e} | at step {end[0]}: {end[1]:.4e} | min {best:.4e} | "
          f"end/start = {ratio:.3f}")
    verdict("overfit", ratio < 0.5, "criterion: final logged loss < 0.5 x first logged loss")


def check_resume(args):
    print(LINE + f"\n4) RESUME CHECK (run to 30 iterations, restart, continue to 50; batch {args.eff_batch})")
    d = os.path.join(args.work_dir, "resume")
    shutil.rmtree(d, ignore_errors=True)
    common = ["--ckpt_dir", d, "--subset", "64", "--val_subset", "4", "--log_every_iters", "10",
              "--save_every_iters", "10", "--run_name", "check_resume"]
    code1, out1 = run_trainer(args, common + ["--max_iters", "30", "--no_resume"], "run 1")
    ck = torch.load(os.path.join(d, "last.pt"), map_location="cpu", weights_only=False) \
        if os.path.isfile(os.path.join(d, "last.pt")) else None
    step1 = ck["global_step"] if ck else None
    print(f"  after run 1: last.pt global_step={step1} next_epoch={ck['extra']['next_epoch'] if ck else None} "
          f"next_iter={ck['extra']['next_iter'] if ck else None}")
    code2, out2 = run_trainer(args, common + ["--max_iters", "50"], "run 2")
    m = re.search(r"\[resume\] from .* global_step=(\d+)", out2)
    resumed_at = int(m.group(1)) if m else None
    first_step2 = train_losses(out2)[0][0] if train_losses(out2) else None
    ck2 = torch.load(os.path.join(d, "last.pt"), map_location="cpu", weights_only=False)
    ids = [re.search(r"id=(\S+)", l).group(1) for l in (out1 + out2).splitlines()
           if l.startswith("[wandb] run name=") and "id=" in l]
    same_run = (len(ids) == 2 and ids[0] == ids[1]) if args.wandb_mode != "disabled" else True
    print(f"  run 2 resumed at global_step={resumed_at}; first logged step {first_step2}; "
          f"final last.pt global_step={ck2['global_step']}; W&B ids {ids or 'n/a'}")
    ok = (code1 == 0 and code2 == 0 and step1 == 30 and resumed_at == 30 and first_step2 == 40
          and ck2["global_step"] == 50 and same_run)
    verdict("resume", ok, "expects stop@30, resume@30, first log @40, stop@50, same W&B run")


def check_timing(args, cfg, device):
    print(LINE + f"\n5) TIMING ({args.timing_iters} iterations after {args.warmup_iters} warm-up, "
                 f"batch {args.eff_batch}, fp32)")
    train_ds, val_ds, conv = build_findnet_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir,
                                                    normalization=cfg["adapter"]["normalization"],
                                                    augment_train=cfg["adapter"]["augment_flips"])
    n_train, n_val = len(train_ds), len(val_ds)
    bs = int(args.eff_batch)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"]["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(cfg["train"]["tf32"])
        torch.backends.cudnn.benchmark = bool(cfg["train"]["cudnn_benchmark"])
        torch.cuda.reset_peak_memory_stats()
    model = build_findnet(**cfg["model"]).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), betas=tuple(cfg["optim"]["betas"]),
                            weight_decay=float(cfg["optim"]["weight_decay"]))
    n_iters = args.warmup_iters + args.timing_iters
    sub = Subset(train_ds, list(range(min(n_train, n_iters * bs))))
    loader = DataLoader(sub, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", drop_last=True)
    loss_kwargs = dict(S=int(cfg["model"]["S"]), **cfg["loss"])
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    done, t_start, finite = 0, None, True
    while done < n_iters:
        for b in loader:
            if done == args.warmup_iters:
                sync()
                t_start = time.time()
            ma, li, gt, nm = (b[k].to(device, non_blocking=True) for k in ("ma", "li", "gt", "nonmetal"))
            out = model(ma, li, nm)
            loss, _ = findnet_loss(ma, gt, nm, *out, **loss_kwargs)
            finite &= bool(torch.isfinite(loss))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
            if done >= n_iters:
                break
    sync()
    s_it = (time.time() - t_start) / args.timing_iters
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")

    # validation speed on a few batches
    model.eval()
    vsub = Subset(val_ds, list(range(min(n_val, 4 * bs))))
    t0 = time.time()
    with torch.no_grad():
        for b in DataLoader(vsub, batch_size=bs, num_workers=args.num_workers):
            model(*(b[k].to(device) for k in ("ma", "li", "nonmetal")))
    sync()
    s_val_img = (time.time() - t0) / len(vsub)

    steps_per_epoch = math.ceil(n_train / bs)
    epochs = int(cfg["train"]["epochs"])
    epoch_h = steps_per_epoch * s_it / 3600
    val_h = n_val * s_val_img / 3600 / max(1, int(cfg["validation"].get("every_epochs") or 1))
    total_h = epochs * (epoch_h + val_h)
    print(f"  device {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'} | "
          f"train slices {n_train} | val slices {n_val} | steps/epoch {steps_per_epoch}")
    print(f"  {s_it:.3f} s/iter (batch {bs}) | peak GPU memory {peak:.1f} GiB | "
          f"val forward {s_val_img:.3f} s/img (model only, excl. metrics)")
    print(f"  per epoch: train {epoch_h:.2f} h + val {val_h:.2f} h | config budget {epochs} epochs -> "
          f"~{total_h:.0f} h total (~{total_h / 23:.1f} Colab sessions of ~23 h) | "
          f"~{23 / max(1e-9, epoch_h + val_h):.1f} epochs per 23 h session")
    verdict("timing", finite and s_it > 0, f"finite loss={finite}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    for k in ("ma_dir", "gt_dir", "li_dir", "mask_dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--work_dir", default="/content/findnet_checks")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--wandb_mode", default="offline", choices=["online", "offline", "disabled"])
    ap.add_argument("--project", default="ct-mar-checks")
    ap.add_argument("--overfit_iters", type=int, default=300)
    ap.add_argument("--warmup_iters", type=int, default=20)
    ap.add_argument("--timing_iters", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=0,
                    help="Batch for checks 3-5; 0 = min(config batch, largest batch that fits in GPU memory).")
    ap.add_argument("--only", nargs="*", choices=["forward", "memory", "overfit", "resume", "timing"])
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.eff_batch = args.batch_size or int(cfg["train"]["batch_size"])  # refined by the memory check

    print(LINE + "\nFINDNET_CHECKS REPORT")
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | device {device} | "
          f"{torch.cuda.get_device_name() if device.type == 'cuda' else ''}")
    print(f"config {args.config} | normalization {cfg['adapter']['normalization']} | "
          f"gaussian_filter {cfg['model']['gaussian_filter']} | batch {cfg['train']['batch_size']}")
    todo = args.only or ["forward", "memory", "overfit", "resume", "timing"]
    checks = {"forward": lambda: check_forward(args, cfg, device), "memory": lambda: check_memory(args, cfg, device),
              "overfit": lambda: check_overfit(args),
              "resume": lambda: check_resume(args), "timing": lambda: check_timing(args, cfg, device)}
    for name in todo:
        try:
            checks[name]()
        except Exception as e:  # report and continue with the other checks
            verdict(name, False, f"{type(e).__name__}: {e}")
    print(LINE)
    for name, ok in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"OVERALL: {'ALL PASS' if all(ok for _, ok in RESULTS) else 'SOME FAILED'}")
    print(LINE)


if __name__ == "__main__":
    main()
