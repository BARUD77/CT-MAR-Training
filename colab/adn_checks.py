"""Real-data checks for the ADN baseline (run on Colab, GPU runtime).

  !python colab/adn_checks.py                       # all checks, default /content/data paths
  !python colab/adn_checks.py --skip_smoke --skip_amp

  1) split: build_datasets with vs without LI gives the identical train/val split (train_adn passes no LI)
  2) does the AAPM GT contain the metal? GT vs MA inside the metal mask (raw HU, before clipping)
  3) unpaired items == same window of the dataset's normalized slice (real files; full 512 + 256 metal crops)
  4) forward on real val slices (all networks; ADNInference shape / finite / range)
  5) timing: 500 real steps (fp32, batch 2) at full 512 and with --crop_size 256: s/step, data-wait,
     GPU utilization (nvidia-smi), peak memory, projected 50-epoch time
  6) fp32 vs bf16 (--amp): loss curves on identical batches with a second fp32 run as the noise floor
  7) validation time per slice, full-val time, projected total; proposes --val_subset if full val is slow
  8) smoke: train_adn.py 1 epoch on 32 MA / 32 clean real slices + 8 val (fp32 and bf16), then --resume
"""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import train_adn as tr  # noqa: E402
from acdnet_checks import GpuSampler, header  # noqa: E402
from baselines.common.data import HU_MAX, HU_MIN, build_datasets  # noqa: E402
from models.adn import ADNInference, to_adn  # noqa: E402
from models.adn.data import _mmap_2d  # noqa: E402


def make_args(a, extra=()):
    return tr.parse_args(["--ma_dir", a.ma_dir, "--gt_dir", a.gt_dir, "--mask_dir", a.mask_dir,
                          "--hu_min", str(HU_MIN), "--hu_max", str(HU_MAX), "--metal_mask_on_gt",
                          "--log_dir", tempfile.mkdtemp(prefix="adn_chk_"), "--num_workers", str(a.num_workers),
                          "--wandb_mode", "disabled", *extra])


def run_steps(args, unpaired, device, n_steps, warmup, amp=False, sample_gpu=True):
    """Fresh (seeded) ADN trained for n_steps on the real unpaired stream; timing + per-step losses."""
    args.amp = amp
    nets = tr.build_networks(args, device)
    opts, _ = tr.build_optimizers(args, nets)
    for n in nets:
        n.train()
    loader = tr.build_train_loader(args, unpaired, 1, 1 + n_steps // tr.steps_per_epoch(args, unpaired))
    it = iter(loader)
    g_hist, d_hist, t_data, t_start = [], [], 0.0, None
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler() if sample_gpu else contextlib.nullcontext()
    with sampler:
        for i in range(n_steps):
            if i == warmup:
                torch.cuda.synchronize()
                t_start, t_data = time.time(), 0.0
            t0 = time.time()
            b = next(it)
            low = to_adn(b["ma"].to(device, non_blocking=True))
            high = to_adn(b["clean"].to(device, non_blocking=True))
            t_data += time.time() - t0
            _, g_total, d_stats = tr.train_step(nets, opts, low, high, args)
            g_hist.append(g_total)
            d_hist.append(d_stats)
        torch.cuda.synchronize()
    elapsed = time.time() - t_start
    del it, loader
    args.amp = False
    return {"s_per_step": elapsed / (n_steps - warmup), "data_wait_frac": t_data / elapsed,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "gpu": sampler.summary() if sample_gpu else "",
            "g": torch.stack(g_hist).cpu().numpy(), "d": torch.stack(d_hist).cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ma_dir", default="/content/data/MA/MA_image")
    ap.add_argument("--gt_dir", default="/content/data/GT/GT")
    ap.add_argument("--li_dir", default="/content/data/LI/LI")
    ap.add_argument("--mask_dir", default="/content/data/mask/metal_mask")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--amp_steps", type=int, default=300)
    ap.add_argument("--val_slices", type=int, default=80)
    ap.add_argument("--gt_metal_slices", type=int, default=300)
    ap.add_argument("--skip_amp", action="store_true")
    ap.add_argument("--skip_smoke", action="store_true")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "Needs a GPU runtime"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False  # trainer default (fp32, no TF32)
    torch.backends.cudnn.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | python {sys.version.split()[0]} "
          f"| CPUs {os.cpu_count()} | num_workers {a.num_workers}")

    args = make_args(a)
    train_ds, val_ds = build_datasets(a.ma_dir, a.gt_dir, None, a.mask_dir)

    # 1) split ------------------------------------------------------------------------------
    header("1) Split with vs without LI (train_adn.py builds the datasets without LI)")
    tr_li, va_li = build_datasets(a.ma_dir, a.gt_dir, a.li_dir, a.mask_dir)
    key = lambda ds: [(p[0], p[1]) for p in ds.pairs]  # noqa: E731
    same = key(tr_li) == key(train_ds) and key(va_li) == key(val_ds)
    print(f"  train {len(train_ds)} / val {len(val_ds)} (no LI) vs train {len(tr_li)} / val {len(va_li)} (with LI)")
    print(f"  [{'PASS' if same else 'FAIL'}] identical (MA, GT) pairs in identical order in both splits")
    del tr_li, va_li

    # 2) GT metal ---------------------------------------------------------------------------
    header("2) Does the GT contain the metal? Raw HU inside the metal mask (before clipping)")
    rng = np.random.default_rng(0)
    pool = [(train_ds, i) for i in range(len(train_ds))] + [(val_ds, i) for i in range(len(val_ds))]
    rows, n_seen = [], 0
    for j in rng.permutation(len(pool)):
        if len(rows) >= a.gt_metal_slices or n_seen > 5 * a.gt_metal_slices:
            break
        ds, i = pool[j]
        n_seen += 1
        p = ds.pairs[i]
        mask_file = p[2] if len(p) >= 3 else None
        m = np.asarray(_mmap_2d(os.path.join(a.mask_dir, mask_file))) if mask_file else None
        if m is None:
            continue
        m = (m > 0) if m.max() > 1 else (m > 0.5)
        if m.sum() < 10:
            continue
        ma = np.asarray(_mmap_2d(os.path.join(a.ma_dir, p[0])), dtype=np.float64)
        gt = np.asarray(_mmap_2d(os.path.join(a.gt_dir, p[1])), dtype=np.float64)
        ring = np.zeros_like(m)
        ys, xs = np.nonzero(m)
        ring[max(ys.min() - 10, 0):ys.max() + 11, max(xs.min() - 10, 0):xs.max() + 11] = True
        ring &= ~m
        rows.append((np.median(gt[m]), np.median(ma[m]), np.mean(np.abs(gt[m] - ma[m])), (gt[m] >= 2500).mean(),
                     (ma[m] >= 2500).mean(), gt[m].max(), ma[m].max(), np.median(np.abs(gt[ring] - ma[ring]))))
    r = np.array(rows)
    if len(r):
        print(f"  {len(r)} slices with >= 10 metal pixels (raw file values, assumed HU)")
        print(f"  inside mask: median GT {np.median(r[:, 0]):.0f} HU (5-95%: {np.percentile(r[:, 0], 5):.0f}.."
              f"{np.percentile(r[:, 0], 95):.0f}) | median MA {np.median(r[:, 1]):.0f} HU")
        print(f"  inside mask: frac of pixels >= 2500 HU: GT {r[:, 3].mean():.3f} | MA {r[:, 4].mean():.3f}")
        print(f"  inside mask: max GT {np.median(r[:, 5]):.0f} HU | max MA {np.median(r[:, 6]):.0f} HU (median over slices)")
        print(f"  mean |GT-MA| inside mask {np.median(r[:, 2]):.0f} HU vs median |GT-MA| in a 10-px box around it "
              f"{np.median(r[:, 7]):.0f} HU")
        has_metal = np.median(r[:, 3]) > 0.5
        print(f"  -> GT {'CONTAINS' if has_metal else 'does NOT contain'} the metal "
              f"({'GT ~ MA inside the mask' if has_metal else 'GT inside the mask is tissue-like'}). "
              "Either way all metrics exclude the dilated metal region, and --metal_mask_on_gt copies MA into "
              "the GT metal region for validation only.")
    else:
        print("  no slices with metal found in the sample")

    # 3) unpaired data equality -------------------------------------------------------------
    header("3) Unpaired items == same window of the dataset's normalized slice (real files)")
    for crop, mcp in ((None, 0.7), (256, 0.7)):
        u = tr.build_unpaired(make_args(a, ["--crop_size", str(crop)] if crop else []), train_ds)
        ok = tot = n_flip = n_metal = 0
        for k in rng.choice(u.n_ma, size=15, replace=False):
            item = u[(1, int(k))]
            ma_idx, cl_idx = u.plan(1, int(k))
            f_ma, f_cl, f_mask = u._files(ma_idx)[0], u._files(cl_idx)[1], u._files(ma_idx)[2]
            ma_mm = _mmap_2d(os.path.join(a.ma_dir, f_ma))
            cl_mm = _mmap_2d(os.path.join(a.gt_dir, f_cl))
            mask_mm = _mmap_2d(os.path.join(a.mask_dir, f_mask)) if (u.metal_crop_prob > 0 and f_mask) else None
            w_ma, w_cl, flip = u.aug_params(1, int(k), ma_mm, cl_mm, mask_mm)
            ref_ma, ref_cl = train_ds[ma_idx][0][0][w_ma], train_ds[cl_idx][1][0][w_cl]
            if flip:
                ref_ma, ref_cl = ref_ma.flip(0), ref_cl.flip(0)
            tot += 1
            ok += int(torch.equal(item["ma"][0], ref_ma) and torch.equal(item["clean"][0], ref_cl))
            n_flip += int(flip)
            if crop and mask_mm is not None:
                n_metal += int(np.asarray(mask_mm)[w_ma].max() > 0)
        print(f"  [{'PASS' if ok == tot else 'FAIL'}] crop={crop or 'full'}: {ok}/{tot} MA+clean items bit-identical "
              f"| flipped {n_flip} | {'MA crops with metal ' + str(n_metal) + '/' + str(tot) if crop else ''}")

    # 4) forward ----------------------------------------------------------------------------
    header("4) Forward on real val slices (untrained networks)")
    g, d_l, d_h = tr.build_networks(args, device)
    infer = ADNInference.from_adn(g, tr.model_kwargs_from_args(args)).eval()
    ma = torch.stack([val_ds[i][0] for i in range(2)]).to(device)
    gt = torch.stack([val_ds[i][1] for i in range(2)]).to(device)
    with torch.no_grad():
        ll, lh, art = g.forward1(to_adn(ma))
        hl, hh = g.forward2(to_adn(gt), art)
        y = infer(ma)
        dl_out = d_l(lh)
    shapes = {k: tuple(v.shape) for k, v in dict(ll=ll, lh=lh, hl=hl, hh=hh, infer=y, d=dl_out).items()}
    fin = all(bool(torch.isfinite(t).all()) for t in (ll, lh, hl, hh, y, dl_out))
    print(f"  {shapes} | finite={fin} | ADNInference range [{y.min().item():.3f}, {y.max().item():.3f}] "
          f"(normalized space; tanh -> [0,1])")
    del g, d_l, d_h, infer, ll, lh, hl, hh, art, y, dl_out
    torch.cuda.empty_cache()

    # 5) timing -----------------------------------------------------------------------------
    timing = {}
    for crop in (None, 256):
        targs = make_args(a, ["--crop_size", str(crop)] if crop else [])
        u = tr.build_unpaired(targs, train_ds)
        spe = tr.steps_per_epoch(targs, u)
        header(f"5) Timing: {a.steps} real steps, fp32, batch {targs.batch_size}, crop={crop or 'full 512'} "
               f"(first {a.warmup} excluded)")
        res = run_steps(targs, u, device, a.steps, a.warmup)
        timing[crop] = (res, spe, targs)
        t_epoch = spe * res["s_per_step"]
        print(f"  {res['s_per_step']:.3f} s/step | data wait {100 * res['data_wait_frac']:.1f}% | peak alloc "
              f"{res['peak_mem_gb']:.2f} GB | {res['gpu']}")
        print(f"  {spe} steps/epoch -> {t_epoch / 60:.1f} min/epoch train | {targs.epochs} epochs = "
              f"{targs.epochs * t_epoch / 3600:.1f} h (train only)")
        print(f"  losses finite: {bool(np.isfinite(res['g']).all() and np.isfinite(res['d']).all())} | last 50 steps: "
              f"G {res['g'][-50:].mean():.3f} | sig(D_l r/f) {res['d'][-50:, 2].mean():.3f}/{res['d'][-50:, 3].mean():.3f}"
              f" | sig(D_h r/f) {res['d'][-50:, 4].mean():.3f}/{res['d'][-50:, 5].mean():.3f}")
        torch.cuda.empty_cache()

    # 6) fp32 vs bf16 -----------------------------------------------------------------------
    if not a.skip_amp:
        header(f"6) fp32 vs bf16 autocast: same init, same batches, {a.amp_steps} steps at full 512")
        targs = make_args(a)
        u = tr.build_unpaired(targs, train_ds)
        f1 = run_steps(targs, u, device, a.amp_steps, a.warmup, sample_gpu=False)
        f2 = run_steps(targs, u, device, a.amp_steps, a.warmup, sample_gpu=False)
        bf = run_steps(targs, u, device, a.amp_steps, a.warmup, amp=True, sample_gpu=False)

        def smooth(x, k=25):
            return np.convolve(x, np.ones(k) / k, mode="valid")

        def rel(x, y):
            return np.abs(smooth(x) - smooth(y)) / np.abs(smooth(x))

        noise, diff = rel(f1["g"], f2["g"]), rel(f1["g"], bf["g"])
        for s in (0, 49, 99, 199, a.amp_steps - 1):
            if s < a.amp_steps:
                print(f"  step {s + 1:>4d}: G fp32 {f1['g'][s]:.4f} | fp32 rerun {f2['g'][s]:.4f} | bf16 {bf['g'][s]:.4f}")
        print(f"  smoothed (25-step) relative diff of G total: fp32 vs fp32 rerun mean {100 * noise.mean():.2f}% "
              f"max {100 * noise.max():.2f}% | fp32 vs bf16 mean {100 * diff.mean():.2f}% max {100 * diff.max():.2f}%")
        for name, j in (("D_l", 0), ("D_h", 1)):
            print(f"  last 100 steps {name} loss: fp32 {f1['d'][-100:, j].mean():.4f} | rerun {f2['d'][-100:, j].mean():.4f}"
                  f" | bf16 {bf['d'][-100:, j].mean():.4f}")
        print(f"  speed: bf16 {bf['s_per_step']:.3f} s/step vs fp32 {f1['s_per_step']:.3f} s/step "
              f"({f1['s_per_step'] / bf['s_per_step']:.2f}x) | peak bf16 {bf['peak_mem_gb']:.2f} GB vs fp32 "
              f"{f1['peak_mem_gb']:.2f} GB")
        match = diff.mean() <= max(3 * noise.mean(), 0.01) and np.isfinite(bf["g"]).all()
        print(f"  -> {'curves match within the fp32 noise floor' if match else 'curves DO NOT match'}; "
              f"--amp {'is an option' if match and bf['s_per_step'] < 0.9 * f1['s_per_step'] else 'not recommended'}")

    # 7) validation -------------------------------------------------------------------------
    header("7) Validation time and projected total")
    g, _, _ = tr.build_networks(args, device)
    infer = ADNInference.from_adn(g, tr.model_kwargs_from_args(args))
    sub = torch.utils.data.Subset(val_ds, list(range(min(a.val_slices, len(val_ds)))))
    tr.run_validation(infer, torch.utils.data.Subset(val_ds, list(range(8))), args, device)  # warm-up
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        tr.run_validation(infer, sub, args, device)
    t_slice = (time.time() - t0) / len(sub)
    t_full = t_slice * len(val_ds)
    res, spe, targs = timing[None]
    E = targs.epochs
    t_train = E * spe * res["s_per_step"]
    t_val = (E + 1) * t_full  # every epoch + final best.pt evaluation
    print(f"  validation: {t_slice:.3f} s/slice (batch {args.val_batch_size}) | full val {len(val_ds)} slices = "
          f"{t_full / 60:.1f} min")
    print(f"  projected (full 512, fp32, --val_every 1, full val): train {t_train / 3600:.1f} h + val "
          f"{t_val / 3600:.1f} h = {(t_train + t_val) / 3600:.1f} h "
          f"(~{int(np.ceil((t_train + t_val) / (24 * 3600 * 0.95)))} x 24 h Colab sessions)")
    if t_full > 180:
        n_sub = 300
        print(f"  -> full val takes > 3 min: propose --val_subset {n_sub} (fixed seeded subset; the last epoch and "
              f"best.pt are still evaluated on the full val set): val cost {(E * n_sub * t_slice + 2 * t_full) / 3600:.1f} h")
    else:
        print("  -> full val is cheap enough; keep --val_subset 0 (full)")
    del g, infer
    torch.cuda.empty_cache()

    # 8) smoke ------------------------------------------------------------------------------
    if not a.skip_smoke:
        base = [sys.executable, "train_adn.py", "--ma_dir", a.ma_dir, "--gt_dir", a.gt_dir, "--mask_dir", a.mask_dir,
                "--hu_min", str(HU_MIN), "--hu_max", str(HU_MAX), "--metal_mask_on_gt", "--wandb_mode", "disabled",
                "--debug_max_train_samples", "64", "--debug_max_val_samples", "8", "--num_workers", str(a.num_workers),
                "--print_every", "4"]
        for amp in (False, True):
            header(f"8) Smoke ({'bf16' if amp else 'fp32'}): 1 epoch on 32 MA / 32 clean real slices + 8 val, then --resume")
            log_dir = tempfile.mkdtemp(prefix=f"adn_smoke_{'bf16' if amp else 'fp32'}_")
            extra = ["--amp"] if amp else []
            r1 = subprocess.run(base + extra + ["--log_dir", log_dir, "--epochs", "1"], capture_output=True, text=True)
            r2 = subprocess.run(base + extra + ["--log_dir", log_dir, "--epochs", "2", "--resume",
                                                os.path.join(log_dir, "last.pt")], capture_output=True, text=True)
            for r in (r1, r2):
                print("\n".join("  " + ln[:240] for ln in r.stdout.splitlines()
                                if ln.startswith(("[Epoch", "[resume]", "[FINAL"))))
                if r.returncode:
                    print(r.stderr[-3000:])
            sig = [tuple(map(float, m)) for m in re.findall(r"sig\(D_l r/f\)=([\d.]+)/([\d.]+) sig\(D_h r/f\)=([\d.]+)/([\d.]+)",
                                                            r1.stdout + r2.stdout)]
            collapse = any(rr > 0.95 and ff < 0.05 for s in sig for rr, ff in ((s[0], s[1]), (s[2], s[3])))
            ok = r1.returncode == 0 and r2.returncode == 0 and "[resume]" in r2.stdout and "[Epoch 2]" in r2.stdout
            fm = json.load(open(os.path.join(log_dir, "final_metrics.json"))) if ok else None
            print(f"  [{'PASS' if ok and not collapse else 'FAIL'}] runs ok={ok} | D collapse (real>0.95 & fake<0.05)="
                  f"{collapse} | final last SSIM "
                  f"{fm['last']['overall']['ssim']:.4f}" if fm else f"  [FAIL] runs ok={ok}")


if __name__ == "__main__":
    main()
