"""Local (synthetic-data) checks for the OSCNet+ baseline.

Run from the repo root:
  python -m models.oscnet.tests.test_oscnetplus [--orig_dir <clone of hongwang01/OSCNet>] [--smoke]

  1) forward shapes at 64 and 512 (batch 2), gradient flow, no aliased parameters
  2) port equivalence vs the official network/oscnetplus.py (only with --orig_dir): identical init,
     identical train-mode loss / gradients, official pretrained net_latest.pt loads strict and gives
     identical eval outputs; sanity on the repo's own sample slice (SynDeepLesion, not our data)
  3) batch layout: default 16 slices x 1 patch -> (16,1,64,64) of 16 distinct slices
  4) --metal_mask_on_gt is a no-op for the OSCNet+ loss; binary-mask assertion
  5) (--smoke) train_oscnetplus.py: 2 epochs x 20 it, val on 8 slices, resume; resume-exactness on CPU
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.data import build_datasets  # noqa: E402
from baselines.common.tests.verify_local_stage2 import make_fake_dataset  # noqa: E402
from models.acdnet.tests.test_acdnet import RESULTS, fake_inputs, report  # noqa: E402
from models.oscnet import OSCNetplus, OSCNetPlusInference, oscnetplus_loss  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = 10


# ---------------------------------------------------------------------------------------------
# 1) shapes
# ---------------------------------------------------------------------------------------------
def check_shapes():
    for s in (64, 512):
        torch.manual_seed(0)
        net = OSCNetplus().to(DEV).eval()
        ma, li, _, metal = fake_inputs(2, s, device=DEV)
        with torch.no_grad():
            X0, ListX, ListA = net(ma * 255, li * 255, 1 - metal)
            wrapper = OSCNetPlusInference().to(DEV).eval()
            wrapper.net.load_state_dict(net.state_dict())
            pred = wrapper(ma, li, metal)
        shp = (2, 1, s, s)
        ok = (tuple(X0.shape) == shp and len(ListX) == S + 1 and len(ListA) == S
              and all(tuple(t.shape) == shp for t in ListX + ListA) and tuple(pred.shape) == shp
              and torch.equal(pred, ListX[-1] / 255) and bool(torch.isfinite(pred).all()))
        report(f"1) forward shapes @ {s}x{s}, batch 2", ok,
               f"X0 {tuple(X0.shape)}, |ListX|={len(ListX)}, |ListA|={len(ListA)}, pred {tuple(pred.shape)}")
        del net, wrapper
    net = OSCNetplus().to(DEV).train()
    ma, li, gt, metal = fake_inputs(4, 64, device=DEV)
    mask = 1 - metal
    loss, *_ = oscnetplus_loss(ma * 255, gt * 255, mask, *net(ma * 255, li * 255, mask), S=S)
    loss.backward()
    no_grad = sorted(n for n, p in net.named_parameters() if p.grad is None)
    report("1b) backward: finite loss, every parameter gets a gradient", not no_grad and torch.isfinite(loss).item(),
           f"loss={loss.item():.4e}; params without grad: {no_grad}")
    params = list(net.parameters())
    aliased = len({p.data_ptr() for p in params}) != len(params) or any(0 in p.stride() for p in params if p.dim())
    report("1c) no aliased / stride-0 parameters", not aliased, f"{sum(p.numel() for p in params):,} parameters")


# ---------------------------------------------------------------------------------------------
# 2) port vs original
# ---------------------------------------------------------------------------------------------
def load_original(orig_dir):
    # network/ has `_init_.py` (not a package) and oscnetplus.py does `from .Fconv_wnet import ...`
    # and loads utils/init_kernel.mat relative to the cwd at import: register a synthetic package.
    pkg = types.ModuleType("oscnet_orig_network")
    pkg.__path__ = [os.path.join(orig_dir, "network")]
    sys.modules["oscnet_orig_network"] = pkg
    cwd = os.getcwd()
    os.chdir(orig_dir)
    try:
        return importlib.import_module("oscnet_orig_network.oscnetplus")
    finally:
        os.chdir(cwd)


def untie(module):
    # What `.cuda()` does in the original train/test scripts: every parameter becomes its own copy.
    module._apply(lambda t: t.clone())
    return module


def orig_opt():
    return types.SimpleNamespace(S=10, T=3, num_M=4, num_Q=32, num_rot=8, etaM=1.0, etaX=5.0, padding=4, inP=5,
                                 sizeP=9, ifini=1.0, cdiv=10.0)


def check_port(orig_dir):
    mod = load_original(orig_dir)
    torch.manual_seed(123)
    orig = untie(mod.OSCNetplus(orig_opt()))
    torch.manual_seed(123)
    ours = OSCNetplus()
    so, sp = orig.state_dict(), ours.state_dict()
    same_init = so.keys() == sp.keys() and all(torch.equal(so[k], sp[k]) for k in so)
    report("2a) identical parameter/buffer names and initial values (same seed) as the original", same_init,
           f"{len(sp)} state_dict entries (incl. Basis buffer {tuple(sp['fcnn.Basis'].shape)})")

    ma, li, gt, metal = fake_inputs(16, 64, seed=1)
    mask = 1 - metal
    outs_o = orig.train()(ma * 255, li * 255, mask)
    outs_p = ours.train()(ma * 255, li * 255, mask)
    lo = oscnetplus_loss(ma * 255, gt * 255, mask, *outs_o, S=S)[0]
    lp = oscnetplus_loss(ma * 255, gt * 255, mask, *outs_p, S=S)[0]
    lo.backward()
    lp.backward()
    go = dict(orig.named_parameters())
    max_gdiff = max(float((go[n].grad - p.grad).abs().max()) for n, p in ours.named_parameters())
    report("2b) train mode (batch 16 x 64^2): identical loss and gradients", torch.equal(lo, lp) and max_gdiff == 0.0,
           f"loss {lo.item():.6e} vs {lp.item():.6e}; max |grad diff| = {max_gdiff:.1e}")

    ckpt_path = os.path.join(orig_dir, "pretrained_model", "model_oscplus", "net_latest.pt")
    if not os.path.exists(ckpt_path):
        report("2c) official checkpoint", False, f"missing {ckpt_path}")
        return
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    orig.load_state_dict(sd, strict=True)
    wrapper = OSCNetPlusInference()
    wrapper.net.load_state_dict(sd, strict=True)
    orig.eval()
    wrapper.eval()
    ma, li, _, metal = fake_inputs(2, 128, seed=2)
    with torch.no_grad():
        o = orig(ma * 255, li * 255, 1 - metal)
        p = wrapper.net(ma * 255, li * 255, 1 - metal)
        pred = wrapper(ma, li, metal)
    flat_o, flat_p = [o[0]] + o[1] + o[2], [p[0]] + p[1] + p[2]
    max_diff = max(float((a - b).abs().max()) for a, b in zip(flat_o, flat_p))
    report("2c) official net_latest.pt loads strict; eval outputs identical to the original",
           max_diff == 0.0 and torch.equal(pred, o[1][-1] / 255), f"max |diff| over all outputs = {max_diff:.1e}")
    check_pretrained_sample(orig_dir, wrapper)


def check_pretrained_sample(orig_dir, wrapper):
    """The repo's own SynDeepLesion test sample, preprocessed like the original test.py."""
    try:
        import h5py
        from PIL import Image
    except ImportError:
        print("[SKIP] 2d) needs h5py + PIL")
        return
    base = os.path.join(orig_dir, "data", "test")
    rel = open(os.path.join(base, "test_640geo_dir.txt")).readline().strip()  # .../gt.h5
    with h5py.File(os.path.join(base, "test_640geo", rel), "r") as f:
        gt = f["image"][()]
    with h5py.File(os.path.join(base, "test_640geo", rel[:-5] + "0.h5"), "r") as f:
        ma, li = f["ma_CT"][()], f["LI_CT"][()]
    m = np.array(Image.fromarray(np.load(os.path.join(base, "testmask.npy"))[:, :, 0]).resize((416, 416),
                                                                                              Image.BILINEAR))
    t = lambda a: torch.tensor(np.clip(a, 0, 1), dtype=torch.float32)[None, None]  # noqa: E731
    ma_t, li_t, gt_t = t(ma), t(li), t(gt)
    metal = (torch.tensor(m.astype(np.float32))[None, None] > 0).float()  # binarized (ours) vs bilinear (orig)
    with torch.no_grad():
        pred = wrapper(ma_t, li_t, metal)
    nm = metal < 0.5

    def psnr(x):  # original test.py convention: clamp to [0, 0.5], data range 0.5, non-metal pixels
        x, y = x.clamp(0, 0.5)[nm], gt_t.clamp(0, 0.5)[nm]
        return float(10 * torch.log10(0.25 / ((x - y) ** 2).mean()))

    p_ma, p_li, p_pred = psnr(ma_t), psnr(li_t), psnr(pred)
    report("2d) pretrained OSCNet+ via OSCNetPlusInference on the repo's sample slice beats MA and LI",
           p_pred > p_li and p_pred > p_ma, f"PSNR (non-metal, orig. convention) MA {p_ma:.2f} | LI {p_li:.2f} | "
           f"OSCNet+ {p_pred:.2f} dB")


# ---------------------------------------------------------------------------------------------
# 3) batch layout
# ---------------------------------------------------------------------------------------------
def check_batch_layout(tmp_root):
    import train_oscnetplus as tr
    dirs = make_fake_dataset(os.path.join(tmp_root, "layout"), n_body=40, n_head=10, shape=(96, 80))
    args = tr.parse_args(["--ma_dir", dirs["MA_image"], "--gt_dir", dirs["GT"], "--li_dir", dirs["LI"],
                          "--mask_dir", dirs["metal_mask"], "--log_dir", os.path.join(tmp_root, "layout_log"),
                          "--num_workers", "0", "--iters_per_epoch", "3"])
    train_ds, _ = build_datasets(dirs["MA_image"], dirs["GT"], dirs["LI"], dirs["metal_mask"])
    loader = tr.build_train_loader(args, train_ds, 1, 1)
    sampler_keys = list(loader.sampler)
    b = tr.batch_to_device(next(iter(loader)), torch.device("cpu"))
    first = [i for i, _ in sampler_keys[:args.batch_slices]]
    ok = (args.batch_slices, args.patches_per_slice) == (16, 1) and tuple(b["ma"].shape) == (16, 1, 64, 64) \
        and len(set(first)) == 16 and len(loader) == 3
    report("3) default batch = 16 distinct slices x 1 patch -> (16,1,64,64) (original MARTrainDataset layout)", ok,
           f"batch {tuple(b['ma'].shape)}, distinct slices in batch 1: {len(set(first))}, "
           f"milestones {args.milestones}, epochs {args.epochs}, it/epoch default 4000")


# ---------------------------------------------------------------------------------------------
# 4) loss / mask
# ---------------------------------------------------------------------------------------------
def check_loss_and_mask():
    ma, li, gt, metal = fake_inputs(4, 64, seed=4)
    net = OSCNetplus().eval()
    with torch.no_grad():
        outs = net(ma * 255, li * 255, 1 - metal)
        l1 = oscnetplus_loss(ma * 255, gt * 255, 1 - metal, *outs, S=S)[0]
        l2 = oscnetplus_loss(ma * 255, torch.where(metal > 0.5, ma, gt) * 255, 1 - metal, *outs, S=S)[0]
    report("4a) OSCNet+ loss identical with/without metal painted on GT", torch.equal(l1, l2))
    m = OSCNetPlusInference().eval()
    try:
        with torch.no_grad():
            m(ma[:1], li[:1], metal[:1] * 0.5)
        ok = False
    except ValueError:
        ok = True
    report("4b) OSCNetPlusInference rejects a non-binary metal mask", ok)


# ---------------------------------------------------------------------------------------------
# 5) smoke + resume
# ---------------------------------------------------------------------------------------------
def run_trainer(dirs, log_dir, extra, cpu=False):
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "train_oscnetplus.py"),
           "--ma_dir", dirs["MA_image"], "--gt_dir", dirs["GT"], "--li_dir", dirs["LI"], "--mask_dir", dirs["metal_mask"],
           "--hu_min", "-1024", "--hu_max", "3072", "--metal_mask_on_gt", "--log_dir", log_dir,
           "--wandb_mode", "disabled", "--print_every", "10"] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"  # an empty value is dropped on Windows and the GPU stays visible
    r = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    return r


def check_smoke(tmp_root):
    dirs = make_fake_dataset(os.path.join(tmp_root, "smoke512"), n_body=72, n_head=18, shape=(512, 512))
    log_dir = os.path.join(tmp_root, "run_smoke")
    common = ["--iters_per_epoch", "20", "--val_every", "1", "--full_val_every", "2", "--val_subset", "4",
              "--debug_max_val_samples", "8", "--num_workers", "2"]
    r = run_trainer(dirs, log_dir, common + ["--epochs", "2"])
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("[Epoch", "[FINAL", "    [", "[data]", "[model]"))]
    print("\n".join("     " + ln for ln in lines))
    ok = (r.returncode == 0 and all(os.path.exists(os.path.join(log_dir, f))
                                    for f in ("last.pt", "best.pt", "final_metrics.json")))
    report("5a) smoke: 2 epochs x 20 it (512^2 fake data), val on 8 slices, finite losses, metrics, checkpoints", ok)
    if not ok:
        return
    ck = torch.load(os.path.join(log_dir, "best.pt"), map_location="cpu")  # default weights_only (as test.py)
    report("5b) best.pt loads with torch.load defaults (weights_only) and has the expected keys",
           {"model_state", "optimizer_state", "scheduler_state", "epoch", "best_ssim", "model_kwargs"} <= set(ck)
           and ck["model_name"] == "oscnetplus", f"model_kwargs={ck['model_kwargs']}")
    r2 = run_trainer(dirs, log_dir, common + ["--epochs", "3", "--resume", os.path.join(log_dir, "last.pt")])
    ok2 = r2.returncode == 0 and "[resume]" in r2.stdout and "[Epoch 3]" in r2.stdout
    report("5c) --resume last.pt continues at epoch 3", ok2,
           next((ln for ln in r2.stdout.splitlines() if ln.startswith("[resume]")), ""))


def check_resume_exact(tmp_root):
    dirs = make_fake_dataset(os.path.join(tmp_root, "resume128"), n_body=40, n_head=10, shape=(128, 128))
    common = ["--iters_per_epoch", "4", "--val_every", "1", "--full_val_every", "0", "--val_subset", "3",
              "--debug_max_val_samples", "4", "--num_workers", "0", "--milestones", "1", "--print_every", "100"]
    a, b = os.path.join(tmp_root, "run_a"), os.path.join(tmp_root, "run_b")
    ra = run_trainer(dirs, a, common + ["--epochs", "2"], cpu=True)
    rb1 = run_trainer(dirs, b, common + ["--epochs", "1"], cpu=True)
    rb2 = run_trainer(dirs, b, common + ["--epochs", "2", "--resume", os.path.join(b, "last.pt")], cpu=True)
    if not (ra.returncode == rb1.returncode == rb2.returncode == 0):
        report("5d) resume exactness", False, "a trainer run failed")
        return
    ca = torch.load(os.path.join(a, "last.pt"), map_location="cpu")
    cb = torch.load(os.path.join(b, "last.pt"), map_location="cpu")
    w_same = all(torch.equal(ca["model_state"][k], cb["model_state"][k]) for k in ca["model_state"])
    oa, ob = ca["optimizer_state"]["state"], cb["optimizer_state"]["state"]
    o_same = all(torch.equal(oa[i][k], ob[i][k]) for i in oa for k in oa[i])
    lr_same = ca["optimizer_state"]["param_groups"][0]["lr"] == cb["optimizer_state"]["param_groups"][0]["lr"]
    sched_same = ca["scheduler_state"] == cb["scheduler_state"]
    # best_ssim: in run B epoch 1 is that invocation's last epoch, so its subset SSIM comes from the full-val
    # pass (different val batch composition -> float32 conv rounding). Not a resume effect.
    ssim_close = abs(ca["best_ssim"] - cb["best_ssim"]) <= 1e-6 * abs(ca["best_ssim"])
    report("5d) CPU: 2 epochs straight == 1 epoch + resume (weights, BN stats, Adam state, LR, scheduler "
           "bit-identical; best_ssim within 1e-6 rel)",
           w_same and o_same and lr_same and sched_same and ssim_close and ca["best_epoch"] == cb["best_epoch"],
           f"lr={ca['optimizer_state']['param_groups'][0]['lr']:.1e} (milestone at 1 crossed), "
           f"best_ssim {ca['best_ssim']:.10f} vs {cb['best_ssim']:.10f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", default=os.environ.get("OSCNET_ORIG_DIR"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true")
    a = ap.parse_args()
    tmp_root = tempfile.mkdtemp(prefix="oscnet_test_")
    try:
        check_shapes()
        if a.orig_dir:
            check_port(a.orig_dir)
        else:
            print("[SKIP] 2) port vs original (pass --orig_dir <OSCNet clone>)")
        check_batch_layout(tmp_root)
        check_loss_and_mask()
        if a.smoke:
            check_smoke(tmp_root)
            check_resume_exact(tmp_root)
    finally:
        if a.keep_tmp:
            print(f"temp dir kept: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
