"""Local (synthetic-data) checks for the ACDNet baseline.

Run from the repo root:
  python -m models.acdnet.tests.test_acdnet [--orig_dir <clone of hongwang01/ACDNet>] [--smoke]

  1) forward shapes at 64 and 512 (batch 2) + gradient flow
  2) port equivalence vs the official acdnet.py (only with --orig_dir): identical init, official
     ACDNet_latest.pt loads strict, identical outputs / loss / gradients
  3) patch == same window cropped from the dataset's full normalized slice (+ flips, mask rules,
     worker independence, resumable draw stream)
  4) evaluate_predictions ignores the metal region of pred and of GT (--metal_mask_on_gt no-op)
  5) binary-mask assertion
  6) (--smoke) train_acdnet.py: 2 epochs x 20 it, val on 8 slices; resume-exactness on CPU
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.data import HU_MAX, HU_MIN, build_datasets  # noqa: E402
from baselines.common.eval_utils import evaluate_predictions  # noqa: E402
from baselines.common.tests.verify_local_stage2 import make_fake_dataset  # noqa: E402
from models.acdnet import ACDNet, ACDNetInference, acdnet_loss  # noqa: E402
from models.acdnet.patch_data import ACDNetPatchDataset, DrawStreamSampler  # noqa: E402

RESULTS = []
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def report(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def fake_inputs(b, s, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    ma = torch.rand(b, 1, s, s, generator=g)
    li = torch.rand(b, 1, s, s, generator=g)
    gt = torch.rand(b, 1, s, s, generator=g)
    metal = torch.zeros(b, 1, s, s)
    metal[:, :, s // 3: s // 3 + max(2, s // 16), s // 2: s // 2 + max(2, s // 12)] = 1
    return [t.to(device) for t in (ma, li, gt, metal)]


# ---------------------------------------------------------------------------------------------
# 1) shapes
# ---------------------------------------------------------------------------------------------
def check_shapes():
    for s in (64, 512):
        net = ACDNet().to(DEV).eval()
        ma, li, _, metal = fake_inputs(2, s, device=DEV)
        with torch.no_grad():
            X0, ListX, ListA, ListX_nonK, ListA_nonK = net(ma * 255, li * 255, 1 - metal)
            wrapper = ACDNetInference().to(DEV).eval()
            wrapper.net.load_state_dict(net.state_dict())
            pred = wrapper(ma, li, metal)
        shp = (2, 1, s, s)
        ok = (tuple(X0.shape) == shp and len(ListX) == 11 and len(ListA) == 10 and len(ListX_nonK) == 2
              and len(ListA_nonK) == 2 and all(tuple(t.shape) == shp for t in ListX + ListA + ListX_nonK + ListA_nonK)
              and tuple(pred.shape) == shp and torch.equal(pred, ListX[-1] / 255))
        report(f"1) forward shapes @ {s}x{s}, batch 2", ok,
               f"X0 {tuple(X0.shape)}, |ListX|={len(ListX)}, |ListA|={len(ListA)}, |nonK|={len(ListX_nonK)}, "
               f"pred {tuple(pred.shape)}")
        del net, wrapper
    # gradient flow in train mode at 64
    net = ACDNet().to(DEV).train()
    ma, li, gt, metal = fake_inputs(4, 64, device=DEV)
    mask = 1 - metal
    outs = net(ma * 255, li * 255, mask)
    loss, *_ = acdnet_loss(ma * 255, gt * 255, mask, *outs, T=10)
    loss.backward()
    no_grad = sorted(n for n, p in net.named_parameters() if p.grad is None)
    expected = sorted(["tau_extend"] + [f"proxNet_K_T.{i}.tau" for i in range(10)])
    report("1b) backward: every parameter gets a gradient except the original's unused ones",
           no_grad == expected and torch.isfinite(loss).item(),
           f"loss={loss.item():.4e}; params without grad: {no_grad[:3]}... ({len(no_grad)}), "
           f"eta3_T grad[0]={net.eta3_T.grad[0].item():.1e} (unused row)")
    n_params = sum(p.numel() for p in net.parameters())
    aliased = len({p.data_ptr() for p in net.parameters()}) != len(list(net.parameters()))
    report("1c) no aliased parameters", not aliased, f"{n_params:,} parameters")


# ---------------------------------------------------------------------------------------------
# 2) port vs original
# ---------------------------------------------------------------------------------------------
def load_original(orig_dir):
    cwd = os.getcwd()
    os.chdir(orig_dir)  # the original loads utils/init_kernel_dir.mat relative to the cwd at import
    try:
        spec = importlib.util.spec_from_file_location("acdnet_original", os.path.join(orig_dir, "acdnet.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def untie(module):
    # What `.cuda()` does in the original train/test scripts: every parameter becomes its own copy.
    module._apply(lambda t: t.clone())
    return module


def check_port(orig_dir):
    mod = load_original(orig_dir)
    opt = SimpleNamespace(T=10, N=6, Np=32, d=32, num_res=3, Mtau=0.5, etaM=1, etaX=5, batchSize=32)
    torch.manual_seed(123)
    orig = untie(mod.ACDNet(opt))
    torch.manual_seed(123)
    ours = ACDNet()
    so, sp = orig.state_dict(), ours.state_dict()
    same_init = so.keys() == sp.keys() and all(torch.equal(so[k], sp[k]) for k in so)
    report("2a) identical parameter names and initial values (same seed) as the original", same_init,
           f"{len(sp)} state_dict entries")

    # Train-mode forward + loss + backward on identical weights/inputs.
    ma, li, gt, metal = fake_inputs(8, 64, seed=1)
    mask = 1 - metal
    outs_o = orig.train()(ma * 255, li * 255, mask)
    outs_p = ours.train()(ma * 255, li * 255, mask)
    lo = acdnet_loss(ma * 255, gt * 255, mask, *outs_o, T=10)[0]
    lp = acdnet_loss(ma * 255, gt * 255, mask, *outs_p, T=10)[0]
    lo.backward()
    lp.backward()
    go = dict(orig.named_parameters())
    max_gdiff = max(float((go[n].grad - p.grad).abs().max()) for n, p in ours.named_parameters() if p.grad is not None)
    report("2b) train mode: identical loss and gradients", torch.equal(lo, lp) and max_gdiff == 0.0,
           f"loss {lo.item():.6e} vs {lp.item():.6e}; max |grad diff| = {max_gdiff:.1e}")

    ckpt_path = os.path.join(orig_dir, "models", "ACDNet_latest.pt")
    if not os.path.exists(ckpt_path):
        report("2c) official checkpoint", False, f"missing {ckpt_path}")
        return
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    orig.load_state_dict(sd, strict=True)
    wrapper = ACDNetInference()
    wrapper.net.load_state_dict(sd, strict=True)
    orig.eval()
    wrapper.eval()
    ma, li, _, metal = fake_inputs(2, 128, seed=2)
    with torch.no_grad():
        o = orig(ma * 255, li * 255, 1 - metal)
        p = wrapper.net(ma * 255, li * 255, 1 - metal)
        pred = wrapper(ma, li, metal)
    flat_o = [o[0]] + o[1] + o[2] + o[3] + o[4]
    flat_p = [p[0]] + p[1] + p[2] + p[3] + p[4]
    max_diff = max(float((a - b).abs().max()) for a, b in zip(flat_o, flat_p))
    report("2c) official ACDNet_latest.pt loads strict; eval outputs identical to the original", max_diff == 0.0
           and torch.equal(pred, o[1][-1] / 255), f"max |diff| over all outputs = {max_diff:.1e}")


# ---------------------------------------------------------------------------------------------
# 3) patches
# ---------------------------------------------------------------------------------------------
def make_patch_test_dataset(root):
    dirs = make_fake_dataset(root, n_body=40, n_head=10, shape=(96, 80))
    rng = np.random.default_rng(1)
    files = sorted(os.listdir(dirs["MA_image"]))
    # Vary on-disk formats: (H,W,1) arrays, float64, LI with mu, masks as uint8 {0,255} / bool / float {0,1}.
    for i, f in enumerate(files):
        tag = f.replace("_metalart_", "_{}_")
        if i % 3 == 0:
            p = os.path.join(dirs["MA_image"], f)
            np.save(p, np.load(p)[..., None].astype(np.float64))
        m = np.zeros((96, 80), np.float32)
        r, c = rng.integers(0, 80), rng.integers(0, 64)
        m[r:r + 10, c:c + 12] = 1
        mp = os.path.join(dirs["metal_mask"], tag.format("metalonlymask"))
        if i % 3 == 1:
            np.save(mp, (m * 255).astype(np.uint8))
        elif i % 3 == 2:
            np.save(mp, m.astype(bool)[..., None])
        else:
            np.save(mp, m)
    return dirs


def check_patches(tmp_root):
    dirs = make_patch_test_dataset(os.path.join(tmp_root, "patch"))
    train_ds, val_ds = build_datasets(dirs["MA_image"], dirs["GT"], dirs["LI"], dirs["metal_mask"])
    pds = ACDNetPatchDataset(train_ds, patch_size=64, patches_per_slice=4, seed=42)
    n_ok, n_tot, flips, positions = 0, 0, set(), set()
    for idx in range(len(train_ds)):
        draw = 1000 + 7 * idx
        full = train_ds[idx]  # (ma, gt, mask, li), (1,H,W)
        got = pds[(idx, draw)]
        params = pds.crop_params(draw, 96, 80)
        for j, (r, c, hf, vf) in enumerate(params):
            flips.add((hf, vf))
            positions.add((r, c))
            for name, ref in (("ma", full[0]), ("gt", full[1]), ("mask", full[2]), ("li", full[3])):
                win = ref[:, r:r + 64, c:c + 64]
                if hf:
                    win = torch.flip(win, dims=[2])
                if vf:
                    win = torch.flip(win, dims=[1])
                n_tot += 1
                n_ok += int(torch.equal(got[name][j], win) and got[name][j].dtype == win.dtype)
    report("3a) every patch == the same window (+same flips) of the dataset's full normalized slice",
           n_ok == n_tot, f"{n_ok}/{n_tot} (slices x 4 patches x 4 modalities); flip combos seen={sorted(flips)}; "
           f"{len(positions)} distinct windows; row range covers 0..32: "
           f"{min(r for r, _ in positions)}..{max(r for r, _ in positions)}")
    report("3b) masks binary {0,1} for uint8 {0,255} / bool / float masks",
           all(set(torch.unique(pds[(i, 5)]["mask"]).tolist()) <= {0.0, 1.0} for i in range(len(train_ds))))

    val_names = {p[0] for p in val_ds.pairs}
    report("3c) patch dataset only sees train slices", not ({p[0] for p in pds.base.pairs} & val_names)
           and pds.base.split == "train")

    n = len(train_ds)
    full_stream = list(DrawStreamSampler(n, 42, 0, 5 * n))
    passes_ok = all(sorted(i for i, _ in full_stream[k * n:(k + 1) * n]) == list(range(n)) for k in range(5))
    resumed = list(DrawStreamSampler(n, 42, 37, 3 * n))
    report("3d) draw stream: each pass covers every slice once; a stream started at draw 37 == full[37:]",
           passes_ok and resumed == full_stream[37:37 + 3 * n])

    # Worker independence: same batches with 0 and 2 workers.
    from torch.utils.data import DataLoader
    from models.acdnet.patch_data import worker_init_fn

    def batches(nw):
        dl = DataLoader(pds, batch_size=8, sampler=DrawStreamSampler(n, 42, 16, 48), num_workers=nw,
                        worker_init_fn=worker_init_fn)
        return [b for b in dl]

    b0, b2 = batches(0), batches(2)
    same = all(torch.equal(x[k], y[k]) for x, y in zip(b0, b2) for k in x)
    report("3e) DataLoader batches identical with 0 and 2 workers", same and len(b0) == 6,
           f"{len(b0)} batches of {tuple(b0[0]['ma'].shape)} -> trainer reshapes to (32,1,64,64)")


# ---------------------------------------------------------------------------------------------
# 4) metrics ignore metal
# ---------------------------------------------------------------------------------------------
def check_metal_exclusion():
    ma, pred, gt, metal = fake_inputs(3, 128, seed=3)
    base = evaluate_predictions(pred, gt, metal, HU_MIN, HU_MAX)
    pred2 = torch.where(metal > 0.5, torch.rand_like(pred) * 3 - 1, pred)  # garbage incl. outside [0,1]
    alt = evaluate_predictions(pred2, gt, metal, HU_MIN, HU_MAX)
    fill = evaluate_predictions(torch.where(metal > 0.5, ma, pred), gt, metal, HU_MIN, HU_MAX)
    gt_painted = evaluate_predictions(pred, torch.where(metal > 0.5, ma, gt), metal, HU_MIN, HU_MAX)
    same = all(base[k] == alt[k] == fill[k] == gt_painted[k] for k in ("ssim", "psnr", "rmse"))
    report("4) evaluate_predictions: SSIM/PSNR/RMSE unchanged by pred metal values, MA fill, or GT metal painting",
           same, f"ssim={base['ssim'][0]:.4f} psnr={base['psnr'][0]:.2f} rmse={base['rmse'][0]:.2f}")

    # Loss: --metal_mask_on_gt is a no-op as the loss is masked to the non-metal region.
    ma, li, gt, metal = fake_inputs(4, 64, seed=4)
    net = ACDNet().eval()
    with torch.no_grad():
        outs = net(ma * 255, li * 255, 1 - metal)
        l1 = acdnet_loss(ma * 255, gt * 255, 1 - metal, *outs, T=10)[0]
        l2 = acdnet_loss(ma * 255, torch.where(metal > 0.5, ma, gt) * 255, 1 - metal, *outs, T=10)[0]
    report("4b) ACDNet loss identical with/without metal painted on GT", torch.equal(l1, l2))


def check_mask_assert():
    ma, li, _, metal = fake_inputs(1, 64)
    m = ACDNetInference().eval()
    try:
        with torch.no_grad():
            m(ma, li, metal * 0.5)
        ok = False
    except ValueError:
        ok = True
    report("5) ACDNetInference rejects a non-binary metal mask", ok)


# ---------------------------------------------------------------------------------------------
# 6) smoke + resume
# ---------------------------------------------------------------------------------------------
def run_trainer(dirs, log_dir, extra, cpu=False):
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "train_acdnet.py"),
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
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[Epoch") or ln.startswith("[FINAL")
             or ln.startswith("    [") or ln.startswith("[data]") or ln.startswith("[model]")]
    print("\n".join("     " + ln for ln in lines))
    ok = (r.returncode == 0 and os.path.exists(os.path.join(log_dir, "last.pt"))
          and os.path.exists(os.path.join(log_dir, "best.pt")) and os.path.exists(os.path.join(log_dir, "final_metrics.json")))
    report("6a) smoke: 2 epochs x 20 it (512^2 fake data), val on 8 slices, finite losses, metrics, checkpoints", ok)
    if ok:
        ck = torch.load(os.path.join(log_dir, "best.pt"), map_location="cpu")  # default weights_only (as test.py)
        report("6b) best.pt loads with torch.load defaults (weights_only) and has the expected keys",
               {"model_state", "optimizer_state", "scheduler_state", "epoch", "best_ssim", "model_kwargs"} <= set(ck),
               f"keys={sorted(ck)}")
        r2 = run_trainer(dirs, log_dir, common + ["--epochs", "3", "--resume", os.path.join(log_dir, "last.pt")])
        ok2 = r2.returncode == 0 and "[resume]" in r2.stdout and "[Epoch 3]" in r2.stdout
        report("6c) --resume last.pt continues at epoch 3", ok2,
               next((ln for ln in r2.stdout.splitlines() if ln.startswith("[resume]")), ""))
        return dirs, log_dir
    return dirs, None


def check_resume_exact(tmp_root):
    dirs = make_fake_dataset(os.path.join(tmp_root, "resume128"), n_body=40, n_head=10, shape=(128, 128))
    common = ["--iters_per_epoch", "6", "--val_every", "1", "--full_val_every", "0", "--val_subset", "3",
              "--debug_max_val_samples", "4", "--num_workers", "0", "--milestones", "1", "--print_every", "100"]
    a, b = os.path.join(tmp_root, "run_a"), os.path.join(tmp_root, "run_b")
    ra = run_trainer(dirs, a, common + ["--epochs", "2"], cpu=True)
    rb1 = run_trainer(dirs, b, common + ["--epochs", "1"], cpu=True)
    rb2 = run_trainer(dirs, b, common + ["--epochs", "2", "--resume", os.path.join(b, "last.pt")], cpu=True)
    if not (ra.returncode == rb1.returncode == rb2.returncode == 0):
        report("6d) resume exactness", False, "a trainer run failed")
        return
    ca = torch.load(os.path.join(a, "last.pt"), map_location="cpu")
    cb = torch.load(os.path.join(b, "last.pt"), map_location="cpu")
    w_same = all(torch.equal(ca["model_state"][k], cb["model_state"][k]) for k in ca["model_state"])
    oa, ob = ca["optimizer_state"]["state"], cb["optimizer_state"]["state"]
    o_same = all(torch.equal(oa[i][k], ob[i][k]) for i in oa for k in oa[i])
    lr_same = ca["optimizer_state"]["param_groups"][0]["lr"] == cb["optimizer_state"]["param_groups"][0]["lr"]
    sched_same = ca["scheduler_state"] == cb["scheduler_state"]
    # best_ssim: in run B epoch 1 is that invocation's last epoch, so its subset SSIM comes from the full-val
    # pass (different val batch composition -> float32 conv rounding ~1e-8 relative). Not a resume effect.
    ssim_close = abs(ca["best_ssim"] - cb["best_ssim"]) <= 1e-6 * abs(ca["best_ssim"])
    report("6d) CPU: 2 epochs straight == 1 epoch + resume (weights, BN stats, Adam state, LR, scheduler "
           "bit-identical; best_ssim within 1e-6 rel)",
           w_same and o_same and lr_same and sched_same and ssim_close and ca["best_epoch"] == cb["best_epoch"],
           f"lr={ca['optimizer_state']['param_groups'][0]['lr']:.1e} (MultiStepLR milestone at 1 crossed), "
           f"best_ssim {ca['best_ssim']:.10f} vs {cb['best_ssim']:.10f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", default=os.environ.get("ACDNET_ORIG_DIR"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true")
    a = ap.parse_args()
    tmp_root = tempfile.mkdtemp(prefix="acdnet_test_")
    try:
        check_shapes()
        if a.orig_dir:
            check_port(a.orig_dir)
        else:
            print("[SKIP] 2) port vs original (pass --orig_dir <ACDNet clone>)")
        check_patches(tmp_root)
        check_metal_exclusion()
        check_mask_assert()
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
