"""Local (synthetic-data) checks for the ADN baseline.

Run from the repo root:
  python -m models.adn.tests.test_adn [--orig_dir <clone of liaohaofu/adn>] [--smoke]

  1) forward shapes of all networks at 512 and 256 (batch 2) + gradient flow of one train step
  2) port equivalence vs the official code (only with --orig_dir): identical init / state_dict keys /
     forwards, decoder_art fuse quirk, official deep_lesion_49.pt loads strict and reproduces the
     official demo outputs (if the checkpoint is in <orig_dir>/runs/deep_lesion/), and one train step
     equals ADNTrain.optimize with the generator's adversarial gradient taken through the pre-update D
  3) data: unpaired crop/slice == same window of the dataset's normalized slice (+ flip), disjoint
     halves, epoch = one pass over the MA half, shuffle mode, metal-centred crops, worker
     independence, resumable stream
  4) size independence: (a) mean-reduced losses, (b) normalization layers + train == eval at batch 1,
     (c) fully convolutional discriminators
  5) evaluate_predictions ignores the metal region of the prediction
  6) (--smoke) train_adn.py: 1 epoch on 32 MA / 32 clean 512^2 slices + 8 val slices, fp32 and bf16,
     resume; resume exactness on CPU
"""
import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.data import HU_MAX, HU_MIN, build_datasets  # noqa: E402
from baselines.common.eval_utils import evaluate_predictions  # noqa: E402
from baselines.common.tests.verify_local_stage2 import make_fake_dataset  # noqa: E402
from models.adn import (ADN, ADN_D_KWARGS, ADN_G_KWARGS, ADN_LOSS_WEIGHTS, ADNInference,  # noqa: E402
                        NLayerDiscriminator, losses as adn_losses)
from models.adn.data import EpochDrawSampler, UnpairedADNDataset, _mmap_2d  # noqa: E402

RESULTS = []
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def report(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def step_args(amp=False):
    return SimpleNamespace(amp=amp)


def make_nets(seed=0, device="cpu"):
    torch.manual_seed(seed)
    return (ADN(**ADN_G_KWARGS).to(device), NLayerDiscriminator(**ADN_D_KWARGS).to(device),
            NLayerDiscriminator(**ADN_D_KWARGS).to(device))


# ---------------------------------------------------------------------------------------------
# 1) shapes
# ---------------------------------------------------------------------------------------------
def check_shapes():
    import train_adn as tr
    g, dl, dh = make_nets(0, DEV)
    infer = ADNInference.from_adn(g, ADN_G_KWARGS)
    for s in (512, 256):
        x = torch.rand(2, 1, s, s, device=DEV) * 2 - 1
        h = torch.rand(2, 1, s, s, device=DEV) * 2 - 1
        with torch.no_grad():
            ll, lh, art = g.forward1(x)
            hl, hh = g.forward2(h, art)
            hlh, lhl = g.forward_lh(hl), g.forward_hl(hl, lh)
            o_dl, o_dh = dl(x), dh(x)
            y = infer(x * 0.5 + 0.5)
        shapes = [tuple(t.shape) for t in (ll, lh, hl, hh, hlh, lhl, y)]
        art_shapes = [tuple(t.shape[1:]) for t in art]
        ok = all(sh == (2, 1, s, s) for sh in shapes) and tuple(o_dl.shape) == tuple(o_dh.shape)
        ok = ok and art_shapes == [(256, s // 4, s // 4), (128, s // 2, s // 2), (64, s, s)]
        rng_ok = all(float(t.abs().max()) <= 1.0 for t in (ll, lh, hl, hh))
        report(f"1) shapes at {s}: ll/lh/hl/hh/hlh/lhl/ADNInference {(2, 1, s, s)}, artifact sides {art_shapes}, "
               f"D {tuple(o_dl.shape)}", ok and rng_ok)
    # gradient flow through one train step (all G and D parameters that should get gradients change)
    nets = make_nets(1, DEV)
    opts, _ = tr.build_optimizers(SimpleNamespace(lr=1e-4, beta1=0.5, beta2=0.999, weight_decay=1e-4,
                                                  step_size=100000, gamma=0.5), nets)
    before = [{k: v.detach().clone() for k, v in n.state_dict().items()} for n in nets]
    x = torch.rand(2, 1, 64, 64, device=DEV) * 2 - 1
    h = torch.rand(2, 1, 64, 64, device=DEV) * 2 - 1
    terms, g_total, d_stats = tr.train_step(nets, opts, x, h, step_args())
    changed = [[k for k, v in n.state_dict().items() if not torch.equal(v, b[k])] for n, b in zip(nets, before)]
    g_static = sorted(k for k in before[0] if k not in changed[0])
    # officially dead (no gradient path): decoder_art.fuse* (decoder_art uses decoder.fuse*) and encoder_art.res*
    # (only the num_sides=3 shallowest encoder_art outputs are used); also untrained in deep_lesion_49.pt
    dead = sorted([f"decoder_art.fuse{i}.{p}" for i in range(3) for p in ("bias", "weight")]
                  + [f"encoder_art.res{i}.model.{j}.conv.{p}" for i in range(4) for j in range(2) for p in ("bias", "weight")])
    ok = (g_static == dead and len(changed[1]) == len(before[1]) and len(changed[2]) == len(before[2]))
    report("1) one train step updates every G and D parameter except the officially dead decoder_art.fuse* and "
           "encoder_art.res*", ok, f"G terms {sorted(terms)}; unchanged G tensors: {len(g_static)}/{len(before[0])}")


# ---------------------------------------------------------------------------------------------
# 2) port vs original
# ---------------------------------------------------------------------------------------------
def check_port(orig_dir):
    sys.path.insert(0, orig_dir)
    from adn.models import ADNTrain
    from adn.networks import ADN as OrigADN, NLayerDiscriminator as OrigD
    from adn.utils import get_config

    torch.manual_seed(0)
    o = OrigADN(**ADN_G_KWARGS)
    torch.manual_seed(0)
    p = ADN(**ADN_G_KWARGS)
    so, sp = o.state_dict(), p.state_dict()
    torch.manual_seed(1)
    od = OrigD(**ADN_D_KWARGS)
    torch.manual_seed(1)
    pd = NLayerDiscriminator(**ADN_D_KWARGS)
    same_init = (list(so) == list(sp) and all(torch.equal(so[k], sp[k]) for k in so)
                 and list(od.state_dict()) == list(pd.state_dict())
                 and all(torch.equal(a, b) for a, b in zip(od.state_dict().values(), pd.state_dict().values())))
    report("2a) seeded init: identical G/D state_dict keys and values", same_init, f"G {len(so)} tensors")

    x, h = torch.rand(2, 1, 64, 64) * 2 - 1, torch.rand(2, 1, 64, 64) * 2 - 1
    with torch.no_grad():
        a1, a2 = o.forward1(x), o.forward2(x, h)
        b1 = p.forward1(x)
        b2 = p.forward2(h, b1[2])
        same_fwd = (all(torch.equal(u, v) for u, v in zip(a1 + a2, b1[:2] + b2))
                    and torch.equal(o.forward_hl(x, h), p.forward_hl(x, h)) and torch.equal(o.forward_lh(x), p.forward_lh(x))
                    and torch.equal(od(x), pd(x)))
        for m in (o, p):
            m.decoder_art.fuse0.weight.add_(1.0)
        dead = torch.equal(o.forward1(x)[0], a1[0]) and torch.equal(p.forward1(x)[0], b1[0])
        for m in (o, p):
            m.decoder.fuse0.weight.add_(1.0)
        live = not torch.equal(o.forward1(x)[0], a1[0]) and not torch.equal(p.forward1(x)[0], b1[0])
    report("2b) identical forward1/forward2/forward_lh/forward_hl/D outputs", same_fwd)
    report("2c) official quirk reproduced: decoder_art uses decoder.fuse*, decoder_art.fuse* are dead", dead and live)

    ck_path = os.path.join(orig_dir, "runs", "deep_lesion", "deep_lesion_49.pt")
    if os.path.isfile(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        g = ADN(**ADN_G_KWARGS)
        g.load_state_dict(ck["model_g"], strict=True)
        d = NLayerDiscriminator(**ADN_D_KWARGS)
        d.load_state_dict(ck["model_dl"], strict=True)
        inf = ADNInference()
        inf.load_state_dict({k: v for k, v in ck["model_g"].items() if k.startswith(("encoder_low.", "decoder."))},
                            strict=True)
        g.eval()
        errs = []
        import glob

        def load(f):  # demo.py load_image for deep_lesion
            im = np.load(f).astype(np.float64)
            im[im < -1000] = -1000
            im = im / 1000 * 0.192 + 0.192
            return torch.tensor(np.clip(im, 0, 0.5) / 0.5 * 2 - 1, dtype=torch.float32)[None, None]

        inv = lambda s: (s * 0.192 / 1000 + 0.192) * 2 - 1  # noqa: E731  (inverse of demo.py to_hu(x*0.5+0.5))
        lows = sorted(glob.glob(os.path.join(orig_dir, "samples", "deep_lesion", "with_art", "*.npy")))
        highs = sorted(glob.glob(os.path.join(orig_dir, "samples", "deep_lesion", "without_art", "*.npy")))
        for lf, hf in zip(lows, highs):
            ln, hn = os.path.basename(lf)[:-4], os.path.basename(hf)[:-4]
            rd = os.path.join(orig_dir, "results", "deep_lesion", f"{ln}_{hn}")
            with torch.no_grad():
                xl = load(lf)
                _, lh, art = g.forward1(xl)
                hl, _ = g.forward2(load(hf), art)
                y = inf(xl * 0.5 + 0.5) * 2 - 1
            errs.append(float(np.abs(lh[0, 0].numpy() - inv(np.load(os.path.join(rd, f"low_{ln}_artifact_reduced.npy")))).max()))
            errs.append(float(np.abs(hl[0, 0].numpy() - inv(np.load(os.path.join(rd, f"high_{hn}_artifact_transferred.npy")))).max()))
            errs.append(float((y - lh).abs().max()))
        report("2d) official deep_lesion_49.pt loads strict (G, D, ADNInference) and reproduces the official demo "
               "outputs (artifact-reduced + artifact-transferred)", len(errs) == 9 and max(errs) < 1e-4,
               f"max |diff| = {max(errs):.2e} in [-1,1] over {len(errs) // 3} sample pairs")
    else:
        print(f"[SKIP] 2d) pretrained check ({ck_path} not found)")

    # 2e) train step vs ADNTrain.optimize. The official step updates D before G.backward (only possible with
    # PyTorch<=1.0 .data semantics); with the D steps deferred, the official step is exactly G-with-pre-update-D
    # followed by the D steps on the same fakes, which is what train_adn.train_step does.
    import train_adn as tr
    opts_cfg = get_config(os.path.join(orig_dir, "config", "adn.yaml"))
    torch.manual_seed(3)
    orig = ADNTrain(opts_cfg["learn"], opts_cfg["loss"], **opts_cfg["model"])
    nets = (ADN(**ADN_G_KWARGS), NLayerDiscriminator(**ADN_D_KWARGS), NLayerDiscriminator(**ADN_D_KWARGS))
    for n, key in zip(nets, ("model_g", "model_dl", "model_dh")):
        n.load_state_dict(getattr(orig, key).state_dict())
    my_opts, _ = tr.build_optimizers(SimpleNamespace(lr=1e-4, beta1=0.5, beta2=0.999, weight_decay=1e-4,
                                                     step_size=100000, gamma=0.5), nets)
    saved = {}
    for key in ("model_dl", "model_dh"):
        m = getattr(orig, key)
        real_step = m._optimizer.step

        def deferred(m=m, key=key, real_step=real_step):
            saved[key] = (real_step, [q.grad.detach().clone() for q in m.parameters()])
        m._optimizer.step = deferred
    x, h = torch.rand(2, 1, 64, 64) * 2 - 1, torch.rand(2, 1, 64, 64) * 2 - 1
    orig.optimize(x, h)
    for key, (real_step, grads) in saved.items():
        m = getattr(orig, key)
        for q, gq in zip(m.parameters(), grads):
            q.grad = gq
        real_step()
    terms, g_total, d_stats = tr.train_step(nets, my_opts, x, h, step_args())
    ol = {k: float(v) for k, v in orig.loss.items()}
    name_map = {"gl_g": "gl_g", "gh_g": "gh_g", "ll": "ll_l1", "hh": "hh_l1", "hlh": "hlh_l1", "art": "art_l1"}
    loss_diff = max(abs(float(terms[k]) - ol[v]) for k, v in name_map.items())
    loss_diff = max(loss_diff, abs(float(d_stats[0]) - ol["gl_d"]), abs(float(d_stats[1]) - ol["gh_d"]))
    pdiff = 0.0
    for n, key in zip(nets, ("model_g", "model_dl", "model_dh")):
        ref = getattr(orig, key).state_dict()
        pdiff = max(pdiff, max(float((v - ref[k]).abs().max()) for k, v in n.state_dict().items()))
    report("2e) train_step == official ADNTrain.optimize (D steps deferred): all 8 loss values and all G/D "
           "parameters after one step", loss_diff < 1e-5 and pdiff < 1e-6,
           f"max loss diff {loss_diff:.1e}, max param diff {pdiff:.1e}; official loss keys {sorted(ol)}")


# ---------------------------------------------------------------------------------------------
# 3) data
# ---------------------------------------------------------------------------------------------
def check_data(tmp_root):
    dirs = make_fake_dataset(os.path.join(tmp_root, "data512"), n_body=40, n_head=10, shape=(512, 512))
    # put a metal blob far from the fake dataset's corner mask in some slices
    for f in sorted(os.listdir(dirs["metal_mask"]))[::3]:
        m = np.zeros((512, 512), np.float32)
        m[300:310, 400:412] = 1
        np.save(os.path.join(dirs["metal_mask"], f), m)
    train_ds, _ = build_datasets(dirs["MA_image"], dirs["GT"], None, dirs["metal_mask"])
    n = len(train_ds)

    def compare(u, keys):
        ok = True
        for e, k in keys:
            item = u[(e, k)]
            ma_idx, clean_idx = u.plan(e, k)
            ma_file, clean_file, mask_file = u._files(ma_idx)[0], u._files(clean_idx)[1], u._files(ma_idx)[2]
            mmaps = [_mmap_2d(os.path.join(d, f)) for d, f in ((train_ds.ma_dir, ma_file), (train_ds.gt_dir, clean_file))]
            mask_mm = _mmap_2d(os.path.join(train_ds.mask_dir, mask_file)) if u.metal_crop_prob > 0 else None
            w_ma, w_cl, flip = u.aug_params(e, k, mmaps[0], mmaps[1], mask_mm)
            ref_ma = train_ds[ma_idx][0][0][w_ma]
            ref_cl = train_ds[clean_idx][1][0][w_cl]
            if flip:
                ref_ma, ref_cl = ref_ma.flip(0), ref_cl.flip(0)
            ok &= torch.equal(item["ma"][0], ref_ma) and torch.equal(item["clean"][0], ref_cl)
            ok &= item["ma_idx"] == ma_idx and item["clean_idx"] == clean_idx
        return ok

    keys = [(e, k) for e in (1, 2) for k in range(0, 20, 3)]
    u_full = UnpairedADNDataset(train_ds, "disjoint", None, 0.7, True, 42)
    u_crop = UnpairedADNDataset(train_ds, "disjoint", 256, 0.7, True, 42)
    u_shuf = UnpairedADNDataset(train_ds, "shuffle", 128, 0.0, False, 42)
    flips = [u_full.aug_params(e, k, np.zeros((4, 4)), np.zeros((4, 4)), None)[2] for e, k in keys]
    report("3a) unpaired item == dataset normalized slice / window (full 512, crop 256 with metal crops + flip, "
           "shuffle crop 128), bit-identical", compare(u_full, keys) and compare(u_crop, keys) and compare(u_shuf, keys),
           f"flips used: {sum(flips)}/{len(flips)}")

    A, B = set(u_full.ma_pool.tolist()), set(u_full.clean_pool.tolist())
    plans = [u_full.plan(1, k) for k in range(u_full.n_ma)]
    plans2 = [u_full.plan(2, k) for k in range(u_full.n_ma)]
    ok = (not (A & B) and len(A | B) == n and abs(len(A) - len(B)) <= 1
          and sorted(p[0] for p in plans) == sorted(A) and all(p[1] in B for p in plans + plans2)
          and [p[0] for p in plans] != [p[0] for p in plans2])
    regions = [train_ds.pairs[i][0].split("_")[1] for i in sorted(A)]
    report("3b) disjoint: halves are disjoint and cover the split; an epoch visits every MA of half A once "
           "(new order each epoch); clean images only from half B",
           ok, f"n={n}: |A|={len(A)} |B|={len(B)}; head in A: {regions.count('head')}")
    sp = [u_shuf.plan(1, k) for k in range(u_shuf.n_ma)]
    report("3c) shuffle: MA and clean are two independent permutations of the full split",
           sorted(p[0] for p in sp) == list(range(n)) and sorted(p[1] for p in sp) == list(range(n))
           and [p[0] for p in sp] != [p[1] for p in sp])

    # metal-centred crops: the crop contains metal whenever the draw chose a metal crop and the slice has metal
    u_m = UnpairedADNDataset(train_ds, "disjoint", 256, 1.0, False, 42)
    n_metal, n_tot = 0, 0
    for k in range(u_m.n_ma):
        ma_idx, _ = u_m.plan(1, k)
        mask = _mmap_2d(os.path.join(train_ds.mask_dir, u_m._files(ma_idx)[2]))
        ma_mm = _mmap_2d(os.path.join(train_ds.ma_dir, u_m._files(ma_idx)[0]))
        w_ma, _, _ = u_m.aug_params(1, k, ma_mm, ma_mm, mask)
        n_metal += bool(np.asarray(mask)[w_ma].max() > 0)
        n_tot += 1
    report("3d) metal_crop_prob=1: every 256 MA crop contains metal (jitter +-64 px)", n_metal == n_tot,
           f"{n_metal}/{n_tot}")

    def batches(nw, start, end, dpe=8):
        dl = DataLoader(u_crop, batch_size=2, sampler=EpochDrawSampler(start, end, dpe), num_workers=nw)
        return [(b["ma"], b["clean"], b["ma_idx"]) for b in dl]
    b0, b2 = batches(0, 1, 2), batches(2, 1, 2)
    same = all(torch.equal(x0, x2) for p, q in zip(b0, b2) for x0, x2 in zip(p, q)) and len(b0) == len(b2) == 8
    report("3e) batches identical with num_workers 0 vs 2 (counter-based RNG; per-worker seeding)", same)
    b_res = batches(0, 2, 2)
    report("3f) stream resumed at epoch 2 == epoch 2 of the uninterrupted stream",
           all(torch.equal(x0, x2) for p, q in zip(b0[4:], b_res) for x0, x2 in zip(p, q)))
    return dirs


# ---------------------------------------------------------------------------------------------
# 4) size independence
# ---------------------------------------------------------------------------------------------
def check_size_independence():
    src = inspect.getsource(adn_losses)
    calls = [ln.strip() for ln in src.splitlines() if "F.l1_loss" in ln or "F.mse_loss" in ln]
    no_red = "reduction=" not in src and ".sum(" not in src
    c = torch.full((1, 1, 62, 62), 0.3)
    C = torch.full((2, 1, 126, 126), 0.3)
    same_val = (abs(float(adn_losses.lsgan_loss(c, True)) - float(adn_losses.lsgan_loss(C, True))) < 1e-7
                and abs(float(torch.nn.functional.l1_loss(c, c * 0)) - float(torch.nn.functional.l1_loss(C, C * 0))) < 1e-7)
    report("4a) every loss term uses mean reduction (F.l1_loss / F.mse_loss defaults, no 'reduction' / sums): "
           "same value for a 62^2 x1 and 126^2 x2 constant map", no_red and same_val and len(calls) >= 5,
           f"{len(calls)} call sites")

    g, dl, _ = make_nets(0, "cpu")
    types = {}
    for name, net in (("G", g), ("D", dl)):
        for m in net.modules():
            t = type(m).__name__
            if "Norm" in t:
                key = f"{name}:{t}"
                if t == "InstanceNorm2d":
                    key += f"(affine={m.affine}, track_running_stats={m.track_running_stats})"
                types[key] = types.get(key, 0) + 1
    no_bn = not any("BatchNorm" in k for k in types)
    x = torch.rand(1, 1, 128, 128) * 2 - 1
    with torch.no_grad():
        g.train()
        dl.train()
        a = (g.forward1(x)[:2], dl(x))
        g.eval()
        dl.eval()
        b = (g.forward1(x)[:2], dl(x))
    same = torch.equal(a[0][0], b[0][0]) and torch.equal(a[0][1], b[0][1]) and torch.equal(a[1], b[1])
    report("4b) no BatchNorm; per-sample InstanceNorm (no running stats) + per-sample LayerNorm; train == eval "
           "output at batch 1", no_bn and same, f"{types}")

    has_linear = any(isinstance(m, torch.nn.Linear) for m in dl.modules())
    sizes = {}
    with torch.no_grad():
        for s in (128, 256, 300, 512):
            sizes[s] = tuple(dl(torch.zeros(1, 1, s, s)).shape[-2:])
    report("4c) discriminators fully convolutional (no Linear; any input size; output map scales with input)",
           not has_linear and sizes[512] == (126, 126) and sizes[256] == (62, 62), f"output maps {sizes}")


# ---------------------------------------------------------------------------------------------
# 5) metal exclusion
# ---------------------------------------------------------------------------------------------
def check_metal_exclusion():
    g = torch.Generator().manual_seed(0)
    gt = torch.rand(2, 1, 128, 128, generator=g) * 0.3 + 0.2
    pred = gt + 0.02 * torch.randn(2, 1, 128, 128, generator=g)
    m = torch.zeros(2, 1, 128, 128)
    m[:, :, 40:50, 60:75] = 1
    a = evaluate_predictions(pred, gt, m, HU_MIN, HU_MAX)
    pred2 = torch.where(m > 0.5, torch.full_like(pred, 0.9), pred)
    b = evaluate_predictions(pred2, gt, m, HU_MIN, HU_MAX)
    report("5) evaluate_predictions: changing the prediction inside the metal mask changes no metric "
           "(SSIM fills the dilated metal in both images; RMSE/PSNR exclude it) -> no MA fill needed", a == b,
           f"ssim {a['ssim']} rmse {a['rmse']}")


# ---------------------------------------------------------------------------------------------
# 6) smoke + resume
# ---------------------------------------------------------------------------------------------
def run_trainer(dirs, log_dir, extra, cpu=False):
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "train_adn.py"),
           "--ma_dir", dirs["MA_image"], "--gt_dir", dirs["GT"], "--mask_dir", dirs["metal_mask"],
           "--hu_min", "-1024", "--hu_max", "3072", "--metal_mask_on_gt", "--log_dir", log_dir,
           "--wandb_mode", "disabled"] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"  # an empty value is dropped on Windows and the GPU stays visible
    r = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    return r


def show(r):
    keep = ("[Epoch", "[FINAL", "    [", "[data]", "[model]", "[resume]")
    print("\n".join("     " + ln[:260] for ln in r.stdout.splitlines() if ln.startswith(keep)))


def check_smoke(tmp_root):
    dirs = make_fake_dataset(os.path.join(tmp_root, "smoke512"), n_body=60, n_head=12, shape=(512, 512))
    common = ["--debug_max_train_samples", "64", "--debug_max_val_samples", "8", "--num_workers", "2",
              "--print_every", "8", "--val_batch_size", "4"]
    for amp in (False, True):
        if amp and DEV.type != "cuda":
            print("[SKIP] 6b) bf16 smoke needs CUDA")
            continue
        log_dir = os.path.join(tmp_root, f"run_smoke_{'bf16' if amp else 'fp32'}")
        # fp32 full-512 training at batch 2 needs > 12 GB; on small GPUs train on 256 crops (val stays full 512)
        small_gpu = DEV.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 16 * 1024 ** 3
        crop = ["--crop_size", "256"] if (small_gpu and not amp) else []
        if crop:
            print("     (GPU < 16 GB: fp32 smoke trains on 256 crops; validation on full 512 slices)")
        r = run_trainer(dirs, log_dir, common + ["--epochs", "1"] + (["--amp"] if amp else []) + crop)
        show(r)
        ok = r.returncode == 0 and all(os.path.exists(os.path.join(log_dir, f))
                                       for f in ("last.pt", "best.pt", "final_metrics.json"))
        detail = ""
        if ok:
            fm = json.load(open(os.path.join(log_dir, "final_metrics.json")))
            detail = f"final last: SSIM {fm['last']['overall']['ssim']:.4f} PSNR {fm['last']['overall']['psnr']:.2f}"
        report(f"6{'b' if amp else 'a'}) smoke {'bf16' if amp else 'fp32'}: 1 epoch on 32 MA / 32 clean 512^2 slices "
               f"(batch 2 -> 16 steps), val on 8 slices, finite losses, metrics, checkpoints", ok, detail)
        if ok and not amp:
            ck = torch.load(os.path.join(log_dir, "best.pt"), map_location="cpu")  # default weights_only (as test.py)
            exp = {"model_state", "adn_full", "optimizer_state", "scheduler_state", "epoch", "best_ssim",
                   "model_kwargs", "monitor_metric", "monitor_value"}
            inf = ADNInference(**{k: v for k, v in ck["model_kwargs"].items()})
            inf.load_state_dict(ck["model_state"], strict=True)
            report("6c) best.pt loads with torch.load defaults (weights_only); model_state loads strict into "
                   "ADNInference; adn_full has G/D_l/D_h", exp <= set(ck) and set(ck["adn_full"]) ==
                   {"model_g", "model_dl", "model_dh"}, f"keys={sorted(ck)}")
            r2 = run_trainer(dirs, log_dir, common + crop + ["--epochs", "2", "--resume", os.path.join(log_dir, "last.pt")])
            show(r2)
            report("6d) --resume last.pt continues at epoch 2", r2.returncode == 0 and "[resume]" in r2.stdout
                   and "[Epoch 2]" in r2.stdout)


def check_resume_exact(tmp_root):
    # CPU training here is not bit-deterministic run to run (two identical uninterrupted runs can differ by
    # ~1e-4 after 2 epochs), so resume is judged against that noise floor: counters / LRs / schedulers must
    # be identical, weights within the floor.
    dirs = make_fake_dataset(os.path.join(tmp_root, "resume64"), n_body=40, n_head=10, shape=(64, 64))
    common = ["--debug_max_train_samples", "12", "--debug_max_val_samples", "4", "--num_workers", "0",
              "--print_every", "100", "--val_subset", "3", "--step_size", "1"]
    a, a2, b = (os.path.join(tmp_root, n) for n in ("run_a", "run_a2", "run_b"))
    runs = [run_trainer(dirs, a, common + ["--epochs", "2"], cpu=True),
            run_trainer(dirs, a2, common + ["--epochs", "2"], cpu=True),
            run_trainer(dirs, b, common + ["--epochs", "1"], cpu=True),
            run_trainer(dirs, b, common + ["--epochs", "2", "--resume", os.path.join(b, "last.pt")], cpu=True)]
    if any(r.returncode for r in runs):
        report("6e) resume exactness", False, "a trainer run failed")
        return
    ca, ca2, cb = (torch.load(os.path.join(d, "last.pt"), map_location="cpu") for d in (a, a2, b))

    def wdiff(x, y):
        return max(float((v - y["adn_full"][n][k]).abs().max()) for n in x["adn_full"] for k, v in x["adn_full"][n].items())

    def counters(c):
        return ({n: [int(s["step"]) for s in c["optimizer_state"][n]["state"].values()] for n in c["optimizer_state"]},
                {n: c["optimizer_state"][n]["param_groups"][0]["lr"] for n in c["optimizer_state"]},
                c["scheduler_state"], c["epoch"])
    floor, d_res = wdiff(ca, ca2), wdiff(ca, cb)
    same_counters = counters(ca) == counters(cb)
    ok = same_counters and d_res <= max(5 * floor, 1e-3)
    report("6e) CPU: 2 epochs straight vs 1 epoch + resume: identical Adam step counts, LRs, schedulers, epoch; "
           "weights within the run-to-run noise floor", ok,
           f"max |w| diff straight-vs-resume {d_res:.1e} | straight-vs-straight {floor:.1e} | lr after 2 epochs "
           f"(StepLR step 1 for the test) = {ca['optimizer_state']['model_g']['param_groups'][0]['lr']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", default=os.environ.get("ADN_ORIG_DIR"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true")
    a = ap.parse_args()
    tmp_root = tempfile.mkdtemp(prefix="adn_test_")
    try:
        check_shapes()
        if a.orig_dir:
            check_port(a.orig_dir)
        else:
            print("[SKIP] 2) port vs original (pass --orig_dir <liaohaofu/adn clone>)")
        check_data(tmp_root)
        check_size_independence()
        check_metal_exclusion()
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
