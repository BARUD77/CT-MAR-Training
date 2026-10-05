"""Stage 2 local verification (CPU, synthetic data only).

  a) metric equivalence: frozen original metric code vs baselines/common/eval_utils.py
  b) checkpoint round trip: states + RNG + continued-training equivalence + atomic-save safety
  c) fake dataset with the real file naming: common/data.py split == original trainer split

Run from the repo root:  python -m baselines.common.tests.verify_local_stage2
"""
import copy
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from scipy import ndimage

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common import checkpoint as ckpt  # noqa: E402
from baselines.common.data import build_datasets, unpack_batch, DATASET_KWARGS  # noqa: E402
from baselines.common.eval_utils import evaluate_predictions, summarize_metrics  # noqa: E402
from baselines.common.tests.original_metric_code import original_val_metrics  # noqa: E402
from baselines.common.tests.original_split_code import original_trainer_datasets  # noqa: E402

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# --------------------------------------------------------------------------------------------
# a) metric equivalence
# --------------------------------------------------------------------------------------------
def random_metal_mask(rng, h, w, n_blobs):
    m = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[:h, :w]
    for _ in range(n_blobs):
        cy, cx = rng.integers(40, h - 40, size=2)
        ry, rx = rng.integers(2, 15, size=2)
        m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    return m.astype(np.float32)


def synthetic_batch(rng, b, h=512, w=512, with_metal=True):
    gt = ndimage.gaussian_filter(rng.random((b, 1, h, w)), sigma=(0, 0, 3, 3)).astype(np.float32)
    gt = (gt - gt.min()) / (gt.max() - gt.min())
    # predictions: GT + noise, deliberately exceeding [0,1] in places to exercise clamping
    pred = gt + rng.normal(0, 0.05, size=gt.shape).astype(np.float32)
    pred[:, :, :20, :20] = 1.3
    pred[:, :, -20:, -20:] = -0.2
    mask = None
    if with_metal:
        mask = np.stack([random_metal_mask(rng, h, w, int(rng.integers(1, 5)))[None] for _ in range(b)])
        mask = torch.from_numpy(mask)
    return torch.from_numpy(pred), torch.from_numpy(gt), mask


def check_original_copy_matches_git():
    """The frozen reference must match the pre-refactor trainer in git (commit cd22fab)."""
    try:
        src = subprocess.run(["git", "show", "cd22fab:trainer_wandb_unclipped.py"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except Exception as e:  # git not available
        report("a0) frozen reference == git cd22fab trainer code", False, f"git show failed: {e}")
        return
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if "# Normalized metrics are computed on clamped" in l)
    end = next(i for i, l in enumerate(lines) if "std_rmse = float(np.std(rmse_list" in l)
    orig_block = [l.strip() for l in lines[start:end + 1] if l.strip()]
    ref_path = os.path.join(os.path.dirname(__file__), "original_metric_code.py")
    with open(ref_path, encoding="utf-8") as f:
        ref_lines = [l.strip() for l in f.read().splitlines() if l.strip()]
    n = len(orig_block)
    found = any(ref_lines[i:i + n] == orig_block for i in range(len(ref_lines) - n + 1))
    report("a0) frozen reference == git cd22fab trainer code", found, f"{n} code lines compared")


def check_metrics():
    rng = np.random.default_rng(0)
    hu_min, hu_max = DATASET_KWARGS["hu_min"], DATASET_KWARGS["hu_max"]
    all_new = {"ssim": [], "rmse": [], "psnr": []}
    all_old = {"ssim": [], "rmse": [], "psnr": []}
    n_imgs, max_abs = 0, 0.0
    exact = True
    cases = [(4, True), (3, True), (1, True), (2, False)]
    for b, with_metal in cases:
        pred, gt, mask = synthetic_batch(rng, b, with_metal=with_metal)
        new = evaluate_predictions(pred, gt, mask, hu_min, hu_max)
        o_ssim, o_psnr, o_rmse, _ = original_val_metrics(pred, gt, mask, hu_min, hu_max)
        old = {"ssim": o_ssim, "rmse": o_rmse, "psnr": o_psnr}
        for k in old:
            exact &= (new[k] == old[k])
            max_abs = max(max_abs, max(abs(a - c) for a, c in zip(new[k], old[k])))
            all_new[k] += new[k]
            all_old[k] += old[k]
        n_imgs += b
    report("a1) per-image SSIM/RMSE/PSNR identical (exact float ==)", exact,
           f"{n_imgs} images 512x512 (incl. 2 without metal mask), max |diff| = {max_abs:.3e}")

    # summaries: original computes mean/std over the concatenated lists
    old_summary = {}
    for k, v in all_old.items():
        old_summary[k] = float(np.mean(v))
        old_summary[f"{k}_std"] = float(np.std(v, ddof=1))
    new_summary = summarize_metrics(all_new)
    report("a2) mean/std identical", new_summary == old_summary,
           ", ".join(f"{k}={new_summary[k]:.6f}" for k in ["ssim", "ssim_std", "rmse", "psnr"]))
    print("     sample per-image values (new vs old):")
    for i in range(3):
        print(f"       img{i}: ssim {all_new['ssim'][i]:.10f} / {all_old['ssim'][i]:.10f} | "
              f"rmse {all_new['rmse'][i]:.6f} / {all_old['rmse'][i]:.6f} | "
              f"psnr {all_new['psnr'][i]:.6f} / {all_old['psnr'][i]:.6f}")


def check_trainer_imports_refactor():
    """The refactored trainer must import eval_utils and no longer contain the inline loop."""
    with open(os.path.join(_REPO_ROOT, "trainer_wandb_unclipped.py"), encoding="utf-8") as f:
        src = f.read()
    ok = ("evaluate_predictions(pred, y_batch, mask_batch, args.hu_min, args.hu_max)" in src
          and "summarize_metrics(" in src and "metal_fill_norm" not in src)
    report("a3) trainer_wandb_unclipped.py calls eval_utils (inline loop removed)", ok)


# --------------------------------------------------------------------------------------------
# b) checkpoint round trip
# --------------------------------------------------------------------------------------------
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.out = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x):
        return self.out(torch.relu(self.bn(self.conv(x))))


def make_training_objects(seed):
    torch.manual_seed(seed)
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + 0.1 * s))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, opt, sched, scaler


def train_steps(model, opt, sched, n):
    for _ in range(n):
        x = torch.rand(2, 1, 16, 16)  # consumes torch RNG -> resume must restore it
        noise = np.random.rand()      # consumes numpy RNG
        jitter = random.random()      # consumes python RNG
        loss = (model(x) - x * (1 + 0.01 * noise + 0.01 * jitter)).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


def states_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(states_equal(x, y) for x, y in zip(a, b))
    return a == b


def check_checkpoint(tmp_root):
    ckpt_dir = os.path.join(tmp_root, "drive_ckpts")
    local_tmp = os.path.join(tmp_root, "local_tmp")
    random.seed(1); np.random.seed(1)
    model, opt, sched, scaler = make_training_objects(0)
    train_steps(model, opt, sched, 5)
    path = os.path.join(ckpt_dir, ckpt.LAST_NAME)
    ckpt.save_checkpoint(path, model, opt, sched, scaler, epoch=3, global_step=5, best_metric=0.8123,
                         wandb_run_id="abc123", extra={"iter_in_epoch": 2}, local_tmp_dir=local_tmp)
    saved_model = copy.deepcopy(model.state_dict())
    saved_opt = copy.deepcopy(opt.state_dict())  # deep copy: optimizer state tensors are updated in place
    saved_sched = copy.deepcopy(sched.state_dict())

    # reference continuation (no reload)
    train_steps(model, opt, sched, 4)
    ref_after = {k: v.clone() for k, v in model.state_dict().items()}
    ref_draws = (torch.rand(3).tolist(), np.random.rand(3).tolist(), [random.random() for _ in range(3)])

    # fresh objects with different init + scrambled RNG, then resume
    random.seed(999); np.random.seed(999); torch.manual_seed(999)
    model2, opt2, sched2, scaler2 = make_training_objects(123)
    meta = ckpt.load_checkpoint(ckpt.find_resume_checkpoint(ckpt_dir), model2, opt2, sched2, scaler2)
    ok_meta = (meta["epoch"] == 3 and meta["global_step"] == 5 and abs(meta["best_metric"] - 0.8123) < 1e-12
               and meta["wandb_run_id"] == "abc123" and meta["extra"] == {"iter_in_epoch": 2})
    report("b1) metadata restored (epoch, step, best, W&B id, extra)", ok_meta, str(meta))
    ok_states = (states_equal(model2.state_dict(), saved_model) and states_equal(opt2.state_dict(), saved_opt)
                 and states_equal(sched2.state_dict(), saved_sched))
    report("b2) model / optimizer / scheduler states identical", ok_states)

    train_steps(model2, opt2, sched2, 4)
    draws = (torch.rand(3).tolist(), np.random.rand(3).tolist(), [random.random() for _ in range(3)])
    report("b3) RNG restored (torch, numpy, python draws match)", draws == ref_draws)
    report("b4) continued training after resume is bit-identical",
           states_equal(model2.state_dict(), ref_after))

    # atomic save: simulate a disconnect during the copy to the checkpoint dir
    before = open(path, "rb").read()
    orig_copy = shutil.copyfile

    def broken_copy(src, dst, *a, **k):
        with open(dst, "wb") as f:
            f.write(b"partial")
        raise IOError("simulated disconnect")

    shutil.copyfile = broken_copy
    try:
        ckpt.save_checkpoint(path, model2, opt2, sched2, scaler2, epoch=9, global_step=99, best_metric=0.9,
                             local_tmp_dir=local_tmp)
        raised = False
    except IOError:
        raised = True
    finally:
        shutil.copyfile = orig_copy
    intact = open(path, "rb").read() == before
    leftovers = [f for f in os.listdir(ckpt_dir) if f.endswith(".tmp")] + os.listdir(local_tmp)
    report("b5) interrupted save leaves previous last.pt intact, no temp files",
           raised and intact and not leftovers, f"raised={raised} intact={intact} leftovers={leftovers}")
    report("b6) improvement rule (higher SSIM is better)",
           ckpt.is_improvement(0.9, None) and ckpt.is_improvement(0.91, 0.9)
           and not ckpt.is_improvement(0.9, 0.9))


# --------------------------------------------------------------------------------------------
# c) fake dataset split
# --------------------------------------------------------------------------------------------
def make_fake_dataset(root, n_body=80, n_head=12, shape=(16, 16), missing_li=()):
    dirs = {k: os.path.join(root, k) for k in ["MA_image", "GT", "LI", "metal_mask"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = [("body", i) for i in range(n_body)] + [("head", 1000 + i) for i in range(n_head)]
    for region, i in ids:
        tag = f"{shape[0]}x{shape[1]}x1"
        np.save(os.path.join(dirs["MA_image"], f"training_{region}_metalart_img{i}_{tag}.npy"),
                rng.uniform(-1500, 4000, shape).astype(np.float32))
        np.save(os.path.join(dirs["GT"], f"training_{region}_nometal_img{i}_{tag}.npy"),
                rng.uniform(-1500, 4000, shape).astype(np.float32))
        if (region, i) not in missing_li:
            np.save(os.path.join(dirs["LI"], f"training_{region}_li_img{i}_{tag}.npy"),
                    rng.uniform(0.0, 0.5, shape).astype(np.float32))
        m = np.zeros(shape, np.float32); m[4:7, 5:9] = 1
        np.save(os.path.join(dirs["metal_mask"], f"training_{region}_metalonlymask_img{i}_{tag}.npy"), m)
    return dirs


def check_split(tmp_root):
    dirs = make_fake_dataset(os.path.join(tmp_root, "fake"))
    tr_new, va_new = build_datasets(dirs["MA_image"], dirs["GT"], dirs["LI"], dirs["metal_mask"])
    tr_old, va_old = original_trainer_datasets(dirs["MA_image"], dirs["GT"], dirs["LI"], dirs["metal_mask"])
    same = tr_new.pairs == tr_old.pairs and va_new.pairs == va_old.pairs
    report("c1) common/data.py split == original trainer split (same order)", same,
           f"train={len(tr_new)} val={len(va_new)}; val[0]={va_new.pairs[0][0]}")
    disjoint = not ({p[0] for p in tr_new.pairs} & {p[0] for p in va_new.pairs})
    report("c2) train/val disjoint", disjoint)

    s_new, s_old = tr_new[0], tr_old[0]
    same_sample = all(torch.equal(a, b) for a, b in zip(s_new, s_old))
    f = unpack_batch(s_new, has_mask=True, has_li=True)
    rng_ok = all(0.0 <= float(f[k].min()) and float(f[k].max()) <= 1.0 for k in ["ma", "gt", "li"])
    mask_ok = set(torch.unique(f["mask"]).tolist()) <= {0.0, 1.0}
    report("c3) identical sample tensors; unpack_batch -> ma/gt/li in [0,1], binary mask",
           same_sample and rng_ok and mask_ok, f"shapes={[tuple(f[k].shape) for k in ['ma','gt','mask','li']]}")

    # Informational: the split depends on which directories are passed (key intersection).
    dirs2 = make_fake_dataset(os.path.join(tmp_root, "fake_missing"), missing_li={("body", 3)})
    tr_a, va_a = build_datasets(dirs2["MA_image"], dirs2["GT"], dirs2["LI"], dirs2["metal_mask"])
    tr_b, va_b = build_datasets(dirs2["MA_image"], dirs2["GT"], None, dirs2["metal_mask"])
    print(f"     note: with one LI file missing, split with LI dir ({len(tr_a)}/{len(va_a)}) vs without "
          f"({len(tr_b)}/{len(va_b)}) differ={[p[:3] for p in va_a.pairs] != va_b.pairs} "
          "-> always pass the same dirs as the reference runs.")


def main():
    tmp_root = tempfile.mkdtemp(prefix="stage2_verify_")
    try:
        print("== a) metric equivalence ==")
        check_original_copy_matches_git()
        check_metrics()
        check_trainer_imports_refactor()
        print("== b) checkpoint round trip ==")
        check_checkpoint(tmp_root)
        print("== c) fake dataset split ==")
        check_split(tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\nSUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed -> {'ALL PASS' if n_fail == 0 else 'FAIL'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
