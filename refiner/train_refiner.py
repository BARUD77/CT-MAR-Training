"""Train the near-metal refiner on regions collected by refiner/collect_crops.py.

Each step draws one 128 x 128 crop per slice, centred on a random pixel within 20 mm of the metal, with random
flips / 90-degree rotations. Loss (only where the gate is active, metal excluded):
    L = sum(g * |base + g * r - gt|) / sum(g)          (normalized image space)
Fixed schedule (AdamW, cosine LR, --epochs); the LAST epoch is saved (no checkpoint selection on validation data).
If --val_dir is given, the gated-region L1 (in HU) on validation regions is printed for monitoring only.

Usage:
  python refiner/train_refiner.py --train_dir D:\\data_npy\\refiner_crops\\oof --val_dir D:\\data_npy\\refiner_crops\\val
      --seed 1 --out refiner_runs\\refiner_oof_s1.pt
  ablations: --no_dist (no distance channel), --r_full 5 --r_zero 10 (gate radius)
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.near_metal_refiner import GatedRefiner, config_dict, gate_from_distance  # noqa: E402

HU_SPAN = 4096.0
CROP = 128


class RegionCrops(Dataset):
    def __init__(self, files, r_zero, train=True, seed=0):
        self.files, self.r_zero, self.train = files, r_zero, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        a = {k: z[k].astype(np.float32) for k in ("ma", "li", "base", "gt", "dist")}
        a["metal"] = z["metal"].astype(np.float32)
        H, W = a["gt"].shape
        rng = self.rng if self.train else np.random.default_rng(i)
        ys, xs = np.nonzero(a["dist"] <= self.r_zero)
        k = rng.integers(len(ys))
        y0 = int(np.clip(ys[k] - CROP // 2, 0, H - CROP))
        x0 = int(np.clip(xs[k] - CROP // 2, 0, W - CROP))
        out = np.stack([a[n][y0:y0 + CROP, x0:x0 + CROP] for n in ("ma", "li", "base", "gt", "dist", "metal")])
        if self.train:
            if rng.random() < 0.5:
                out = out[:, :, ::-1]
            if rng.random() < 0.5:
                out = out[:, ::-1, :]
            out = np.rot90(out, rng.integers(4), axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(out))


def worker_init(wid):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2 ** 32)


def step_loss(model, x):
    ma, li, base, gt, dist, metal = (x[:, j:j + 1] for j in range(6))
    out, r, g = model(ma, li, base, dist, metal, return_parts=True)
    w = g * (1.0 - metal)
    loss = (w * (out - gt).abs()).sum() / w.sum().clamp_min(1.0)
    base_l1 = (w * (base - gt).abs()).sum() / w.sum().clamp_min(1.0)
    return loss, base_l1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_dist", action="store_true")
    ap.add_argument("--r_full", type=float, default=10.0)
    ap.add_argument("--r_zero", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--max_files", type=int, default=0, help="for testing")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(os.path.join(args.train_dir, "*.npz")))
    if args.max_files:
        files = files[:args.max_files]
    vfiles = sorted(glob.glob(os.path.join(args.val_dir, "*.npz"))) if args.val_dir else []
    if args.max_files:
        vfiles = vfiles[:max(1, args.max_files // 4)]
    print(f"train regions {len(files)} ({args.train_dir}) | val regions {len(vfiles)} | {dev} | seed {args.seed}")
    model = GatedRefiner(use_dist=not args.no_dist, base=args.width, r_full=args.r_full, r_zero=args.r_zero).to(dev)
    print(f"refiner {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | config {config_dict(model)}")
    dl = DataLoader(RegionCrops(files, args.r_zero, True, args.seed), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, worker_init_fn=worker_init,
                    persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))
    vdl = DataLoader(RegionCrops(vfiles, args.r_zero, False), batch_size=args.batch, num_workers=0) if vfiles else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s / max(total, 1), 1.0))))
    log, t0 = [], time.time()
    for ep in range(args.epochs):
        model.train()
        tl = tb = 0.0
        for x in dl:
            x = x.to(dev, non_blocking=True)
            loss, b = step_loss(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tl += float(loss); tb += float(b)
        row = {"epoch": ep + 1, "train_l1_hu": tl / len(dl) * HU_SPAN, "train_base_l1_hu": tb / len(dl) * HU_SPAN}
        if vdl is not None:
            model.eval()
            vl = vb = 0.0
            with torch.no_grad():
                for x in vdl:
                    l, b = step_loss(model, x.to(dev))
                    vl += float(l); vb += float(b)
            row.update(val_l1_hu=vl / len(vdl) * HU_SPAN, val_base_l1_hu=vb / len(vdl) * HU_SPAN)
        log.append(row)
        print("  " + " | ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items())
              + f" | {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config_dict(model), "args": vars(args), "log": log,
                "train_dir": args.train_dir, "n_train": len(files)}, args.out)
    with open(os.path.splitext(args.out)[0] + "_log.json", "w") as f:
        json.dump({"args": vars(args), "log": log}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
