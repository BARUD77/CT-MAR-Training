"""Train ACDNet (Wang et al., IJCAI 2022; https://github.com/hongwang01/ACDNet) on our AAPM split.

Follows the original recipe: 64x64 random patches (4 per slice x 8 slices = 32 per batch) from the
train split with random h/v flips, 800 iterations per epoch, 300 epochs, Adam(lr 2e-4, betas
(0.5, 0.999)), MultiStepLR([50,100,150,200], 0.5) stepped per epoch, the original multi-stage loss,
fp32. Images are fed in the original x255 scale; the output is divided by 255 for evaluation and
never clamped. Validation runs on full 512x512 val slices through `evaluate_predictions`.

W&B / logging / checkpoint conventions follow trainer_wandb_unclipped.py.

Example:
  python train_acdnet.py --ma_dir .../MA_image --gt_dir .../GT --li_dir .../LI --mask_dir .../metal_mask \
      --hu_min -1024 --hu_max 3072 --metal_mask_on_gt --run_name alpha_acdnet --log_dir <drive>/alpha_acdnet
  # continue after a Colab disconnect:
  python train_acdnet.py <same args> --resume <drive>/alpha_acdnet/last.pt
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from baselines.common.checkpoint import atomic_torch_save
from baselines.common.data import HU_MAX, HU_MIN, build_datasets, unpack_batch
from baselines.common.eval_utils import evaluate_predictions, summarize_metrics
from models.acdnet import ACDNET_SCALE, ACDNetInference, acdnet_loss, nonmetal_from_metal
from models.acdnet.patch_data import ACDNetPatchDataset, DrawStreamSampler, worker_init_fn

MODEL_NAME = "acdnet"
MODEL_ARG_NAMES = ("T", "N", "Np", "d", "num_res", "Mtau", "etaM", "etaX")
# Arguments that define the training stream; they must not change when resuming.
STREAM_ARG_NAMES = ("seed", "iters_per_epoch", "batch_slices", "patches_per_slice", "patch_size")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ACDNet baseline")
    # data
    p.add_argument('--ma_dir', type=str, required=True)
    p.add_argument('--gt_dir', type=str, required=True)
    p.add_argument('--li_dir', type=str, required=True)
    p.add_argument('--mask_dir', type=str, required=True)
    p.add_argument('--hu_min', type=float, default=HU_MIN,
                   help='Must equal the fixed baseline window (baselines/common/data.py).')
    p.add_argument('--hu_max', type=float, default=HU_MAX,
                   help='Must equal the fixed baseline window (baselines/common/data.py).')
    p.add_argument('--metal_mask_on_gt', action='store_true',
                   help='Paint MA metal onto GT (as in trainer_wandb_unclipped.py). No-op for the ACDNet loss '
                        '(masked to non-metal) and for evaluate_predictions (metal excluded); kept for parity.')
    # model (original defaults)
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--N', type=int, default=6)
    p.add_argument('--Np', type=int, default=32)
    p.add_argument('--d', type=int, default=32)
    p.add_argument('--num_res', type=int, default=3)
    p.add_argument('--Mtau', type=float, default=0.5)
    p.add_argument('--etaM', type=float, default=1.0)
    p.add_argument('--etaX', type=float, default=5.0)
    # loss weights (original defaults)
    p.add_argument('--Xl2', type=float, default=1.0)
    p.add_argument('--Xl1', type=float, default=5e-4)
    p.add_argument('--Al1', type=float, default=5e-4)
    # optimization (original defaults)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--iters_per_epoch', type=int, default=800)
    p.add_argument('--batch_slices', type=int, default=8, help='Slices per batch')
    p.add_argument('--patches_per_slice', type=int, default=4, help='Patches per slice (batch = 8 x 4 = 32)')
    p.add_argument('--patch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--milestones', type=int, nargs='+', default=[50, 100, 150, 200])
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--amp', action='store_true',
                   help='bf16 autocast for the training forward (loss in fp32). Off by default (original is fp32); '
                        'only use it if colab/acdnet_checks.py shows matching 200-iteration loss curves.')
    p.add_argument('--tf32', action='store_true',
                   help='Allow TF32 matmul/conv. Off by default so training is true fp32 like the original.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--print_every', type=int, default=100)
    # validation
    p.add_argument('--val_every', type=int, default=5, help='Validate on the fixed subset every N epochs')
    p.add_argument('--full_val_every', type=int, default=25, help='Validate on the full val set every N epochs '
                   '(and always at the last epoch)')
    p.add_argument('--val_subset', type=int, default=300, help='Size of the fixed, seeded val subset used for best.pt')
    p.add_argument('--val_batch_size', type=int, default=4)
    p.add_argument('--debug_max_val_samples', type=int, default=None,
                   help='Debug only: truncate the val set to its first N slices')
    # logging / checkpoints
    p.add_argument('--log_dir', type=str, required=True, help='Directory for last.pt / best.pt (e.g. on Drive)')
    p.add_argument('--run_name', type=str, default=None, help='W&B run name')
    p.add_argument('--project', type=str, default='ct-mar', help='W&B project name')
    p.add_argument('--entity', type=str, default=None, help='W&B entity (team/user) if needed')
    p.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    p.add_argument('--resume', type=str, default=None, help='Path to last.pt to continue exactly from')
    p.add_argument('--local_tmp_dir', type=str, default=None,
                   help='Local dir for the temp file of atomic checkpoint writes (default: system temp)')
    args = p.parse_args(argv)

    if args.hu_min != HU_MIN or args.hu_max != HU_MAX:
        raise ValueError(f"--hu_min/--hu_max must be {HU_MIN}/{HU_MAX} (fixed for all baselines, CLAUDE.md).")
    return args


def model_kwargs_from_args(args):
    return {k: getattr(args, k) for k in MODEL_ARG_NAMES}


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def region_of(pair):
    return "head" if "_head_" in os.path.basename(pair[0]).lower() else "body"


def val_subset_indices(n, k, seed):
    """Fixed, seeded subset of the val set (sorted indices)."""
    if k <= 0 or k >= n:
        return list(range(n))
    return sorted(np.random.default_rng(seed).choice(n, size=k, replace=False).tolist())


def build_train_loader(args, train_ds, start_epoch, end_epoch):
    """One DataLoader over the deterministic draw stream for epochs [start_epoch, end_epoch]."""
    patch_ds = ACDNetPatchDataset(train_ds, patch_size=args.patch_size,
                                  patches_per_slice=args.patches_per_slice, seed=args.seed)
    draws_per_epoch = args.iters_per_epoch * args.batch_slices
    n_epochs = max(0, end_epoch - start_epoch + 1)
    sampler = DrawStreamSampler(len(train_ds), args.seed, start_draw=(start_epoch - 1) * draws_per_epoch,
                                num_draws=n_epochs * draws_per_epoch)
    kw = dict(batch_size=args.batch_slices, sampler=sampler, num_workers=args.num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=True, worker_init_fn=worker_init_fn,
              generator=torch.Generator().manual_seed(args.seed))
    if args.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(patch_ds, **kw)


def batch_to_device(batch, device):
    """(B_slices, P, 1, p, p) -> (B_slices*P, 1, p, p). Checks the mask on CPU (no GPU sync)."""
    out = {k: v.flatten(0, 1) for k, v in batch.items()}
    nonmetal_from_metal(out["mask"])  # raises if not binary
    return {k: v.to(device, non_blocking=True) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------
# Train step
# ----------------------------------------------------------------------------------------------
def train_step(model, optimizer, b, args):
    """One ACDNet update on a device batch dict (normalized [0,1], mask 1 = metal).

    Returns detached (loss, loss_l2X, loss_l1X, loss_l1A) tensors (no host sync).
    """
    ma, li, gt, metal = b["ma"], b["li"], b["gt"], b["mask"]
    if args.metal_mask_on_gt:
        gt = torch.where(metal > 0.5, ma, gt)
    mask = 1.0 - metal                      # non-metal mask (binary checked in batch_to_device)
    Xma, XLI, GT = ma * ACDNET_SCALE, li * ACDNET_SCALE, gt * ACDNET_SCALE
    with torch.autocast(device_type=Xma.device.type, dtype=torch.bfloat16, enabled=args.amp):
        X0, ListX, ListA, ListX_nonK, ListA_nonK = model.net(Xma, XLI, mask)
    if args.amp:
        X0 = X0.float()
        ListX, ListA = [t.float() for t in ListX], [t.float() for t in ListA]
        ListX_nonK, ListA_nonK = [t.float() for t in ListX_nonK], [t.float() for t in ListA_nonK]
    loss, l2X, l1X, l1A = acdnet_loss(Xma, GT, mask, X0, ListX, ListA, ListX_nonK, ListA_nonK,
                                      T=args.T, Xl2=args.Xl2, Xl1=args.Xl1, Al1=args.Al1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach(), l2X.detach(), l1X.detach(), l1A.detach()


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def run_validation(model, ds, args, device):
    """Full-slice fp32 validation; returns per-image {'psnr','ssim','rmse'} lists in ds order."""
    was_training = model.training
    model.eval()
    loader = DataLoader(ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'))
    out = {"psnr": [], "ssim": [], "rmse": []}
    for batch in loader:
        f = unpack_batch(batch, has_mask=True, has_li=True)
        x = f["ma"].to(device, non_blocking=True)
        y = f["gt"].to(device, non_blocking=True)
        m = f["mask"].to(device, non_blocking=True)
        li = f["li"].to(device, non_blocking=True)
        if args.metal_mask_on_gt:
            y = torch.where(m > 0.5, x, y)
        pred = model(x, li, m).float()  # ListX[-1] / 255, not clamped
        # evaluate_predictions excludes the (dilated) metal region from SSIM, RMSE and PSNR, so the
        # metal-region output of ACDNet (unconstrained by its non-metal-masked loss) does not matter.
        per = evaluate_predictions(pred, y, m, HU_MIN, HU_MAX)
        for k in out:
            out[k].extend(per[k])
    model.train(was_training)
    return out


def summarize_split(per_img, regions, idx=None):
    """Overall + body/head summaries (same keys as the trainer) for an optional index subset."""
    if idx is not None:
        per_img = {k: [v[i] for i in idx] for k, v in per_img.items()}
        regions = [regions[i] for i in idx]
    if len(per_img["ssim"]) != len(regions):
        raise RuntimeError(f"val metrics ({len(per_img['ssim'])}) / region labels ({len(regions)}) mismatch")
    overall = summarize_metrics(per_img)
    overall["n"] = len(regions)
    by_region = {}
    for region in ("body", "head"):
        ridx = [i for i, r in enumerate(regions) if r == region]
        if ridx:
            by_region[region] = summarize_metrics({k: [v[i] for i in ridx] for k, v in per_img.items()})
            by_region[region]["n"] = len(ridx)
    return overall, by_region


def metrics_payload(overall, by_region, prefix=""):
    payload = {f"{prefix}{k}": overall[k] for k in ("psnr", "psnr_std", "ssim", "ssim_std", "rmse", "rmse_std")}
    payload[f"{prefix}val_n"] = overall["n"]
    for region, rs in by_region.items():
        for k in ("psnr", "psnr_std", "ssim", "ssim_std", "rmse", "rmse_std", "n"):
            payload[f"{prefix}val_{region}/{k}"] = rs[k]
    return payload


def print_metrics(tag, overall, by_region):
    print(f"{tag} PSNR: {overall['psnr']:.2f} \u00b1 {overall['psnr_std']:.2f} "
          f"SSIM: {overall['ssim']:.4f} \u00b1 {overall['ssim_std']:.4f} "
          f"RMSE: {overall['rmse']:.2f} \u00b1 {overall['rmse_std']:.2f} HU (n={overall['n']})")
    for region, rs in by_region.items():
        print(f"    [{region} n={rs['n']}] "
              f"PSNR: {rs['psnr']:.2f} \u00b1 {rs['psnr_std']:.2f} "
              f"SSIM: {rs['ssim']:.4f} \u00b1 {rs['ssim_std']:.4f} "
              f"RMSE: {rs['rmse']:.2f} \u00b1 {rs['rmse_std']:.2f} HU")


# ----------------------------------------------------------------------------------------------
# W&B
# ----------------------------------------------------------------------------------------------
def init_wandb_run(args, run_id):
    """Same wandb.init arguments as trainer_wandb_unclipped.py; resumes the run id stored in last.pt."""
    if args.wandb_mode == 'disabled':
        print("[wandb] disabled")
        return None
    import wandb
    kw = dict(project=args.project, entity=args.entity, name=args.run_name, config=vars(args),
              dir=args.log_dir, save_code=True, reinit=True, mode=args.wandb_mode)
    if run_id:
        kw.update(id=run_id, resume="allow")
    run = wandb.init(**kw)
    print(f"[wandb] run name={run.name} id={run.id}{' (resumed)' if run_id else ''}")
    return run


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.log_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # as in the original train.py
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    # ---------------- Data ----------------
    train_ds, val_ds = build_datasets(args.ma_dir, args.gt_dir, args.li_dir, args.mask_dir)
    if args.debug_max_val_samples is not None:
        val_ds = Subset(val_ds, list(range(min(args.debug_max_val_samples, len(val_ds)))))
        val_pairs = [val_ds.dataset.pairs[i] for i in val_ds.indices]
    else:
        val_pairs = val_ds.pairs
    val_regions = [region_of(p) for p in val_pairs]
    sub_idx = val_subset_indices(len(val_ds), args.val_subset, args.seed)
    val_sub_ds = Subset(val_ds, sub_idx)
    print(f"[data] train slices={len(train_ds)} | val={len(val_ds)} | val subset={len(sub_idx)} "
          f"| {args.batch_slices} slices x {args.patches_per_slice} patches of {args.patch_size}^2 "
          f"| {args.iters_per_epoch} it/epoch")

    # ---------------- Model / optim ----------------
    model_kwargs = model_kwargs_from_args(args)
    model = ACDNetInference(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    print(f"[model] ACDNet {model_kwargs} | params={sum(p.numel() for p in model.parameters()):,}")

    start_epoch, best_ssim, best_epoch, run_id = 1, -float('inf'), None, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        saved = ckpt.get("args", {})
        bad = [k for k in STREAM_ARG_NAMES + MODEL_ARG_NAMES if k in saved and saved[k] != getattr(args, k)]
        if bad:
            raise ValueError(f"--resume: these args differ from the checkpoint and would break exact "
                             f"continuation: {[(k, saved[k], getattr(args, k)) for k in bad]}")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_ssim = float(ckpt["best_ssim"])
        best_epoch = ckpt.get("best_epoch")
        run_id = ckpt.get("wandb_run_id")
        print(f"[resume] {args.resume}: continuing at epoch {start_epoch} | best SSIM {best_ssim:.4f} "
              f"(epoch {best_epoch}) | lr {optimizer.param_groups[0]['lr']:.2e}")

    run = init_wandb_run(args, run_id)
    run_id = run.id if run is not None else None

    best_path = os.path.join(args.log_dir, "best.pt")
    last_path = os.path.join(args.log_dir, "last.pt")

    def ckpt_payload(epoch):
        return {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "best_ssim": float(best_ssim), "best_epoch": best_epoch,
                "model_name": MODEL_NAME, "model_kwargs": model_kwargs, "wandb_run_id": run_id, "args": vars(args)}

    # ---------------- Train ----------------
    loader = build_train_loader(args, train_ds, start_epoch, args.epochs)
    train_iter = iter(loader) if start_epoch <= args.epochs else None
    last_full = None  # (epoch, per_img) of the most recent full validation
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        sums = torch.zeros(4, device=device)
        t_epoch, t_data = time.time(), 0.0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for it in range(args.iters_per_epoch):
            t0 = time.time()
            b = batch_to_device(next(train_iter), device)
            t_data += time.time() - t0
            parts = train_step(model, optimizer, b, args)
            sums += torch.stack(parts)
            if (it + 1) % args.print_every == 0 or it == 0:
                cur = [float(v) for v in parts]
                if not all(math.isfinite(v) for v in cur):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch} it {it}: {cur}")
                print(f"[Epoch:{epoch:>3d}/{args.epochs}] train:{it + 1:0>5d}/{args.iters_per_epoch:0>5d}, "
                      f"Loss={cur[0]:5.2e}, Lossl2X={cur[1]:5.2e}, Lossl1X={cur[2]:5.2e}, Lossl1A={cur[3]:5.2e}, "
                      f"lr={lr:.2e}")
        avg = (sums / args.iters_per_epoch).tolist()
        if not all(math.isfinite(v) for v in avg):
            raise RuntimeError(f"Non-finite mean loss in epoch {epoch}: {avg}")
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_time = time.time() - t_epoch
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == 'cuda' else 0.0
        scheduler.step()  # per epoch, as in the original

        # ---------------- Validation ----------------
        do_full = (args.full_val_every > 0 and epoch % args.full_val_every == 0) or epoch == args.epochs
        do_sub = do_full or (args.val_every > 0 and epoch % args.val_every == 0)
        t_val = time.time()
        sub_summary = full_summary = None
        if do_full:
            per_full = run_validation(model, val_ds, args, device)
            last_full = (epoch, per_full)
            full_summary = summarize_split(per_full, val_regions)
            sub_summary = summarize_split(per_full, val_regions, sub_idx)  # subset metrics from the same pass
        elif do_sub:
            per_sub = run_validation(model, val_sub_ds, args, device)
            sub_summary = summarize_split(per_sub, [val_regions[i] for i in sub_idx])
        val_time = time.time() - t_val

        improved = False
        if sub_summary is not None and sub_summary[0]["ssim"] > best_ssim:
            best_ssim, best_epoch, improved = float(sub_summary[0]["ssim"]), epoch, True
            atomic_torch_save({**ckpt_payload(epoch), "monitor_metric": "ssim", "monitor_value": best_ssim},
                              best_path, local_tmp_dir=args.local_tmp_dir)
        # Rolling last.pt, saved after the best bookkeeping so --resume restores best_ssim.
        atomic_torch_save(ckpt_payload(epoch), last_path, local_tmp_dir=args.local_tmp_dir)

        # ---------------- Log ----------------
        payload = {"loss": avg[0], "loss_l2X": avg[1], "loss_l1X": avg[2], "loss_l1A": avg[3], "lr": lr,
                   "epoch": epoch, "time/train_s": train_time, "time/s_per_it": train_time / args.iters_per_epoch,
                   "time/data_wait_frac": t_data / max(train_time, 1e-9), "time/val_s": val_time,
                   "gpu/peak_mem_gb": peak_mem_gb}
        if sub_summary is not None:
            payload.update(metrics_payload(*sub_summary))
            payload["best_ssim"] = best_ssim
        if full_summary is not None:
            payload.update(metrics_payload(*full_summary, prefix="full_val/"))
        if run is not None:
            run.log(payload, step=epoch)
        print(f"[Epoch {epoch}] Loss: {avg[0]:.4e} (l2X {avg[1]:.3e} l1X {avg[2]:.3e} l1A {avg[3]:.3e}) "
              f"lr {lr:.2e} | {train_time / args.iters_per_epoch:.3f} s/it, data wait "
              f"{100 * t_data / max(train_time, 1e-9):.1f}%, peak {peak_mem_gb:.2f} GB | val {val_time:.1f}s")
        if sub_summary is not None:
            print_metrics(f"[Epoch {epoch}] val subset{' (best)' if improved else ''}", *sub_summary)
        if full_summary is not None:
            print_metrics(f"[Epoch {epoch}] FULL val", *full_summary)

    del train_iter, loader

    # ---------------- Final: full-set metrics for best and last ----------------
    final_epoch = max(args.epochs, start_epoch - 1)  # epoch of the weights currently in `model`
    if last_full is None or last_full[0] != final_epoch:
        last_full = (final_epoch, run_validation(model, val_ds, args, device))
    results = {"last": {"epoch": last_full[0], "per_image": last_full[1]}}
    if best_epoch is not None and os.path.exists(best_path):
        if best_epoch == last_full[0]:
            results["best"] = {"epoch": best_epoch, "per_image": last_full[1]}
        else:
            best_model = ACDNetInference(**model_kwargs).to(device)
            best_model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True)["model_state"])
            results["best"] = {"epoch": best_epoch, "per_image": run_validation(best_model, val_ds, args, device)}
            del best_model
    final = {}
    for tag in ("best", "last"):
        if tag not in results:
            continue
        overall, by_region = summarize_split(results[tag]["per_image"], val_regions)
        final[tag] = {"epoch": results[tag]["epoch"], "overall": overall, "by_region": by_region}
        print_metrics(f"[FINAL {tag} (epoch {results[tag]['epoch']})] full val", overall, by_region)
        if run is not None:
            for k, v in metrics_payload(overall, by_region, prefix=f"final_{tag}/").items():
                run.summary[k] = v
            run.summary[f"final_{tag}/epoch"] = results[tag]["epoch"]
    with open(os.path.join(args.log_dir, "final_metrics.json"), "w") as f:
        json.dump(final, f, indent=2)
    if run is not None:
        run.finish()
    return final


if __name__ == '__main__':
    main()
