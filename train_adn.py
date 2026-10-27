"""Train ADN (Liao et al., IEEE TMI 2020; https://github.com/liaohaofu/adn) as an UNPAIRED baseline.

Official recipe (config/adn.yaml + runs/adn.yaml DeepLesion overrides, see models/adn/losses.py):
  G: ADN (base_ch 64, num_down 2, num_residual 4, num_sides 3, instance/instance/layer norms, fuse),
  D: two PatchGAN NLayerDiscriminators (ndf 64, n_layers 2, instance norm), LSGAN,
  loss weights ll 20, hh 20, hlh 20, art 20 (gt_art), gl 1, gh 1 (lh = lhl = 0),
  one Adam per network (lr 1e-4, betas (0.5, 0.999), weight_decay 1e-4), StepLR(1e5, 0.5) stepped per
  epoch (= no decay), batch size 2, 50 epochs, random vertical flip.
Data: MA images only from half A and clean (GT) images only from half B of our train split
(--unpaired_mode disjoint) or from two independent shuffles (shuffle); no LI, no metal mask in the model
or losses. Images are mapped from our normalized [0,1] space to ADN's [-1,1] (x*2-1) and back.

Update order: the original stepped each discriminator BEFORE the generator backward and relied on
PyTorch<=1.0 `.data` in-place updates, so the generator's adversarial gradient flowed through the
already-updated D weights (with activations from the old D). Current PyTorch rejects that graph (the
official ADNTrain.optimize raises "modified by an inplace operation"). Here G is updated first with
both Ds frozen (same loss values as the original: D before its update), then each D is updated on
the same detached fakes and the same real images, exactly as in the original.

Validation, W&B and checkpoint conventions follow trainer_wandb_unclipped.py / train_acdnet.py:
ADNInference (encoder_low + decoder) on full 512x512 val slices through `evaluate_predictions`.

Example:
  python train_adn.py --ma_dir .../MA_image --gt_dir .../GT --mask_dir .../metal_mask \
      --hu_min -1024 --hu_max 3072 --metal_mask_on_gt --run_name alpha_adn --log_dir <drive>/alpha_adn
  # continue after a Colab disconnect:
  python train_adn.py <same args> --resume <drive>/alpha_adn/last.pt
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
from baselines.common.eval_utils import evaluate_predictions
from models.adn import (ADN, ADN_BATCH_SIZE, ADN_D_KWARGS, ADN_G_KWARGS, ADN_LEARN, ADN_LOSS_WEIGHTS,
                        ADN_NUM_EPOCHS, ADNInference, NLayerDiscriminator, discriminator_loss, generator_losses,
                        to_adn)
from models.adn.data import UNPAIRED_MODES, EpochDrawSampler, UnpairedADNDataset, worker_init_fn
from train_acdnet import (init_wandb_run, metrics_payload, print_metrics, region_of, summarize_split,
                          val_subset_indices)

MODEL_NAME = "adn"
# Arguments that define the model / training stream; they must not change when resuming.
STREAM_ARG_NAMES = ("seed", "batch_size", "unpaired_mode", "crop_size", "metal_crop_prob", "no_random_flip",
                    "debug_max_train_samples", "output_act")
G_LOSS_NAMES = ("gl_g", "gh_g", "ll", "hh", "hlh", "art")
D_STAT_NAMES = ("loss_dl", "loss_dh", "dl_real_mean", "dl_fake_mean", "dh_real_mean", "dh_fake_mean")
VIS_HU = (-1000.0, 1000.0)  # display window for the W&B val image triplets only


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ADN (unpaired) baseline")
    # data
    p.add_argument('--ma_dir', type=str, required=True)
    p.add_argument('--gt_dir', type=str, required=True)
    p.add_argument('--mask_dir', type=str, required=True,
                   help='Metal masks: val metrics and (crops only) the MA crop location; never used in the losses')
    p.add_argument('--hu_min', type=float, default=HU_MIN,
                   help='Must equal the fixed baseline window (baselines/common/data.py).')
    p.add_argument('--hu_max', type=float, default=HU_MAX,
                   help='Must equal the fixed baseline window (baselines/common/data.py).')
    p.add_argument('--metal_mask_on_gt', action='store_true',
                   help='Validation only (as in trainer_wandb_unclipped.py): paint MA metal onto GT. The unpaired '
                        'clean training images are never modified.')
    p.add_argument('--unpaired_mode', type=str, default='disjoint', choices=UNPAIRED_MODES)
    p.add_argument('--crop_size', type=int, default=None, help='Optional independent random crops (default: full slices)')
    p.add_argument('--metal_crop_prob', type=float, default=0.7,
                   help='With --crop_size: probability that the MA crop is centred near a random metal pixel')
    p.add_argument('--no_random_flip', action='store_true', help='Disable the official random vertical flip')
    # model
    p.add_argument('--output_act', type=str, default='tanh', choices=['tanh', 'linear'],
                   help='Last decoder activation. tanh (official) is exact for clipped data (targets in [-1,1]).')
    # optimization (official defaults)
    p.add_argument('--epochs', type=int, default=ADN_NUM_EPOCHS)
    p.add_argument('--batch_size', type=int, default=ADN_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=ADN_LEARN['lr'])
    p.add_argument('--beta1', type=float, default=ADN_LEARN['beta1'])
    p.add_argument('--beta2', type=float, default=ADN_LEARN['beta2'])
    p.add_argument('--weight_decay', type=float, default=ADN_LEARN['weight_decay'])
    p.add_argument('--step_size', type=int, default=int(ADN_LEARN['step_size']), help='StepLR step (epochs)')
    p.add_argument('--gamma', type=float, default=ADN_LEARN['gamma'])
    p.add_argument('--amp', action='store_true',
                   help='bf16 autocast for G/D forwards (losses in fp32). Off by default; only use it if '
                        'colab/adn_checks.py shows matching loss curves.')
    p.add_argument('--tf32', action='store_true', help='Allow TF32 matmul/conv (off: true fp32)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--print_every', type=int, default=500)
    p.add_argument('--debug_max_train_samples', type=int, default=None,
                   help='Debug only: use the first N train slices (split into the two halves as usual)')
    # validation
    p.add_argument('--val_every', type=int, default=1, help='Validate every N epochs (and at the last epoch)')
    p.add_argument('--val_subset', type=int, default=0,
                   help='Size of a fixed seeded val subset used for per-epoch validation / best.pt (0 = full val)')
    p.add_argument('--val_batch_size', type=int, default=8)
    p.add_argument('--image_every', type=int, default=5, help='Log val image triplets every N epochs (0 = never)')
    p.add_argument('--num_val_images', type=int, default=4)
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
    if args.crop_size is not None and args.crop_size % 4:
        raise ValueError("--crop_size must be divisible by 4 (two stride-2 downsamplings)")
    return args


def model_kwargs_from_args(args):
    return {**ADN_G_KWARGS, "output_act": args.output_act}


# ----------------------------------------------------------------------------------------------
# Model / optimizers
# ----------------------------------------------------------------------------------------------
def build_networks(args, device):
    """(G, D_l, D_h), seeded. D_l = official model_dl (judges pred_lh vs clean images), D_h = model_dh."""
    torch.manual_seed(args.seed)
    g = ADN(**model_kwargs_from_args(args)).to(device)
    d_l = NLayerDiscriminator(**ADN_D_KWARGS).to(device)
    d_h = NLayerDiscriminator(**ADN_D_KWARGS).to(device)
    return g, d_l, d_h


def build_optimizers(args, nets):
    opts = [torch.optim.Adam(n.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                             weight_decay=args.weight_decay) for n in nets]
    scheds = [torch.optim.lr_scheduler.StepLR(o, step_size=args.step_size, gamma=args.gamma) for o in opts]
    return opts, scheds


def _set_requires_grad(nets, flag):
    for n in nets:
        for p in n.parameters():
            p.requires_grad_(flag)


def train_step(nets, opts, low, high, args, weights=ADN_LOSS_WEIGHTS):
    """One ADN update. low (MA) / high (clean) are unpaired (B,1,h,w) batches in ADN's [-1,1].

    Returns (g_terms, g_total, d_stats) as detached tensors (no host sync).
    """
    g, d_l, d_h = nets
    opt_g, opt_dl, opt_dh = opts
    dev = low.device.type
    # Generator: both Ds frozen (no D grads), adversarial terms through the current (pre-update) Ds.
    _set_requires_grad((d_l, d_h), False)
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=args.amp):
        terms, preds = generator_losses(g, d_l, d_h, low, high, weights)
    terms = {k: v.float() for k, v in terms.items()}
    g_total = sum(terms.values())
    opt_g.zero_grad(set_to_none=True)
    g_total.backward()
    opt_g.step()
    # Discriminators: same detached fakes (from before the G step) and the same real images.
    _set_requires_grad((d_l, d_h), True)
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=args.amp):
        loss_dl, rl, fl = discriminator_loss(d_l, preds["pred_lh"], high)
        loss_dh, rh, fh = discriminator_loss(d_h, preds["pred_hl"], low)
    loss_dl, loss_dh = loss_dl.float(), loss_dh.float()
    opt_dl.zero_grad(set_to_none=True)
    opt_dh.zero_grad(set_to_none=True)
    (loss_dl + loss_dh).backward()  # disjoint parameter sets -> identical to two separate backwards
    opt_dl.step()
    opt_dh.step()
    with torch.no_grad():
        sig = lambda t: torch.sigmoid(t.float()).mean()  # noqa: E731
        d_stats = torch.stack([loss_dl.detach(), loss_dh.detach(), sig(rl), sig(fl), sig(rh), sig(fh)])
    return {k: v.detach() for k, v in terms.items()}, g_total.detach(), d_stats


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def build_train_loader(args, unpaired_ds, start_epoch, end_epoch):
    """One DataLoader over the deterministic draw stream for epochs [start_epoch, end_epoch]."""
    steps = steps_per_epoch(args, unpaired_ds)
    sampler = EpochDrawSampler(start_epoch, end_epoch, steps * args.batch_size)
    kw = dict(batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=True, worker_init_fn=worker_init_fn,
              generator=torch.Generator().manual_seed(args.seed))
    if args.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(unpaired_ds, **kw)


def steps_per_epoch(args, unpaired_ds):
    """One epoch = one pass over the MA pool (half A in disjoint mode), drop_last."""
    n = unpaired_ds.n_ma // args.batch_size
    if n < 1:
        raise ValueError(f"MA pool ({unpaired_ds.n_ma}) smaller than batch size {args.batch_size}")
    return n


def build_unpaired(args, train_ds):
    return UnpairedADNDataset(train_ds, mode=args.unpaired_mode, crop_size=args.crop_size,
                              metal_crop_prob=args.metal_crop_prob, random_flip=not args.no_random_flip,
                              seed=args.seed)


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def run_validation(model, ds, args, device, keep_idx=()):
    """Full-slice fp32 validation of ADNInference.

    Returns (per-image {'psnr','ssim','rmse'} lists in ds order, {i: (ma, pred, gt) numpy} for keep_idx).
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'))
    out = {"psnr": [], "ssim": [], "rmse": []}
    keep, keep_idx, i0 = {}, set(keep_idx), 0
    for batch in loader:
        f = unpack_batch(batch, has_mask=True, has_li=False)
        x = f["ma"].to(device, non_blocking=True)
        y = f["gt"].to(device, non_blocking=True)
        m = f["mask"].to(device, non_blocking=True)
        if args.metal_mask_on_gt:
            y = torch.where(m > 0.5, x, y)
        pred = model(x).float()
        # evaluate_predictions excludes the (dilated) metal region from SSIM (filled with 100 HU in both
        # images), RMSE and PSNR, so ADN's metal-region output does not enter any metric.
        per = evaluate_predictions(pred, y, m, HU_MIN, HU_MAX)
        for k in out:
            out[k].extend(per[k])
        for j in range(x.size(0)):
            if i0 + j in keep_idx:
                keep[i0 + j] = tuple(t[j, 0].cpu().numpy() for t in (x, pred, y))
        i0 += x.size(0)
    model.train(was_training)
    return out, keep


def triplet_image(keep):
    """Rows of [MA | ADN | GT] in the VIS_HU display window, uint8."""
    lo = (VIS_HU[0] - HU_MIN) / (HU_MAX - HU_MIN)
    hi = (VIS_HU[1] - HU_MIN) / (HU_MAX - HU_MIN)
    rows = [np.concatenate([np.clip((a - lo) / (hi - lo), 0, 1) for a in keep[i]], axis=1) for i in sorted(keep)]
    return (np.concatenate(rows, axis=0) * 255).astype(np.uint8)


def vis_indices(regions, k, seed):
    """Fixed seeded val indices for the image triplets, split between body and head when possible."""
    rng = np.random.default_rng([seed, 7])
    body = [i for i, r in enumerate(regions) if r == "body"]
    head = [i for i, r in enumerate(regions) if r == "head"]
    n_head = min(len(head), k // 2)
    pick = list(rng.choice(head, n_head, replace=False)) if n_head else []
    pick += list(rng.choice(body, min(len(body), k - n_head), replace=False)) if body else []
    return sorted(int(i) for i in pick)


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
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    # ---------------- Data ----------------
    # Same directories as the MA-only reference runs (no LI) -> same split.
    train_ds, val_ds = build_datasets(args.ma_dir, args.gt_dir, None, args.mask_dir)
    if args.debug_max_train_samples is not None:
        train_ds.pairs = train_ds.pairs[:args.debug_max_train_samples]
    if args.debug_max_val_samples is not None:
        val_ds = Subset(val_ds, list(range(min(args.debug_max_val_samples, len(val_ds)))))
        val_pairs = [val_ds.dataset.pairs[i] for i in val_ds.indices]
    else:
        val_pairs = val_ds.pairs
    unpaired = build_unpaired(args, train_ds)
    n_steps = steps_per_epoch(args, unpaired)
    val_regions = [region_of(p) for p in val_pairs]
    sub_idx = val_subset_indices(len(val_ds), args.val_subset, args.seed)
    full_every_epoch = len(sub_idx) == len(val_ds)
    vis_full = vis_indices(val_regions, args.num_val_images, args.seed)
    print(f"[data] train slices={len(train_ds)} | MA pool={unpaired.n_ma} clean pool={len(unpaired.clean_pool)} "
          f"({args.unpaired_mode}) | val={len(val_ds)} | val subset={len(sub_idx)} | batch {args.batch_size} "
          f"| {n_steps} steps/epoch | crop={args.crop_size or 'full'}"
          f"{f' (metal_crop_prob {args.metal_crop_prob})' if args.crop_size else ''} "
          f"| random_flip={not args.no_random_flip}")

    # ---------------- Model / optim ----------------
    model_kwargs = model_kwargs_from_args(args)
    nets = build_networks(args, device)
    g, d_l, d_h = nets
    infer = ADNInference.from_adn(g, model_kwargs)
    opts, scheds = build_optimizers(args, nets)
    n_par = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    print(f"[model] ADN {model_kwargs} | G params={n_par(g):,} (inference {n_par(infer):,}) | "
          f"D {ADN_D_KWARGS} params={n_par(d_l):,} x2 | loss weights {ADN_LOSS_WEIGHTS}")

    start_epoch, best_ssim, best_epoch, run_id = 1, -float('inf'), None, None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        saved = ckpt.get("args", {})
        bad = [k for k in STREAM_ARG_NAMES if k in saved and saved[k] != getattr(args, k)]
        if bad:
            raise ValueError(f"--resume: these args differ from the checkpoint and would break exact "
                             f"continuation: {[(k, saved[k], getattr(args, k)) for k in bad]}")
        for net, key in zip(nets, ("model_g", "model_dl", "model_dh")):
            net.load_state_dict(ckpt["adn_full"][key])
        for o, s, key in zip(opts, scheds, ("model_g", "model_dl", "model_dh")):
            o.load_state_dict(ckpt["optimizer_state"][key])
            s.load_state_dict(ckpt["scheduler_state"][key])
        start_epoch = int(ckpt["epoch"]) + 1
        best_ssim = float(ckpt["best_ssim"])
        best_epoch = ckpt.get("best_epoch")
        run_id = ckpt.get("wandb_run_id")
        print(f"[resume] {args.resume}: continuing at epoch {start_epoch} | best SSIM {best_ssim:.4f} "
              f"(epoch {best_epoch}) | lr {opts[0].param_groups[0]['lr']:.2e}")

    run = init_wandb_run(args, run_id)
    run_id = run.id if run is not None else None
    if run is not None:
        run.config.update({"adn_g_kwargs": model_kwargs, "adn_d_kwargs": ADN_D_KWARGS,
                           "adn_loss_weights": ADN_LOSS_WEIGHTS, "steps_per_epoch": n_steps}, allow_val_change=True)

    best_path = os.path.join(args.log_dir, "best.pt")
    last_path = os.path.join(args.log_dir, "last.pt")
    keys = ("model_g", "model_dl", "model_dh")

    def ckpt_payload(epoch):
        return {"epoch": epoch, "model_state": infer.state_dict(),
                "adn_full": {k: n.state_dict() for k, n in zip(keys, nets)},
                "optimizer_state": {k: o.state_dict() for k, o in zip(keys, opts)},
                "scheduler_state": {k: s.state_dict() for k, s in zip(keys, scheds)},
                "best_ssim": float(best_ssim), "best_epoch": best_epoch, "model_name": MODEL_NAME,
                "model_kwargs": model_kwargs, "d_kwargs": ADN_D_KWARGS, "loss_weights": ADN_LOSS_WEIGHTS,
                "wandb_run_id": run_id, "args": vars(args)}

    # ---------------- Train ----------------
    loader = build_train_loader(args, unpaired, start_epoch, args.epochs)
    train_iter = iter(loader) if start_epoch <= args.epochs else None
    last_full = None  # (epoch, per_img) of the most recent full validation
    for n in nets:
        n.train()
    for epoch in range(start_epoch, args.epochs + 1):
        lrs = [o.param_groups[0]["lr"] for o in opts]
        g_sums = torch.zeros(len(G_LOSS_NAMES) + 1, device=device)
        d_sums = torch.zeros(len(D_STAT_NAMES), device=device)
        t_epoch, t_data = time.time(), 0.0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for it in range(n_steps):
            t0 = time.time()
            b = next(train_iter)
            low = to_adn(b["ma"].to(device, non_blocking=True))
            high = to_adn(b["clean"].to(device, non_blocking=True))
            t_data += time.time() - t0
            terms, g_total, d_stats = train_step(nets, opts, low, high, args)
            g_vec = torch.stack([g_total] + [terms[k] for k in G_LOSS_NAMES])
            g_sums += g_vec
            d_sums += d_stats
            if (it + 1) % args.print_every == 0 or it == 0:
                cur_g, cur_d = g_vec.tolist(), d_stats.tolist()
                if not all(math.isfinite(v) for v in cur_g + cur_d):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch} it {it}: G {cur_g} D {cur_d}")
                print(f"[Epoch:{epoch:>3d}/{args.epochs}] train:{it + 1:0>5d}/{n_steps:0>5d}, G={cur_g[0]:.4f} "
                      + " ".join(f"{k}={v:.4f}" for k, v in zip(G_LOSS_NAMES, cur_g[1:]))
                      + f" | D_l={cur_d[0]:.4f} D_h={cur_d[1]:.4f} sig(D_l r/f)={cur_d[2]:.3f}/{cur_d[3]:.3f} "
                        f"sig(D_h r/f)={cur_d[4]:.3f}/{cur_d[5]:.3f}")
        g_avg = (g_sums / n_steps).tolist()
        d_avg = (d_sums / n_steps).tolist()
        if not all(math.isfinite(v) for v in g_avg + d_avg):
            raise RuntimeError(f"Non-finite mean loss in epoch {epoch}: G {g_avg} D {d_avg}")
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_time = time.time() - t_epoch
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == 'cuda' else 0.0
        for s in scheds:
            s.step()  # per epoch, as in the original (update_lr)

        # ---------------- Validation ----------------
        do_val = (args.val_every > 0 and epoch % args.val_every == 0) or epoch == args.epochs
        do_full = do_val and (full_every_epoch or epoch == args.epochs)
        do_img = run is not None and args.image_every > 0 and (epoch % args.image_every == 0 or epoch == 1)
        t_val = time.time()
        sub_summary = full_summary = keep = None
        if do_full:
            per_full, keep = run_validation(infer, val_ds, args, device, vis_full if do_img else ())
            last_full = (epoch, per_full)
            full_summary = summarize_split(per_full, val_regions)
            sub_summary = full_summary if full_every_epoch else summarize_split(per_full, val_regions, sub_idx)
        elif do_val:
            vis_sub = [sub_idx.index(i) for i in vis_full if i in sub_idx] if do_img else ()
            per_sub, keep = run_validation(infer, Subset(val_ds, sub_idx), args, device, vis_sub)
            sub_summary = summarize_split(per_sub, [val_regions[i] for i in sub_idx])
        elif do_img:
            _, keep = run_validation(infer, Subset(val_ds, vis_full), args, device, range(len(vis_full)))
        val_time = time.time() - t_val

        improved = False
        if sub_summary is not None and sub_summary[0]["ssim"] > best_ssim:
            best_ssim, best_epoch, improved = float(sub_summary[0]["ssim"]), epoch, True
            atomic_torch_save({**ckpt_payload(epoch), "monitor_metric": "ssim", "monitor_value": best_ssim},
                              best_path, local_tmp_dir=args.local_tmp_dir)
        # Rolling last.pt, saved after the best bookkeeping so --resume restores best_ssim.
        atomic_torch_save(ckpt_payload(epoch), last_path, local_tmp_dir=args.local_tmp_dir)

        # ---------------- Log ----------------
        payload = {"loss": g_avg[0], **{f"loss_{k}": v for k, v in zip(G_LOSS_NAMES, g_avg[1:])},
                   **{f"gan/{k}": v for k, v in zip(D_STAT_NAMES, d_avg)},
                   "gan/lr_g": lrs[0], "gan/lr_dl": lrs[1], "gan/lr_dh": lrs[2], "epoch": epoch,
                   "time/train_s": train_time, "time/s_per_it": train_time / n_steps,
                   "time/data_wait_frac": t_data / max(train_time, 1e-9), "time/val_s": val_time,
                   "gpu/peak_mem_gb": peak_mem_gb}
        if sub_summary is not None:
            payload.update(metrics_payload(*sub_summary))
            payload["best_ssim"] = best_ssim
        if full_summary is not None and not full_every_epoch:
            payload.update(metrics_payload(*full_summary, prefix="full_val/"))
        if run is not None:
            if keep:
                import wandb
                payload["val_images"] = wandb.Image(triplet_image(keep),
                                                    caption=f"MA | ADN | GT (HU {VIS_HU[0]:.0f}..{VIS_HU[1]:.0f})")
            run.log(payload, step=epoch)
        print(f"[Epoch {epoch}] Loss: {g_avg[0]:.4e} ("
              + " ".join(f"{k} {v:.3e}" for k, v in zip(G_LOSS_NAMES, g_avg[1:]))
              + f") D_l {d_avg[0]:.3e} D_h {d_avg[1]:.3e} sig(D_l r/f) {d_avg[2]:.3f}/{d_avg[3]:.3f} "
                f"sig(D_h r/f) {d_avg[4]:.3f}/{d_avg[5]:.3f} | lr {lrs[0]:.2e} | {train_time / n_steps:.3f} s/it, "
                f"data wait {100 * t_data / max(train_time, 1e-9):.1f}%, peak {peak_mem_gb:.2f} GB | val {val_time:.1f}s")
        if sub_summary is not None:
            tag = "val" if full_every_epoch else "val subset"
            print_metrics(f"[Epoch {epoch}] {tag}{' (best)' if improved else ''}", *sub_summary)
        if full_summary is not None and not full_every_epoch:
            print_metrics(f"[Epoch {epoch}] FULL val", *full_summary)

    del train_iter, loader

    # ---------------- Final: full-set metrics for best and last ----------------
    final_epoch = max(args.epochs, start_epoch - 1)  # epoch of the weights currently in `infer`
    if last_full is None or last_full[0] != final_epoch:
        last_full = (final_epoch, run_validation(infer, val_ds, args, device)[0])
    results = {"last": {"epoch": last_full[0], "per_image": last_full[1]}}
    if best_epoch is not None and os.path.exists(best_path):
        if best_epoch == last_full[0]:
            results["best"] = {"epoch": best_epoch, "per_image": last_full[1]}
        else:
            best_model = ADNInference(**model_kwargs).to(device)
            best_model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True)["model_state"])
            results["best"] = {"epoch": best_epoch, "per_image": run_validation(best_model, val_ds, args, device)[0]}
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
    print("\u2705 Training complete!")
    return final


if __name__ == '__main__':
    main()
