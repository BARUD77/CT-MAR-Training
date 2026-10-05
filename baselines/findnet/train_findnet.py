"""Train FIND-Net (Tasharofi et al., MICCAI 2025) on my AAPM CT-MAR train split, validate on my val split.

- Data:        baselines/common/data.py (fixed dataset args) via baselines/findnet/dataset_adapter.py
- Model:       original FINDNet from third_party/findnet via baselines/findnet/model_wrapper.py
- Loss:        original stage-wise loss (DICDNet train script == FIND-Net paper Eq. 6)
- Optimizer:   AdamW + linear warmup + cosine annealing (FIND-Net paper), stepped per iteration
- Validation:  baselines/common/eval_utils.py on the FINAL stage output converted back to [0,1];
               best.pt selected by validation masked SSIM
- Checkpoints: baselines/common/checkpoint.py; last.pt every epoch and every N iterations, best.pt on
               improvement; automatic resume from <ckpt_dir>/last.pt (incl. mid-epoch position)
- Logging:     baselines/common/logging_utils.py (W&B run name "findnet_<run_name>", resumes the same run)

Run from anywhere (paths are resolved relative to the repo):
  python baselines/findnet/train_findnet.py --config baselines/findnet/config.yaml [overrides]
Re-running the same command after a disconnect resumes automatically.
"""
import argparse
import copy
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common import checkpoint as ckpt  # noqa: E402
from baselines.common.data import HU_MIN, HU_MAX, DATASET_KWARGS  # noqa: E402
from baselines.common.eval_utils import evaluate_predictions, summarize_metrics  # noqa: E402
from baselines.findnet.dataset_adapter import build_findnet_datasets  # noqa: E402
from baselines.findnet.model_wrapper import build_findnet, final_output, findnet_commit  # noqa: E402

METHOD = "findnet"
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
# Config sections that must not change when resuming a run (a change is reported as a warning).
RESUME_CHECKED_SECTIONS = ("model", "adapter", "loss", "optim", "schedule")


# ------------------------------------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------------------------------------
def findnet_loss(Xma, Xgt, mask, X0, ListX, ListA, S, stage_weight=0.1, Xl2=1.0, Xl1=5e-4, Al1=5e-4):
    """Original training loss (DICDNet train_DICDNet.py, used unchanged by FIND-Net; paper Eq. 6).

    mask = non-metal mask. ListX has S+1 entries (ListX[-1] = final), ListA has S entries.
    The intermediate stages X_0..X_{S-1}, A_0..A_{S-1} and X0 are weighted by `stage_weight`; the
    final X and the last A get weight 1 (the last A is therefore counted with 1 + stage_weight, as in
    the original code). MSE terms are means, L1 terms are sums, exactly as in the original.
    """
    assert len(ListX) == S + 1 and len(ListA) == S, (len(ListX), len(ListA), S)
    w = stage_weight
    newAgt = mask * (Xma - Xgt)
    newXgt = mask * Xgt
    loss_l2Xs = 0
    loss_l1Xs = 0
    loss_l1As = 0
    for j in range(S):
        loss_l2Xs = loss_l2Xs + w * F.mse_loss(ListX[j] * mask, newXgt)
        loss_l1Xs = loss_l1Xs + w * torch.sum(torch.abs(ListX[j] * mask - newXgt))
        loss_l1As = loss_l1As + w * torch.sum(torch.abs(mask * ListA[j] - newAgt))
    loss_l1Xf = torch.sum(torch.abs(ListX[-1] * mask - newXgt))
    loss_l1Af = torch.sum(torch.abs(mask * ListA[-1] - newAgt))
    loss_l2Xf = F.mse_loss(ListX[-1] * mask, newXgt)
    loss_l1X0 = w * torch.sum(torch.abs(X0 * mask - newXgt))
    loss_l2X0 = w * F.mse_loss(X0 * mask, newXgt)
    loss_l2X = loss_l2Xs + loss_l2Xf + loss_l2X0
    loss_l1X = loss_l1Xs + loss_l1Xf + loss_l1X0
    loss_l1A = loss_l1As + loss_l1Af
    loss = Xl2 * loss_l2X + Xl1 * loss_l1X + Al1 * loss_l1A
    parts = {"l2X": loss_l2X.detach(), "l1X": loss_l1X.detach(), "l1A": loss_l1A.detach(),
             "l2X_final": loss_l2Xf.detach()}
    return loss, parts


# ------------------------------------------------------------------------------------------------
# Config / CLI
# ------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train FIND-Net baseline (auto-resumes from <ckpt_dir>/last.pt).")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--ma_dir"); p.add_argument("--gt_dir"); p.add_argument("--li_dir"); p.add_argument("--mask_dir")
    p.add_argument("--ckpt_dir", help="Checkpoint directory (e.g. a Google Drive folder).")
    p.add_argument("--local_tmp_dir", help="Local staging dir for atomic checkpoint writes.")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--save_every_iters", type=int)
    p.add_argument("--log_every_iters", type=int)
    p.add_argument("--val_every_epochs", type=int, help="Validate every k epochs (and always after the last epoch).")
    p.add_argument("--seed", type=int)
    p.add_argument("--max_iters", type=int, default=0,
                   help="Stop once the TOTAL optimizer step count (across resumes) reaches this. 0 = no limit.")
    p.add_argument("--subset", type=int, default=0, help="Train on only the first N slices of the train split.")
    p.add_argument("--val_subset", type=int, default=0, help="Validate on only the first N val slices.")
    p.add_argument("--project"); p.add_argument("--entity"); p.add_argument("--run_name")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"])
    p.add_argument("--no_resume", action="store_true", help="Ignore an existing last.pt (it will be overwritten).")
    return p.parse_args(argv)


def load_config(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = {
        ("data", "ma_dir"): args.ma_dir, ("data", "gt_dir"): args.gt_dir,
        ("data", "li_dir"): args.li_dir, ("data", "mask_dir"): args.mask_dir,
        ("data", "num_workers"): args.num_workers,
        ("checkpoint", "ckpt_dir"): args.ckpt_dir, ("checkpoint", "local_tmp_dir"): args.local_tmp_dir,
        ("checkpoint", "save_every_iters"): args.save_every_iters,
        ("train", "epochs"): args.epochs, ("train", "batch_size"): args.batch_size, ("train", "seed"): args.seed,
        ("train", "log_every_iters"): args.log_every_iters,
        ("validation", "every_epochs"): args.val_every_epochs,
        ("wandb", "project"): args.project, ("wandb", "entity"): args.entity,
        ("wandb", "run_name"): args.run_name, ("wandb", "mode"): args.wandb_mode,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            cfg[section][key] = value
    for key in ("ma_dir", "gt_dir", "li_dir", "mask_dir"):
        if not cfg["data"].get(key):
            raise ValueError(f"data.{key} must be set in the config or via --{key}")
    if not cfg["checkpoint"].get("ckpt_dir"):
        raise ValueError("checkpoint.ckpt_dir must be set in the config or via --ckpt_dir")
    if cfg["train"].get("precision", "fp32") != "fp32":
        raise ValueError("Only precision: fp32 is implemented.")
    return cfg


def config_diffs(old, new, sections=RESUME_CHECKED_SECTIONS):
    diffs = []
    for s in sections:
        o, n = (old or {}).get(s, {}), new.get(s, {})
        for k in sorted(set(o) | set(n)):
            if o.get(k) != n.get(k):
                diffs.append(f"{s}.{k}: {o.get(k)!r} -> {n.get(k)!r}")
    return diffs


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
def make_lr_lambda(warmup_steps, total_steps, min_lr_ratio):
    """Linear warmup 0 -> 1, then cosine 1 -> min_lr_ratio (same form as trainer_wandb_unclipped.py)."""
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def epoch_order(n, seed, epoch):
    """Deterministic shuffle per epoch, so a mid-epoch resume continues the same order."""
    g = torch.Generator()
    g.manual_seed(int(seed) * 100003 + int(epoch))
    return torch.randperm(n, generator=g).tolist()


class Logger:
    """W&B wrapper; mode 'disabled' never imports wandb."""

    def __init__(self, cfg, run_id, extra_config):
        self.run = None
        mode = cfg["wandb"].get("mode", "online")
        if mode == "disabled":
            print("[wandb] disabled")
            return
        from baselines.common.logging_utils import init_wandb
        import wandb
        self.run = init_wandb(METHOD, cfg["wandb"]["project"], config={**cfg, **extra_config},
                              run_name=cfg["wandb"].get("run_name"), entity=cfg["wandb"].get("entity"),
                              run_id=run_id, log_dir=os.path.join(cfg["checkpoint"]["ckpt_dir"], "wandb"),
                              mode=mode)
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        print(f"[wandb] run name={self.run.name} id={self.run.id} mode={mode}"
              f"{' (resumed)' if run_id else ''}")

    @property
    def run_id(self):
        return self.run.id if self.run is not None else None

    def log(self, d):
        if self.run is not None:
            self.run.log(d)

    def finish(self):
        if self.run is not None:
            self.run.finish()


@torch.no_grad()
def validate(model, val_ds, conv, cfg, device):
    model.eval()
    loader = DataLoader(val_ds, batch_size=int(cfg["validation"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["data"]["num_workers"]), pin_memory=(device.type == "cuda"))
    per_image = {"ssim": [], "rmse": [], "psnr": []}
    loss_sum, n_batches = 0.0, 0
    for b in loader:
        ma, li, gt, nm = (b[k].to(device, non_blocking=True) for k in ("ma", "li", "gt", "nonmetal"))
        out = model(ma, li, nm)
        vloss, _ = findnet_loss(ma, gt, nm, *out, S=int(cfg["model"]["S"]), **cfg["loss"])
        loss_sum += float(vloss.item())
        n_batches += 1
        pred_norm = conv.to_norm(final_output(out)).float()
        r = evaluate_predictions(pred_norm, b["gt_norm"].to(device), b["metal_mask"].to(device), HU_MIN, HU_MAX)
        for k in per_image:
            per_image[k].extend(r[k])
    model.train()
    return summarize_metrics(per_image), loss_sum / max(1, n_batches), len(per_image["ssim"])


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args)
    tc, cc = cfg["train"], cfg["checkpoint"]
    ckpt_dir = cc["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    seed = int(tc["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tc["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(tc["tf32"])
        torch.backends.cudnn.benchmark = bool(tc["cudnn_benchmark"])

    # ---------------- data ----------------
    dc = cfg["data"]
    train_ds, val_ds, conv = build_findnet_datasets(
        dc["ma_dir"], dc["gt_dir"], dc["li_dir"], dc["mask_dir"],
        normalization=cfg["adapter"]["normalization"], augment_train=cfg["adapter"]["augment_flips"],
        subset=args.subset, val_subset=args.val_subset)
    bs = int(tc["batch_size"])
    steps_per_epoch = math.ceil(len(train_ds) / bs)

    # ---------------- model / optim / schedule ----------------
    model = build_findnet(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    oc, sc = cfg["optim"], cfg["schedule"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(oc["lr"]), betas=tuple(oc["betas"]),
                                  weight_decay=float(oc["weight_decay"]))
    total_steps = int(tc["epochs"]) * steps_per_epoch
    warmup_steps = int(round(float(sc["warmup_epochs"]) * steps_per_epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps, float(sc["min_lr_ratio"])))

    # ---------------- resume ----------------
    st = {"next_epoch": 1, "next_iter": 0, "global_step": 0, "best_ssim": None, "wandb_run_id": None,
          "epochs_since_improve": 0, "epoch_loss_sum": 0.0, "epoch_loss_count": 0}
    resume_path = None if args.no_resume else ckpt.find_resume_checkpoint(ckpt_dir)
    if resume_path:
        meta = ckpt.load_checkpoint(resume_path, model, optimizer, scheduler, None, map_location="cpu")
        ex = meta["extra"]
        st.update({k: ex[k] for k in ("next_epoch", "next_iter", "epochs_since_improve",
                                      "epoch_loss_sum", "epoch_loss_count") if k in ex})
        st.update(global_step=meta["global_step"], best_ssim=meta["best_metric"],
                  wandb_run_id=meta["wandb_run_id"])
        for d in config_diffs(ex.get("config"), cfg):
            print(f"[resume] WARNING config changed since checkpoint: {d}")
        if ex.get("steps_per_epoch") not in (None, steps_per_epoch):
            print(f"[resume] WARNING steps_per_epoch changed: {ex.get('steps_per_epoch')} -> {steps_per_epoch}")
        print(f"[resume] from {resume_path}: global_step={st['global_step']} epoch={st['next_epoch']} "
              f"iter_in_epoch={st['next_iter']} best_ssim={st['best_ssim']}")
    else:
        print(f"[resume] no checkpoint in {ckpt_dir}; starting from scratch"
              + (" (--no_resume)" if args.no_resume else ""))

    with open(os.path.join(ckpt_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "cli": vars(args)}, f, sort_keys=False)

    run_info = {"findnet_commit": findnet_commit(), "n_params": n_params, "dataset_kwargs": DATASET_KWARGS,
                "n_train": len(train_ds), "n_val": len(val_ds), "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps, "warmup_steps": warmup_steps, "cli": vars(args)}
    logger = Logger(cfg, st["wandb_run_id"], run_info)
    st["wandb_run_id"] = logger.run_id or st["wandb_run_id"]

    print(f"[findnet] commit={run_info['findnet_commit']} gaussian_filter={cfg['model']['gaussian_filter']} "
          f"params={n_params:,} device={device}")
    print(f"[findnet] train={len(train_ds)} val={len(val_ds)} batch={bs} steps/epoch={steps_per_epoch} "
          f"epochs={tc['epochs']} total_steps={total_steps} warmup_steps={warmup_steps} "
          f"normalization={cfg['adapter']['normalization']}")

    def save(name, next_epoch, next_iter, epoch_for_meta):
        ckpt.save_checkpoint(
            os.path.join(ckpt_dir, name), model, optimizer, scheduler, None,
            epoch=epoch_for_meta, global_step=st["global_step"], best_metric=st["best_ssim"],
            wandb_run_id=st["wandb_run_id"], local_tmp_dir=cc.get("local_tmp_dir"),
            extra={"next_epoch": next_epoch, "next_iter": next_iter,
                   "epochs_since_improve": st["epochs_since_improve"],
                   "epoch_loss_sum": st["epoch_loss_sum"], "epoch_loss_count": st["epoch_loss_count"],
                   "steps_per_epoch": steps_per_epoch, "config": copy.deepcopy(cfg),
                   "method": METHOD, "findnet_commit": run_info["findnet_commit"]})

    # ---------------- training ----------------
    max_iters = int(args.max_iters or 0)
    save_every = int(cc.get("save_every_iters") or 0)
    log_every = int(tc.get("log_every_iters") or 50)
    patience = int(tc.get("early_stop_patience") or 0)
    loss_kwargs = dict(S=int(cfg["model"]["S"]), **cfg["loss"])
    pin = device.type == "cuda"
    stopped_by_max_iters = False
    win = {"loss": 0.0, "l2X": 0.0, "l1X": 0.0, "l1A": 0.0, "n": 0, "t0": time.time()}

    model.train()
    for epoch in range(int(st["next_epoch"]), int(tc["epochs"]) + 1):
        start_iter = int(st["next_iter"]) if epoch == int(st["next_epoch"]) else 0
        order = epoch_order(len(train_ds), seed, epoch)[start_iter * bs:]
        loader = DataLoader(train_ds, batch_size=bs, sampler=order, num_workers=int(dc["num_workers"]),
                            pin_memory=pin, drop_last=False)
        it = start_iter
        for b in loader:
            if max_iters and st["global_step"] >= max_iters:
                stopped_by_max_iters = True
                break
            ma, li, gt, nm = (b[k].to(device, non_blocking=True) for k in ("ma", "li", "gt", "nonmetal"))
            out = model(ma, li, nm)
            loss, parts = findnet_loss(ma, gt, nm, *out, **loss_kwargs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss {loss.item()} at global_step={st['global_step']}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            st["global_step"] += 1
            it += 1
            lv = float(loss.item())
            st["epoch_loss_sum"] += lv
            st["epoch_loss_count"] += 1
            win["loss"] += lv
            for k in ("l2X", "l1X", "l1A"):
                win[k] += float(parts[k])
            win["n"] += 1

            if st["global_step"] % log_every == 0:
                n = win["n"]
                sec_per_it = (time.time() - win["t0"]) / n
                lr = optimizer.param_groups[0]["lr"]
                logger.log({"global_step": st["global_step"], "train/loss": win["loss"] / n,
                            "train/loss_l2X": win["l2X"] / n, "train/loss_l1X": win["l1X"] / n,
                            "train/loss_l1A": win["l1A"] / n, "train/lr": lr, "train/epoch": epoch,
                            "train/sec_per_iter": sec_per_it})
                print(f"[train] epoch {epoch} it {it}/{steps_per_epoch} step {st['global_step']} "
                      f"loss {win['loss'] / n:.4e} lr {lr:.3e} {sec_per_it:.3f}s/it", flush=True)
                win.update(loss=0.0, l2X=0.0, l1X=0.0, l1A=0.0, n=0, t0=time.time())

            if save_every and st["global_step"] % save_every == 0:
                save(ckpt.LAST_NAME, epoch, it, epoch)

        if stopped_by_max_iters:
            save(ckpt.LAST_NAME, epoch, it, epoch)
            print(f"[stop] reached --max_iters {max_iters} at global_step={st['global_step']} "
                  f"(epoch {epoch}, iter {it}); saved {ckpt.LAST_NAME}")
            break

        # ---------------- validation (every k epochs and after the last epoch) ----------------
        epoch_loss = st["epoch_loss_sum"] / max(1, st["epoch_loss_count"])
        st["epoch_loss_sum"], st["epoch_loss_count"] = 0.0, 0
        logger.log({"global_step": st["global_step"], "train/epoch_loss": epoch_loss, "train/epoch": epoch})
        val_every = max(1, int(cfg["validation"].get("every_epochs") or 1))
        if epoch % val_every != 0 and epoch != int(tc["epochs"]):
            print(f"[epoch] {epoch} done, train_loss {epoch_loss:.4e} (no validation this epoch)", flush=True)
            save(ckpt.LAST_NAME, epoch + 1, 0, epoch)
            continue

        t_val = time.time()
        val, val_loss, n_val = validate(model, val_ds, conv, cfg, device)
        improved = ckpt.is_improvement(val["ssim"], st["best_ssim"])
        if improved:
            st["best_ssim"] = val["ssim"]
            st["epochs_since_improve"] = 0
        else:
            st["epochs_since_improve"] += 1
        logger.log({"epoch": epoch, "val/ssim": val["ssim"], "val/ssim_std": val["ssim_std"],
                    "val/psnr": val["psnr"], "val/psnr_std": val["psnr_std"],
                    "val/rmse": val["rmse"], "val/rmse_std": val["rmse_std"], "val/loss": val_loss,
                    "val/best_ssim": st["best_ssim"], "val/train_epoch_loss": epoch_loss,
                    "val/lr": optimizer.param_groups[0]["lr"], "val/global_step": st["global_step"]})
        print(f"[val] epoch {epoch} ({n_val} imgs, {time.time() - t_val:.0f}s) "
              f"SSIM {val['ssim']:.4f} ± {val['ssim_std']:.4f} | PSNR {val['psnr']:.2f} ± {val['psnr_std']:.2f} | "
              f"RMSE {val['rmse']:.2f} ± {val['rmse_std']:.2f} HU | val_loss {val_loss:.4e} | "
              f"train_loss {epoch_loss:.4e}" + (" | NEW BEST" if improved else ""), flush=True)

        if improved:
            save(ckpt.BEST_NAME, epoch + 1, 0, epoch)
        save(ckpt.LAST_NAME, epoch + 1, 0, epoch)

        if patience > 0 and st["epochs_since_improve"] >= patience:
            print(f"[stop] early stopping: no val SSIM improvement for {patience} epochs "
                  f"(best {st['best_ssim']:.4f})")
            break
    else:
        print(f"[done] all {tc['epochs']} epochs completed. best val SSIM = {st['best_ssim']}")

    logger.finish()


if __name__ == "__main__":
    main()
