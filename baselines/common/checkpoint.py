"""Crash-safe checkpointing shared by ALL baselines.

A checkpoint holds: model, optimizer, scheduler, GradScaler (optional), RNG states (python, numpy,
torch CPU, all CUDA devices), epoch, global step, best metric, W&B run id, and a free-form `extra`
dict (e.g. position inside the epoch, config).

Writes are atomic with respect to disconnects: the checkpoint is first serialized to a LOCAL temp
file, then copied to `<dest>.tmp` inside the checkpoint directory (e.g. Google Drive), its size is
checked, and finally `os.replace` renames it onto `<dest>`. An interrupted save therefore leaves at
most a stale `.tmp` file; the previous `<dest>` stays intact.

Policy (used by the training scripts): save LAST_NAME every epoch and every N iterations; save
BEST_NAME whenever validation masked SSIM improves.
"""
import os
import random
import shutil
import tempfile

import numpy as np
import torch

LAST_NAME = "last.pt"
BEST_NAME = "best.pt"
FORMAT_VERSION = 1


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
        else:
            print(f"[checkpoint] WARNING: saved {len(cuda_states)} CUDA RNG states but "
                  f"{torch.cuda.device_count()} devices are visible; CUDA RNG not restored.")


def atomic_torch_save(obj, dest_path, local_tmp_dir=None):
    """torch.save to a local temp file, copy next to dest as .tmp, verify size, rename onto dest."""
    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    os.makedirs(dest_dir, exist_ok=True)
    if local_tmp_dir is not None:
        os.makedirs(local_tmp_dir, exist_ok=True)
    fd, local_tmp = tempfile.mkstemp(suffix=".pt", dir=local_tmp_dir)
    os.close(fd)
    dest_tmp = dest_path + ".tmp"
    try:
        torch.save(obj, local_tmp)
        shutil.copyfile(local_tmp, dest_tmp)
        with open(dest_tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        src_size, dst_size = os.path.getsize(local_tmp), os.path.getsize(dest_tmp)
        if src_size != dst_size:
            raise IOError(f"Checkpoint copy size mismatch ({src_size} vs {dst_size}) for {dest_tmp}")
        os.replace(dest_tmp, dest_path)
    finally:
        if os.path.exists(local_tmp):
            os.remove(local_tmp)
        if os.path.exists(dest_tmp):
            os.remove(dest_tmp)


def save_checkpoint(path, model, optimizer, scheduler=None, scaler=None, *, epoch, global_step,
                    best_metric, wandb_run_id=None, extra=None, local_tmp_dir=None):
    state = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": None if best_metric is None else float(best_metric),
        "wandb_run_id": wandb_run_id,
        "extra": extra or {},
    }
    atomic_torch_save(state, path, local_tmp_dir=local_tmp_dir)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, *,
                    restore_rng=True, map_location="cpu"):
    """Load a checkpoint written by save_checkpoint into the given objects.

    Returns dict with epoch, global_step, best_metric, wandb_run_id, extra.
    """
    # weights_only=False: the file contains RNG states (numpy arrays/tuples). Only load files
    # written by this module.
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if restore_rng and state.get("rng") is not None:
        restore_rng_state(state["rng"])
    return {
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "best_metric": state.get("best_metric"),
        "wandb_run_id": state.get("wandb_run_id"),
        "extra": state.get("extra", {}),
    }


def find_resume_checkpoint(ckpt_dir):
    """Path to ckpt_dir/last.pt if it exists, else None."""
    path = os.path.join(ckpt_dir, LAST_NAME)
    return path if os.path.isfile(path) else None


def is_improvement(value, best, min_delta=0.0):
    """Higher-is-better (masked SSIM)."""
    return best is None or value > best + min_delta
