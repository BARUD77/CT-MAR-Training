"""Weights & Biases helpers shared by ALL baselines."""
import os
import time


def wandb_login(secret_name="WANDB_API_KEY"):
    """Log in to W&B without hardcoding the key.

    Order: Colab secret `secret_name` (google.colab.userdata) -> environment variable
    `secret_name` -> existing local W&B credentials. The key is never printed.
    Returns True if a key was found and passed to wandb.login.
    """
    import wandb

    key = None
    try:
        from google.colab import userdata  # type: ignore
        try:
            key = userdata.get(secret_name)
        except Exception:
            key = None
    except ImportError:
        pass
    if not key:
        key = os.environ.get(secret_name)
    if key:
        wandb.login(key=key, relogin=True)
        return True
    print(f"[wandb] No '{secret_name}' Colab secret or env var found; using existing login if any.")
    return False


def make_run_name(method, run_name=None):
    """Run names are always prefixed with the method name, e.g. 'findnet_<name or timestamp>'."""
    suffix = run_name if run_name else time.strftime("%Y%m%d-%H%M%S")
    prefix = f"{method}_"
    return suffix if suffix.startswith(prefix) else prefix + suffix


def init_wandb(method, project, config=None, run_name=None, entity=None, run_id=None,
               log_dir=None, mode=None, tags=None):
    """Start (or resume) a W&B run.

    Args:
        method: method name used as run-name prefix and tag (e.g. "findnet").
        run_id: W&B run id from a checkpoint; if given the run is resumed (resume="allow").
        mode: "online" / "offline" / "disabled"; default: WANDB_MODE env var or "online".
    Returns the wandb Run. Store `run.id` in checkpoints to resume the same run later.
    """
    import wandb

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    kwargs = dict(
        project=project,
        entity=entity,
        name=make_run_name(method, run_name),
        config=config,
        dir=log_dir,
        tags=list(tags or []) + [method],
        mode=mode or os.environ.get("WANDB_MODE", "online"),
    )
    if run_id:
        kwargs.update(id=run_id, resume="allow")
    return wandb.init(**kwargs)
