"""FIND-Net model import from third_party/findnet (vendored, commit in third_party/findnet/COMMIT.txt).

The architecture, stage count, kernel sizes and layers are NOT changed. Two load-time adjustments,
both without editing any file in third_party/:

1. Gaussian filtering switch. The repo selects the variant with the module-level flag
   `Gaussian_filter` in Model/ffc.py (default False = "FIND-Net (No-GF)"). The README says to edit the
   file to get full FIND-Net. Here Model/ffc.py is read, the single line `Gaussian_filter = False` is
   replaced by `Gaussian_filter = True` IN MEMORY, and the module is executed under its real name
   `Model.ffc` before Model.ProxNet / Model.findnet import it.

2. Aliased parameters (optional, `materialize_aliased_params`). etaM_S, etaX_S, K_q and every Mnet.tau
   are nn.Parameters created from `.expand()`ed tensors (one memory location shared by all elements),
   and K0 / K wrap the same module-level tensor. Current PyTorch refuses to update them in place
   (optimizer.step) or load into them (load_state_dict). Each parameter is replaced by a contiguous
   clone with identical values, i.e. same initialization, independent storage -- which is what the
   authors' released checkpoints show (all those entries differ after training).

FIND-Net also loads `utils/init_kernel.mat` relative to the CWD and imports `Model.*` absolutely, so
the import runs with third_party/findnet on sys.path and as CWD (restored afterwards).
"""
import importlib
import os
import sys
import types
from types import SimpleNamespace

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINDNET_DIR = os.path.join(_REPO_ROOT, "third_party", "findnet")

_LOADED_VARIANT = None  # gaussian_filter flag of the FIND-Net modules currently in sys.modules


def findnet_commit():
    path = os.path.join(FINDNET_DIR, "COMMIT.txt")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("commit:"):
                return line.split(":", 1)[1].strip()
    return None


def load_findnet_class(gaussian_filter=True):
    """Import and return the original FINDNet class with the requested Gaussian-filter variant."""
    global _LOADED_VARIANT
    if _LOADED_VARIANT is not None:
        if _LOADED_VARIANT != bool(gaussian_filter):
            raise RuntimeError("FIND-Net was already imported with gaussian_filter="
                               f"{_LOADED_VARIANT}; one variant per process.")
        return sys.modules["Model.findnet"].FINDNet

    if not os.path.isdir(os.path.join(FINDNET_DIR, "Model")):
        raise FileNotFoundError(f"FIND-Net code not found at {FINDNET_DIR}")
    for name in ("Model", "Model.ffc", "Model.ProxNet", "Model.findnet"):
        if name in sys.modules:
            raise RuntimeError(f"A module named '{name}' is already imported; cannot load FIND-Net.")

    if FINDNET_DIR not in sys.path:
        sys.path.insert(0, FINDNET_DIR)
    old_cwd = os.getcwd()
    os.chdir(FINDNET_DIR)
    try:
        importlib.import_module("Model")  # namespace package (no __init__.py)

        ffc_path = os.path.join(FINDNET_DIR, "Model", "ffc.py")
        with open(ffc_path, encoding="utf-8") as f:
            src = f.read()
        flag_line = "Gaussian_filter = False"
        if src.count(flag_line) != 1:
            raise RuntimeError(f"Expected exactly one '{flag_line}' in {ffc_path}; code changed?")
        if gaussian_filter:
            src = src.replace(flag_line, "Gaussian_filter = True")
        ffc_mod = types.ModuleType("Model.ffc")
        ffc_mod.__file__ = ffc_path
        ffc_mod.__package__ = "Model"
        sys.modules["Model.ffc"] = ffc_mod
        exec(compile(src, ffc_path, "exec"), ffc_mod.__dict__)
        if ffc_mod.Gaussian_filter != bool(gaussian_filter):
            raise RuntimeError("Gaussian_filter flag was not applied.")

        findnet_mod = importlib.import_module("Model.findnet")
    finally:
        os.chdir(old_cwd)
    _LOADED_VARIANT = bool(gaussian_filter)
    return findnet_mod.FINDNet


def materialize_parameters(model):
    """Give every parameter its own contiguous storage (values unchanged)."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.clone().contiguous()
    return model


def build_findnet(S=10, num_M=32, num_Q=32, T=3, etaM=1.0, etaX=5.0, gaussian_filter=True,
                  materialize_aliased_params=True):
    """Build the original FINDNet. Call: model(ma, li, nonmetal_mask) -> (X0, ListX, ListA).

    The returned module is the original class (state_dict keys identical to the released checkpoints).
    """
    FINDNet = load_findnet_class(gaussian_filter)
    args = SimpleNamespace(S=int(S), num_M=int(num_M), num_Q=int(num_Q), T=int(T),
                           etaM=float(etaM), etaX=float(etaX))
    model = FINDNet(args)
    if materialize_aliased_params:
        materialize_parameters(model)
    return model


def final_output(outputs):
    """Final-stage image from the model's (X0, ListX, ListA) output."""
    return outputs[1][-1]
