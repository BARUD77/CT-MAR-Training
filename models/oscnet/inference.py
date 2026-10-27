"""OSCNet+ wrapper for our normalized [0,1] data (1 = metal in our masks).

Reuses the ACDNet conventions: x255 input scale and the binary-checked metal -> non-metal conversion.
"""
import torch.nn as nn

from models.acdnet.inference import ACDNET_SCALE, nonmetal_from_metal

from .oscnetplus import OSCNetplus

# The original pipeline feeds images in [0,255]-scaled units (Dataset.py: normalize(...) * 255) and
# divides the output by 255; step sizes / thresholds were tuned for that range (same as ACDNet).
OSCNETPLUS_SCALE = ACDNET_SCALE

OSCNETPLUS_DEFAULT_KWARGS = dict(S=10, T=3, num_M=4, num_Q=32, num_rot=8, etaM=1.0, etaX=5.0, padding=4, inP=5,
                                 sizeP=9, ifini=1, cdiv=10.0)


class OSCNetPlusInference(nn.Module):
    """forward(ma, li, metal_mask) -> ListX[-1] / 255 in our normalized space (no clamping).

    ma, li: (B,1,H,W) normalized [0,1]; metal_mask: (B,1,H,W) binary, 1 = metal.
    The full OSCNet+ outputs (for the loss) are available via `self.net(...)` on x255 inputs.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model_kwargs = {**OSCNETPLUS_DEFAULT_KWARGS, **kwargs}
        self.net = OSCNetplus(**self.model_kwargs)

    def forward(self, ma, li, metal_mask):
        non_metal = nonmetal_from_metal(metal_mask)
        _, ListX, _ = self.net(ma * OSCNETPLUS_SCALE, li * OSCNETPLUS_SCALE, non_metal)
        return ListX[-1] / OSCNETPLUS_SCALE
