"""ACDNet wrapper for our normalized [0,1] data (1 = metal in our masks)."""
import torch
import torch.nn as nn

from .acdnet import ACDNet

# The original pipeline feeds images in [0,255]-scaled units (Dataset.py: normalize(...) * 255) and
# divides the output by 255; the step sizes / thresholds were tuned for that range.
ACDNET_SCALE = 255.0

ACDNET_DEFAULT_KWARGS = dict(T=10, N=6, Np=32, d=32, num_res=3, Mtau=0.5, etaM=1.0, etaX=5.0)


def nonmetal_from_metal(metal_mask):
    """Our masks are 1 = metal; ACDNet expects 1 = non-metal. Asserts the mask is binary."""
    if not bool(((metal_mask == 0) | (metal_mask == 1)).all()):
        vals = torch.unique(metal_mask)
        raise ValueError(f"Metal mask must be binary {{0,1}}; got values {vals[:10].tolist()}"
                         f"{' ...' if vals.numel() > 10 else ''}")
    return 1.0 - metal_mask


class ACDNetInference(nn.Module):
    """forward(ma, li, metal_mask) -> ListX[-1] / 255 in our normalized space (no clamping).

    ma, li: (B,1,H,W) normalized [0,1]; metal_mask: (B,1,H,W) binary, 1 = metal.
    The full ACDNet outputs (for the loss) are available via `self.net(...)` on x255 inputs.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model_kwargs = {**ACDNET_DEFAULT_KWARGS, **kwargs}
        self.net = ACDNet(**self.model_kwargs)

    def forward(self, ma, li, metal_mask):
        non_metal = nonmetal_from_metal(metal_mask)
        _, ListX, _, _, _ = self.net(ma * ACDNET_SCALE, li * ACDNET_SCALE, non_metal)
        return ListX[-1] / ACDNET_SCALE
