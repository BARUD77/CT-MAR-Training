"""ADN inference wrapper for our normalized [0,1] data: artifact image -> artifact-reduced image.

Only the artifact-image content encoder (`encoder_low`) and the clean-image generator (`decoder`) are
used, i.e. the official ADN.forward_lh / ADNTest output `pred_lh`.

Value mapping (official DeepLesion preprocessing, adn/datasets/deep_lesion.py):
  normalize:   clip(x, min, max) -> (x - min) / (max - min) -> * 2 - 1
  denormalize: y * 0.5 + 0.5     -> * (max - min) + min
Our dataset already does the clip + min-max step (HU window [-1024, 3072], clip_hu=True), so the
input is x * 2 - 1 and the output is y * 0.5 + 0.5, back in our normalized space. The output is not
clamped (evaluate_predictions clamps to [0, 1]).
"""
import torch.nn as nn

from .losses import ADN_G_KWARGS
from .networks import Decoder, Encoder


def to_adn(x):
    """Our normalized [0,1] -> ADN [-1,1]."""
    return x * 2.0 - 1.0


def from_adn(y):
    """ADN [-1,1] -> our normalized [0,1]."""
    return y * 0.5 + 0.5


class ADNInference(nn.Module):
    """forward(x): (B,1,H,W) MA in normalized [0,1] (H, W divisible by 4) -> prediction, same space.

    state_dict keys (encoder_low.*, decoder.*) equal the corresponding ADN keys, so
    `ADNInference().load_state_dict({k: v for k, v in adn_sd.items() if k.startswith(('encoder_low.', 'decoder.'))})`
    works for our checkpoints and for the official `model_g` state.
    """

    def __init__(self, output_act="tanh", **kwargs):
        super().__init__()
        g = {**ADN_G_KWARGS, **kwargs}
        self.model_kwargs = {**g, "output_act": output_act}
        n = g["num_down"] + g["num_residual"] + 1 if g["num_sides"] == "all" else g["num_sides"]
        self.encoder_low = Encoder(g["input_ch"], g["base_ch"], g["num_down"], g["num_residual"],
                                   g["res_norm"], g["down_norm"])
        self.decoder = Decoder(g["input_ch"], g["base_ch"], g["num_down"], g["num_residual"], n,
                               g["res_norm"], g["up_norm"], g["fuse"], output_act)

    @classmethod
    def from_adn(cls, adn, model_kwargs):
        """Wrap the encoder_low / decoder modules of a full ADN (shared, not copied)."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.model_kwargs = dict(model_kwargs)
        obj.encoder_low = adn.encoder_low
        obj.decoder = adn.decoder
        return obj

    def forward(self, x):
        code, _ = self.encoder_low(to_adn(x))
        return from_adn(self.decoder(code))
