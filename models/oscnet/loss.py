"""OSCNet+ training loss, ported verbatim from the official train.py (hongwang01/OSCNet, TMI 2023).

All tensors are in the x255 scale. `mask` is the NON-metal mask (1 = non-metal).
L1 terms use torch.sum, L2 terms use F.mse_loss (mean), as in the original.
"""
import torch
import torch.nn.functional as F


def oscnetplus_loss(Xma, GT, mask, X0, ListX, ListA, S, Xl2=1.0, Xl1=5e-4, Al1=5e-4):
    """Returns (loss, loss_l2X, loss_l1X, loss_l1A)."""
    loss_l2Xt = 0
    loss_l1Xt = 0
    loss_l1At = 0
    newAgt = mask * (Xma - GT)
    newXgt = mask * GT
    for j in range(S):
        loss_l2Xt = loss_l2Xt + 0.1 * F.mse_loss(ListX[j] * mask, newXgt)
        loss_l1Xt = loss_l1Xt + 0.1 * torch.sum(torch.abs(ListX[j] * mask - newXgt))
        loss_l1At = loss_l1At + 0.1 * torch.sum(torch.abs(mask * ListA[j] - newAgt))
    loss_l1Xf = torch.sum(torch.abs((ListX[-1] * mask - newXgt)))
    loss_l1Af = torch.sum(torch.abs(mask * ListA[-1] - newAgt))
    loss_l2Xf = F.mse_loss(ListX[-1] * mask, newXgt)
    loss_l1X0 = 0.1 * torch.sum(torch.abs(X0 * mask - newXgt))
    loss_l2X0 = 0.1 * F.mse_loss(X0 * mask, newXgt)
    loss_l2X = loss_l2Xt + loss_l2Xf + loss_l2X0
    loss_l1X = loss_l1Xt + loss_l1Xf + loss_l1X0
    loss_l1A = loss_l1At + loss_l1Af
    loss = Xl2 * loss_l2X + Xl1 * loss_l1X + Al1 * loss_l1A
    return loss, loss_l2X, loss_l1X, loss_l1A
