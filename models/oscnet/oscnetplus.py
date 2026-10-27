# Port of OSCNet+ from the official implementation (network/oscnetplus.py):
#   H. Wang, Q. Xie, D. Zeng, J. Ma, D. Meng, Y. Zheng, "OSCNet: Orientation-Shared Convolutional
#   Network for CT Metal Artifact Learning", IEEE TMI 2023 (and MICCAI 2022 "Orientation-Shared
#   Convolution Representation for CT Metal Artifact Learning").  https://github.com/hongwang01/OSCNet
#   Original author: Hong Wang (hongwang9209@hotmail.com).
# The network math is unchanged. Changes vs. the original file:
#   * Hyper-parameters are keyword arguments (defaults = the official OSCNet+ settings, README:
#     `--model oscplus --cdiv 10`, other values from train.py) instead of an argparse namespace;
#     Mnet/Xnet take (channels, T) instead of `args`.
#   * The kernel initialisation (utils/init_kernel.mat, key 'C9') is loaded relative to this file
#     inside __init__ instead of from the working directory at import time.
#   * Legacy calls replaced by equivalents: torch.FloatTensor / torch.Tensor -> torch.tensor(float32),
#     unused `torch.autograd.Variable` import removed, module-level `filter` (shadowed a builtin)
#     renamed; plain-tensor attributes that are not parameters/buffers (etaM, etaX, C_q_const,
#     tau_const, tau0) are dropped (they are not in the state_dict).
#   * Parameters created from expanded tensors are .clone()'d at init (etaM_S, etaX_S, C_q, Mnet.tau).
#     In the original they are stride-0 views on CPU; the original `OSCNetplus(opt).cuda()` copies
#     them into dense tensors, so they were trained as independent entries. Cloning reproduces that
#     on any device and avoids in-place optimizer updates on overlapping memory.
# Parameter names are identical to the original, so the official net_latest.pt loads with strict=True.
import os

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fconv import Fconv_PCA

_KERNEL_INIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_kernel.mat")


def _load_kernel_init():
    kernel = io.loadmat(_KERNEL_INIT_PATH)["C9"]  # 3*32*9*9
    return torch.tensor(np.asarray(kernel[0:1, :, :, :]), dtype=torch.float32)


# Filtering on the metal-affected CT image for initializing B^(0) and Z^(0), refer to the supplementary material.
def _mean_filter_3x3():
    return (torch.ones(3, 3, dtype=torch.float32) / 9).unsqueeze(dim=0).unsqueeze(dim=0)


class OSCNetplus(nn.Module):
    def __init__(self, S=10, T=3, num_M=4, num_Q=32, num_rot=8, etaM=1.0, etaX=5.0, padding=4, inP=5,
                 sizeP=9, ifini=1, cdiv=10.0):
        super().__init__()
        self.S = S                  # Stage number S includes the initialization process
        self.iter = self.S - 1      # not include the initialization process
        self.cdiv = cdiv            # controlling the update speed for filter
        self.num_rot = num_rot      # the number of rotation directions
        self.num_M = num_M          # the number of feature map in every rotation
        self.num_Q = num_Q          # for concatenation channel (see TMI2021 DICDNet)

        # Stepsize
        self.etaM_S = self.make_eta(self.iter, torch.tensor([float(etaM)]))
        self.etaX_S = self.make_eta(self.S, torch.tensor([float(etaX)]))

        # kernel
        kernel = _load_kernel_init()
        smallkernel = kernel[:, :self.num_M, :, :]
        allrot_kernel = smallkernel.repeat(1, self.num_rot, 1, 1)
        self.C0 = nn.Parameter(data=allrot_kernel, requires_grad=True)  # used in initialization process
        # filter parameterization
        self.fcnn = Fconv_PCA(sizeP=sizeP, inNum=1, outNum=self.num_M, tranNum=num_rot, inP=inP, padding=padding,
                              ifIni=ifini, bias=True, Smooth=True, iniScale=1.0, cdiv=self.cdiv)

        # filter for initializing X and Q
        C_q_const = _mean_filter_3x3().expand(self.num_Q, 1, -1, -1)  # size: self.num_Q*1*3*3
        self.C_q = nn.Parameter(C_q_const.clone(), requires_grad=True)

        # proxNet
        self.proxNet_X_0 = Xnet(num_Q + 1, T)  # used in initialization process
        self.proxNet_X_S = self.make_Xnet(self.S, num_Q + 1, T)
        self.proxNet_M_S = self.make_Mnet(self.S, num_M * num_rot, T)
        self.proxNet_X_last_layer = Xnet(num_Q + 1, T)  # fine-tune at the last

        # for sparsity
        self.tau = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    @staticmethod
    def make_Xnet(iters, channels, T):
        return nn.Sequential(*[Xnet(channels, T) for _ in range(iters)])

    @staticmethod
    def make_Mnet(iters, channels, T):
        return nn.Sequential(*[Mnet(channels, T) for _ in range(iters)])

    @staticmethod
    def make_eta(iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        return nn.Parameter(data=const_f.clone(), requires_grad=True)

    def forward(self, CT_ma, LIct, Mask):
        """CT_ma, LIct: (B,1,H,W) in the x255 scale; Mask: (B,1,H,W) NON-metal mask (1 = non-metal).

        Returns X0, ListX (S+1 entries), ListA (S entries).
        """
        batchSize, h, w = CT_ma.size()[0], CT_ma.size()[2], CT_ma.size()[3]
        # save mid-updating results
        ListX = []  # storage all the estimated CT image
        ListA = []  # storage all the estimated artifact layer
        input = CT_ma
        Q00 = F.conv2d(LIct, self.C_q, stride=1, padding=1)  # dual variable Q
        input_ini = torch.cat((LIct, Q00), dim=1)
        XQ_ini = self.proxNet_X_0(input_ini)
        X0 = XQ_ini[:, :1, :, :]
        Q0 = XQ_ini[:, 1:, :, :]

        # 1st iteration: Updating X0-->M1
        A_hat = Mask * (input - X0)
        A_hat_cut = F.relu(A_hat - self.tau)  # for sparsity
        Epsilon = F.conv_transpose2d(A_hat_cut, self.C0 / 10, stride=1,
                                     padding=4)  # /10 for controlling the updating speed
        M1 = self.proxNet_M_S[0](Epsilon)
        A, C = self.fcnn(M1, X0, CT_ma)
        # 1st iteration: Updating M1-->X1
        A_hat = input - A.reshape(batchSize, 1, h, w)
        X_mid = (1 - self.etaX_S[0] * Mask / 10) * X0 + self.etaX_S[0] * Mask / 10 * A_hat
        input_concat = torch.cat((X_mid, Q0), dim=1)
        XQ = self.proxNet_X_S[0](input_concat)
        X1 = XQ[:, :1, :, :]
        Q1 = XQ[:, 1:, :, :]
        ListX.append(X1)
        ListA.append(A)
        X = X1
        Q = Q1
        M = M1
        for i in range(self.iter):
            # M-net
            A_hat = Mask * (input - X)
            Epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d(
                (Mask * A - A_hat).reshape(1, -1, h, w), C / self.cdiv, groups=batchSize, stride=1,
                padding=4).reshape(batchSize, -1, h, w)
            M = self.proxNet_M_S[i + 1](M - Epsilon)

            # X-net
            A = F.conv2d(M.reshape(1, -1, h, w), C / self.cdiv, groups=batchSize, stride=1,
                         padding=4).reshape(batchSize, 1, h, w)
            ListA.append(A)
            X_hat = input - A
            X_mid = (1 - self.etaX_S[i + 1, :] * Mask / 10) * X + self.etaX_S[i + 1, :] * Mask / 10 * X_hat
            input_concat = torch.cat((X_mid, Q), dim=1)
            XQ = self.proxNet_X_S[i + 1](input_concat)
            X = XQ[:, :1, :, :]
            Q = XQ[:, 1:, :, :]
            A, C = self.fcnn(M, X, CT_ma)
            ListX.append(X)
        XQ_adjust = self.proxNet_X_last_layer(XQ)
        X = XQ_adjust[:, :1, :, :]
        ListX.append(X)
        return X0, ListX, ListA


def _resblocks(channels, T):
    return nn.Sequential(*[
        nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1),
                      nn.BatchNorm2d(channels),
                      nn.ReLU(),
                      nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1),
                      nn.BatchNorm2d(channels))
        for _ in range(T)])


# proxNet_M
class Mnet(nn.Module):
    def __init__(self, channels, T):
        super().__init__()
        self.channels = channels  # num_M * num_rot
        self.T = T                # the number of resblocks in each proxNet
        self.layer = _resblocks(self.channels, self.T)
        tau_const = torch.tensor([0.5]).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(
            -1, self.channels, -1, -1)
        self.tau = nn.Parameter(tau_const.clone(), requires_grad=True)  # for sparsity

    def forward(self, input):
        M = input
        for i in range(self.T):
            M = F.relu(M + self.layer[i](M))
        M = F.relu(M - self.tau)
        return M


# proxNet_X
class Xnet(nn.Module):
    def __init__(self, channels, T):
        super().__init__()
        self.channels = channels  # num_Q + 1
        self.T = T
        self.layer = _resblocks(self.channels, self.T)

    def forward(self, input):
        X = input
        for i in range(self.T):
            X = F.relu(X + self.layer[i](X))
        return X
