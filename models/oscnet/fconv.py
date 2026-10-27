# Port of the parts of network/Fconv_wnet.py used by OSCNet+, from the official implementation:
#   H. Wang, Q. Xie, D. Zeng, J. Ma, D. Meng, Y. Zheng, "OSCNet: Orientation-Shared Convolutional
#   Network for CT Metal Artifact Learning", IEEE TMI 2023.  https://github.com/hongwang01/OSCNet
#   (original author: Hong Wang, hongwang9209@hotmail.com; the filter parameterization follows
#   Xie et al., "Fourier series expansion based filter parametrization for equivariant convolutions")
# The math is unchanged. Changes vs. the original file:
#   * Only WNet, Fconv_PCA, GetBasis_PCA and MaskC are kept (the other classes are not used by
#     OSCNetplus); the unused PIL import is removed.
#   * Fconv_PCA.__init__ no longer calls Getini_reg: its result (`iniw`) was never used, the call only
#     consumed numpy's global RNG. No parameter or buffer depends on it.
#   * Legacy calls replaced by equivalents: torch.FloatTensor(np) -> torch.tensor(np, dtype=float32),
#     torch.norm -> torch.linalg.vector_norm, `== None` -> `is None`.
# Parameter/buffer names are identical to the original (Basis, wnet.*, c).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WNet(nn.Module):
    """Predicts per-sample basis coefficients from (current X, metal-affected CT)."""

    def __init__(self, h, w):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 6, 3, 1, 1)
        self.conv3 = nn.Conv2d(6, 6, 3, 1, 1)
        self.f1 = nn.Linear(6, h * w)
        self.h = h
        self.w = w

    def forward(self, Ximg, Xma):
        batchSize = Ximg.size()[0]
        height, width = Ximg.size()[2], Ximg.size()[3]
        output1 = self.conv1(torch.cat((Ximg, Xma), dim=1))
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = F.avg_pool2d(output3, (height, width))  # global average over the whole input
        output8 = F.relu(self.f1(output4.reshape(batchSize, -1)))
        output8 = output8.reshape(-1, self.h, self.w)
        norm = torch.linalg.vector_norm(output8, ord=2, dim=2)
        norm_re = norm.unsqueeze(dim=2).expand(-1, -1, self.w)
        output8 = torch.div(output8, norm_re + 1e-6)
        weight = output8.reshape(-1, self.h, 1, 1, self.w)
        return weight


class Fconv_PCA(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0, cdiv=1):
        super().__init__()
        if inP is None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.cdiv = cdiv
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)  # Rank: 25=inp*inp=5*5
        self.register_buffer("Basis", Basis)
        self.wnet = WNet(outNum, Basis.size(3))
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        # (original: iniw = Getini_reg(...) * iniScale -- unused, see header)
        if padding is None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1, inNum, 1, 1), requires_grad=bias)

    def forward(self, input, Ximg, Xma):  # input: Map, Ximg: Xrecon, Xma: metal-corrupted images
        batchSize, h, w = Ximg.size()[0], Ximg.size()[2], Ximg.size()[3]
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        weights = self.wnet(Ximg, Xma)
        Basis_per = self.Basis.permute(3, 0, 1, 2).reshape(self.Basis.size(3), -1)
        tempW = torch.matmul(weights, Basis_per).reshape(
            batchSize, outNum, 1, 1, self.sizeP, self.sizeP, self.tranNum).permute(0, 1, 6, 2, 3, 4, 5)
        Num = tranNum // expand
        tempWList = [torch.cat(
            [tempW[:, :, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, :, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=4)
            for i in range(expand)]
        tempW = torch.cat(tempWList, dim=2)
        _filter = tempW.reshape([batchSize, outNum * tranNum, self.sizeP, self.sizeP])
        _bias = self.c.repeat([1, 1, inNum, 1]).reshape([1, inNum * self.expand, 1, 1])
        M_re = input.reshape(1, -1, h, w)
        output = F.conv2d(M_re, _filter / self.cdiv, groups=batchSize, stride=1, padding=self.padding)
        outf = output + _bias
        outf_re = outf.reshape(batchSize, 1, h, w)
        return outf_re, _filter


def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth=True):
    if inP is None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(Mask, 2)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)

    BasisC = np.reshape(BasisC, [sizeP * sizeP * tranNum, inP * inP])
    BasisS = np.reshape(BasisS, [sizeP * sizeP * tranNum, inP * inP])

    BasisR = np.concatenate((BasisC, BasisS), axis=1)

    U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))

    Rank = np.sum(S > 0.0001)
    BasisR = np.matmul(np.matmul(BasisR, U[:, :Rank]), np.diag(1 / np.sqrt(S[:Rank] + 0.0000000001)))
    BasisR = np.reshape(BasisR, [sizeP, sizeP, tranNum, Rank])

    temp = np.reshape(BasisR, [sizeP * sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis=0) ** 2, axis=0) + np.std(np.sum(temp ** 2 * sizeP * sizeP, axis=0),
                                                              axis=0)) / np.mean(
        np.sum(temp, axis=0) ** 2 + np.sum(temp ** 2 * sizeP * sizeP, axis=0), axis=0)
    Trod = 1
    Ind = var < Trod
    Rank = np.sum(Ind)
    Weight = 1 / np.maximum(var, 0.04) / 25
    if Smooth:  # after svd, freq
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight, 0), 0), 0) * BasisR

    return torch.tensor(BasisR, dtype=torch.float32), Rank, Weight


def MaskC(SizeP):
    p = (SizeP - 1) / 2
    x = np.arange(-p, p + 1) / p
    X, Y = np.meshgrid(x, x)
    C = X ** 2 + Y ** 2
    Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)
    return X, Y, Mask
