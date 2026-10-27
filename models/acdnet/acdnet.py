# Port of ACDNet from the official implementation:
#   H. Wang, Y. Li, D. Meng, Y. Zheng, "Adaptive Convolutional Dictionary Network for CT Metal
#   Artifact Reduction", IJCAI 2022.  https://github.com/hongwang01/ACDNet  (acdnet.py)
# The network math is unchanged. Changes vs. the original file:
#   * Hyper-parameters are keyword arguments (defaults = original train.py defaults) instead of an
#     argparse namespace; the unused `args.batchSize` attribute is dropped.
#   * The dictionary initialisation (utils/init_kernel_dir.mat, key 'C9') is loaded relative to
#     this file inside __init__ instead of from the working directory at import time.
#   * Deprecated / legacy calls replaced by equivalents: torch.FloatTensor / torch.Tensor ->
#     torch.tensor(dtype=float32), torch.norm -> torch.linalg.vector_norm, unused
#     `torch.autograd.Variable` import removed, module-level `filter` (shadowed a builtin) renamed.
#   * Every parameter created from an expanded or shared tensor is .clone()'d at init (eta1_T/2_T/3_T,
#     etaX_nonK/etaM_nonK, Cp, D/D0, convert_conv_layer1/2, tau/tau_extend, Mnet.tau, Knet.tau).
#     In the original these parameters alias each other's memory on CPU, but the original
#     `ACDNet(opt).cuda()` copies every parameter separately, so they were trained as independent
#     tensors (confirmed in the released ACDNet_latest.pt: eta rows, Cp filters, D vs D0 and
#     conv1 vs conv2 all differ). Cloning reproduces that effective behaviour on any device and
#     avoids in-place optimizer updates on overlapping memory.
# Parameter names are identical to the original, so the official state_dict loads with strict=True.
import os

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F

_KERNEL_INIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_kernel_dir.mat")


def _load_dictionary_init():
    # Initialize the common dictionary D with a simple Gaussian kernel (3*64*9*9 -> 1*64*9*9).
    dini = io.loadmat(_KERNEL_INIT_PATH)["C9"]
    return torch.tensor(np.asarray(dini[0:1, :, :, :]), dtype=torch.float32)


# Filtering on the XLI for initializing P^(0) and X^(0), refer to Sec 1.2 in the supplementary material.
def _mean_filter_3x3():
    return (torch.ones(3, 3, dtype=torch.float32) / 9).unsqueeze(dim=0).unsqueeze(dim=0)


class ACDNet(nn.Module):
    def __init__(self, T=10, N=6, Np=32, d=32, num_res=3, Mtau=0.5, etaM=1.0, etaX=5.0):
        super().__init__()
        self.T = T                       # stage number T includes the initialization process
        self.iters = self.T - 1          # not include the initialization process
        self.d = d                       # number d of kernels in the dictionary D
        self.N = N                       # channel dimension N of the feature map M
        self.Np = Np                     # channel expansion dimension Np (Sec 1.1 in SM)
        self.num_res = num_res

        # Stepsize
        self.etaM = torch.tensor([float(etaM)])                       # initialization
        self.etaX = torch.tensor([float(etaX)])                       # initialization
        self.etaK = torch.tensor([5.0])                                # initialization
        self.eta1_T = self.make_eta(self.T, self.etaM)                 # learnable
        self.eta2_T = self.make_eta(self.T, self.etaX)
        self.eta3_T = self.make_eta(self.T, self.etaK)
        Dic = _load_dictionary_init()
        self.D = nn.Parameter(data=Dic[:, :self.d, :, :].clone(), requires_grad=True)

        # proxNet
        self.proxNet_X_T = self.make_Xnet(self.T)
        self.proxNet_M_T = self.make_Mnet(self.T, Mtau)
        self.proxNet_K_T = self.make_Knet(self.T)
        self.proxNet_X_last_layer = Xnet(Np, num_res)                  # fine-tune at the last

        tau_const = torch.tensor([1.0])
        self.tau = nn.Parameter(tau_const.clone(), requires_grad=True)          # for sparse feature map
        self.tau_extend = nn.Parameter(tau_const.clone(), requires_grad=True)   # unused (kept for state_dict)

        # filter for initializing X and P
        Cp_const = _mean_filter_3x3().expand(self.Np, 1, -1, -1)       # size: Np*1*3*3
        self.Cp = nn.Parameter(Cp_const.clone(), requires_grad=True)

        # For initializing M^(0), refer to Sec 1.2 in SM
        self.etaX_nonK = self.make_eta(2, self.etaX)
        self.etaM_nonK = self.make_eta(1, self.etaM)
        self.D0 = nn.Parameter(data=Dic[:, :self.N, :, :].clone(), requires_grad=True)  # used in initialization
        self.proxNet_X_0 = Xnet(Np, num_res)                           # used in initialization process
        self.proxNet_M_init = self.make_Mnet(2, Mtau)
        self.proxNet_X_init = self.make_Xnet(2)
        convert = torch.eye(self.N, self.N)
        big_convert = convert.unsqueeze(dim=2).unsqueeze(dim=3)        # N*N*1*1
        self.convert_conv_layer1 = nn.Parameter(data=big_convert.clone(), requires_grad=True)
        self.convert_conv_layer2 = nn.Parameter(data=big_convert.clone(), requires_grad=True)

    def make_Xnet(self, iters):
        return nn.Sequential(*[Xnet(self.Np, self.num_res) for _ in range(iters)])

    def make_Mnet(self, iters, Mtau):                                  # Mnet channels No. N
        return nn.Sequential(*[Mnet(self.N, self.num_res, Mtau) for _ in range(iters)])

    def make_Knet(self, iters):
        return nn.Sequential(*[Knet(self.N, self.d) for _ in range(iters)])

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f.clone(), requires_grad=True)
        return eta

    def forward(self, Xma, XLI, Mask):  # Mask: non-metal region
        ## First using non-K version to initialize M and X, refer to Sec 1.2 in SM
        input = Xma
        b, h, w = input.size()[0], input.size()[2], input.size()[3]
        ListX = []
        ListA = []
        ListX_nonK = []
        ListA_nonK = []
        # initialize P0 and X0
        P00 = F.conv2d(XLI, self.Cp, stride=1, padding=1)
        input_ini = torch.cat((XLI, P00), dim=1)
        XP_ini = self.proxNet_X_0(input_ini)
        X0 = XP_ini[:, :1, :, :]
        P0 = XP_ini[:, 1:, :, :]

        # updating M0--->M1
        ES = Mask * (input - X0)
        EDM = F.relu(ES - self.tau)                                    # for sparse rain layer
        GM = F.conv_transpose2d(EDM, self.D0 / 10, stride=1, padding=4)  # /10 for controlling the updating speed
        M = self.proxNet_M_init[0](GM)
        DM = F.conv2d(M, self.D0 / 10, stride=1, padding=4)
        # Updating X0-->X1
        EB = input - DM
        GX = X0 - EB
        X1 = X0 - self.etaX_nonK[0, :] / 10 * Mask * GX
        input_dual = torch.cat((X1, P0), dim=1)
        out_dual = self.proxNet_X_init[0](input_dual)
        X = out_dual[:, :1, :, :]
        P = out_dual[:, 1:, :, :]
        ListX_nonK.append(X)
        ListA_nonK.append(DM)

        # formal updating of non-K version for M-net
        ES = input - X
        EDM = DM - ES
        GM = F.conv_transpose2d(Mask * EDM, self.D0 / 10, stride=1, padding=4)
        input_new = M - self.etaM_nonK[0, :] / 10 * GM
        M = self.proxNet_M_init[1](input_new)

        # formal updating of non-K version for B-net
        DM = F.conv2d(M, self.D0 / 10, stride=1, padding=4)

        EX = input - DM
        GX = X - EX
        x_dual = X - self.etaX_nonK[1, :] / 10 * Mask * GX
        input_dual = torch.cat((x_dual, P), dim=1)
        out_dual = self.proxNet_X_init[1](input_dual)
        X = out_dual[:, :1, :, :]
        P = out_dual[:, 1:, :, :]
        M1 = F.conv2d(F.relu(M), self.convert_conv_layer1, stride=1, padding=0)
        M = F.conv2d(M1, self.convert_conv_layer2, stride=1, padding=0)
        ListX_nonK.append(X)
        ListA_nonK.append(DM)

        ########### the computation process above is corresponding to the initialization process in Sec 1.2 of SM

        # Using adaptive version to initialize M, X, and K
        # 1st iteration: Updating X0, M0-->K0
        M_re = M.reshape(1, b * self.N, h, w)
        D_re = self.D.reshape(1, 1 * self.d, 9, 9).expand(b * self.N, -1, -1, -1).reshape(b * self.N * 1 * self.d, 1, 9, 9)
        DM = F.conv2d(M_re, D_re / 10, groups=b * self.N, stride=1, padding=4)
        DM_re = DM.reshape(b, self.N, 1, self.d, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, self.N * self.d, 1 * h * w)
        DM_re_trans = DM_re
        A_hat = input - X
        A_hatmask = Mask * A_hat
        A_hat_re = A_hatmask.reshape(b, 1 * h * w).unsqueeze(dim=2)
        GK = torch.bmm(DM_re_trans, A_hat_re).squeeze(dim=2)
        GK_re = GK.reshape(b, self.N, self.d)
        K = self.proxNet_K_T[0](GK_re)

        # 1st iteration: Updating X0, M0, K0-->M1
        D_per = self.D.permute(1, 0, 2, 3).reshape(self.d, -1)
        KD = torch.matmul(K, D_per / 10).reshape(b, self.N, 1, 9, 9).permute(0, 2, 1, 3, 4).reshape(b * 1, self.N, 9, 9)
        A = F.conv2d(M_re, KD, groups=b, stride=1, padding=4)
        A_re = A.reshape(b, 1, h, w)
        A_hat_cut = (Mask * A_hat - Mask * A_re).reshape(1, 1 * b, h, w)

        Epsilon = F.conv_transpose2d(A_hat_cut, KD, groups=b, stride=1, padding=4).reshape(b, self.N, h, w)
        Epsilon_re = Epsilon.reshape(b, self.N, h, w)
        GM = M + self.eta1_T[0, :] / 10 * Epsilon_re
        M = self.proxNet_M_T[0](GM)

        # 1st iteration: Updating K0, X0, M1-->X1
        M_re = M.reshape(1, b * self.N, h, w)
        A = F.conv2d(M_re, KD, groups=b, stride=1, padding=4)
        A_re = A.reshape(b, 1, h, w)

        X_hat = input - A_re
        X_mid = (1 - self.eta2_T[0] / 10 * Mask) * X + self.eta2_T[0, :] / 10 * Mask * X_hat
        inputX_concat = torch.cat((X_mid, P), dim=1)
        X_dual = self.proxNet_X_T[0](inputX_concat)
        X = X_dual[:, :1, :, :]
        P = X_dual[:, 1:, :, :]
        ListX.append(X)
        ListA.append(A_re)
        D_re = self.D.reshape(1, 1 * self.d, 9, 9).expand(b * self.N, -1, -1, -1).reshape(b * self.N * 1 * self.d, 1, 9, 9)
        for i in range(self.iters):
            # K-Net
            DM = F.conv2d(M_re, D_re / 10, groups=b * self.N, stride=1, padding=4)
            DM_re = DM.reshape(b, self.N, 1, self.d, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, self.N * self.d, 1 * h * w)
            DM_re_trans = DM_re
            A = F.conv2d(M_re, KD, groups=b, stride=1, padding=4)
            A_re = A.reshape(b, 1, h, w)
            A_hat = input - X - A_re
            A_hatmask = Mask * A_hat
            A_hat_re = A_hatmask.reshape(b, 1 * h * w).unsqueeze(dim=2)
            GK = torch.bmm(DM_re_trans, A_hat_re).squeeze(dim=2)
            GK_re = GK.reshape(b, self.N, self.d)
            K = self.proxNet_K_T[i + 1](K + self.eta3_T[i + 1, :] / (h * w * 50) * GK_re)

            # M-net
            A_re = torch.bmm(K.reshape(b, 1, self.N * self.d), DM_re).reshape(b, 1, h, w)
            KD = torch.matmul(K, D_per / 10).reshape(b, self.N, 1, 9, 9).permute(0, 2, 1, 3, 4).reshape(b * 1, self.N, 9, 9)
            A_hat_cut = (Mask * input - Mask * X - Mask * A_re).reshape(1, 1 * b, h, w)
            Epsilon = F.conv_transpose2d(A_hat_cut, KD, groups=b, stride=1, padding=4).reshape(b, self.N, h, w)
            Epsilon_re = Epsilon.reshape(b, self.N, h, w)
            GM = M + self.eta1_T[i + 1, :] / 10 * Epsilon_re
            M = self.proxNet_M_T[i + 1](GM)

            # X-net
            M_re = M.reshape(1, b * self.N, h, w)
            A = F.conv2d(M_re, KD, groups=b, stride=1, padding=4)
            A_re = A.reshape(b, 1, h, w)
            X_hat = input - A_re
            ListA.append(A_re)
            X_mid = (1 - self.eta2_T[i + 1, :] * Mask / 10) * X + self.eta2_T[i + 1, :] * Mask / 10 * X_hat
            inputX_concat = torch.cat((X_mid, P), dim=1)
            X_dual = self.proxNet_X_T[i + 1](inputX_concat)
            X = X_dual[:, :1, :, :]
            P = X_dual[:, 1:, :, :]
            ListX.append(X)
        XP_adjust = self.proxNet_X_last_layer(X_dual)
        X = XP_adjust[:, :1, :, :]
        ListX.append(X)
        return X0, ListX, ListA, ListX_nonK, ListA_nonK


# proxNet_M
class Mnet(nn.Module):
    def __init__(self, N, num_res, Mtau):
        super().__init__()
        self.channels = N
        self.L = num_res
        self.layer = self.make_resblock(self.L)
        tau0 = torch.tensor([float(Mtau)])
        tau_const = tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        self.tau = nn.Parameter(tau_const.clone(), requires_grad=True)  # for sparse rain map

    def make_resblock(self, num_res):
        layers = []
        for _ in range(num_res):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        M = input
        for i in range(self.L):
            M = F.relu(M + self.layer[i](M))
        M = F.relu(M - self.tau)
        return M


# proxNet_K
class Knet(nn.Module):
    def __init__(self, N, d):
        super().__init__()
        self.d = d
        self.N = N
        self.layer = self.make_resblock(1)
        tau0 = torch.tensor([0.1])
        tau_const = tau0.unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.N, self.d)
        self.tau = nn.Parameter(tau_const.clone(), requires_grad=True)  # unused in forward (as in the original)

    def make_resblock(self, num_resK):
        layers = []
        for _ in range(num_resK):
            layers.append(nn.Sequential(
                nn.Linear(self.d * self.N, self.d * self.N),
                nn.ReLU(),
                nn.Linear(self.d * self.N, self.d * self.N),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        K = input
        b = K.size()[0]
        K = K.reshape(-1, self.d * self.N)
        for i in range(1):
            K = F.relu(K + self.layer[i](K))
        K = K.reshape(b, self.N, self.d)
        norm = torch.linalg.vector_norm(K, ord=2, dim=2)
        norm_re = norm.unsqueeze(dim=2).expand(-1, -1, self.d)
        K = torch.div(K, norm_re + 1e-6)
        return K


# proxNet_X
class Xnet(nn.Module):
    def __init__(self, Np, num_res):
        super().__init__()
        self.channels = Np + 1
        self.L = num_res
        self.layer = self.make_resblock(num_res)

    def make_resblock(self, L):
        layers = []
        for _ in range(L):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.L):
            X = F.relu(X + self.layer[i](X))
        return X
