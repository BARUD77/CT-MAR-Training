# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import sys
sys.path.append("/content/drive/MyDrive/MAR/Pix2PIxGAN")
from swin_unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys



class SwinUnet(nn.Module):
    def __init__(self, config, img_size=256, in_channels=2, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.in_channels = in_channels
        self.zero_head = zero_head
        self.config = config

        # 1️⃣ Swin Transformer 作为 U-Net 编码器
        self.swin_unet = SwinTransformerSys(
           img_size=config.DATA.IMG_SIZE,
           patch_size=config.MODEL.SWIN.PATCH_SIZE,
           in_chans=config.MODEL.SWIN.IN_CHANS,
           num_classes=config.MODEL.SWIN.NUM_CLASSES,  
           embed_dim=config.MODEL.SWIN.EMBED_DIM,
           depths=config.MODEL.SWIN.DEPTHS,
           depths_decoder=config.MODEL.SWIN.DECODER_DEPTHS,
           num_heads=config.MODEL.SWIN.NUM_HEADS,
           window_size=config.MODEL.SWIN.WINDOW_SIZE,
           mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
           qkv_bias=config.MODEL.SWIN.QKV_BIAS,
           drop_rate=config.MODEL.DROP_RATE,
           drop_path_rate=config.MODEL.DROP_PATH_RATE,
           ape=config.MODEL.SWIN.APE,
           patch_norm=config.MODEL.SWIN.PATCH_NORM,
           use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

        # 2️⃣ 替换 Softmax，使用 Tanh 或 ReLU 输出像素值
        self.final_activation = nn.Tanh()  # 适用于 [-1, 1] 归一化数据
        # self.final_activation = nn.ReLU()  # 如果像素范围是 [0, 1]

    def forward(self, x):
        logits = self.swin_unet(x)  # 通过 Swin Transformer U-Net

        return self.final_activation(logits)  # 让输出是连续像素值

import torch
import torch.nn as nn

class SwinUnetResidual(nn.Module):
    def __init__(self, config, img_size=256, in_channels=1, zero_head=False, vis=False):
        super(SwinUnetResidual, self).__init__()
        self.in_channels = in_channels
        self.zero_head = zero_head
        self.config = config

        # 1️⃣ Swin Transformer 作为 U-Net 编码器
        self.swin_unet = SwinTransformerSys(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=1,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            depths_decoder=config.MODEL.SWIN.DECODER_DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

        # 2️⃣ 使用 Tanh 作为最终激活函数（可以改为 ReLU）
        self.final_activation = nn.Tanh()

    def forward(self, x):
        """
        输入 x: [batch_size, channels, height, width]
        输出: self.final_activation(logits) 与输入 x 的第二个通道（channel=1）的残差连接
        """
        logits = self.swin_unet(x)  # 通过 Swin Transformer U-Net 生成预测结果
        logits = self.final_activation(logits)  # 经过激活函数

        # 3️⃣ 残差连接：取输入的第二个通道，并与 logits 相加
        residual = x[:, 0:1, :, :]  # 取输入的第二个通道，保持维度 [batch, 1, H, W]
        output = logits + residual  # 残差连接

        return output
