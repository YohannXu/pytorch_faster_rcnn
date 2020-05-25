# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-06 16:17:59
# Description: RPN的预测部分

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import type_check
from easydict import EasyDict


class RPNHead(nn.Module):
    """
    对每个anchors进行class预测和offset预测
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(RPNHead, self).__init__()
        strides = cfg.RPN.STRIDES
        sizes = cfg.RPN.SIZES
        ratios = cfg.RPN.RATIOS
        in_channel = cfg.FPN.OUT_CHANNEL

        # 每个位置上的anchors数量
        if len(strides) == 1:
            self.num_anchors_per_location = len(sizes) * len(ratios)
        else:
            self.num_anchors_per_location = len(ratios)

        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channel, self.num_anchors_per_location, kernel_size=1, stride=1, padding=0)
        self.reg_layer = nn.Conv2d(in_channel, self.num_anchors_per_location * 4, kernel_size=1, stride=1, padding=0)

        for l in [self.conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @type_check(object, list)
    def forward(self, features):
        """
        Args:
            features: list[Tensor], 假定每个Tensor的大小为N x C x H x W
        Return:
            令A = self.num_anchors_per_location
            logits: list[Tensor], N x A x H x W
            bbox_regs: list[Tensor], N x A * 4 x H x W
        """
        logits = []
        bbox_regs = []
        for feature in features:
            x = F.relu(self.conv(feature))
            logits.append(self.cls_layer(x))
            bbox_regs.append(self.reg_layer(x))
        return logits, bbox_regs
