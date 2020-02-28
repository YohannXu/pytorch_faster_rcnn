# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-27 19:27:39
# Description: FPN模块

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


class FPN(nn.Module):
    """
    特征金字塔,融合低层高细节低语义和高层低细节高语义特征,进行更好的预测
    """

    @type_check(str)
    def __init__(self, cfg):
        """
        Args:
            cfg: str, 配置文件
        """
        super(FPN, self).__init__()
        self.in_channels = cfg.FPN.IN_CHANNELS
        self.out_channel = cfg.FPN.OUT_CHANNEL
        self.inner_names = []
        self.layer_names = []

        for index, in_channel in enumerate(self.in_channels, 1):
            inner_name = 'fpn_inner_{}'.format(index)
            layer_name = 'fpn_layer_{}'.format(index)

            inner_block = nn.Conv2d(in_channel, self.out_channel, 1)
            layer_block = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1)
            nn.init.kaiming_normal_(inner_block.weight, a=1)
            nn.init.kaiming_normal_(layer_block.weight, a=1)
            nn.init.constant_(inner_block.bias, 0)
            nn.init.constant_(layer_block.bias, 0)
            self.add_module(inner_name, inner_block)
            self.add_module(layer_name, layer_block)
            self.inner_names.append(inner_name)
            self.layer_names.append(layer_name)
        self.last_block = nn.MaxPool2d(1, 2, 0)

    @type_check(list)
    def forward(self, x):
        last_inner = getattr(self, self.inner_names[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_names[-1])(last_inner))
        for feature, inner_name, layer_name in zip(x[:-1][::-1], self.inner_names[:-1][::-1], self.layer_names[:-1][::-1]):
            top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest')
            lateral = getattr(self, inner_name)(feature)
            last_inner = top_down + lateral
            results.insert(0, getattr(self, layer_name)(last_inner))
        results.append(self.last_block(results[-1]))

        return results
