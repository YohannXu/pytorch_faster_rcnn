# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-27 19:27:33
# Description: 参数固定的BatchNorm2d

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

class BatchNorm2d(nn.Module):
    """
    参数固定的BatchNorm2d
    因为在进行目标检测任务训练时,对显存占用大,因此batch size较小,bn没法有效训练,因此固定
    """

    @type_check(object, int)
    def __init__(self, num_channels):
        """
        Args:
            num_channels: int, 通道数
        """
        super(BatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(num_channels))
        self.register_buffer('bias', torch.zeros(num_channels))
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

    @type_check(object, torch.Tensor)
    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
