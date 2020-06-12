# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-19 14:59:00
# Description: roi概率预测与回归预测部分

import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from ...utils import type_check
from .pool import Pool


class RoIHead(nn.Module):
    """
    得到rpn阶段每个proposal的概率预测和回归预测
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(RoIHead, self).__init__()
        self.cfg = cfg
        resolution = self.cfg.ROI_BOX.RESOLUTION
        self.pool = Pool(cfg)

        in_channel = self.cfg.FPN.OUT_CHANNEL
        hidden_dim = self.cfg.ROI_BOX.HIDDEN_DIM
        self.fc1 = nn.Linear(in_channel * resolution ** 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        num_classes = self.cfg.ROI_BOX.NUM_CLASSES
        self.cls_layer = nn.Linear(hidden_dim, num_classes)
        self.reg_layer = nn.Linear(hidden_dim, num_classes * 4)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.cls_layer.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_layer.bias, 0)
        nn.init.normal_(self.reg_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_layer.bias, 0)

    @type_check(object, list, list)
    def forward(self, features, proposals):
        """
        Args:
            features: list[Tensor], 长度为FPN层数, 假定大小为N x C x H x W
            proposals: list[Tensor], 长度为FPN层数, 假定大小为n x 4
        Return:
            logits: Tensor, num x num_classes, num为所有proposals的数量
            bbox_regs: Tensor, num x num_classes * 4
        """
        # 对proposals进行roi align
        x = self.pool(features, proposals)  # num x C x resolution x resolution
        x = x.view(x.shape[0], -1)  # num x C * resolution ** 2
        x = F.relu(self.fc1(x))  # num x hidden_dim
        x = F.relu(self.fc2(x))  # num x hidden_dim

        logits = self.cls_layer(x)  # num x num_classes
        bbox_regs = self.reg_layer(x)  # num x num_classes * 4

        return logits, bbox_regs
