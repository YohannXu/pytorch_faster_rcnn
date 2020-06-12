# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-31 16:35:21
# Description: model.py

import torch
import torch.nn as nn
from easydict import EasyDict

from faster_rcnn.backbone import ResNet
from faster_rcnn.rois.box_roi import BoxRoI
from faster_rcnn.rpn import RPN
from faster_rcnn.utils import type_check


class Model(nn.Module):
    """
    Faster RCNN 目标检测模型
    """
    @type_check(object, EasyDict, bool, bool)
    def __init__(self, cfg, pretrained=True, is_train=True):
        super(Model, self).__init__()
        self.backbone = ResNet(cfg, pretrained)
        self.rpn = RPN(cfg, is_train)
        self.roi = BoxRoI(cfg, is_train)
        # 如果使用混合精度, 需要手动将其转换为半精度运算
        # self.roi.head.pool.forward = amp.half_function(self.roi.head.pool.forward)

    @type_check(object, torch.Tensor, torch.Tensor, list, list)
    def forward(self, images, sizes=None, cats=None, bboxes=None):
        features = self.backbone(images)

        proposals, rpn_loss = self.rpn(features, sizes, bboxes)

        if self.training:
            roi_loss = self.roi(features, proposals, sizes, cats, bboxes)
            return rpn_loss, roi_loss
        else:
            boxes, probs, labels = self.roi(features, proposals, sizes)
            return boxes, probs, labels
