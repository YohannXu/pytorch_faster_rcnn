# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-31 16:35:21
# Description: model.py

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from faster_rcnn.backbone import ResNet
from faster_rcnn.rpn import RPN
from faster_rcnn.rois.box_roi import BoxRoI
from faster_rcnn.utils import type_check
from easydict import EasyDict
from apex import amp


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
