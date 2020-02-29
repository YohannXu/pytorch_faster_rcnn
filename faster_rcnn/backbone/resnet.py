# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-26 22:31:20
# Description: 构建resnet

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from ..utils import type_check, load_state_dict_from_url
from .fpn import FPN
from .batch_norm import BatchNorm2d


@type_check(int, int, int)
def conv1x1(in_planes, out_planes, stride=1):
    """
    卷积核尺寸为1的卷积函数
    Args:
        in_planes: int, 输入通道数
        out_planes: int, 输出通道数
        stride: int, 卷积核步长,默认为1,当需要降采样时为2
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@type_check(int, int, int)
def conv3x3(in_planes, out_planes, stride=1):
    """
    卷积核尺寸为3的卷积函数
    Args:
        in_planes: int, 输入通道数
        out_planes: int, 输出通道数
        stride: int, 卷积核步长,默认为1,当需要降采样时为2
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BottleNeck(nn.Module):
    """
    Resnet网络基本单元
    若不需要降采样,网络结构为:         若需要降采样,网络结构为:
    input ---       in_planes          input -------       in_planes
      |     |                            |         |
      |  conv1x1    planes               |      conv1x1    planes
      |     |                            |         |
      |  conv3x3    planes            conv1x1   conv3x3    planes
      |     |                            |         |
      |  conv1x1    planes x 4           |      conv1x1    planes x 4
      |     |                            |         |
      + -----                            +----------
    output          planes x 4         output              planes x 4
    由于目标检测的训练过程中,batch size通常只有2~4,因此会影响bn的训练,故加载预训练模型参数后,就固定bn参数
    """

    @type_check(object, int, int, int, nn.Module)
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Args:
            in_planes: int, 输入通道数
            planes: int, 中间层通道数
            stride: int, 卷积步长,默认为1,当需要降采样时为2
            downsample: nn.Module, 用于降采样,默认为None,需要降采样时传入对应降采样模块
        """
        super(BottleNeck, self).__init__()
        # 输出通道数为中间层通道数的4倍
        self.expansion = 4
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    @type_check(object, torch.Tensor)
    def forward(self, x):
        shortcut = x

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)
        return out


class Stem(nn.Module):
    """
    ResNet的前端部分,由一个较大的卷积核及最大池化组成
    """

    def __init__(self):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        nn.init.kaiming_normal_(self.conv.weight, a=1)

    @type_check(object, torch.Tensor)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class ResNet(nn.Module):
    """
    ResNet网络,由一个stem和4个stage构成
    """

    @type_check(object, EasyDict, bool)
    def __init__(self, cfg, pretrained=True):
        """
        Args:
            cfg: EasyDict, config文件
            pretrained: bool, 是否使用fpn
        """
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.use_fpn = cfg.FPN.ON
        num_layers = cfg.BACKBONE.NUM_LAYERS
        suffix = cfg.BACKBONE.SUFFIX
        assert num_layers in [50, 101, 152], 'num_layers should be in [50, 101, 152]'
        if num_layers == 50:
            self.num_layers = [3, 4, 6, 3]
        elif num_layers == 101:
            self.num_layers = [3, 4, 23, 3]
        elif num_layers == 152:
            self.num_layers = [3, 8, 36, 3]

        self.stem = Stem()

        self.layer1 = self.make_stage(0)
        self.layer2 = self.make_stage(1)
        self.layer3 = self.make_stage(2)
        self.layer4 = self.make_stage(3)

        # 冻结部分参数
        for layer_id in range(self.cfg.BACKBONE.NUM_FROZEN):
            if layer_id == 0:
                m = self.stem
            else:
                m = getattr(self, 'layer{}'.format(layer_id))

            for p in m.parameters():
                p.requires_grad = False

        # 构建FPN模块
        if self.use_fpn:
            self.fpn = FPN(self.cfg)

        # 加载预训练权重
        if pretrained:
            print('loading pretrained weights')
            weights = load_state_dict_from_url(num_layers, suffix)
            self.load_state_dict(weights, strict=False)

    @type_check(object, int)
    def make_stage(self, stage_id):
        """
        构建4个stage
        每个stage的第一个BottleNeck都包含降采样,但stage1只需要增加通道数,其他stage既需要尺寸减半也需要增加通道数
        Args:
            stage_id: int, stage索引
        """
        layers = []

        if stage_id == 0:
            downsample = nn.Sequential(
                conv1x1(64, 256),
                BatchNorm2d(256)
            )
            layers.append(BottleNeck(64, 64, 1, downsample))
            for _ in range(self.num_layers[stage_id] - 1):
                layers.append(BottleNeck(256, 64))
        else:
            plane = 128 * 2 ** (stage_id - 1)
            downsample = nn.Sequential(
                conv1x1(2 * plane, 4 * plane, 2),
                BatchNorm2d(4 * plane)
            )
            layers.append(BottleNeck(2 * plane, plane, 2, downsample))
            for _ in range(self.num_layers[stage_id] - 1):
                layers.append(BottleNeck(4 * plane, plane))
        return nn.Sequential(*layers)

    @type_check(object, torch.Tensor)
    def forward(self, x):
        x = self.stem(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.use_fpn:
            x = self.fpn([x1, x2, x3, x4])
            return x
        else:
            return [x4]
