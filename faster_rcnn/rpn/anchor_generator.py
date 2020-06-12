# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-06 16:18:04
# Description: rpn网络生成anchors

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

from ..utils import type_check


class AnchorGenerator(nn.Module):
    """
    用于生成anchors
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(AnchorGenerator, self).__init__()
        # anchors步长
        self.strides = cfg.RPN.STRIDES
        # anchors尺寸
        self.sizes = cfg.RPN.SIZES
        # anchors长宽比
        self.ratios = cfg.RPN.RATIOS

        assert len(self.strides) > 0, 'len(strides) should be larger than zero'
        assert len(self.sizes) > 0, 'len(sizes) should be larger than zero'
        assert len(self.ratios) > 0, 'len(ratios) should be larger than zero'

        # 确定在feature map的每个像素值位置上需要生成的anchors数量
        if len(self.strides) == 1:
            self.num_anchors_per_location = len(self.sizes) * len(self.ratios)
        else:
            assert len(self.strides) == len(self.sizes), 'len(strides) should equal to len(sizes) when use FPN'
            self.num_anchors_per_location = len(self.ratios)

        # 先生成基础anchors
        if len(self.strides) == 1:
            self.cell_anchors = [self.base_anchors(self.strides[0], self.sizes, self.ratios)]
        else:
            self.cell_anchors = [self.base_anchors(stride, [size], self.ratios) for stride, size in zip(self.strides, self.sizes)]

    @type_check(object, int, list, list)
    def base_anchors(self, stride, sizes, ratios):
        """
        根据传入的步长及尺寸生成基础anchors
        Args:
            stride: int, anchors步长
            sizes: list, anchors尺寸, 假定长度为N
            ratios: list, anchors长宽比, 假定长度为M
        Return:
            anchors: Tensor, M * N x 4
        """
        # TODO 这部分代码由maskrcnn_benchmark代码转换而来
        # 因为最后生成的anchors的中心点坐标为((stride-1)/2, (stride-1)/2),边长分别为sizes/sqrt(ratios)和sizes*sqrt(ratios),所以是否可以省略中间的计算过程,直接得到结果
        sizes = np.array(sizes)  # N
        ratios = np.array(ratios)  # M
        scale = sizes / stride  # N
        x_ctr, y_ctr = (stride - 1) / 2, (stride - 1) / 2
        area = stride * stride
        ws = np.round(np.sqrt(area / ratios))  # M
        hs = np.round(ws * ratios)  # M
        ws = ws[:, np.newaxis] * scale[np.newaxis, :]  # M x N
        hs = hs[:, np.newaxis] * scale[np.newaxis, :]  # M x N
        ws = ws.reshape(-1, 1)  # M * N
        hs = hs.reshape(-1, 1)  # M * N
        x1 = x_ctr - (ws - 1) / 2  # M * N
        y1 = y_ctr - (hs - 1) / 2  # M * N
        x2 = x_ctr + (ws - 1) / 2  # M * N
        y2 = y_ctr + (hs - 1) / 2  # M * N
        anchors = np.hstack([x1, y1, x2, y2])  # M * N x 4
        anchors = torch.from_numpy(anchors).float()
        return anchors

    @type_check(object, torch.Size, int, torch.Tensor)
    def grid_anchors(self, grid_size, stride, anchor):
        """
        将base_anchors在整个feature map上进行滑动,得到所有anchors
        Args:
            grid_size: list, feature map尺寸,假定为[h, w]
            stride: int, anchors步长
            anchor: Tensor, 基础anchors, 假定大小为A x 4
        Return:
            anchors: Tensor, 所有的anchors, w * h * A x 4
        """
        grid_height, grid_width = grid_size
        shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32)  # w
        shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32)  # h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)  # h x w, h x w
        shift_x = shift_x.reshape(-1)  # h * w
        shift_y = shift_y.reshape(-1)  # h * w
        # 得到所有偏移量
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)  # h * w x 4
        anchors = shifts.view(-1, 1, 4) + anchor.view(1, -1, 4)  # h * w x A x 4
        anchors = anchors.reshape(-1, 4)  # w * h * A x 4
        return anchors

    @type_check(object, list)
    def forward(self, features):
        """
        根据是否使用FPN来生成anchors
        若未使用FPN,只输入1层feature map
        若使用FPN,则输入5层feature map
        Args:
            features: list[Tensor], 输入feature map
        Return:
            anchors: list[Tensor], 输出anchors
        """
        batch_size = features[0].shape[0]
        grid_sizes = [feature.shape[2:] for feature in features]
        # 在feature map上进行滑动
        if len(self.strides) == 1:
            anchors = [[self.grid_anchors(grid_size, self.strides[0], self.cell_anchors[0]) for grid_size in grid_sizes]] * batch_size
        else:
            anchors = [[self.grid_anchors(grid_size, stride, anchor)
                       for grid_size, stride, anchor in zip(grid_sizes, self.strides, self.cell_anchors)]] * batch_size

        return anchors


@type_check(torch.Tensor, torch.Tensor)
def filter_anchors(anchors, size):
    """
    去除位于图片范围外的anchors,返回一个mask
    Args:
        anchors, Tensor, 输入anchors
        size: Tensor, 图片原始尺寸
    """
    height, width = size
    inds_inside = (
        (anchors[..., 0] >= 0)
        & (anchors[..., 1] >= 0)
        & (anchors[..., 2] < width)
        & (anchors[..., 3] < height)
    )
    return inds_inside
