# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-19 14:59:10
# Description: 池化流程

import torch
import torch.nn as nn
from easydict import EasyDict
from torchvision.ops import RoIAlign

from ...utils import type_check


class Pool(nn.Module):
    """
    RoIAlign流程实现
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(Pool, self).__init__()
        self.cfg = cfg
        # roi_align输出尺寸
        self.resolution = self.cfg.ROI_BOX.RESOLUTION
        # backbone输出的的降采样系数
        ratios = self.cfg.ROI_BOX.RATIOS
        # 插值时的采样点数量
        sample_ratio = self.cfg.ROI_BOX.SAMPLE_RATIO

        # 对不同降采样系数的输入, 使用不同的RoiAlign
        for index, ratio in enumerate(ratios):
            name = 'roi_align_{}'.format(index)
            roi_align = RoIAlign(
                (self.resolution, self.resolution),
                ratio,
                sample_ratio
            )
            self.add_module(name, roi_align)

        self.num_level = len(ratios)
        self.level_min = -torch.log2(torch.tensor(ratios[0])).item()
        self.level_max = -torch.log2(torch.tensor(ratios[-1])).item()
        self.base_size = 224
        self.base_level = 4
        self.eps = 1e-6

    @type_check(object, torch.Tensor)
    def set_level(self, rois):
        """
        根据每个roi的面积为每个roi分配不同的RoIAlign
        Args:
            rois: Tensor, 假定大小为N x 5
        Return:
            levels: Tensor, N
        """
        # 计算面积
        areas = (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1)
        # 计算边长
        sizes = areas.sqrt()

        levels = torch.floor(self.base_level + torch.log2(sizes / self.base_size + self.eps))
        levels = levels.clamp(min=self.level_min, max=self.level_max).to(torch.int64) - self.level_min

        return levels

    @type_check(object, list)
    def convert(self, proposals):
        """
        将不同图片的proposals拼接为Tensor
        Args:
            proposals, list[Tensor], 长度为batch size, 假定大小为N x 4
        Return:
            rois: Tensor, num x 5, num为所有张量长度之和
        """
        concated_proposals = torch.cat(proposals, dim=0)
        device = concated_proposals.device
        dtype = concated_proposals.dtype
        # 每个proposals对应的图片索引
        inds = torch.cat(
            [
                torch.full((len(proposal), 1), i, device=device, dtype=dtype) for i, proposal in enumerate(proposals)
            ],
            dim=0
        )
        # 将索引与proposals连接
        rois = torch.cat([inds, concated_proposals], dim=1)

        return rois

    @type_check(object, list, list)
    def forward(self, features, proposals):
        """
        执行RoIAlign, 将不定尺寸的proposals变为固定尺寸大小
        Args:
            features: list[Tensor], 长度为FPN层数, 假定大小为N x C x H x W
            proposals: list[Tensor], 长度为N, 假定大小为n x 4
        Return:
            results: Tensor, num x C x resolution x resolution, num为所有proposals的数量
        """
        rois = self.convert(proposals)
        levels = self.set_level(rois)

        num_rois = rois.shape[0]
        num_channel = features[0].shape[1]
        device = rois.device
        dtype = rois.dtype

        results = torch.zeros(
            (num_rois, num_channel, self.resolution, self.resolution),
            dtype=dtype,
            device=device
        )

        # FPN共有5层输出, 这里只使用了前4层
        for level_id, feature in enumerate(features[:-1]):
            inds = torch.nonzero(levels == level_id).squeeze(1)
            roi_align = getattr(self, 'roi_align_{}'.format(level_id))
            roi = rois[inds]
            results[inds] = roi_align(feature, roi)

        return results
