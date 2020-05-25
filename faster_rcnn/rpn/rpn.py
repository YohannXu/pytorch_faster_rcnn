# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-06 16:17:44
# Description: RPN流程部分

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
import math
from easydict import EasyDict
from .anchor_generator import AnchorGenerator, filter_anchors
from .head import RPNHead
from .loss import RPNLoss
from torchvision.ops import nms


class RPN(nn.Module):
    """
    执行RPN流程
    """

    @type_check(object, EasyDict, bool)
    def __init__(self, cfg, is_train=True):
        super(RPN, self).__init__()
        self.anchor_generator = AnchorGenerator(cfg)
        self.rpn_head = RPNHead(cfg)
        self.loss = RPNLoss(cfg)
        self.cfg = cfg
        self.post_nms_top_n = self.cfg.RPN.POST_NMS_TOP_N_TRAIN if self.training else self.cfg.RPN.POST_NMS_TOP_N_TEST
        self.pre_nms_top_n = self.cfg.RPN.PRE_NMS_TOP_N_TRAIN if self.training else self.cfg.RPN.PRE_NMS_TOP_N_TEST
        self.post_top_n = self.cfg.RPN.POST_TOP_N_TRAIN if self.training else self.cfg.RPN.POST_TOP_N_TEST
        self.nms_threshold = self.cfg.RPN.NMS_THRESHOLD

    @type_check(object, torch.Tensor, int)
    def permute(self, t, C):
        """
        调整概率预测张量或偏移量预测张量的形状,使其顺序与anchors顺序相同
        Args:
            t: Tensor, 假定形状为N x A * C x H x W
            C: 目标通道数,当输入张量为概率预测张量时,C取1;当输入张量为偏移量预测张量时,C取4
        Return:
            t: Tensor, N x H * W * A x C
        """
        N, _, H, W = t.shape
        t = t.view(N, -1, C, H, W)  # N x A x C x H x W
        t = t.permute(0, 3, 4, 1, 2)  # N x H x W x A x C
        t = t.reshape(N, -1, C)  # N x H * W * A x C
        return t

    @type_check(object, torch.Tensor, torch.Tensor)
    def decode(self, anchors, offsets):
        """
        根据预测offsets对anchors进行调整,得到proposals
        Args:
            anchors: Tensor, N * top_n x 4
            offsets: Tensor, N * top_n x 4
        Return:
            proposals: Tensor, N * top_n x 4
        """
        # offset最大阈值
        max_offset = math.log(1000. / 16)
        anchors = anchors.to(offsets.dtype)

        # 计算anchors的宽、高及中心点坐标
        ws = anchors[:, 2] - anchors[:, 0] + 1
        hs = anchors[:, 3] - anchors[:, 1] + 1
        x_ctr = anchors[:, 0] + 0.5 * ws
        y_ctr = anchors[:, 1] + 0.5 * hs

        wx, wy, ww, wh = self.cfg.RPN.BBOX_REG_WEIGHTS

        dx = offsets[:, 0] / wx
        dy = offsets[:, 1] / wy
        dw = offsets[:, 2] / ww
        dh = offsets[:, 3] / wh

        dw = dw.clamp(max=max_offset)
        dh = dh.clamp(max=max_offset)

        # 进行调整
        x_ctr = x_ctr + dx * ws
        y_ctr = y_ctr + dy * hs
        ws = ws * torch.exp(dw)
        hs = hs * torch.exp(dh)

        # xywh -> xyxy
        x1 = x_ctr - 0.5 * ws
        y1 = y_ctr - 0.5 * hs
        x2 = x_ctr + 0.5 * ws - 1
        y2 = y_ctr + 0.5 * hs - 1

        proposals = torch.stack([x1, y1, x2, y2], dim=1)

        return proposals

    @type_check(object, torch.Tensor, torch.Tensor, int)
    def iou(self, area, proposals, index):
        """
        计算某一个proposal和之后所有proposals之间的iou
        Args:
            area: Tensor, 提前计算好的所有proposals的面积, 避免重复计算, 假定形状为num x 4
            proposals: Tensor, 假定形状为num x 4
            index: int, 特定proposal的索引
        Return:
            iou: Tensor, num
        """
        x_tl = torch.max(proposals[index, 0], proposals[index + 1:, 0])
        y_tl = torch.max(proposals[index, 1], proposals[index + 1:, 1])
        x_br = torch.min(proposals[index, 2], proposals[index + 1:, 2])
        y_br = torch.min(proposals[index, 3], proposals[index + 1:, 3])

        inter = (x_br - x_tl + 1).clamp(0) * (y_br - y_tl + 1).clamp(0)
        iou = inter / (area[index] + area[index + 1] - inter)

        return iou

    @type_check(object, torch.Tensor, torch.Tensor)
    def nms(self, proposals, logits):
        """
        执行NMS, 保留一定数量的proposals, 返回其对应索引
        ps: 速度慢，未使用
        Args:
            proposals: Tensor, 假定大小为N x 4
            logits: Tensor, 假定大小为N
        Return:
            proposals: Tensor, top_n x 4
            logits: Tensor, top_n
        """
        # 根据logits排序
        sorted_inds = logits.argsort(descending=True)
        proposals = proposals[sorted_inds]
        logits = logits[sorted_inds]

        # 计算所有proposals的面积
        area = (proposals[:, 2] - proposals[:, 0] + 1) * (proposals[:, 3] - proposals[:, 1] + 1)

        # NMS流程
        keep = []
        # 遍历索引
        num_proposals = proposals.shape[0]
        for i in range(num_proposals):
            if logits[i] > 0:
                if i != num_proposals - 1:
                    ious = self.iou(area, proposals, i)
                    logits[i + 1:][ious > self.threshold] = -1
                keep.append(i)
        keep = torch.LongTensor(keep)

        # 取前N个
        post_nms_top_n = min(self.post_nms_top_n, keep.shape[0])
        keep = keep[:post_nms_top_n]

        return proposals[keep], logits[keep]

    @type_check(object, tuple, torch.Tensor, torch.Tensor, torch.Tensor)
    def forward_for_feature_map(self, anchors, logits, bbox_regs, sizes):
        """
        处理FPN中一层feature map对应的anchors以及概率预测和回归预测
        Args:
            anchors: tuple(Tensor), 长度为N, 假定每个tensor的形状为 H * W * A x 4
            logits: Tensor, 假定大小为N x A x H x W
            bbox_regs: Tensor, 假定大小为N x A * 4 x H x W
            sizes: Tensor, 原始图片尺寸, 假定大小为N x 2
        Return:
            list[tuple(proposal, logit)]
                proposal: Tensor, top_n x 4
                logit: Tensor, top_n x 4
        """
        device = logits.device
        N, A, H, W = logits.shape
        # 调整概率预测和回归预测的形状, 使其顺序与anchors顺序相同
        logits = self.permute(logits, 1).view(N, -1)  # N x A x H x W -> N x H * W * A x 1 -> N x H * W * A
        bbox_regs = self.permute(bbox_regs, 4)  # N x A * 4 x H x W -> N x H * W * A x 4

        # 对每张图片, 根据概率选出前N个概率预测及其索引
        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        logits = logits.sigmoid()
        logits, topk_inds = logits.topk(pre_nms_top_n, dim=1, sorted=True)  # N x top_n, N x top_n

        # 根据索引得到对应回归预测
        batch_inds = torch.arange(N, device=device)[:, None]
        bbox_regs = bbox_regs[batch_inds, topk_inds]  # N x top_n x 4

        # 根据索引得到对应anchors
        anchors = torch.cat(anchors, dim=0).view(N, -1, 4)  # N * H * W * A x 4 -> N x H * W * A x 4
        anchors = anchors[batch_inds, topk_inds]  # N x top_n x 4
        anchors = anchors.to(device)

        # 得到proposals
        proposals = self.decode(anchors.view(-1, 4), bbox_regs.view(-1, 4))  # N * top_n x 4
        proposals = proposals.view(N, -1, 4)  # N x top_n x 4

        # 使用列表存储结果, 因为不同图片保留下来的proposals数量可能不同
        # 所以不能拼接为一个张量返回
        results = []
        for proposal, logit, size in zip(proposals, logits, sizes):
            # 将proposals裁剪到图片范围内
            proposal[:, 0].clamp_(0, size[1] - 1)
            proposal[:, 1].clamp_(0, size[0] - 1)
            proposal[:, 2].clamp_(0, size[1] - 1)
            proposal[:, 3].clamp_(0, size[0] - 1)

            keep = nms(proposal, logit, self.nms_threshold)
            post_nms_top_n = min(self.post_nms_top_n, keep.shape[0])
            keep = keep[:post_nms_top_n]
            proposal = proposal[keep]
            logit = logit[keep]

            # 速度慢，未使用
            # proposal, logit = self.nms(proposal, logit, self.nms_threshold)

            results.append((proposal, logit))
        return results

    @type_check(object, list, list, list, torch.Tensor, list)
    def inference(self, anchors, logits, bbox_regs, sizes, targets=None):
        """
        rpn部分的inference阶段
        在anchors基础上经过计算得到proposals
        Args:
            anchors: list[list[Tensor]], 外层长度为batch size, 内层长度为FPN层数, 假定大小为H * W * A x 4
            logits: list[Tensor], 长度为FPN层数, 假定大小为N x A x H x W
            bbox_regs: list[Tensor], 长度为FPN层数, 假定大小为N x A * 4 x H x W
            sizes: Tensor, 图片缩放后尺寸, 假定大小为N x 2
            targets: Tensor, 假定大小为N x M x 4, M为一个batch中真实检测框的最大数目
        Return:
            concated_proposals: list[Tensor]
        """
        # 调整anchors顺序
        anchors = list(zip(*anchors))  # list[tuple(Tensor)], 外层长度为FPN层数, 内层长度为batch size

        # 遍历FPN每一层对应的anchor, 概率预测和回归预测
        # 得到proposals和对应logits
        # list[list[tuple(proposal, logit)]], 外层长度为FPN层数, 内层长度为batch size
        results = []
        for anchor, logit, bbox_reg in zip(anchors, logits, bbox_regs):
            result = self.forward_for_feature_map(anchor, logit, bbox_reg, sizes)
            results.append(result)

        # 调整results顺序
        results = list(zip(*results))  # list[tuple(tuple(proposal, logit))], 外层长度为batch size, 内层长度为FPN层数

        # 将同一图片不同FPN层数的proposals和logits拼接为一个张量
        # 因为不同图片保留下来的proposals数量可能不同, 因此用list保存结果
        concated_proposals = []
        concated_logits = []
        for result in results:
            proposal_list = []
            logit_list = []
            for proposal, logit in result:
                proposal_list.append(proposal)
                logit_list.append(logit)
            proposal = torch.cat(proposal_list, dim=0)
            logit = torch.cat(logit_list, dim=0)
            concated_proposals.append(proposal)
            concated_logits.append(logit)

        # 为每张图片保留概率较高的proposals
        if self.cfg.RPN.POST_PER_BATCH and self.training:
            # 在整个batch上保留前N个
            # 每张图片保留下来的proposals数量
            num_proposals = [proposal.shape[0] for proposal in concated_proposals]
            all_logits = torch.cat(concated_logits)
            post_top_n = min(all_logits.shape[0], self.post_top_n)
            # 得到前N个索引
            _, inds = all_logits.topk(post_top_n, sorted=True)
            # 索引转为二值化mask
            top_inds = torch.zeros_like(all_logits, dtype=torch.bool)
            top_inds[inds] = True
            # 将索引划分到每张图片
            top_inds = top_inds.split(num_proposals)
            # 根据索引保留对应proposals
            for i in range(len(concated_proposals)):
                concated_proposals[i] = concated_proposals[i][top_inds[i]]
        else:
            # 每张图片保留前N个
            for i in range(len(concated_proposals)):
                post_top_n = min(concated_proposals[i].shape[0], self.post_top_n)
                _, top_inds = concated_logits[i].topk(post_top_n, sorted=True)
                concated_proposals[i] = concated_proposals[i][top_inds]

        # 如果是训练模式, 则将每张图片的target添加到对应的proposals中
        # 这是为了防止在roi部分进行target和proposals匹配时出现所有proposals和所有target的iou均为0, 导致训练无法正常进行
        if self.training:
            for i, target in enumerate(targets):
                concated_proposals[i] = torch.cat([concated_proposals[i], target], dim=0)

        return concated_proposals

    @type_check(object, list, torch.Tensor, list)
    def forward(self, features, sizes, targets=None):
        """
        Args:
            features: list[Tensor], backbone输出, 长度为FPN层数, 假定大小为N x C x H x W
            sizes: Tensor, 图片缩放后尺寸, 假定大小为N x 2
            targets: Tensor, 真实检测框, 假定大小为N x M x 4
        Return:
            proposals: list[Tensor], RPN部分得到的proposals, 长度为N
            loss: dict{'rpn_cls_loss', 'rpn_reg_loss'}, rpn部分的分类损失和回归损失
        """
        # 生成anchors
        # list[list[Tensor]], 外层长度为batch size, 内层长度为FPN层数
        anchors = self.anchor_generator(features)

        # 得到概率预测和回归预测
        # list[Tensor], N x A x H x W 和 N x A * 4 x H x W
        logits, bbox_regs = self.rpn_head(features)

        with torch.no_grad():
            proposals = self.inference(anchors, logits, bbox_regs, sizes, targets)

        if self.training:
            loss = self.loss(anchors, logits, bbox_regs, sizes, targets)
        else:
            loss = {}

        return proposals, loss
