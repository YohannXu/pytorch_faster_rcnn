# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-19 14:59:05
# Description: RoI部分的损失函数

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import type_check
from easydict import EasyDict


def smooth_l1_loss(pred, target, beta=1./9, size_average=True):
    n = torch.abs(pred - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


class RoILoss(nn.Module):
    """
    RoI部分的损失函数计算流程
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(RoILoss, self).__init__()
        self.cfg = cfg

    @type_check(object, torch.Tensor, torch.Tensor)
    def match(self, proposals, targets):
        """
        将proposals和targets进行匹配
        Args:
            proposals: Tensor, 假定大小为N x 4
            targets: Tensor, 假定大小为M x 4
        Return:
            inds: Tensor, N
        """
        # 先计算proposals与targets之间的iou
        x_tl = torch.max(proposals[:, 0][:, None], targets[:, 0][None, :])
        y_tl = torch.max(proposals[:, 1][:, None], targets[:, 1][None, :])
        x_br = torch.min(proposals[:, 2][:, None], targets[:, 2][None, :])
        y_br = torch.min(proposals[:, 3][:, None], targets[:, 3][None, :])
        inter = (x_br - x_tl + 1).clamp(0) * (y_br - y_tl + 1).clamp(0)
        area1 = (proposals[:, 2] - proposals[:, 0] + 1) * (proposals[:, 3] - proposals[:, 1] + 1)
        area2 = (targets[:, 2] - targets[:, 0] + 1) * (targets[:, 3] - targets[:, 1] + 1)
        ious = inter / (area1[:, None] + area2[None, :] - inter)

        # 前景背景阈值
        fg_threshold = self.cfg.ROI_BOX.FG_IOU_THRESHOLD
        bg_threshold = self.cfg.ROI_BOX.BG_IOU_THRESHOLD

        # 先得到每个proposal对应iou最高的target的索引和iou值
        vals, inds = ious.max(dim=1)  # N
        ori_inds = inds.clone()
        # iou值低于背景阈值的proposal为负样本
        inds[vals < bg_threshold] = -1
        # iou值位于两个阈值之间的proposal会被丢弃
        inds[(vals >= bg_threshold) & (vals < fg_threshold)] = -2

        # 有时可能会存在所有proposal均为负样本的情况
        # 因此还需要得到每个target对应iou最高的proposal的索引
        # 将对应的proposal标记为正样本,保证训练能够正常进行
        # 得到每个target对应最高iou值
        vals, _ = ious.max(dim=0)  # M
        # 找出每个target对应最高iou值对应的proposals索引
        # 因为对同一个target,可能存在多个proposals,都具有最高iou值
        # 所以需要通过这种方式得到索引
        post_inds = torch.nonzero(ious.permute(1, 0) == vals[:, None])[:, 1]
        # 对索引对应的proposals的正负样本划分进行更新
        inds[post_inds] = ori_inds[post_inds]

        return inds

    @type_check(object, torch.Tensor)
    def sample(self, matches):
        """
        采样出用于计算损失的正负proposals索引
        Args:
            matches: Tensor
        Return:
            pos_inds: Tensor, 正proposals索引
            neg_inds: Tensor, 负proposals索引
        """
        # 得到所有正负proposals索引
        pos_inds = torch.nonzero(matches >= 0).squeeze(1)
        neg_inds = torch.nonzero(matches == -1).squeeze(1)

        # 得到训练所需proposals数量和正proposals数量
        num_per_image = self.cfg.ROI_BOX.NUM_PER_IMAGE
        positive_ratio = self.cfg.ROI_BOX.POSITIVE_RATIO
        num_pos = int(num_per_image * positive_ratio)

        # 如果实际正proposals数量大于所需正proposals数量,则进行采样
        if pos_inds.shape[0] > num_pos:
            perm = torch.randperm(pos_inds.shape[0], device=pos_inds.device)[:num_pos]
            pos_inds = pos_inds[perm]
        else:
            num_pos = pos_inds.shape[0]

        # 计算训练所需负proposals数量
        num_neg = num_per_image - num_pos
        # 如果实际负proposals数量大于所需负proposals数量,则进行采样
        if neg_inds.shape[0] > num_neg:
            perm = torch.randperm(neg_inds.shape[0], device=neg_inds.device)[:num_neg]
            neg_inds = neg_inds[perm]
        else:
            num_neg = neg_inds.shape[0]

        return pos_inds, neg_inds

    @type_check(object, torch.Tensor, torch.Tensor)
    def encode(self, proposals, targets):
        """
        计算正proposals与对应targets之间的真实偏移量
        Args:
            anchors: Tensor, 假定大小为N x 4
            targets: Tensor, 假定大小为N x 4
        Return:
            offsets: Tensor, N x 4
        """
        ws = proposals[:, 2] - proposals[:, 0] + 1
        hs = proposals[:, 3] - proposals[:, 1] + 1
        x_ctr = proposals[:, 0] + 0.5 * ws
        y_ctr = proposals[:, 1] + 0.5 * hs

        t_ws = targets[:, 2] - targets[:, 0] + 1
        t_hs = targets[:, 3] - targets[:, 1] + 1
        t_x_ctr = targets[:, 0] + 0.5 * t_ws
        t_y_ctr = targets[:, 1] + 0.5 * t_hs

        wx, wy, ww, wh = self.cfg.ROI_BOX.BBOX_REG_WEIGHTS

        tx = wx * (t_x_ctr - x_ctr) / ws
        ty = wy * (t_y_ctr - y_ctr) / hs
        tw = ww * torch.log(t_ws / ws)
        th = wh * torch.log(t_hs / hs)

        offsets = torch.stack([tx, ty, tw, th], dim=1)

        return offsets

    @type_check(object, list, list, list)
    def subsample(self, proposals, cats, targets):
        """
        采样出正负proposals以及正proposals对应的真实回归值
        Args:
            proposals: list[Tensor], 长度为batch size, 假定大小为num x 4
            cats: 每张图片中检测框的真实类别, 假定大小为N x M
            targets: 每张图片中检测框的真实坐标, 假定大小为N x M x 4
        Return:
            sampled_proposals: list[Tensor], (num_pos + num_neg) x 4
            sampled_logits_targets: list[Tensor], (num_pos + num_neg)
            sampled_bbox_regs_targets: list[Tensor], num_pos x 4
        """
        sampled_proposals = []
        sampled_logits_targets = []
        sampled_bbox_regs_targets = []

        for proposal, cat, target in zip(proposals, cats, targets):
            # 先去掉填充的0
            # inds = cat > 0
            # cat = cat[inds]
            # target = target[inds]
            # 匹配
            match = self.match(proposal, target)
            # 采样
            pos_inds, neg_inds = self.sample(match)
            # 计算得到真实回归
            bbox_regs_targets = self.encode(proposal[pos_inds], target[match[pos_inds]])
            # 得到真实概率分布
            logits_targets = torch.cat([cat[match[pos_inds]], torch.zeros_like(neg_inds, dtype=torch.long)], dim=0)

            sampled_proposals.append(proposal[torch.cat([pos_inds, neg_inds], dim=0)])
            sampled_logits_targets.append(logits_targets)
            sampled_bbox_regs_targets.append(bbox_regs_targets)

        return sampled_proposals, sampled_logits_targets, sampled_bbox_regs_targets

    @type_check(object, torch.Tensor, torch.Tensor, list, list)
    def forward(self, logits, bbox_regs, logits_targets, bbox_regs_targets):
        """
        计算roi loss
        Args:
            logits: Tensor, num x num_classes
            bbox_regs: Tensor, num x num_classes * 4
            logits_targets: list
            bbox_regs_targets: list
        """
        logits_targets = torch.cat(logits_targets, dim=0)
        bbox_regs_targets = torch.cat(bbox_regs_targets, dim=0)
        roi_cls_loss = F.cross_entropy(logits, logits_targets)

        # 得到正proposals索引
        pos_inds = (logits_targets > 0).nonzero().squeeze(1)
        # 得到正样本真实类别
        pos_label = logits_targets[pos_inds]
        # 计算出每个正roi对应的回归预测索引
        map_inds = 4 * pos_label[:, None] + torch.tensor([0, 1, 2, 3], device=logits.device)

        roi_reg_loss = smooth_l1_loss(
            bbox_regs[pos_inds[:, None], map_inds],
            bbox_regs_targets,
            size_average=False,
            beta=1
        ) / logits_targets.numel()

        loss = {}
        loss['roi_cls_loss'] = roi_cls_loss
        loss['roi_reg_loss'] = roi_reg_loss

        return loss
