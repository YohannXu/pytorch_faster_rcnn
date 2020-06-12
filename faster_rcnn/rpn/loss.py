# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-06 16:17:54
# Description: RPN部分的损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from ..utils import type_check
from .anchor_generator import filter_anchors


@type_check(torch.Tensor, torch.Tensor, float, bool)
def smooth_l1_loss(pred, target, beta=1./9, size_average=True):
    """
    smooth l1 损失函数
    当预测值与真实值差距较小时,使用二次函数
    当预测值与真实值差距较大时,使用线性函数
    """
    n = torch.abs(pred - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


class RPNLoss(nn.Module):
    """
    RPN部分损失函数的计算流程
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        super(RPNLoss, self).__init__()
        self.cfg = cfg

    @type_check(object, torch.Tensor, torch.Tensor)
    def match(self, anchors, targets):
        """
        将anchors和target进行匹配
        只有当anchor与某个target iou足够高时,才负责预测
        Args:
            anchors: Tensor, 假定大小为N x 4
            targets: Tensor, 假定大小为M x 4
        Return:
            inds: Tensor, N
        """
        anchors = anchors.to(targets.device)
        # 先计算anchors和targets之间的IoU
        x_tl = torch.max(anchors[:, 0][:, None], targets[:, 0][None, :])  # N x M
        y_tl = torch.max(anchors[:, 1][:, None], targets[:, 1][None, :])  # N x M
        x_rb = torch.min(anchors[:, 2][:, None], targets[:, 2][None, :])  # N x M
        y_rb = torch.min(anchors[:, 3][:, None], targets[:, 3][None, :])  # N x M

        inter = (x_rb - x_tl + 1).clamp(0) * (y_rb - y_tl + 1).clamp(0)  # N x M
        area1 = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)  # N
        area2 = (targets[:, 2] - targets[:, 0] + 1) * (targets[:, 3] - targets[:, 1] + 1)  # M
        ious = inter / (area1[:, None] + area2[None, :] - inter)  # N x M

        fg_threshold = self.cfg.RPN.FG_IOU_THRESHOLD
        bg_threshold = self.cfg.RPN.BG_IOU_THRESHOLD

        # 先得到每个anchor对应iou最高的target的索引和iou值
        vals, inds = ious.max(dim=1)  # N
        ori_inds = inds.clone()
        # iou值低于背景阈值的anchor为负样本
        inds[vals < bg_threshold] = -1
        # iou值位于两个阈值之间的anchor会被丢弃
        inds[(vals >= bg_threshold) & (vals < fg_threshold)] = -2
        # 有时可能会存在所有anchors均为负样本的情况
        # 因此还需要得到每个target对应iou最高的anchor的索引
        # 将对应的anchor标记为正样本,保证训练能够正常进行
        # 得到每个target对应最高iou值
        vals, _ = ious.max(dim=0)  # M
        # 找出每个target对应最高iou值对应的anchors索引
        # 因为对同一个target,可能存在多个anchors,都具有最高iou值
        # 所以需要通过这种方式得到索引
        post_inds = torch.nonzero(ious.permute(1, 0) == vals[:, None])[:, 1]
        # 对索引对应的anchors的正负样本划分进行更新
        inds[post_inds] = ori_inds[post_inds]

        return inds

    @type_check(object, torch.Tensor)
    def sample(self, matches):
        """
        采样出用于计算损失的正负anchors索引
        Args:
            matches: Tensor
        Return:
            pos_inds: Tensor, 正anchors索引
            neg_inds: Tensor, 负anchors索引
        """
        # 得到所有正负anchors索引
        pos_inds = torch.nonzero(matches >= 0).squeeze(1)
        neg_inds = torch.nonzero(matches == -1).squeeze(1)

        # 得到训练所需anchors数量和正anchors数量
        num_per_image = self.cfg.RPN.NUM_PER_IMAGE
        positive_ratio = self.cfg.RPN.POSITIVE_RATIO
        num_pos = int(num_per_image * positive_ratio)

        # 如果实际正anchors数量大于所需正anchors数量,则进行采样
        if pos_inds.shape[0] > num_pos:
            perm = torch.randperm(pos_inds.shape[0], device=pos_inds.device)[:num_pos]
            pos_inds = pos_inds[perm]
        else:
            num_pos = pos_inds.shape[0]

        # 计算训练所需负anchors数量
        num_neg = num_per_image - num_pos
        # 如果实际负anchors数量大于所需负anchors数量,则进行采样
        if neg_inds.shape[0] > num_neg:
            perm = torch.randperm(neg_inds.shape[0], device=neg_inds.device)[:num_neg]
            neg_inds = neg_inds[perm]
        else:
            num_neg = neg_inds.shape[0]

        return pos_inds, neg_inds

    @type_check(object, torch.Tensor, torch.Tensor)
    def encode(self, anchors, targets):
        """
        计算正anchors与对应targets之间的真实偏移量
        Args:
            anchors: Tensor, 假定大小为N x 4
            targets: Tensor, 假定大小为N x 4
        Return:
            offsets: Tensor, N x 4
        """
        ws = anchors[:, 2] - anchors[:, 0] + 1
        hs = anchors[:, 3] - anchors[:, 1] + 1
        x_ctr = anchors[:, 0] + 0.5 * ws
        y_ctr = anchors[:, 1] + 0.5 * hs

        t_ws = targets[:, 2] - targets[:, 0] + 1
        t_hs = targets[:, 3] - targets[:, 1] + 1
        t_x_ctr = targets[:, 0] + 0.5 * t_ws
        t_y_ctr = targets[:, 1] + 0.5 * t_hs

        wx, wy, ww, wh = self.cfg.RPN.BBOX_REG_WEIGHTS

        tx = wx * (t_x_ctr - x_ctr) / ws
        ty = wy * (t_y_ctr - y_ctr) / hs
        tw = ww * torch.log(t_ws / ws)
        th = wh * torch.log(t_hs / hs)

        offsets = torch.stack([tx, ty, tw, th], dim=1)

        return offsets

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

    @type_check(object, list, list, list, torch.Tensor, list)
    def forward(self, anchors, logits, bbox_regs, sizes, targets):
        """
        Args:
            anchors: list[list[Tensor]], 生成的anchors, 外层长度为batch_size, 内层长度为FPN层数, 张量大小假定为H * W * A x 4
            logits: list[Tensor], 概率预测张量, 长度为FPN层数, 张量大小假定为N x A x H x W
            bbox_regs: list[Tensor], 偏移量预测张量, 长度为FPN层数, 张量大小假定为N x A * 4 x H x W
            targets: Tensor, 假定大小为N x M x 4
        """
        # 拼接anchors
        anchors = [torch.cat(anchor, dim=0) for anchor in anchors]  # list[Tensor], 长度为batch size, 张量大小为sum x 4, sum为anchors总数
        # 调整logits形状
        logits = torch.cat([self.permute(logit, 1) for logit in logits], dim=1)  # N x sum x 1
        # 调整bbox_regs形状
        bbox_regs = torch.cat([self.permute(bbox_reg, 4) for bbox_reg in bbox_regs], dim=1)  # N x sum x 4

        sampled_logits = []
        sampled_logits_targets = []
        sampled_bbox_regs = []
        sampled_bbox_regs_targets = []

        # 遍历每张图片的anchor、logit、bbox_reg、target
        for anchor, logit, bbox_reg, size, target in zip(anchors, logits, bbox_regs, sizes, targets):
            mask = filter_anchors(anchor, size)
            anchor = anchor.to(target.device)
            # anchor与target匹配
            match = self.match(anchor, target)
            match[~mask] = -2
            # 得到正负anchors索引
            pos_inds, neg_inds = self.sample(match)
            # 得到正anchor对应的预测偏移量
            sampled_bbox_regs.append(bbox_reg[pos_inds])
            # 计算出正anchor对应的真实偏移量
            bbox_regs_targets = self.encode(anchor[pos_inds], target[match[pos_inds]])
            sampled_bbox_regs_targets.append(bbox_regs_targets)
            # 得到正负anchor对应的预测概率
            sampled_inds = torch.cat([pos_inds, neg_inds])
            sampled_logits.append(logit[sampled_inds])
            # 计算出正负anchor对应的真实概率
            logits_targets = (match[sampled_inds] >= 0).to(torch.float32)
            sampled_logits_targets.append(logits_targets)

        # 对采样结果进行拼接
        logits = torch.cat(sampled_logits, dim=0).squeeze(1)
        logits_targets = torch.cat(sampled_logits_targets, dim=0)
        bbox_regs = torch.cat(sampled_bbox_regs, dim=0)
        bbox_regs_targets = torch.cat(sampled_bbox_regs_targets, dim=0)

        # 计算loss
        cls_loss = F.binary_cross_entropy_with_logits(logits, logits_targets)
        reg_loss = smooth_l1_loss(bbox_regs, bbox_regs_targets, size_average=False) / logits.numel()

        return {'rpn_cls_loss': cls_loss, 'rpn_reg_loss': reg_loss}
