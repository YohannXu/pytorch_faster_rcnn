# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-19 14:58:56
# Description: roi流程

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torchvision.ops import nms

from ...utils import type_check
from .head import RoIHead
from .loss import RoILoss


class BoxRoI(nn.Module):
    """
    faster rcnn roi流程
    """

    @type_check(object, EasyDict, bool)
    def __init__(self, cfg, is_train=True):
        super(BoxRoI, self).__init__()
        self.cfg = cfg
        self.head = RoIHead(cfg)
        self.loss = RoILoss(cfg)
        self.prob_threshold = self.cfg.ROI_BOX.PROB_THRESHOLD
        self.nms_threshold = self.cfg.ROI_BOX.NMS_THRESHOLD

    @type_check(object, torch.Tensor, torch.Tensor)
    def decode(self, proposals, bbox_regs):
        """
        根据回归预测对proposals进行调整, 得到boxes
        Args:
            proposals: Tensor, 假定大小为N x 4
            bbox_regs: Tensor, 假定大小为N x num_classes * 4
        Return:
            boxes: Tensor, N x num_classes * 4
        """
        max_offset = math.log(1000 / 16)

        ws = proposals[:, 2] - proposals[:, 0] + 1
        hs = proposals[:, 3] - proposals[:, 1] + 1
        x_ctr = proposals[:, 0] + 0.5 * ws
        y_ctr = proposals[:, 1] + 0.5 * hs

        wx, wy, ww, wh = self.cfg.ROI_BOX.BBOX_REG_WEIGHTS

        dx = bbox_regs[:, 0::4] / wx  # N x num_classes
        dy = bbox_regs[:, 1::4] / wy  # N x num_classes
        dw = bbox_regs[:, 2::4] / ww  # N x num_classes
        dh = bbox_regs[:, 3::4] / wh  # N x num_classes

        dw = dw.clamp(max=max_offset)
        dh = dh.clamp(max=max_offset)

        x_ctr = x_ctr[:, None] + dx * ws[:, None]  # N x num_classes
        y_ctr = y_ctr[:, None] + dy * hs[:, None]  # N x num_classes
        ws = ws[:, None] * torch.exp(dw)  # N x num_classes
        hs = hs[:, None] * torch.exp(dh)  # N x num_classes

        boxes = torch.zeros_like(bbox_regs)

        x1 = x_ctr - 0.5 * ws  # N x num_classes
        y1 = y_ctr - 0.5 * hs  # N x num_classes
        x2 = x_ctr + 0.5 * ws - 1  # N x num_classes
        y2 = y_ctr + 0.5 * hs - 1  # N x num_classes

        boxes[:, 0::4] = x1
        boxes[:, 1::4] = y1
        boxes[:, 2::4] = x2
        boxes[:, 3::4] = y2

        return boxes

    @type_check(object, torch.Tensor, torch.Tensor, int)
    def iou(self, area, boxes, index):
        """
        计算某一个box和之后所有boxes之间的iou
        Args:
            area: Tensor, 提前计算好的所有boxes的面积, 避免重复计算, 假定形状为num x 4
            boxes: Tensor, 假定形状为num x 4
            index: int, 特定proposal的索引
        Return:
            iou: Tensor, num
        """
        x_tl = torch.max(boxes[index, 0], boxes[index + 1:, 0])
        y_tl = torch.max(boxes[index, 1], boxes[index + 1:, 1])
        x_br = torch.min(boxes[index, 2], boxes[index + 1:, 2])
        y_br = torch.min(boxes[index, 3], boxes[index + 1:, 3])

        inter = (x_br - x_tl + 1).clamp(0) * (y_br - y_tl + 1).clamp(0)
        iou = inter / (area[index] + area[index + 1] - inter)

        return iou

    @type_check(object, torch.Tensor, torch.Tensor)
    def nms(self, boxes, probs):
        """
        执行NMS, 保留一定数量的boxes
        Args:
            boxes: Tensor, 假定大小为N x 4
            probs: Tensor, 假定大小为N
        Return:
            boxes: Tensor, n x 4
            probs: Tensor, n
        """
        # 根据logits排序
        sorted_inds = probs.argsort(descending=True)
        boxes = boxes[sorted_inds]
        probs = probs[sorted_inds]

        # 计算所有boxes的面积
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

        # NMS流程
        keep = []
        # 遍历索引
        num_proposals = boxes.shape[0]
        for i in range(num_proposals):
            if probs[i] > 0:
                if i != num_proposals - 1:
                    ious = self.iou(area, boxes, i)
                    probs[i + 1:][ious > self.nms_threshold] = -1
                keep.append(i)
        keep = torch.LongTensor(keep)

        boxes = boxes[keep]
        probs = probs[keep]

        return boxes, probs

    @type_check(object, list, torch.Tensor, torch.Tensor, torch.Tensor)
    def inference(self, proposals, logits, bbox_regs, sizes):
        """
        roi部分的inference流程
        Args:
            proposals: list[Tensor]
            logits: Tensor, 概率预测, num x num_classes
            bbox_regs: Tensor, 回归预测, num x num_classes
            sizes: Tensor, N x 2
        Return:
            det_boxes: list[Tensor]
            det_probs: list[Tensor]
            det_labels: list[Tensor]
        """
        probs = F.softmax(logits, dim=1)
        device = probs.device

        # 得到每张图片的proposals数量
        num_per_image = [proposal.shape[0] for proposal in proposals]
        proposals = torch.cat(proposals, dim=0)

        # 得到最终预测boxes
        boxes = self.decode(proposals, bbox_regs)

        # 分配到每张图片
        boxes = boxes.split(num_per_image, dim=0)
        probs = probs.split(num_per_image, dim=0)

        det_probs = []
        det_boxes = []
        det_labels = []

        for prob, box, size in zip(probs, boxes, sizes):
            # 裁剪到图片范围内
            box[:, 0::4].clamp_(min=0, max=size[1] - 1)
            box[:, 1::4].clamp_(min=0, max=size[0] - 1)
            box[:, 2::4].clamp_(min=0, max=size[1] - 1)
            box[:, 3::4].clamp_(min=0, max=size[0] - 1)

            inds = prob > self.prob_threshold
            num_classes = prob.shape[1]

            prob_per_image = []
            box_per_image = []
            label_per_image = []

            # 遍历每个类别, 索引从1开始, 跳过背景
            for class_id in range(1, num_classes):
                # 得到预测概率大于阈值的索引
                ind = inds[:, class_id].nonzero().squeeze(1)
                if ind.shape[0] == 0:
                    continue
                prob_for_class = prob[ind, class_id]
                box_for_class = box[ind, class_id * 4: (class_id + 1) * 4]

                # 去掉右下角大于左上角的boxes
                # keep = (box_for_class[:, 2] > box_for_class[:, 0]) & (box_for_class[:, 3] > box_for_class[:, 1])
                # box_for_class = box_for_class[keep]
                # prob_for_class = prob_for_class[keep]

                # 进行nms
                keep = nms(box_for_class, prob_for_class, self.nms_threshold)
                box_for_class = box_for_class[keep]
                prob_for_class = prob_for_class[keep]
                # box_for_class, prob_for_class = self.nms(box_for_class, prob_for_class)
                label_for_class = torch.full((box_for_class.shape[0],), class_id, dtype=torch.int64, device=device)

                box_per_image.append(box_for_class)
                prob_per_image.append(prob_for_class)
                label_per_image.append(label_for_class)

            # 把所有类别的结果拼接起来
            try:
                box = torch.cat(box_per_image, dim=0)
                prob = torch.cat(prob_per_image, dim=0)
                label = torch.cat(label_per_image, dim=0)
            except Exception:
                box = torch.zeros(0, 4)
                prob = torch.zeros(0)
                label = torch.zeros(0)

            # 保留预测概率靠前的boxes
            num_box = self.cfg.ROI_BOX.DETECTIONS_PER_IMAGE
            if box.shape[0] > num_box:
                sorted_inds = prob.argsort(descending=True)
                box = box[sorted_inds][:num_box]
                prob = prob[sorted_inds][:num_box]
                label = label[sorted_inds][:num_box]

            det_boxes.append(box)
            det_probs.append(prob)
            det_labels.append(label)

        return det_boxes, det_probs, det_labels

    @type_check(object, list, list, torch.Tensor, list, list)
    def forward(self, features, proposals, sizes, cats=None, targets=None):
        """
        Args:
            features: list[Tensor], 长度为FPN层数, 假定大小为N x C x H x W
            proposals: list[Tensor], 长度为batch size, num x 4
            cats: Tensor, 输入图片真实检测框类别, N x M
            sizes: Tensor, 输入图片缩放后尺寸, N x 2
            targets: Tensor, 输入图片真实检测框坐标, N x M x 4
        Return:
            train:
                loss: dict{'roi_cls_loss', 'roi_reg_loss'}
            eval:
                boxes: list[Tensor], num x 4
                probs: list[Tensor], num
                labels: list[Tensor], num
        """
        if self.training:
            with torch.no_grad():
                proposals, logits_targets, bbox_regs_targets = self.loss.subsample(proposals, cats, targets)

        logits, bbox_regs = self.head(features, proposals)

        if self.training:
            loss = self.loss(logits, bbox_regs, logits_targets, bbox_regs_targets)
            return loss

        with torch.no_grad():
            boxes, probs, labels = self.inference(proposals, logits, bbox_regs, sizes)
            return boxes, probs, labels
