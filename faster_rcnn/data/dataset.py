# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-05 15:01:30
# Description: COCO格式数据集

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from PIL import Image
from easydict import EasyDict
import math
import json
from ..utils import type_check


class COCODataset(Dataset):
    """
    COCO格式数据集
    用于训练及验证
    """

    @type_check(object, str, str, T.Compose, bool)
    def __init__(self, root='/datasets/coco/', anno_file=None, transforms=None, is_train=True):
        """
        Args:
            root: str, 数据集根目录
            anno_file: str, 标注文件名
            transforms: 图片预训练
            is_train: bool, 训练还是验证
        """
        super(COCODataset, self).__init__()
        # TODO
        # 改成从cfg文件中读取图片路径
        if is_train:
            self.root = root + 'train2014'
        else:
            self.root = root + 'val2014'

        self.transforms = transforms
        # COCO数据集中类别索引不是从1~80,因此手动调整到0~80
        with open('faster_rcnn/data/classes.json') as f:
            self.classes = json.load(f)

        # 加载数据集
        self.coco = COCO(anno_file)
        # 得到所有图片索引
        ids = list(sorted(self.coco.imgs.keys()))
        # 将不包含bbox标记的图片去掉
        self.ids = []
        for img_id in ids:
            anno_ids = self.coco.getAnnIds(img_id)
            if anno_ids:
                annos = self.coco.loadAnns(anno_ids)
                # 如果所有annos的宽高都大于1,就保留该图片
                # 只要有一个小于等于1,就丢弃该图片
                # TODO 可以只丢弃该anno
                if not all(any(scale <= 1 for scale in anno['bbox'][2:]) for anno in annos):
                    self.ids.append(img_id)

    def __len__(self):
        return len(self.ids)

    @type_check(object, int)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image_name = self.coco.loadImgs(ids=img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, image_name)).convert('RGB')
        target = self.coco.imgToAnns[img_id]

        # 提取出annotation中与目标检测相关的部分
        bbox = []
        cat = []
        for ann in target:
            bbox.append(ann['bbox'])
            cat.append(self.classes[str(ann['category_id'])])
        bbox = np.array(bbox)
        bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:]
        cat = np.array(cat)

        data = {
            'ori_image': None,
            'image': image,
            'bbox': bbox,
            'cat': cat,
            'name': image_name
        }

        if self.transforms:
            data = self.transforms(data)

        return data


class InferenceDataset(Dataset):
    """
    推理时的数据集类
    """

    @type_check(object, str, T.Compose)
    def __init__(self, image_dir, transforms=None):
        """
        Args:
            image_dir: str, 需要推理的图片文件夹路径
            transforms: 图片预处理
        """
        super(InferenceDataset, self).__init__()
        self.image_names = glob('{}/*'.format(image_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    @type_check(object, int)
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(image_name).convert('RGB')

        bbox = np.zeros((0, 4))
        cat = np.array([])
        data = {
            'ori_image': image,
            'image': image,
            'bbox': bbox,
            'cat': cat,
            'name': image_name
        }

        if self.transforms:
            data = self.transforms(data)

        return data


class Collater():
    """
    用于拼接一个batch中的数据
    """

    @type_check(object, EasyDict)
    def __init__(self, cfg):
        self.cfg = cfg

    @type_check(object, list)
    def __call__(self, batch):
        ori_images = [item['ori_image'] for item in batch]
        origin_images = [item['image'] for item in batch]
        origin_bboxes = [item['bbox'] for item in batch]
        origin_cats = [item['cat'] for item in batch]
        ratios = [item['ratio'] for item in batch]
        names = [item['name'] for item in batch]

        # 计算一个batch中图片的最大尺寸
        max_w, max_h = 0, 0
        for image in origin_images:
            h, w = image.shape[1:3]

            max_w = w if w > max_w else max_w
            max_h = h if h > max_h else max_h

        base_size = self.cfg.DATASET.BASE
        max_w = base_size * math.ceil(max_w / base_size)
        max_h = base_size * math.ceil(max_h / base_size)

        images = torch.zeros(len(origin_images), 3, max_h, max_w, dtype=torch.float32)
        for i, image in enumerate(origin_images):
            h, w = image.shape[1:3]
            images[i, :, :h, :w] = image

        # 计算一个batch中图片bboxes的最大数目
        num_max = max(bbox.shape[0] for bbox in origin_bboxes)
        bboxes = torch.zeros(len(origin_bboxes), num_max, 4, dtype=torch.float32)
        for i, bbox in enumerate(origin_bboxes):
            num = bbox.shape[0]
            bboxes[i, :num] = bbox

        num_max = max(cat.shape[0] for cat in origin_cats)
        cats = torch.zeros(len(origin_cats), num_max, dtype=torch.long)
        for i, cat in enumerate(origin_cats):
            num = cat.shape[0]
            cats[i, :num] = cat

        sizes = torch.zeros(len(origin_images), 2, dtype=torch.float32)
        for i, image in enumerate(origin_images):
            sizes[i, 0] = image.size(1)
            sizes[i, 1] = image.size(2)

        data = {
            'ori_images': ori_images,
            'images': images,
            'bboxes': bboxes,
            'cats': cats,
            'sizes': sizes,
            'ratios': ratios,
            'names': names
        }

        return data
