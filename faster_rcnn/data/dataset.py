# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-05 15:01:30
# Description: COCO格式数据集

import bisect
import itertools
import json
import math
import os
from glob import glob

import torch
from easydict import EasyDict
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import BatchSampler, Dataset
from torchvision.transforms import transforms as T

from ..utils import type_check


class COCODataset(Dataset):
    """
    COCO格式数据集
    用于训练及验证
    """

    @type_check(object, EasyDict, T.Compose, bool)
    def __init__(self, cfg, transforms=None, is_train=True):
        """
        Args:
            cfg: str, 配置文件
            transforms: 图片预训练
            is_train: bool, 训练还是验证
        """
        super(COCODataset, self).__init__()
        if is_train:
            self.root = cfg.DATASET.TRAIN_ROOT
            self.anno_file = cfg.DATASET.TRAIN_ANNO
        else:
            self.root = cfg.DATASET.VAL_ROOT
            self.anno_file = cfg.DATASET.VAL_ANNO

        self.transforms = transforms
        # COCO数据集中类别索引不是从1~80,因此手动调整到1~80
        with open('faster_rcnn/data/classes.json') as f:
            self.classes = json.load(f)

        # 加载数据集
        self.coco = COCO(self.anno_file)
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
        bbox = torch.tensor(bbox)
        bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:] - 1
        cat = torch.tensor(cat)

        data = {
            'image': image,
            'bbox': bbox,
            'cat': cat,
            'name': image_name,
            'img_id': img_id
        }

        if self.transforms:
            data = self.transforms(data)

        return data

    def get_info(self, index):
        img_id = self.ids[index]
        info = self.coco.imgs[img_id]
        return info


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

        data = {
            'ori_image': image,
            'image': image,
            'name': image_name
        }

        if self.transforms:
            data = self.transforms(data)

        return data


class DataSampler(BatchSampler):
    """
    加载数据时的Sampler
    将长宽比相近的图片放到同一个batch中，降低显存占用
    """

    @type_check(object, Dataset, EasyDict, int, bool)
    def __init__(self, dataset, cfg, start_iter=0, is_train=True):
        """
        Args:
            dataset: 数据集
            cfg: 配置文件
            start_iter: 当前迭代次数
            is_train: 训练还是验证
        """

        # 得到所有图片的长宽比
        aspect_ratios = self.compute_aspect_ratios(dataset)
        # 根据长宽比对图片进行分组
        group_thresholds = cfg.DATASET.GROUP_THRESHOLD
        self.groups = torch.as_tensor(self.divide(aspect_ratios, group_thresholds))
        # 组id
        self.group_ids = torch.unique(self.groups).sort(0)[0]
        self.dataset = dataset
        self.start_iter = start_iter
        self.is_train = is_train
        if self.is_train:
            self.batch_size = cfg.DATASET.TRAIN_BATCH_SIZE
            self.num_iters = cfg.TRAIN.NUM_ITERS
        else:
            self.batch_size = cfg.DATASET.VAL_BATCH_SIZE
            self.num_iters = len(dataset)

    @type_check(object, Dataset)
    def compute_aspect_ratios(self, dataset):
        """
        计算所有图片的长宽比, 正序
        """
        aspect_ratios = []
        for i in range(len(dataset)):
            info = dataset.get_info(i)
            aspect_ratio = info['height'] / info['width']
            aspect_ratios.append(aspect_ratio)
        return aspect_ratios

    @type_check(object, list, list)
    def divide(self, ratios, thresholds=[1]):
        """
        根据长宽比及阈值将图片分为多个组
        默认划分为长宽比小于1及大于等于1两个组
        Args:
            ratios: 所有图片长宽比
            thresholds: 分组阈值
        """
        thresholds = sorted(thresholds)
        groups = list(map(lambda ratio: bisect.bisect_right(thresholds, ratio), ratios))
        return groups

    def prepare_batches(self):
        """
        根据分组生成batch
        """
        mask = self.sample_ids >= 0
        self.groups = self.groups[self.sample_ids]
        # 得到每一组的索引
        clusters = [(self.groups == i) & mask for i in self.group_ids]
        permuted_clusters = [self.sample_ids[idx] for idx in clusters]

        # 在每一组内划分batch
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        # 将所有组的batch汇总
        merged = tuple(itertools.chain.from_iterable(splits))
        # 得到每个batch中第一张图片的索引
        first_element_of_batch = [t[0].item() for t in merged]
        # 得到图片索引在sample_ids中的位置
        inv_sampled_ids_map = {v: k for k, v in enumerate(self.sample_ids.tolist())}
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )
        # 根据位置进行排序
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # 得到排序后的batch
        batches = [merged[i].tolist() for i in permutation_order]

        return batches

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iters:
            # 训练状态, 打乱图片顺序
            # 每过一个epoch, 重新打乱顺序
            if self.is_train:
                self.sample_ids = torch.randperm(len(self.dataset))
            else:
                self.sample_ids = torch.arange(len(self.dataset))
            batches = self.prepare_batches()
            for batch in batches:
                yield batch
                iteration += 1
                if iteration > self.num_iters:
                    break

    def __len__(self):
        return self.num_iters


class Collater():
    """
    用于拼接一个batch中的数据
    """

    @type_check(object, EasyDict, bool)
    def __init__(self, cfg, is_train_or_val=True):
        self.cfg = cfg
        self.is_train_or_val = is_train_or_val

    @type_check(object, list)
    def __call__(self, batch):
        if self.is_train_or_val:
            origin_images = [item['image'] for item in batch]
            bboxes = [item['bbox'] for item in batch]
            cats = [item['cat'] for item in batch]
            ratios = [item['ratio'] for item in batch]
            names = [item['name'] for item in batch]
            img_ids = [item['img_id'] for item in batch]
        else:
            ori_images = [item['ori_image'] for item in batch]
            origin_images = [item['image'] for item in batch]
            ratios = [item['ratio'] for item in batch]
            names = [item['name'] for item in batch]

        # 拼接缩放后图片
        # 计算一个batch中图片的最大尺寸
        max_w, max_h = 0, 0
        for image in origin_images:
            h, w = image.shape[1:3]

            max_w = w if w > max_w else max_w
            max_h = h if h > max_h else max_h

        # 将最大尺寸调整为基准尺寸的倍数
        base_size = self.cfg.DATASET.BASE
        max_w = base_size * math.ceil(max_w / base_size)
        max_h = base_size * math.ceil(max_h / base_size)

        images = torch.zeros(len(origin_images), 3, max_h, max_w, dtype=torch.float32)
        for i, image in enumerate(origin_images):
            h, w = image.shape[1:3]
            images[i, :, :h, :w] = image

        # 拼接图片尺寸
        sizes = torch.zeros(len(origin_images), 2, dtype=torch.float32)
        for i, image in enumerate(origin_images):
            sizes[i, 0] = image.size(1)
            sizes[i, 1] = image.size(2)

        if self.is_train_or_val:
            data = {
                'images': images,
                'bboxes': bboxes,
                'cats': cats,
                'sizes': sizes,
                'ratios': ratios,
                'names': names,
                'img_ids': img_ids
            }
        else:
            data = {
                'ori_images': ori_images,
                'images': images,
                'sizes': sizes,
                'ratios': ratios,
                'names': names
            }

        return data
