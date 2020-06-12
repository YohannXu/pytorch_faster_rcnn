# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-05 15:01:37
# Description: transforms.py


import random

import torchvision.transforms.functional as F
from easydict import EasyDict
from PIL import Image
from torchvision import transforms as T

from ..utils import type_check


@type_check(EasyDict, bool)
def build_transforms(cfg, is_train=True):
    """
    数据预处理
    """
    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD

    if is_train:
        min_size = cfg.DATASET.MIN_SIZE
        max_size = cfg.DATASET.MAX_SIZE
        flip_horizontal_prob = cfg.DATASET.FLIP_HORIZONTAL_PROB
        brightness = cfg.DATASET.BRIGHTNESS
        contrast = cfg.DATASET.CONTRAST
        saturation = cfg.DATASET.SATURATION
        hue = cfg.DATASET.HUE

        transforms = T.Compose(
            [
                Resize(min_size, max_size),
                RandomHorizontalFlip(flip_horizontal_prob),
                ColorJitter(
                    brightness,
                    contrast,
                    saturation,
                    hue
                ),
                ToTensor(),
                Normalize(mean, std)
            ]
        )
    else:
        min_size = cfg.DATASET.TEST_MIN_SIZE
        max_size = cfg.DATASET.TEST_MAX_SIZE
        transforms = T.Compose(
            [
                Resize(min_size, max_size),
                ToTensor(),
                Normalize(mean, std)
            ]
        )

    return transforms


class Resize():
    """
    图片缩放
    """

    @type_check(object, int, int, int)
    def __init__(self, min_size, max_size, interpolation=Image.BILINEAR):
        """
        Args:
            min_size: int, 图片尺寸最小值
            max_size: int, 图片尺寸最大值
            interpolation: 插值算法
        """
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']

        w, h = image.size

        if self.min_size / min(w, h) * max(w, h) > self.max_size:
            ratio = self.max_size / max(w, h)
        else:
            ratio = self.min_size / min(w, h)

        resize_w = int(w * ratio)
        resize_h = int(h * ratio)

        image = image.resize((resize_w, resize_h), self.interpolation)

        ratio_w = resize_w / w
        ratio_h = resize_h / h

        inputs['image'] = image
        inputs['ratio'] = [ratio_w, ratio_h]

        if 'bbox' in inputs:
            bbox = inputs['bbox']
            bbox[:, 0::2] *= ratio_w
            bbox[:, 1::2] *= ratio_h
            inputs['bbox'] = bbox

        return inputs


class RandomHorizontalFlip():
    """
    随机水平翻转
    """

    @type_check(object, float)
    def __init__(self, prob=0.5):
        self.prob = prob

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']
        bbox = inputs['bbox']

        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = image.size
            bbox[:, 0] = w - bbox[:, 0]
            bbox[:, 2] = w - bbox[:, 2]
            bbox[:, [0, 2]] = bbox[:, [2, 0]]

        inputs['image'] = image
        inputs['bbox'] = bbox

        return inputs


class RandomVerticalFlip():
    """
    随机垂直翻转
    """

    @type_check(object, float)
    def __init__(self, prob=0.5):
        self.prob = prob

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']
        bbox = inputs['bbox']

        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            w, h = image.size
            bbox[:, 1] = h - bbox[:, 1]
            bbox[:, 3] = h - bbox[:, 3]
            bbox[:, [1, 3]] = bbox[:, [3, 1]]

        inputs['image'] = image
        inputs['bbox'] = bbox

        return inputs


class ColorJitter():
    """
    随机色彩调整
    """

    @type_check(object, float, float, float, float)
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Args:
            brightness:, float, 亮度
            contrast: float, 对比度
            saturation: float, 饱和度
            hue: float, 色调
        """
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']
        image = self.color_jitter(image)
        inputs['image'] = image

        return inputs


class RandomCrop():
    pass


class MixUp():
    pass


class ToTensor():
    """
    将部分输入内容转为张量
    """

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']

        image = F.to_tensor(image)

        inputs['image'] = image

        return inputs


class Normalize():
    """
    对图片进行标准化处理
    """

    @type_check(object, list, list)
    def __init__(self, mean, std):
        """
        Args:
            mean: float, 均值
            std: float, 标准差
        """
        self.mean = mean
        self.std = std

    @type_check(object, dict)
    def __call__(self, inputs):
        image = inputs['image']

        image = image[[2, 1, 0]] * 255
        image = F.normalize(image, self.mean, self.std)

        inputs['image'] = image

        return inputs
