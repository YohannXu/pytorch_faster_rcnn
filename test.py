# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-29 21:20:01
# Description: test.py

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from faster_rcnn.backbone import ResNet
from faster_rcnn.data import COCODataset, Collater, build_transforms

from default import cfg

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    backbone = ResNet(cfg)
    images = torch.randn(3, 3, 224, 224)
    dataset = COCODataset(
        anno_file='/datasets/coco/annotations/instances_train2014.json',
        transforms=build_transforms(cfg),
        is_train=True
    )
    collater = Collater(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collater,
        shuffle=False,
        num_workers=4
    )

    backbone = backbone.to(device)
    for data in dataloader:

        images = data['images'].to(device)
        res = backbone(images)
        print(res)

        break
