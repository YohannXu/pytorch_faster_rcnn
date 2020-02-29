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

from faster_rcnn.backbone import ResNet

from default import cfg

if __name__ == '__main__':
    backbone = ResNet(cfg)
    images = torch.randn(3, 3, 224, 224)
    res = backbone(images)
    print(res)
