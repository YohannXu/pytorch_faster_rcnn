# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-03-31 16:29:30
# Description: 推理流程

import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from default import cfg
from faster_rcnn.data import Collater, InferenceDataset, build_transforms
from faster_rcnn.utils import last_checkpoint
from model import Model

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

CATEGORIES = [
    "__background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def inference():
    is_train = False
    is_train_or_val = False
    save_dir = 'results'

    dataset = InferenceDataset(
        image_dir='infer_images',
        transforms=build_transforms(cfg, is_train=False)
    )
    collater = Collater(cfg, is_train_or_val)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collater,
        shuffle=False,
        num_workers=2
    )
    model = Model(cfg, is_train=is_train).to(device)

    checkpoint = last_checkpoint(cfg)
    if checkpoint:
        print('loading {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('weight not found')
        sys.exit()

    model.eval()
    for data in dataloader:
        ori_images = data['ori_images']
        images = data['images'].to(device)
        sizes = data['sizes'].to(device)
        ratios = data['ratios']
        names = data['names']

        with torch.no_grad():
            boxes, probs, labels = model(images, sizes)

        for ori_image, box, prob, label, ratio, name in zip(ori_images, boxes, probs, labels, ratios, names):
            ori_image = np.array(ori_image)
            box[:, 0::2] /= ratio[0]
            box[:, 1::2] /= ratio[1]

            for b, p, l in zip(box, prob, label):
                if p > 0.7:
                    cv2.rectangle(ori_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    # 如果box太靠上, 将文字放在box下方
                    y = b[1] - 10
                    if b[1] < 10:
                        y += 30
                    cv2.putText(ori_image, '{}_{:.2f}'.format(CATEGORIES[l], p), (b[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite('{}/{}'.format(save_dir, os.path.basename(name)), ori_image)


if __name__ == '__main__':
    import time
    start = time.time()
    inference()
    end = time.time()
    print(end - start)
