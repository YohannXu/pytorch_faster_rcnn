# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-05-10 11:01:20
# Description: test.py

import datetime
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def speed_test():
    is_train = False
    is_train_or_val = False
    save_dir = 'test_results'

    model = Model(cfg, is_train).to(device)
    dataset = InferenceDataset(
        image_dir='/datasets/coco/test2014',
        transforms=build_transforms(cfg, is_train=is_train)
    )
    collater = Collater(cfg, is_train_or_val)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collater,
        shuffle=False,
        num_workers=2
    )

    checkpoint = last_checkpoint(cfg)
    if checkpoint:
        print('loading {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('weight not found')
        sys.exit()

    # checkpoint = 'model_pruned.pth'
    # checkpoint = torch.load(checkpoint)['state_dict']
    # model = torch.load('pruned_model.pth').to(device)

    model.eval()
    infer_times = []
    start_time = time.time()
    for data in tqdm(dataloader):
        start_infer_time = time.time()
        ori_images = data['ori_images']
        images = data['images'].to(device)
        sizes = data['sizes'].to(device)
        ratios = data['ratios']
        names = data['names']

        with torch.no_grad():
            boxes, probs, labels = model(images, sizes)
        infer_times.append(time.time() - start_infer_time)

        for ori_image, box, prob, label, ratio, name in zip(ori_images, boxes, probs, labels, ratios, names):
            ori_image = np.array(ori_image)
            box[:, 0::2] /= ratio[0]
            box[:, 1::2] /= ratio[1]

            for b, p, l in zip(box, prob, label):
                if p > 0.7:
                    cv2.rectangle(ori_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(ori_image, '{}_{:.2f}'.format(CATEGORIES[l], p), (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite('{}/{}'.format(save_dir, os.path.basename(name)), ori_image)

    total_time = time.time() - start_time
    avg_time = total_time / len(dataset)
    print('total time: {} on {} images, avg time: {:.4f}s'.format(str(datetime.timedelta(seconds=int(total_time))), len(dataset), avg_time))

    infer_times = np.array(infer_times)
    total_infer_time = infer_times.sum()
    avg_infer_time = infer_times.mean()
    print('total infer time: {} on {} images, avg infer time: {:.4f}'.format(
        str(datetime.timedelta(seconds=int(total_infer_time))), len(dataset), avg_infer_time))


if __name__ == '__main__':
    speed_test()
