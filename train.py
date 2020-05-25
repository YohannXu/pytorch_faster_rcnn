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

import math
import time
import datetime
import shutil
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from faster_rcnn.data import COCODataset, Collater, build_transforms, DataSampler
from faster_rcnn.utils import WarmupMultiStepLR, Metric, last_checkpoint
from default import cfg
from model import Model
from apex import amp

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train():
    is_train = True
    model = Model(cfg, is_train=is_train).to(device)

    # 构建优化器及学习率下降策略
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.OPTIMIZER.BASE_LR
        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        if 'bias' in key:
            lr = cfg.OPTIMIZER.BASE_LR * cfg.OPTIMIZER.BIAS_LR
            weight_decay = cfg.OPTIMIZER.BIAS_WEIGHT_DECAY
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    optimizer = optim.SGD(params, lr, momentum=cfg.OPTIMIZER.MOMENTUM)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')

    global_step = 1
    # 加载权重
    checkpoint = last_checkpoint(cfg)
    if 'model_final.pth' in checkpoint:
        print('training has completed!')
        sys.exit()
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        metric = checkpoint['metric']
        global_step = checkpoint['global_step']
    else:
        metric = Metric()
        global_step = 1

    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.OPTIMIZER.STEPS,
        cfg.OPTIMIZER.GAMMA,
        cfg.OPTIMIZER.WARMUP_FACTOR,
        cfg.OPTIMIZER.WARMUP_ITERS,
        cfg.OPTIMIZER.WARMUP_METHOD,
        global_step - 2
    )

    # 加载训练数据集
    dataset = COCODataset(
        cfg,
        transforms=build_transforms(cfg),
        is_train=is_train
    )
    collater = Collater(cfg)
    sampler = DataSampler(dataset, cfg, global_step, is_train)
    dataloader = DataLoader(
        dataset,
        collate_fn=collater,
        batch_sampler=sampler,
        num_workers=cfg.DATASET.NUM_WORKERS
    )

    # summary保存路径
    if os.path.exists(cfg.TRAIN.LOGDIR):
        shutil.rmtree(cfg.TRAIN.LOGDIR)
    writer = SummaryWriter(log_dir=cfg.TRAIN.LOGDIR)

    # 日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler('train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    model.train()
    max_iters = cfg.TRAIN.NUM_ITERS

    start = time.time()
    for data in dataloader:
        images = data['images'].to(device)
        cats = [cat.to(device) for cat in data['cats']]
        sizes = data['sizes'].to(device)
        bboxes = [bbox.to(device) for bbox in data['bboxes']]

        rpn_loss, roi_loss = model(images, sizes, cats, bboxes)

        loss = rpn_loss['rpn_cls_loss'] + rpn_loss['rpn_reg_loss'] + \
            roi_loss['roi_cls_loss'] + roi_loss['roi_reg_loss']

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - start
        start = time.time()

        metric.update('rpn_cls_loss', rpn_loss['rpn_cls_loss'])
        metric.update('rpn_reg_loss', rpn_loss['rpn_reg_loss'])
        metric.update('roi_cls_loss', roi_loss['roi_cls_loss'])
        metric.update('roi_reg_loss', roi_loss['roi_reg_loss'])
        metric.update('loss', loss)
        metric.update('time', batch_time)

        # 写入summary文件
        writer.add_scalar('Loss/loss', loss, global_step)
        writer.add_scalar('Loss/rpn_cls_loss',
                          rpn_loss['rpn_cls_loss'], global_step)
        writer.add_scalar('Loss/rpn_reg_loss',
                          rpn_loss['rpn_reg_loss'], global_step)
        writer.add_scalar('Loss/roi_cls_loss',
                          roi_loss['roi_cls_loss'], global_step)
        writer.add_scalar('Loss/roi_reg_loss',
                          roi_loss['roi_reg_loss'], global_step)

        eta_time = (max_iters - global_step) * metric['time'].global_avg
        eta_time = str(datetime.timedelta(seconds=int(eta_time)))
        # 打印日志
        if global_step % cfg.TRAIN.LOG_INTERVAL == 0:
            message = metric.delimiter.join(
                [
                    'step: {global_step}',
                    'eta: {eta}',
                    '{metric}',
                    'lr: {lr:.6f}',
                    'max mem: {memory:.0f}'
                ]
            ).format(
                global_step=global_step,
                eta=eta_time,
                metric=str(metric),
                lr=optimizer.param_groups[0]['lr'],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            )
            logger.info(message)

        # 定时保存模型
        if global_step % cfg.TRAIN.SAVE_INTERVAL == 0:
            checkpoint = {}
            checkpoint['state_dict'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['metric'] = metric
            checkpoint['global_step'] = global_step

            if not os.path.exists(cfg.OUTPUT):
                os.makedirs(cfg.OUTPUT)
            torch.save(checkpoint, '{}/model_{:04d}.pth'.format(
                cfg.OUTPUT, global_step))

        global_step += 1

        if global_step == max_iters:
            torch.save(model.state_dict(), '{}/model_final.pth'.format(cfg.OUTPUT))


if __name__ == '__main__':
    train()
