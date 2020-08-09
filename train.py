# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-29 21:20:01
# Description: 训练流程

import datetime
import logging
import os
import shutil
import sys
import time

import torch
import torch.optim as optim
from apex import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from default import cfg
from faster_rcnn.data import COCODataset, Collater, DataSampler, build_transforms
from faster_rcnn.utils import Metric, WarmupMultiStepLR, last_checkpoint
from model import Model

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def train():
    is_train = True
    model = Model(cfg, is_train=is_train).to(device)

    # 构建优化器
    lr = cfg.OPTIMIZER.BASE_LR
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY
        if 'bias' in key:
            lr = cfg.OPTIMIZER.BASE_LR * cfg.OPTIMIZER.BIAS_LR
            weight_decay = cfg.OPTIMIZER.BIAS_WEIGHT_DECAY
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
    optimizer = optim.SGD(params, lr, momentum=cfg.OPTIMIZER.MOMENTUM)
    # 混合精度
    model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.TRAIN.MIX_LEVEL)

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

    # 学习率衰减策略
    if cfg.BACKBONE.SUFFIX == 'pkl':
        steps = cfg.OPTIMIZER.PKL_STEPS
    else:
        steps = cfg.OPTIMIZER.PTH_STEPS
    scheduler = WarmupMultiStepLR(
        optimizer,
        steps,
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
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metric': metric,
                'global_step': global_step
            }

            if not os.path.exists(cfg.OUTPUT):
                os.makedirs(cfg.OUTPUT)
            torch.save(checkpoint, '{}/model_{:04d}.pth'.format(
                cfg.OUTPUT, global_step))

        global_step += 1

        if global_step == max_iters:
            torch.save({'state_dict': model.state_dict()}, '{}/model_final.pth'.format(cfg.OUTPUT))


if __name__ == '__main__':
    train()
