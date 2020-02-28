# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-28 23:30:08
# Description: 配置文件

from easydict import EasyDict as edict

cfg = edict()

cfg.OUTPUT = 'saved_models'

cfg.DATASET = edict()
# 图片尺寸必须是32的整数,确保在降采样时均为偶数,经过FPN升采样时不会出现尺寸不匹配
cfg.DATASET.BASE = 32
cfg.DATASET.MIN_SIZE = 800
cfg.DATASET.MAX_SIZE = 1333
cfg.DATASET.TEST_MIN_SIZE = 800
cfg.DATASET.TEST_MAX_SIZE = 1333
cfg.DATASET.STD = (1., 1., 1.)
cfg.DATASET.MEAN = (102.9801, 115.9465, 122.7717)
cfg.DATASET.FLIP_HORIZONTAL_PROB = 0
cfg.DATASET.BRIGHTNESS = 0
cfg.DATASET.CONTRAST = 0
cfg.DATASET.SATURATION = 0
cfg.DATASET.HUE = 0

cfg.BACKBONE = edict()
cfg.BACKBONE.NUM_LAYERS = 50
cfg.BACKBONE.SUFFIX = 'pkl'
cfg.BACKBONE.NUM_FROZEN = 2

cfg.FPN = edict()
cfg.FPN.ON = True
cfg.FPN.IN_CHANNELS = [256, 512, 1024, 2048]
cfg.FPN.OUT_CHANNEL = 256

cfg.TRAIN = edict()
cfg.TRAIN.EPOCHES = 20
cfg.TRAIN.NUM_ITERS = 200000
cfg.TRAIN.SAVE_INTERVAL = 2000
cfg.TRAIN.LOGDIR = 'tensorboard'

cfg.OPTIMIZER = edict()
cfg.OPTIMIZER.BASE_LR = 0.02
cfg.OPTIMIZER.WEIGHT_DECAY = 0.0001
cfg.OPTIMIZER.BIAS_LR = 2
cfg.OPTIMIZER.BIAS_WEIGHT_DECAY = 0
cfg.OPTIMIZER.MOMENTUM = 0.9
cfg.OPTIMIZER.STEPS = (30000,)
cfg.OPTIMIZER.GAMMA = 0.1
cfg.OPTIMIZER.WARMUP_FACTOR = 1.0 / 3
cfg.OPTIMIZER.WARMUP_ITERS = 500
cfg.OPTIMIZER.WARMUP_METHOD = 'linear'

cfg.RPN = edict()
# anchors步长
cfg.RPN.STRIDES = (4, 8, 16, 32, 64)
# anchor尺寸
cfg.RPN.SIZES = (32, 64, 128, 256, 512)
# anchor长宽比
cfg.RPN.RATIOS = (0.5, 1.0, 2.0)

# NMS前每层feature map对应的anchors数量
cfg.RPN.PRE_NMS_TOP_N = 2000
# NMS后每层feature map对应的anchors数量
cfg.RPN.POST_NMS_TOP_N = 2000
# NMS阈值
cfg.RPN.NMS_THRESHOLD = 0.5
# NMS后处理是否在batch上做
cfg.RPN.POST_PER_BATCH = True
# NMS后处理保留的proposals数量
cfg.RPN.POST_TOP_N = 2000
# RPN前景阈值
cfg.RPN.FG_IOU_THRESHOLD = 0.7
# RPN背景阈值
cfg.RPN.BG_IOU_THRESHOLD = 0.3
# 计算RPN loss时每张图片的proposals数量
cfg.RPN.NUM_PER_IMAGE = 256
# 计算RPN loss时每张图片的proposals正样本比率
cfg.RPN.POSITIVE_RATIO = 0.5
# bbox在进行decoder和encoder时的权重
cfg.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

cfg.ROI_BOX = {}
# ROI_BOX前景阈值
cfg.ROI_BOX.FG_IOU_THRESHOLD = 0.5
# ROI_BOX背景阈值
cfg.ROI_BOX.BG_IOU_THRESHOLD = 0.5
# ROI_BOX每张图片的proposals数量
cfg.ROI_BOX.NUM_PER_IMAGE = 512
# ROI_BOX每张图片的proposals正样本比率
cfg.ROI_BOX.POSITIVE_RATIO = 0.25
# bbox在ROI_BOX阶段进行decoder和encoder时的权重
cfg.ROI_BOX.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# RoIAlign输出尺寸
cfg.ROI_BOX.RESOLUTION = 7
# backbone输出的降采样系数
cfg.ROI_BOX.RATIOS = (0.25, 0.125, 0.0625, 0.03125)
#
cfg.ROI_BOX.SAMPLE_RATIO = 2
# 全连接层维度
cfg.ROI_BOX.HIDDEN_DIM = 1024
cfg.ROI_BOX.NUM_CLASSES = 81
#
cfg.ROI_BOX.PROB_THRESHOLD = 0.05
#
cfg.ROI_BOX.NMS_THRESHOLD = 0.5
cfg.ROI_BOX.DETECTIONS_PER_IMAGE = 100
