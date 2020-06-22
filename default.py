# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-28 23:30:08
# Description: 配置文件

from easydict import EasyDict as edict

cfg = edict()

# 权重保存路径
cfg.OUTPUT = 'saved_models'

cfg.DATASET = edict()
# 图片尺寸必须是32的整数,确保在降采样时均为偶数,经过FPN升采样后不会出现尺寸不匹配的情况
cfg.DATASET.BASE = 32
cfg.DATASET.MIN_SIZE = 800
cfg.DATASET.MAX_SIZE = 1333
cfg.DATASET.TEST_MIN_SIZE = 800
cfg.DATASET.TEST_MAX_SIZE = 1333
# 图片归一化标准差
cfg.DATASET.STD = (1., 1., 1.)
# 图片归一化均值
# 使用caffe2预训练权重
cfg.DATASET.PKL_MEAN = (102.9801, 115.9465, 122.7717)
# 使用pytorch预训练权重
cfg.DATASET.PTH_MEAN = (0.4815, 0.4547, 0.4038)
# 水平翻转概率
cfg.DATASET.FLIP_HORIZONTAL_PROB = 0.0
# 亮度调整
cfg.DATASET.BRIGHTNESS = 0.0
# 对比度调整
cfg.DATASET.CONTRAST = 0.0
# 饱和度调整
cfg.DATASET.SATURATION = 0.0
# 色度调整
cfg.DATASET.HUE = 0.0
# 训练集图片根目录
cfg.DATASET.TRAIN_ROOT = '/datasets/coco/train2014'
# 训练集annotation路径
cfg.DATASET.TRAIN_ANNO = '/datasets/coco/annotations/instances_train2014.json'
# 验证集图片根目录
cfg.DATASET.VAL_ROOT = '/datasets/coco/val2014'
# 验证集annotation路径
cfg.DATASET.VAL_ANNO = '/datasets/coco/annotations/instances_val2014.json'
# 图片分组阈值, 将图片根据长宽比进行分组
# 从每组中生成batch, 避免长宽比差距过大的图片分到同一batch中, 造成显存占用过大的情况
cfg.DATASET.GROUP_THRESHOLD = [1]
cfg.DATASET.TRAIN_BATCH_SIZE = 2
cfg.DATASET.VAL_BATCH_SIZE = 1
# 数据集加载线程数
cfg.DATASET.NUM_WORKERS = 4

cfg.BACKBONE = edict()
# backbone层数
cfg.BACKBONE.NUM_LAYERS = 50
# 预训练权重, 可选pkl或pth, 分别代表caffe2和pytorch预训练权重
cfg.BACKBONE.SUFFIX = 'pkl'
# 冻结部分参数
cfg.BACKBONE.NUM_FROZEN = 2

cfg.FPN = edict()
cfg.FPN.ON = True
# FPN每一层的输入通道数
cfg.FPN.IN_CHANNELS = [256, 512, 1024, 2048]
# FPN输出通道数
cfg.FPN.OUT_CHANNEL = 256

cfg.TRAIN = edict()
# 迭代次数
cfg.TRAIN.NUM_ITERS = 90000
# 日志打印间隔
cfg.TRAIN.LOG_INTERVAL = 20
# 权重保存间隔
cfg.TRAIN.SAVE_INTERVAL = 1000
# tensorboard保存路径
cfg.TRAIN.LOGDIR = 'tensorboard'
# 混合精度
cfg.TRAIN.MIX_LEVEL = 'O0'

cfg.OPTIMIZER = edict()
# 基础学习率
cfg.OPTIMIZER.BASE_LR = 0.0025
# 权重衰减
cfg.OPTIMIZER.WEIGHT_DECAY = 0.0001
# 偏置系数
cfg.OPTIMIZER.BIAS_LR = 2
# 偏置衰减
cfg.OPTIMIZER.BIAS_WEIGHT_DECAY = 0
# 动量参数
cfg.OPTIMIZER.MOMENTUM = 0.9
# 学习率下降迭代次数
# 预训练权重为caffe2
cfg.OPTIMIZER.PKL_STEPS = (60000, 80000)
# 预训练权重为pytorch
cfg.OPTIMIZER.PTH_STEPS = (25000, 75000)
# 学习率下降系数
cfg.OPTIMIZER.GAMMA = 0.1
# 训练开始时, 学习率从较低值逐渐增加到基础学习率, 避免初始学习率过大出现不稳定的情况
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

# 训练时NMS前每层feature map保留的anchors数量
cfg.RPN.PRE_NMS_TOP_N_TRAIN = 2000
# 测试时NMS前每层feature map保留的anchors数量
cfg.RPN.PRE_NMS_TOP_N_TEST = 1000
# 训练时NMS后每层feature map保留的anchors数量
cfg.RPN.POST_NMS_TOP_N_TRAIN = 2000
# 测试时NMS后每层feature map保留的anchors数量
cfg.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS阈值
cfg.RPN.NMS_THRESHOLD = 0.7
# 后处理针对整个batch还是每张图片
cfg.RPN.POST_PER_BATCH = True
# 训练时后处理保留的proposals数量
cfg.RPN.POST_TOP_N_TRAIN = 2000
# 推断时后处理保留的proposals数量
cfg.RPN.POST_TOP_N_TEST = 1000
# RPN前景阈值
cfg.RPN.FG_IOU_THRESHOLD = 0.7
# RPN背景阈值
cfg.RPN.BG_IOU_THRESHOLD = 0.3
# 计算RPN loss时每张图片的proposals数量
cfg.RPN.NUM_PER_IMAGE = 256
# 计算RPN loss时每张图片的proposals正样本比率
cfg.RPN.POSITIVE_RATIO = 0.5
# bbox在RPN阶段进行decoder和encoder时的权重
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
# RoIAlign参数
cfg.ROI_BOX.SAMPLE_RATIO = 2
# 全连接层维度
cfg.ROI_BOX.HIDDEN_DIM = 1024
# 类别数量
cfg.ROI_BOX.NUM_CLASSES = 81
# 推断时概率阈值
cfg.ROI_BOX.PROB_THRESHOLD = 0.05
# NMS阈值
cfg.ROI_BOX.NMS_THRESHOLD = 0.5
# 每张图片的最大检测数量
cfg.ROI_BOX.DETECTIONS_PER_IMAGE = 100
