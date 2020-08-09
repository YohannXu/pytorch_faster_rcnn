# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-27 23:41:13
# Description: 提供各种工具

import os
import pickle
import shutil
import sys
import tempfile
from collections import deque
from functools import wraps
from glob import glob
from inspect import signature
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from tqdm import tqdm


def type_check(*types):
    """
    用于输入参数类型检查
    Args:
        *types: 目标函数的参数类型
    """
    def decorate(func):
        sig = signature(func)
        # 获取目标函数的参数名和参数类型
        # {name: type}
        arg_types = sig.bind_partial(*types).arguments
        params = sig.parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取目标函数输入参数
            # {name: value}
            input_args = sig.bind(*args, **kwargs)
            for name, value in input_args.arguments.items():
                if name in arg_types:
                    if not isinstance(value, arg_types[name]):
                        if params[name].KEYWORD_ONLY and params[name].default == value:
                            pass
                        else:
                            raise TypeError('Argument {} must be {}'.format(name, arg_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


pkl_url = 'https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/'
model_urls = {
    'pth': {
        50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    },
    'pkl': {
        50: pkl_url + 'R-50.pkl',
        101: pkl_url + 'R-101.pkl',
        152: pkl_url + 'R-152.pkl'
    }
}


@type_check(int, str)
def load_state_dict_from_url(num_layers, suffix='pkl'):
    """
    加载resnet预训练权重
    Args:
        num_layers: int, resnet网络层数
        suffix: str, 权重扩展名
    """
    assert suffix in ['pth', 'pkl'], 'Only support .pth or .pkl file'

    if suffix == 'pth':
        weights = {}
        weight = torch.hub.load_state_dict_from_url(model_urls[suffix][num_layers])
        for name, value in weight.items():
            if name == 'conv1.weight':
                name = 'stem.conv.weight'
            if name == 'bn1.weight':
                name = 'stem.bn.weight'
            if name == 'bn1.bias':
                name = 'stem.bn.bias'
            weight.requires_grad = True
            weights[name] = value
    else:
        # 建立保存地址
        save_dir = '~/.cache/torch/checkpoints'
        save_dir = os.path.expanduser(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        url = model_urls[suffix][num_layers]
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        # 得到保存文件名
        cached_file = os.path.join(save_dir, filename)
        # 如果权重未下载,就先下载权重
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            u = urlopen(url)
            meta = u.info()
            # 获取文件大小
            if hasattr(meta, 'getheaders'):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")
            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])
            # 建立临时文件
            cached_file = os.path.expanduser(cached_file)
            cached_dir = os.path.dirname(cached_file)
            f = tempfile.NamedTemporaryFile(delete=False, dir=cached_dir)

            try:
                with tqdm(total=file_size, disable=False, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    while True:
                        buf = u.read(8192)
                        if len(buf) == 0:
                            break
                        f.write(buf)
                        pbar.update(len(buf))

                f.close()
                shutil.move(f.name, cached_dir)
            finally:
                f.close()
                if os.path.exists(f.name):
                    os.remove(f.name)

        # 加载权重
        c2_weights = pickle.load(open(cached_file, 'rb'), encoding='latin1')

        weights = {}
        # 调整权重名称,与定义的backbone网络保持一致
        for name, value in sorted(c2_weights.items()):
            name = name.replace("_", ".")
            name = name.replace(".w", ".weight")
            name = name.replace(".bn", "_bn")
            name = name.replace(".b", ".bias")
            name = name.replace("_bn.s", "_bn.scale")
            name = name.replace(".biasranch", ".branch")
            name = name.replace("bbox.pred", "bbox_pred")
            name = name.replace("cls.score", "cls_score")
            name = name.replace("res.conv1_", "conv1_")

            name = name.replace("_bn.scale", "_bn.weight")

            name = name.replace("conv1_bn.", "bn1.")

            name = name.replace("res2.", "layer1.")
            name = name.replace("res3.", "layer2.")
            name = name.replace("res4.", "layer3.")
            name = name.replace("res5.", "layer4.")

            name = name.replace(".branch2a.", ".conv1.")
            name = name.replace(".branch2a_bn.", ".bn1.")
            name = name.replace(".branch2b.", ".conv2.")
            name = name.replace(".branch2b_bn.", ".bn2.")
            name = name.replace(".branch2c.", ".conv3.")
            name = name.replace(".branch2c_bn.", ".bn3.")

            name = name.replace(".branch1.", ".downsample.0.")
            name = name.replace(".branch1_bn.", ".downsample.1.")

            name = name.replace("downsample.0.gn.s", "downsample.1.weight")
            name = name.replace("downsample.0.gn.bias", "downsample.1.bias")
            name = name.replace('fc1000', 'fc')

            if name == 'conv1.weight':
                name = 'stem.conv.weight'

            if name == 'bn1.weight':
                name = 'stem.bn.weight'
            if name == 'bn1.bias':
                name = 'stem.bn.bias'
            weight = torch.from_numpy(value)
            weight.requires_grad = True
            weights[name] = weight

    return weights


class Average():
    def __init__(self):
        self.val = 0
        self.count = 0
        self.sum = 0

    @type_check(object, float)
    def update(self, val):
        self.val = val
        self.count += 1
        self.sum += val

    @property
    def avg(self):
        return self.sum / self.count


class SmoothAverage():
    @type_check(object, int)
    def __init__(self, window_size=100):
        self.deque = deque(maxlen=window_size)
        self.val = 0
        self.sum = 0
        self.count = 0

    @type_check(object, float)
    def update(self, val):
        self.val = val
        self.deque.append(val)
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        return torch.tensor(list(self.deque)).mean().item()

    @property
    def global_avg(self):
        return self.sum / self.count


class Metric(dict):
    def __init__(self, metric=SmoothAverage, delimiter='    '):
        self['rpn_cls_loss'] = metric()
        self['rpn_reg_loss'] = metric()
        self['roi_cls_loss'] = metric()
        self['roi_reg_loss'] = metric()
        self['loss'] = metric()
        self['time'] = metric()
        self.delimiter = delimiter

    def update(self, k, v):
        if k not in self:
            raise KeyError('{}'.format(k))
        if isinstance(v, torch.Tensor):
            v = v.item()
        self[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.items():
            loss_str.append(
                '{}: {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(loss_str)


def last_checkpoint(cfg):
    """
    返回最新权重文件名
    """
    checkpoints = glob('{}/*.pth'.format(cfg.OUTPUT))
    final_model = '{}/model_final.pth'.format(cfg.OUTPUT)
    if final_model in checkpoints:
        return final_model
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(
            os.path.basename(x).split('.')[0].split('_')[-1]
        ))
        return checkpoints[-1]
    return checkpoints
