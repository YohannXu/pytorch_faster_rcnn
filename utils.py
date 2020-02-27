# -*- coding: utf-8 -*-
# Author: yohannxu
# Email: yuhannxu@gmail.com
# CreateTime: 2020-02-27 23:41:13
# Description: 提供各种工具

import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import wraps
from inspect import signature


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

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取目标函数输入参数
            # {name: value}
            input_args = sig.bind(*args, **kwargs)
            for name, value in input_args.arguments.items():
                if name in arg_types:
                    if not isinstance(value, arg_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, arg_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate
