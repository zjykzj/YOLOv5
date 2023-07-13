# -*- coding: utf-8 -*-

"""
@date: 2023/6/29 下午5:15
@file: build.py
@author: zj
@description: 
"""
from typing import Dict

import math

import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Optimizer

from yolo.utils.general import colorstr
from yolo.utils.log import LOGGER


def build_optimizer(model, cfg: Dict):
    # Config
    name = cfg['OPTIM']['NAME']
    lr = cfg['OPTIM']['LR']
    momentum = cfg['OPTIM']['MOMENTUM']
    decay = cfg['OPTIM']['DECAY']

    nbs = 64  # nominal batch size
    batch_size = cfg['DATA']['BATCH_SIZE']
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    decay *= batch_size * accumulate / nbs  # scale weight_decay

    # Optimizer
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def build_lr_scheduler(optimizer: Optimizer, cfg: Dict):
    # Config
    # cosine LR scheduler
    name = cfg['LR_SCHEDULER']['NAME']
    # final OneCycleLR learning rate (lr0 * lrf)
    lrf = cfg['LR_SCHEDULER']['LRF']
    # total training epochs
    epochs = cfg['LR_SCHEDULER']['EPOCH']

    # Scheduler
    if name == 'cosine':
        lf = one_cycle(1, lrf, epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    return scheduler, lf
