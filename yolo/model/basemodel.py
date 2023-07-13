# -*- coding: utf-8 -*-

"""
@date: 2023/6/29 下午2:23
@file: basemodel.py
@author: zj
@description: 
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from common import Conv, DWConv

from yolo.utils.log import LOGGER
from yolo.utils.check import check_version
from yolo.utils.torch_utils import time_sync, fuse_conv_and_bn, model_info

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.num_classes = nc  # number of classes
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

        self.num_outputs = nc + 5  # number of outputs per anchor
        self.num_all_anchors = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2  # number of anchors

        self.grid = [torch.empty(0) for _ in range(self.num_all_anchors)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.num_all_anchors)]  # init anchor grid
        self.register_buffer('anchors',
                             torch.tensor(anchors).float().view(self.num_all_anchors, -1, 2))  # shape(nl,na,2)

        self.module = nn.ModuleList(
            nn.Conv2d(in_ch, self.num_outputs * self.num_anchors, 1) for in_ch in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.num_all_anchors):
            x[i] = self.module[i](x[i])  # conv
            B, _, H, W = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(B, self.num_anchors, self.num_outputs, H, W).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(W, H, i)

                # Detect (boxes only)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_classes + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(B, self.num_anchors * H * W, self.num_outputs))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        device_ = self.anchors[i].device
        type_ = self.anchors[i].dtype

        shape = 1, self.num_anchors, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=device_, dtype=type_), torch.arange(nx, device=device_, dtype=type_)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility

        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.num_anchors, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False):
        return self._forward_once(x, profile)  # single-scale inference, train

    def _forward_once(self, x, profile=False):
        outputs, dt = [], []  # outputs
        for module in self.model:
            if module.f != -1:  # if not from previous layer
                x = outputs[module.f] if isinstance(module.f, int) else \
                    [x if j == -1 else outputs[j] for j in module.f]  # from earlier layers
            if profile:
                self._profile_one_layer(module, x, dt)

            x = module(x)  # run

            outputs.append(x if module.layer_idx in self.save else None)  # save output
        return x

    def _profile_one_layer(self, module: Module, x: Tensor, dt: List):
        is_final = module == self.model[-1]  # is final layer, copy input as inplace fix
        # FLOPs
        total_ops = \
            thop.profile(module, inputs=(x.copy() if is_final else x,), verbose=False)[0] / 1E9 * 2 if thop else 0

        t = time_sync()
        for _ in range(10):
            module(x.copy() if is_final else x)
        dt.append((time_sync() - t) * 100)

        if module == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {total_ops:10.2f} {module.num_params:10.0f}  {module.type}')
        if is_final:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self
