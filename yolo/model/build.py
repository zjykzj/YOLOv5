# -*- coding: utf-8 -*-

"""
@date: 2023/7/21 下午1:54
@file: build.py
@author: zj
@description: 
"""

import os

from pathlib import Path

import torch

from . import check_amp

from ..utils.downloads import attempt_download
from ..utils.misc import colorstr, intersect_dicts
from ..utils.logger import LOGGER
from ..utils.file_util import check_suffix
from ..utils.torch_util import torch_distributed_zero_first

from yolov5 import Model
from loss import ComputeLoss

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def build_model(hyp, opt, nc, device):
    cfg, weights, resume, freeze = opt.cfg, opt.weights, opt.resume, opt.freeze

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        ckpt = None
        csd = None
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    return pretrained, model, amp, ckpt, csd


def build_criterion(model):
    return ComputeLoss(model)
