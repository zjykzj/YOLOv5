# -*- coding: utf-8 -*-

"""
@date: 2023/7/3 下午2:19
@file: build.py
@author: zj
@description: 
"""
from typing import Dict

import os

from pathlib import Path
from contextlib import contextmanager

import torch
import torch.distributed as dist

from yolov5 import Model
from yolo.utils.log import LOGGER

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def build_model(cfg: Dict, device: torch.device):
    # Model
    pretrained = cfg['MODEL']['PRETRAINED']
    if pretrained:
        ckpt = torch.load(pretrained, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model.load_state_dict(ckpt, strict=False)  # load
        LOGGER.info(f'Loading pretrained from {pretrained}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    return model


def build_criterion(model):
    from loss import ComputeLoss
    return ComputeLoss(model)
