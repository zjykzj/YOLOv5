# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:19
@file: build.py
@author: zj
@description: 
"""
from typing import Dict

import os
import torch

import numpy as np

from .dataset.cocodataset import COCODataset
from .dataset.vocdataset import VOCDataset
from .evaluate.cocoevaluator import COCOEvaluator
from .evaluate.vocevaluator import VOCEvaluator
from .transform import Transform
from .target import Target

from yolo.utils.log import LOGGER


def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Data preprocess
    # [B, H, W, C] -> [B, C, H, W] -> Normalize
    images = torch.from_numpy(np.array(images, dtype=float)).permute(0, 3, 1, 2).contiguous() / 255

    if not isinstance(targets[0], Target):
        targets = torch.stack(targets)

    return images, targets


import math


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def build_data(model, cfg: Dict, data_root: str, is_train: bool = False, is_distributed: bool = False):
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    img_size = cfg['TRAIN']['IMGSIZE'] if is_train else cfg['TEST']['IMGSIZE']
    img_size = check_img_size(img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    data_type = cfg['DATA']['TYPE']
    max_det_num = cfg['DATA']['MAX_NUM_LABELS']

    sampler = None
    transform = Transform(cfg, is_train=is_train)
    dataset_name = cfg['TRAIN']['DATASET_NAME'] if is_train else cfg['TEST']['DATASET_NAME']

    evaluator = None
    if 'PASCAL VOC' == data_type:
        dataset = VOCDataset(root=data_root,
                             name=dataset_name,
                             train=is_train,
                             transform=transform,
                             target_transform=None,
                             target_size=img_size,
                             max_det_nums=max_det_num
                             )
        if not is_train:
            VOCdevkit_dir = os.path.join(data_root, cfg['TEST']['VOC'])
            year = cfg['TEST']['YEAR']
            split = cfg['TEST']['SPLIT']
            evaluator = VOCEvaluator(dataset.classes, VOCdevkit_dir, year=year, split=split)
    elif 'COCO' == data_type:
        dataset = COCODataset(root=data_root,
                              name=dataset_name,
                              train=is_train,
                              transform=transform,
                              target_transform=None,
                              target_size=img_size,
                              max_det_nums=max_det_num
                              )
        if not is_train:
            evaluator = COCOEvaluator(dataset.coco)
    else:
        raise ValueError(f"{data_type} doesn't supports")

    if is_distributed and is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg['DATA']['BATCH_SIZE'],
                                             shuffle=(sampler is None and is_train),
                                             num_workers=cfg['DATA']['WORKERS'],
                                             sampler=sampler,
                                             pin_memory=True,
                                             collate_fn=custom_collate
                                             )

    return dataloader, sampler, evaluator, img_size
