# -*- coding: utf-8 -*-

"""
@date: 2023/8/2 上午10:05
@file: classify.py
@author: zj
@description: 
"""

import torch
import argparse

from pathlib import Path

from yolo import ROOT
from yolo.model.yolov5 import Model, ClassificationModel, DetectionModel
from yolo.utils.fileutil import check_yaml
from yolo.utils.torchutil import select_device
from yolo.utils.misc import print_args
from yolo.utils.torchutil import profile
from yolo.utils.logger import LOGGER

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--cutoff', type=int, default=None, help='Model layer cutoff index for Classify() head')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)
    if isinstance(model, DetectionModel):
        LOGGER.warning("WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model yolov5s-cls.pt'")
        model = ClassificationModel(model=model, nc=1000, cutoff=opt.cutoff or 10)  # convert to classification model
    assert isinstance(model, ClassificationModel)

    # Options
    if opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    else:  # report fused model summary
        model.fuse()
