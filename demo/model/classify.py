# -*- coding: utf-8 -*-

"""
@date: 2023/8/2 上午10:05
@file: classify.py
@author: zj
@description:

Usage - Show Model Info:
    $ python demo/model/classify.py --cfg yolov5s.yaml

Usage - Save Model:
    $ python demo/model/classify.py --cfg yolov5s.yaml --save

"""
import os.path

import torch
import argparse

from datetime import datetime
from pathlib import Path

from yolo import ROOT
from yolo.model.yolov5 import Model, ClassificationModel, DetectionModel
from yolo.utils.fileutil import check_yaml, increment_path
from yolo.utils.torchutil import select_device
from yolo.utils.gitutil import check_git_info
from yolo.utils.misc import print_args, colorstr
from yolo.utils.torchutil import profile
from yolo.utils.logger import LOGGER

GIT_INFO = check_git_info()


def cls_save(opt, model):
    ckpt = {
        'model': model,
        'git': GIT_INFO,  # {remote, branch, commit} if a git repo
        'date': datetime.now().isoformat()
    }

    # Save
    save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    m_name = os.path.splitext(os.path.basename(opt.cfg))[0] + "-cls.pt"
    m_pth = os.path.join(save_dir, m_name)

    LOGGER.info(f"save to {colorstr(m_pth)}")
    torch.save(ckpt, m_pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--cutoff', type=int, default=None, help='Model layer cutoff index for Classify() head')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')

    parser.add_argument('--save', action='store_true', help='save cls model')
    parser.add_argument('--project', default=ROOT / 'runs/cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
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

    elif opt.save:
        cls_save(opt, model)

    else:  # report fused model summary
        model.fuse()
