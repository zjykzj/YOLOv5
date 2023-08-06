# -*- coding: utf-8 -*-

"""
@date: 2023/8/1 下午6:06
@file: t_yolov2_v2.py
@author: zj
@description: 
"""

import argparse
import os.path
from pathlib import Path
from copy import deepcopy

import torch

from yolo import ROOT
from yolo.model.yolov5 import Model
from yolo.utils.fileutil import check_yaml
from yolo.utils.misc import print_args, colorstr
from yolo.utils.logger import LOGGER
from yolo.utils.torchutil import select_device, profile
from yolo.data.auxiliary import check_dataset


def model_save(opt, device):
    model = Model(opt.cfg).to(device)

    hyp = "configs/hyps/hyp.scratch-low.yaml"
    hyp = check_yaml(hyp)

    data = 'configs/data/coco128.yaml'
    data_dict = check_dataset(data)  # check if None
    single_cls = False
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = None  # attach class weights
    model.names = names

    ckpt = {
        'model': deepcopy(model),
    }
    model_path = Path(os.path.basename(opt.cfg).replace(".yaml", ".pt"))
    LOGGER.info(colorstr("save model to ") + str(model_path.resolve()))
    torch.save(ckpt, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    parser.add_argument('--save', action='store_true', help='save model')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'configs' / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    elif opt.save:
        model_save(opt, device)

    else:  # report fused model summary
        model.fuse()
