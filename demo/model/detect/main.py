# -*- coding: utf-8 -*-

"""
@date: 2023/8/4 下午6:02
@file: main.py
@author: zj
@description:

Usage - Show Model Info:
    $ python demo/model/detect/main.py --cfg yolov5l.yaml

"""

import argparse
import os.path
from pathlib import Path

import torch

from yolo import ROOT
from yolo.utils.fileutil import check_yaml
from yolo.utils.misc import print_args, colorstr
from yolo.utils.torchutil import select_device, profile
from yolo.utils.gitutil import check_git_info

GIT_INFO = check_git_info()


def create_model(cfg, device):
    assert isinstance(cfg, str) and cfg.endswith('.yaml')
    cfg_name = os.path.basename(cfg)

    if cfg_name in ['yolov5l.yaml', 'yolov5m.yaml', 'yolov5n.yaml', 'yolov5s.yaml', 'yolov5x.yaml']:
        from yolov5 import Model
        model = Model(cfg).to(device)
    elif cfg_name in ['yolov3.yaml', 'yolov3-tiny.yaml', ]:
        from yolov3 import Model
        model = Model(cfg).to(device)
    else:
        raise ValueError(f"No supported {colorstr(cfg)}.")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')

    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    # model = Model(opt.cfg).to(device)
    model = create_model(opt.cfg, device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'configs' / 'models').rglob('yolo*.yaml'):
            try:
                # _ = Model(cfg)
                _ = create_model(cfg.name, device)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
