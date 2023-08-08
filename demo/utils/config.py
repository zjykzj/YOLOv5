# -*- coding: utf-8 -*-

"""
@date: 2023/8/1 下午5:54
@file: config.py
@author: zj
@description: 
"""

import os
import argparse

import numpy as np
from pathlib import Path

from yolo.utils.fileutil import check_yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))


def tt():
    print(FILE, ROOT, RANK)
    print(FILE.parents[0])
    print(FILE.parents[1])
    print(FILE.parent)

    dataset_dir = ROOT.parent / 'datasets'
    print(dataset_dir)

    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
    print('ROOT:', ROOT)
    DATASETS_DIR = Path(os.getenv('YOLOv5_DATASETS_DIR', ROOT.parent / 'datasets'))  # global datasets directory
    print('DATASETS_DIR:', DATASETS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')

    opt = parser.parse_args()
    cfg = check_yaml(opt.cfg)  # check YAML
    print(cfg)

    # 加载配置文件
    import yaml  # for torch hub

    with open(cfg, encoding='ascii', errors='ignore') as f:
        yaml = yaml.safe_load(f)  # model dict
    print(yaml)
    print(np.array(yaml['anchors']).shape)
    print(np.array(yaml['anchors'])[0].shape)
    print(np.array(yaml['backbone'], dtype=object).shape)
    print(np.array(yaml['head'], dtype=object).shape)
