# -*- coding: utf-8 -*-

"""
@date: 2023/8/1 下午5:54
@file: config.py
@author: zj
@description: 
"""

import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))

if __name__ == '__main__':
    print(FILE, ROOT, RANK)
    print(FILE.parents[0])
    print(FILE.parents[1])
    print(FILE.parent)

    dataset_dir = ROOT.parent / 'datasets'
    print(dataset_dir)
