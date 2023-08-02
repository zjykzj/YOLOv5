# -*- coding: utf-8 -*-

"""
@date: 2023/8/2 上午10:05
@file: classify.py
@author: zj
@description: 
"""

import os
import torch

from copy import deepcopy
from pathlib import Path

from yolo import RANK, LOCAL_RANK, DATASETS_DIR
from yolo.data.dataloaders import create_classification_dataloader
from yolo.model.yolov5 import Model, ClassificationModel

if __name__ == '__main__':
    cfg = 'configs/models/yolov5s.yaml'
    model = Model(cfg)
    print(model)

    model = ClassificationModel(model=model, nc=1000, cutoff=10)
    print(model)

    # Dataloaders
    data = Path('imagenet')
    data_dir = data if data.is_dir() else (DATASETS_DIR / data)
    print("data_dir", data_dir)
    imgsz = 224
    bs = 64
    nw = min(os.cpu_count() - 1, 8)
    WORLD_SIZE = 1
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    trainloader = create_classification_dataloader(path=data_dir / 'train',
                                                   imgsz=imgsz,
                                                   batch_size=bs // WORLD_SIZE,
                                                   augment=True,
                                                   cache=False,
                                                   rank=LOCAL_RANK,
                                                   workers=nw)
    test_dir = data_dir / 'test' if (data_dir / 'test').exists() else data_dir / 'val'  # data/test or data/val
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(path=test_dir,
                                                      imgsz=imgsz,
                                                      batch_size=bs // WORLD_SIZE * 2,
                                                      augment=False,
                                                      cache=False,
                                                      rank=-1,
                                                      workers=nw)

    model.names = trainloader.dataset.classes  # attach class names
    model.transforms = testloader.dataset.torch_transforms  # attach inference transforms

    ckpt = {
        'model': deepcopy(model).half(),  # deepcopy(de_parallel(model)).half(),
    }

    # Save last, best and delete
    torch.save(ckpt, "tmp-cls.pt")
