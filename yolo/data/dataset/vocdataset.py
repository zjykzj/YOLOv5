# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:20
@file: vocdataset.py
@author: zj
@description: 
"""
from typing import Optional, List, Union
import os
import cv2
import glob
import copy

import numpy as np
from numpy import ndarray

import torch
from torch.utils.data.dataset import T_co

from ..target import Target
from yolo.util.box_utils import label2yolobox

from .basedataset import BaseDataset
from ..transforms import Transform


class VOCDataset(BaseDataset):
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self,
                 root: str,
                 name: str,
                 train: bool = True,
                 transform: Transform = None,
                 target_size: int = 416,
                 max_det_nums: int = 50):
        super().__init__(target_size)
        self.root = root
        self.name = name
        self.train = train
        self.transform = transform
        assert self.transform is not None
        assert isinstance(self.transform, Transform)
        self.target_size = target_size
        self.max_det_nums = max_det_nums

        image_dir = os.path.join(root, name, 'images')
        label_dir = os.path.join(root, name, 'labels')

        self.image_path_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        assert len(self.image_path_list) == len(self.label_path_list)

        label_list = list()
        for image_path, label_path in zip(self.image_path_list, self.label_path_list):
            img_name = os.path.basename(image_path).rstrip('.jpg')
            label_name = os.path.basename(label_path).rstrip('.txt')
            assert img_name == label_name

            image = cv2.imread(image_path)
            img_h, img_w = image.shape[:2]

            sub_label_list = list()
            # [[cls_id, x_center, y_center, box_w, box_h], ]
            # The coordinate size is relative to the width and height of the image
            boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
            if len(boxes.shape) == 1:
                boxes = [boxes]
            for label, xc, yc, box_w, box_h in boxes:
                # xc/yc/box_w/box_h -> x1/y1/box_w/box_h
                x_min = (xc - 0.5 * box_w) * img_w
                y_min = (yc - 0.5 * box_h) * img_h
                assert x_min >= 0 and y_min >= 0
                box_w = box_w * img_w
                box_h = box_h * img_h

                # 转换成原始大小，方便后续图像预处理阶段进行转换和调试
                sub_label_list.append([label, x_min, y_min, box_w, box_h])
            label_list.append(np.array(sub_label_list, dtype=float))

        self.label_list = label_list
        self.num_classes = len(self.classes)

    def __getitem__(self, index) -> T_co:
        image, labels, shapes = self.build_image(index)

        target = self.build_target(labels, img_info, img_file)

        return image, target

    def build_image(self, index: int):
        # img_file = self.image_path_list[index]
        labels = copy.deepcopy(self.label_list[index])

        if self.transform.is_mosaic:
            image, labels = self._load_mosaic(index)
            shapes = None
            image, labels, _ = self.transform.forward(image, labels)
        else:
            image, (h0, w0), (h, w) = self._load_image(index)

            image, labels, pad = self.transform.forward(image, labels)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.ascontiguousarray(image)

        return image, labels, shapes

    def build_target(self, labels: ndarray, img_info: List, img_id: Union[int, str]):
        assert isinstance(labels, ndarray)
        target = torch.zeros((self.max_det_nums, 5))
        if len(labels) > 0:
            # 将数值缩放到[0, 1]区间
            labels[:, 1:] = labels[:, 1:] / self.target_size
            # [x1, y1, w, h] -> [xc, yc, w, h]
            labels[:, 1:] = label2yolobox(labels[:, 1:])

            for i, label in enumerate(labels[:self.max_det_nums]):
                target[i, :] = torch.from_numpy(label)

        if self.train:
            return target
        else:
            image_name = os.path.splitext(os.path.basename(img_id))[0]
            target = Target(target, img_info, image_name)
        return target
