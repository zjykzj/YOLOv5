# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 下午5:59
@file: cocodataset.py
@author: zj
@description: 
"""
from typing import Optional, List, Union
import cv2
import os.path

import numpy as np
from numpy import ndarray
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset

from ..transform import Transform
from yolo.util.box_utils import label2yolobox
from ..target import Target


def get_coco_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )
    coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors


class COCODataset(Dataset):
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self,
                 root: str,
                 name: str = 'val2017',
                 train: bool = True,
                 transform: Transform = None,
                 target_transform: Transform = None,
                 target_size: int = 416,
                 max_det_nums=50,
                 min_size: int = 1,
                 model_type: str = 'YOLO',
                 ):
        self.root = root
        self.name = name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size
        # 单张图片预设的最大真值边界框数目
        self.max_det_nums = max_det_nums
        self.min_size = min_size
        self.model_type = model_type

        if 'train' in self.name:
            json_file = 'instances_train2017.json'
        elif 'val' in self.name:
            json_file = 'instances_val2017.json'
        else:
            raise ValueError(f"{name} does not match any files")
        annotation_file = os.path.join(self.root, 'annotations', json_file)
        self.coco = COCO(annotation_file)

        # 获取图片ID列表
        self.ids = self.coco.getImgIds()
        # 获取类别ID
        self.class_ids = sorted(self.coco.getCatIds())

    def __getitem__(self, index):
        # 获取ID
        img_id = self.ids[index]
        # 获取图像路径
        img_file = os.path.join(self.root, 'images', self.name, '{:012}'.format(img_id) + '.jpg')
        # 获取标注框信息
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                label = self.class_ids.index(anno['category_id'])
                # bbox: [x1, y1, w, h]
                bbox = anno['bbox']

                labels.append([label, *bbox])
        labels = np.array(labels)

        image, img_info, labels = self.build_image(index, img_file, labels)
        target = self.build_target(labels, img_info, img_id)
        return image, target

    def build_image(self, index: int, img_file: str, labels: ndarray):
        # 读取图像
        image = cv2.imread(img_file)

        # import copy
        # src_img = copy.deepcopy(image)
        # if len(labels) > 0:
        #     for box in labels[:, 1:]:
        #         x_min, y_min, box_w, box_h = box
        #         cv2.rectangle(src_img, (int(x_min), int(y_min)), (int(x_min + box_w), int(y_min + box_h)),
        #                       (0, 0, 255), 1)
        # cv2.imshow('src_img', src_img)
        # # cv2.imwrite('src_img.jpg', src_img)

        img_info = None
        if self.transform is not None:
            image, labels, img_info = self.transform(index, self.target_size, image, labels)

        # dst_img = copy.deepcopy(image).astype(np.uint8)
        # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        # if len(labels) > 0:
        #     for box in labels[:, 1:]:
        #         x_min, y_min, box_w, box_h = box
        #         cv2.rectangle(dst_img, (int(x_min), int(y_min)), (int(x_min + box_w), int(y_min + box_h)),
        #                       (0, 0, 255), 1)
        # cv2.imshow('dst_img', dst_img)
        # cv2.waitKey(0)
        # # cv2.imwrite("dst_img.jpg", dst_img)

        return image, img_info, labels

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
            target = Target(target, img_info, img_id)
        return target

    def __len__(self):
        return len(self.ids)

    def set_img_size(self, img_size):
        self.target_size = img_size

    def get_img_size(self):
        return self.target_size
