# -*- coding: utf-8 -*-

"""
@date: 2023/6/28 下午5:41
@file: basedataset.py
@author: zj
@description: 
"""

import cv2
import random

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class BaseDataset(Dataset):

    def __init__(self, target_size: int = 640):
        self.target_size = target_size

        self.img_paths = []
        self.labels = []
        self.indices = range(len(self.img_paths))

        self.mosaic_border = [-self.target_size // 2, -self.target_size // 2]

    def __getitem__(self, index) -> T_co:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)

    def set_img_size(self, img_size):
        self.target_size = img_size

    def get_img_size(self):
        return self.target_size

    def _load_image(self, index: int):
        img = cv2.imread(self.img_paths[index])
        h0, w0 = img.shape[:2]

        r = self.target_size / max(h0, w0)
        if r != 1:  # if sizes are not equal
            h = int(h0 * r)
            w = int(w0 * r)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LIENAR)

        return img, (h0, w0), img.shape[:2]

    def _load_mosaic(self, index):
        indices = [index] + random.choice(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)

        s = self.target_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        labels4 = []
        for i, index in enumerate(indices):
            # Load image
            img, (_, _), (h, w) = self._load_image(self, index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        return img4, labels4
