# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午2:35
@file: transform.py
@author: zj
@description: 
"""
from typing import Dict, List

import cv2

from numpy import ndarray
import numpy as np


def pad_and_resize(src_img, labels, dst_size, jitter=0.0, random_replacing=False):
    """
    src_img: [H, W, 3]
    labels: [K, 5] cls_id/x1/y1/b_w/b_h
    """
    src_h, src_w = src_img.shape[:2]

    dh = jitter * src_h
    dw = jitter * src_w
    new_ratio = (src_w + np.random.uniform(low=-dw, high=dw)) / (src_h + np.random.uniform(low=-dh, high=dh))
    if new_ratio < 1:
        # 高大于宽
        # 设置目标大小为高，等比例缩放宽，剩余部分进行填充
        dst_h = dst_size
        dst_w = dst_size * new_ratio
    else:
        # 宽大于等于高
        # 设置目标大小为宽，等比例缩放高，剩余部分进行填充
        dst_w = dst_size
        dst_h = dst_size / new_ratio
    dst_w = int(dst_w)
    dst_h = int(dst_h)

    # 计算ROI填充到结果图像的左上角坐标
    if random_replacing:
        dx = int(np.random.uniform(dst_size - dst_w))
        dy = int(np.random.uniform(dst_size - dst_h))
    else:
        # 等比例进行上下或者左右填充
        dx = (dst_size - dst_w) // 2
        dy = (dst_size - dst_h) // 2

    # 先进行图像缩放，然后创建目标图像，填充ROI区域
    resized_img = cv2.resize(src_img, (dst_w, dst_h))
    padded_img = np.ones((dst_size, dst_size, 3), dtype=np.uint8) * 127
    padded_img[dy:dy + dst_h, dx:dx + dst_w, :] = resized_img

    if len(labels) > 0:
        # 进行缩放以及填充后需要相应的修改坐标位置
        labels[:, 1] = labels[:, 1] / src_w * dst_w + dx
        labels[:, 2] = labels[:, 2] / src_h * dst_h + dy
        # 对于宽/高而言，仅需缩放对应比例即可，不需要增加填充坐标
        labels[:, 3] = labels[:, 3] / src_w * dst_w
        labels[:, 4] = labels[:, 4] / src_h * dst_h

    img_info = [src_h, src_w, dst_h, dst_w, dx, dy, dst_size]
    return padded_img, labels, img_info


def left_right_flip(img: ndarray, labels: ndarray):
    dst_img = cv2.flip(img, 1)

    if len(labels) > 0:
        h, w = img.shape[:2]
        # 左右翻转，所以宽/高不变，变换左上角坐标(x1, y1)和右上角坐标(x2, y1)进行替换
        # x2 = x1 + box_w
        x2 = labels[:, 1] + labels[:, 3]
        # y1/box_w/bo_h不变，仅变换x1 = w - x2
        labels[:, 1] = w - x2

    return dst_img, labels


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def color_dithering(src_img: ndarray, hue: float, saturation: float, exposure: float):
    """
    src_img: 图像 [H, W, 3]
    hue: 色调
    saturation: 饱和度
    exposure: 曝光度
    """
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    img = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue

    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0

    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)

    return img


def bgr2rgb(img: ndarray):
    # BGR -> RGB
    return img[:, :, ::-1]


def print_info(index, img, bboxes, bboxes_xxyy, img_size, img_info):
    print(f"index: {index}")
    print(f"img: {img.shape}")
    print(f"bboxes:\n{bboxes}")
    print(f"bboxes_xxyy:\n{bboxes_xxyy}")
    print(f"img_size: {img_size}")
    print(f"img_info: {img_info}")
    print(f"bboxes_xxyy <= img_size:\n{bboxes_xxyy <= float(img_size)}")


class Transform(object):

    def __init__(self, cfg: Dict, is_train: bool = True):
        self.is_train = is_train

        # 空间抖动
        self.jitter = cfg['AUGMENTATION']['JITTER']
        assert self.jitter > 0
        # 随机放置
        self.random_placing = cfg['AUGMENTATION']['RANDOM_PLACING']
        # 左右翻转
        self.is_flip = cfg['AUGMENTATION']['RANDOM_FLIP']
        # 颜色抖动
        self.is_color = cfg['AUGMENTATION']['IS_COLOR']
        self.hue = cfg['AUGMENTATION']['HUE']
        self.saturation = cfg['AUGMENTATION']['SATURATION']
        self.exposure = cfg['AUGMENTATION']['EXPOSURE']
        # 颜色空间转换
        self.is_rgb = cfg['AUGMENTATION']['IS_RGB']

    def __call__(self, index: int, img_size: int, img: ndarray, labels: ndarray):
        return self.forward(index, img_size, img, labels)

    def forward(self, index: int, img_size: int, img: ndarray, labels: ndarray):
        assert len(img.shape) == 3 and img.shape[2] == 3
        assert len(labels) == 0 or (len(labels.shape) == 2 or labels.shape[1] == 5)

        if self.is_train:
            img, labels, img_info = self._get_train(index, img_size, img, labels)
        else:
            img, labels, img_info = self._get_val(index, img_size, img, labels)
        return img, labels, img_info

    def _get_train(self, index: int, img_size: int, img: ndarray, labels: ndarray):
        """
        1. Pad+Resize
        2. BGR2RGB
        3. Horizontal Flip
        4. Color Jitter
        """
        img, labels, img_info = pad_and_resize(img, labels, img_size, self.jitter, self.random_placing)
        if self.is_rgb:
            img = bgr2rgb(img)
        if self.is_flip and np.random.randn() > 0.5:
            img, labels = left_right_flip(img, labels)
        if self.is_color:
            img = color_dithering(img, self.hue, self.saturation, self.exposure)

        return img, labels, img_info

    def _get_val(self, index: int, img_size: int, img: ndarray, labels: ndarray):
        """
        1. Pad+Resize
        2. BGR2RGB
        """
        img, labels, img_info = pad_and_resize(img, labels, img_size, jitter=0., random_replacing=False)
        if self.is_rgb:
            img = bgr2rgb(img)

        return img, labels, img_info
