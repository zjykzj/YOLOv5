# -*- coding: utf-8 -*-

"""
@date: 2023/6/28 上午10:45
@file: albumentations.py
@author: zj
@description: 
"""

import cv2


def t_augment_hsv():
    from yolo.data.augmentations import augment_hsv

    img = cv2.imread("../../assets/coco/bus.jpg")
    cv2.imshow("src", img)
    augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5)
    cv2.imshow("dst", img)
    cv2.waitKey(0)


def t_hist_equalize():
    from yolo.data.augmentations import hist_equalize

    img = cv2.imread("../../assets/coco/bus.jpg")
    cv2.imshow("src", img)
    img = hist_equalize(img, clahe=True, bgr=False)
    cv2.imshow("dst", img)
    cv2.waitKey(0)


def t_letterbox():
    from yolo.data.augmentations import letterbox

    img = cv2.imread("../../assets/coco/bus.jpg")
    cv2.imshow("src", img)
    img, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False,
                                     scaleup=True, stride=32)
    cv2.imshow("dst", img)
    cv2.waitKey(0)


def t_random_perspective():
    from yolo.data.augmentations import random_perspective

    for _ in range(5):
        img = cv2.imread("../../assets/coco/bus.jpg")
        cv2.imshow("src", img)
        img, targets = random_perspective(img, targets=(),
                                          segments=(),
                                          degrees=10,
                                          translate=.1,
                                          scale=.1,
                                          shear=10,
                                          perspective=0.0,
                                          border=(0, 0))
        cv2.imshow("dst", img)
        cv2.waitKey(0)


def t_copy_paste():
    from yolo.data.augmentations import copy_paste

    img = cv2.imread("../../assets/coco/bus.jpg")
    cv2.imshow("src", img)

    labels = ()
    segments = ()
    img, labels, segments = copy_paste(img, labels, segments, p=0.5)

    cv2.imshow("dst", img)
    cv2.waitKey(0)


def t_cutout():
    from yolo.data.augmentations import cutout

    img = cv2.imread("../../assets/coco/bus.jpg")
    cv2.imshow("src", img)

    labels = ()
    labels = cutout(img, labels, p=0.5)

    cv2.imshow("dst", img)
    cv2.waitKey(0)


def t_mixup():
    from yolo.data.augmentations import mixup, letterbox

    img1 = cv2.imread("../../assets/coco/bus.jpg")
    cv2.imshow("src1", img1)
    img2 = cv2.imread("../../assets/coco/zidane.jpg")
    cv2.imshow("src2", img2)

    img1, ratio, (dw, dh) = letterbox(img1, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False,
                                      scaleup=True, stride=32)
    cv2.imshow("letterbox1", img1)
    img2, ratio, (dw, dh) = letterbox(img2, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False,
                                      scaleup=True, stride=32)
    cv2.imshow("letterbox2", img2)

    for _ in range(10):
        labels = ()
        img, labels = mixup(img1, labels, img2, labels)

        cv2.imshow("dst", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # t_augment_hsv()
    # t_hist_equalize()
    # t_letterbox()
    # t_random_perspective()
    # t_copy_paste()
    # t_cutout()
    t_mixup()
