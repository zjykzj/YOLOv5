# -*- coding: utf-8 -*-

"""
@date: 2023/4/23 下午4:10
@file: cocoevaluator.py
@author: zj
@description: 
"""
from typing import List
import json
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo.util.box_utils import yolobox2label
from .evaluator import Evaluator


class COCOEvaluator(Evaluator):

    def __init__(self, coco: COCO):
        super().__init__()
        self.cocoGt = coco

        self.class_ids = sorted(self.cocoGt.getCatIds())

        self._init_list()

    def _init_list(self):
        self.ids = list()
        self.data_list = list()

    def put(self, outputs: List[List], img_info: List):
        assert isinstance(img_info, list)
        assert len(img_info) == 8, len(img_info)

        id_ = int(img_info[-1])
        self.ids.append(id_)

        for output in outputs:
            x1 = float(output[0])
            y1 = float(output[1])
            x2 = float(output[2])
            y2 = float(output[3])
            # 分类标签
            label = self.class_ids[int(output[6])]
            # 转换到原始图像边界框坐标
            box = yolobox2label([y1, x1, y2, x2], img_info[:6])
            # [y1, x1, y2, x2] -> [x1, y1, w, h]
            bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
            # 置信度 = 目标置信度 * 分类置信度
            score = float(output[4].data.item() * output[5].data.item())  # object score * class score
            # 保存计算结果
            A = {"image_id": id_, "category_id": label, "bbox": bbox,
                 "score": score, "segmentation": []}  # COCO json format
            self.data_list.append(A)

    def result(self):
        annType = ['segm', 'bbox', 'keypoints']

        ap50_95, ap50 = 0, 0

        # 计算完成所有测试图像的预测结果后
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(self.data_list) > 0:
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(self.data_list, open(tmp, 'w'))
            cocoDt = self.cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.cocoGt, cocoDt, annType[1])
            cocoEval.params.imgIds = self.ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # AP50_95, AP50
            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]

        self._init_list()
        return ap50_95, ap50
