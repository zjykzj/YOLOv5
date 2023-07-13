# -*- coding: utf-8 -*-

"""
@date: 2023/4/23 上午9:55
@file: vocevaluator.py
@author: zj
@description:
"""
from typing import List

import os
import numpy as np

from .evaluator import Evaluator
from .voc_eval import voc_eval
from yolo.util.box_utils import yolobox2label


def do_python_eval(all_boxes_dict, classes, VOCdevkit_dir, year=2012, split='val'):
    annotation_path = os.path.join(VOCdevkit_dir, f'VOC{year}', 'Annotations', '{:s}.xml')
    imageset_file = os.path.join(VOCdevkit_dir, f'VOC{year}', 'ImageSets', 'Main', f'{split}.txt')

    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    aps = []

    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        detections = all_boxes_dict[cls]
        rec, prec, ap = voc_eval(detections, annotation_path, imageset_file, cls,
                                 ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

    return np.mean(aps)


class VOCEvaluator(Evaluator):

    def __init__(self, classes, VOCdevkit_dir, year=2012, split='val', save=False):
        super().__init__()
        self.classes = classes
        self.VOCdevkit_dir = VOCdevkit_dir
        self.year = year
        self.split = split
        self.save = save

        self.all_boxes_dict = dict()
        self._init_dict()

    def _init_dict(self):
        self.all_boxes_dict.clear()
        for cls_name in self.classes:
            self.all_boxes_dict[cls_name] = list()

    def put(self, outputs: List[List], img_info: List):
        assert isinstance(img_info, list)
        assert len(img_info) == 8, len(img_info)
        # super().put(outputs, img_info)

        image_name = str(img_info[-1])

        for output in outputs:
            assert len(output) == 7, output
            x1 = float(output[0])
            y1 = float(output[1])
            x2 = float(output[2])
            y2 = float(output[3])
            y1, x1, y2, x2 = yolobox2label([y1, x1, y2, x2], img_info[:6])
            # 置信度 = 目标置信度 * 分类置信度
            # object score * class score
            score = float(output[4].data.item() * output[5].data.item())
            # 分类标签
            label_name = self.classes[int(output[6])]

            self.all_boxes_dict[label_name].append([
                image_name, score, max(0., x1), max(0., y1), max(0., x2), max(0., y2)
                # image_name, score, x1, y1, x2, y2
            ])

    def result(self):
        # super().result()
        if self.save:
            save_dir = os.path.join(self.VOCdevkit_dir, 'results')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            for cls_name, all_boxes in self.all_boxes_dict.items():
                result_cls_path = os.path.join(save_dir, f'{cls_name}.txt')
                with open(result_cls_path, 'w') as f:
                    for item in all_boxes:
                        item = [str(x) for x in item]
                        line = ' '.join(item)
                        f.write(line + '\n')
                print(f"Save to {result_cls_path}")
        ap50 = do_python_eval(self.all_boxes_dict,
                              self.classes,
                              self.VOCdevkit_dir,
                              self.year,
                              self.split)
        self._init_dict()

        return -1., ap50
