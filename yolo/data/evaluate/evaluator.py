# -*- coding: utf-8 -*-

"""
@date: 2023/4/26 下午2:13
@file: evaluator.py
@author: zj
@description: 
"""
from typing import List
from abc import ABC


class Evaluator(ABC):

    def put(self, outputs: List[List], img_info: List):
        pass

    def result(self):
        pass
