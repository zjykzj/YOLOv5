# -*- coding: utf-8 -*-

"""
@date: 2023/7/26 下午6:26
@file: onecycle.py
@author: zj
@description: 
"""

import math

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1
