# -*- coding: utf-8 -*-

"""
@date: 2023/7/26 下午2:58
@file: __init__.py
@author: zj
@description: 
"""

import os


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
