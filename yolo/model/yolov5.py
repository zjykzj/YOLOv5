# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""
from typing import Dict, List

import os
import sys
import math
import argparse
import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from yolo.utils.autoanchor import check_anchor_order
from yolo.utils.general import check_yaml, make_divisible, print_args, colorstr
from yolo.utils.torch_utils import initialize_weights, profile, select_device

from yolo.utils.log import LOGGER

from basemodel import Detect, BaseModel
from common import Conv, Bottleneck, C3, SPP, SPPF
from common import Concat, Contract, Expand

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


class DetectionModel(BaseModel):
    def __init__(self,
                 cfg='yolov5s.yaml',
                 in_channel: int = 3,
                 num_classes: int = None,
                 anchors: List = None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            cfg = check_yaml(cfg)  # check YAML
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.anchors = anchors

        # Define model
        in_channel = self.yaml['in_channel'] = self.yaml.get('in_channel', in_channel)  # input channels
        if num_classes and num_classes != self.yaml['num_classes']:
            LOGGER.info(f"Overriding model.yaml num_classes={self.cfg['num_classes']} with num_classes={num_classes}")
            self.yaml['num_classes'] = num_classes  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), in_channels=[in_channel])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        module = self.model[-1]  # Detect()
        if isinstance(module, Detect):
            s = 256  # 2x min stride
            module.inplace = self.inplace
            forward = lambda x: self.forward(x)
            module.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, in_channel, s, s))])  # forward
            check_anchor_order(module)
            module.anchors /= module.stride.view(-1, 1, 1)

            self.stride = module.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    # YOLOv5 detection model

    def forward(self, x, profile=False):
        return self._forward_once(x, profile)  # single-scale inference, train

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.module, m.stride):  # from
            b = mi.bias.view(m.num_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.num_classes] += math.log(0.6 / (m.num_classes - 0.99999)) if cf is None else torch.log(
                cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


def parse_model(yaml: Dict, in_channels: List[int]):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")

    anchors, nc, depth_multiple, width_multiple, act = \
        yaml['anchors'], yaml['nc'], yaml['depth_multiple'], yaml['width_multiple'], yaml.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    num_outputs = num_anchors * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, out_channel = [], [], in_channels[-1]  # layers, savelist, ch out
    for layer_idx, (f, number, module, args) in enumerate(
            yaml['backbone'] + yaml['head']):  # from, number, module, args
        module = eval(module) if isinstance(module, str) else module  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        number = number_ = max(round(number * depth_multiple), 1) if number > 1 else number  # depth gain
        if module in {Conv, Bottleneck, C3, SPP, SPPF}:
            in_channel, out_channel = in_channels[f], args[0]
            if out_channel != num_outputs:  # if not output
                out_channel = make_divisible(out_channel * width_multiple, 8)

            args = [in_channel, out_channel, *args[1:]]
        elif module is nn.BatchNorm2d:
            args = [in_channels[f]]
        elif module is Concat:
            out_channel = sum(in_channels[x] for x in f)
        # TODO: channel, gw, gd
        elif module in {Detect, }:
            args.append([in_channels[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif module is Contract:
            out_channel = in_channels[f] * args[0] ** 2
        elif module is Expand:
            out_channel = in_channels[f] // args[0] ** 2
        else:
            out_channel = in_channels[f]

        module_ = nn.Sequential(*(module(*args) for _ in range(number))) if number > 1 else module(*args)  # module
        type_ = str(module)[8:-2].replace('__main__.', '')  # module type
        num_params = sum(x.numel() for x in module_.parameters())  # number params
        module_.layer_idx, module_.f, module_.type, module_.num_params = layer_idx, f, type_, num_params  # attach index, 'from' index, type, number params
        LOGGER.info(f'{layer_idx:>3}{str(f):>18}{number_:>3}{num_params:10.0f}  {type_:<40}{str(args):<30}')  # print

        save.extend(x % layer_idx for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(module_)
        if layer_idx == 0:
            in_channels = []
        in_channels.append(out_channel)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    opt.cfg = os.path.join(ROOT, "model/hub", opt.cfg)
    model = Model(opt.cfg).to(device)

    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        yaml_dir = os.path.join(ROOT, 'model/hub')
        for cfg in Path(yaml_dir).rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
