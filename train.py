# -*- coding: utf-8 -*-

"""
@date: 2023/6/29 下午5:11
@file: train.py
@author: zj
@description: 
"""
from typing import Dict, List

import os
import platform

import argparse
from argparse import Namespace
from pathlib import Path
from subprocess import check_output

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from yolo.data.build import build_data
from yolo.model.build import build_model, build_criterion
from yolo.optim.build import build_optimizer, build_lr_scheduler
from yolo.utils.general import check_version, init_seeds
from yolo.utils.log import LOGGER

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def smart_DDP(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def train(args: Namespace, cfg: Dict, device: torch.device):
    # Config
    init_seeds(args.seed + 1 + RANK, deterministic=True)
    cuda = device.type != 'cpu'
    save_dir = cfg['TRAIN']['OUTPUT_DIR']

    # ----------------------------------------------- Base
    # Model
    model = build_model(cfg, device)
    # Criterion
    criterion = build_criterion(model)  # init loss class
    # Data
    train_loader, sampler, evaluator, img_size = build_data(model, cfg, args.data, is_train=True)
    # Optimizer
    optimizer = build_optimizer(model, cfg)
    # Lr_scheduler
    scheduler, lf = build_lr_scheduler(optimizer, cfg)

    # ----------------------------------------------- Upgrade
    # Resume
    global start_epoch
    start_epoch = int(cfg['TRAIN']['START_EPOCH'])
    epochs = int(cfg['TRAIN']['MAX_EPOCHS'])
    eval_epoch = int(cfg['TRAIN']['EVAL_EPOCH'])
    best_fitness, start_epoch = 0.0, 0
    if args.resume:
        def resume():
            if os.path.isfile(args.resume):
                LOGGER.info("=> resume: '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=device)
                global start_epoch
                start_epoch = checkpoint['epoch']

                if not args.distributed:
                    state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
                else:
                    state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)

                if hasattr(checkpoint, 'optimizer'):
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if hasattr(checkpoint, 'lr_scheduler'):
                    scheduler.load_state_dict(checkpoint['lr_scheduler'])

                LOGGER.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))
            else:
                LOGGER.info("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # SyncBatchNorm
    if cuda and args.sync_bn and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)
        LOGGER.info('Using DistributedDataParallel()')

    # ----------------------------------------------- Train
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    LOGGER.info(f'Image sizes {img_size} train, {img_size} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()




        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch YOLO Training')
    parser.add_argument('data', metavar='DIR', type=str, help='path to dataset')

    parser.add_argument('-c', "--cfg", metavar='CFG', type=str, default='configs/yolov3_voc.cfg',
                        help='Path to config file (Default: configs/yolov3_voc.cfg)')

    parser.add_argument('-r', '--resume', metavar='RESUME', type=str, default=None,
                        help='Path to latest checkpoint (Default: None)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='Evaluate model on validation set')

    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int,
                        help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--sync_bn', action='store_true', help='Enabling apex sync BN.')

    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-p', '--print-freq', metavar='N', type=int, default=10, help='Print frequency (Default: 10)')
    args = parser.parse_args()
    LOGGER.info(f"args: {args}")

    # load cfg
    LOGGER.info(f"cfg file: {args.cfg}")
    with open(args.cfg, 'r') as f:
        import yaml
        cfg = yaml.safe_load(f)

    return args, cfg


def git_describe(path=ROOT):  # path must be a directory
    # Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''


from datetime import datetime


def file_date(path=__file__):
    # Return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def main():
    args, cfg = parse_args()

    # Init
    batch_size = cfg['DATA']['BATCH_SIZE']
    device = select_device(args.device, batch_size=batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid BATCH_SIZE'
        assert batch_size % WORLD_SIZE == 0, f'--batch-size {batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    train(args, cfg, device)


if __name__ == '__main__':
    main()
