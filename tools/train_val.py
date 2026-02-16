"""
MonoDLE / CenterNet3D Training Script.

Features:
  - DDP (DistributedDataParallel) multi-GPU training  (``--ddp``)
  - AMP (Automatic Mixed Precision)                   (``--amp``)
  - torch.compile graph-mode acceleration              (``--compile``)
  - timm backbone / optimizer integration              (via yaml config)
  - EMA (Exponential Moving Average) of weights        (``--ema``)
  - Deterministic seeding                              (``--seed``)
  - YOLO3D-compatible output directory layout

Usage:
    # Single GPU (legacy DataParallel)
    python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml

    # Single GPU with AMP + EMA + compile
    python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \\
        --amp --ema --compile

    # Multi-GPU DDP (e.g. 2 GPUs)
    torchrun --nproc_per_node=2 tools/train_val.py \\
        --config experiments/kitti/monodle_kitti.yaml --ddp --amp --ema

    # Evaluation only
    python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml -e

    # Use YOLO3D-style output layout: runs/monodle/<name>
    python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \\
        --project runs/monodle --name train
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

import torch
import torch.distributed as dist

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import (
    create_logger,
    set_random_seed,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MonoDLE CenterNet3D — Monocular 3D Object Detection',
    )
    parser.add_argument('--config', dest='config', required=True,
                        help='Path to experiment yaml config')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False,
                        help='Run evaluation only (no training)')

    # ── Output layout (matches YOLO3D convention) ─────────────────────
    parser.add_argument('--project', type=str, default=None,
                        help='Project directory (default from yaml or runs/monodle)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name / subfolder (default from yaml or "train")')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Explicit output directory (overrides --project/--name)')

    # ── Training enhancements ─────────────────────────────────────────
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='Use DistributedDataParallel (launch with torchrun)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (FP16)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enable torch.compile (PyTorch >= 2.0)')
    parser.add_argument('--compile-backend', type=str, default='inductor',
                        help='torch.compile backend (default: inductor)')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Enable Exponential Moving Average of weights')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                        help='EMA decay factor (default: 0.9999)')

    # ── Reproducibility ───────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides yaml random_seed)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Enable deterministic algorithms (default: True)')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                        help='Disable deterministic algorithms for speed')

    return parser.parse_args()


def _resolve_output_dir(args, cfg):
    """Resolve the output directory in YOLO3D-compatible format.

    Priority: ``--output_dir`` > ``--project/--name`` > yaml ``output_dir``
              > ``runs/monodle/train``
    """
    if args.output_dir:
        return args.output_dir

    project = args.project or cfg.get('project', 'runs/monodle')
    name = args.name or cfg.get('name', 'train')
    return os.path.join(project, name)


def _save_args_yaml(output_dir, args, cfg):
    """Save the effective config + CLI args to ``args.yaml`` (like YOLO3D)."""
    merged = dict(cfg)
    merged['cli'] = {
        'ddp': args.ddp,
        'amp': args.amp,
        'compile': args.compile,
        'compile_backend': args.compile_backend,
        'ema': args.ema,
        'ema_decay': args.ema_decay,
        'seed': args.seed,
        'deterministic': args.deterministic,
    }
    path = os.path.join(output_dir, 'args.yaml')
    with open(path, 'w') as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)


# ─── DDP worker entry point ──────────────────────────────────────────

def main():
    args = parse_args()
    assert os.path.exists(args.config), 'Config not found: %s' % args.config
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # ── Distributed setup ─────────────────────────────────────────────
    rank = 0
    world_size = 1
    if args.ddp:
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        setup_ddp(rank, world_size)

    # ── Seed ──────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else cfg.get('random_seed', 444)
    set_random_seed(seed + rank, deterministic=args.deterministic)

    # ── Output directory (YOLO3D layout) ──────────────────────────────
    output_dir = _resolve_output_dir(args, cfg)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(
        output_dir,
        'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    logger = create_logger(log_file, rank=rank)

    # Log config summary
    if is_main_process():
        logger.info('Config: %s' % args.config)
        logger.info('Output: %s' % output_dir)
        logger.info('DDP: %s  |  AMP: %s  |  compile: %s  |  EMA: %s' %
                     (args.ddp, args.amp, args.compile, args.ema))
        logger.info('Seed: %d  |  Deterministic: %s' % (seed, args.deterministic))
        _save_args_yaml(output_dir, args, cfg)

    # Inject output_dir into trainer/tester configs
    cfg.setdefault('trainer', {})['output_dir'] = output_dir
    cfg.setdefault('tester', {})['output_dir'] = output_dir

    # Resolve relative checkpoint paths against output_dir
    for section in ['trainer', 'tester']:
        for key in ['checkpoint', 'resume_model', 'pretrain_model', 'checkpoints_dir']:
            path = cfg.get(section, {}).get(key)
            if path and not os.path.isabs(path) and not os.path.exists(path):
                alt = os.path.join(output_dir, path)
                if os.path.exists(alt) or os.path.exists(alt + '.pth'):
                    cfg[section][key] = alt

    # ── Build dataloader ──────────────────────────────────────────────
    train_loader, test_loader, train_sampler = build_dataloader(
        cfg['dataset'], distributed=args.ddp,
    )

    # ── Build model ───────────────────────────────────────────────────
    model = build_model(cfg['model'])

    # ── Evaluate only ─────────────────────────────────────────────────
    if args.evaluate_only:
        if is_main_process():
            logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()
        if args.ddp:
            cleanup_ddp()
        return

    # ── Build optimizer ───────────────────────────────────────────────
    optimizer = build_optimizer(cfg['optimizer'], model)

    # ── Build LR scheduler ────────────────────────────────────────────
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1,
    )

    # ── EMA ───────────────────────────────────────────────────────────
    ema = None
    if args.ema:
        from lib.helpers.ema_helper import ModelEMA
        ema = ModelEMA(model, decay=args.ema_decay)
        if is_main_process():
            logger.info('EMA enabled (decay=%.6f)' % args.ema_decay)

    # ── Train ─────────────────────────────────────────────────────────
    if is_main_process():
        logger.info('###################  Training  ##################')
        logger.info('Batch Size: %d (x%d GPUs)' % (cfg['dataset']['batch_size'], world_size))
        logger.info('Learning Rate: %f' % cfg['optimizer']['lr'])

    trainer = Trainer(
        cfg=cfg['trainer'],
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        lr_scheduler=lr_scheduler,
        warmup_lr_scheduler=warmup_lr_scheduler,
        logger=logger,
        train_sampler=train_sampler,
        rank=rank,
        world_size=world_size,
        use_amp=args.amp,
        compile_model=args.compile,
        compile_backend=args.compile_backend,
        ema=ema,
    )
    trainer.train()

    # ── Final evaluation ──────────────────────────────────────────────
    if is_main_process():
        logger.info('###################  Evaluation  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=trainer.model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()

    if args.ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()