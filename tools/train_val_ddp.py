"""
DDP (DistributedDataParallel) training entry-point for MonoDLE.

Usage (single-node, multi-GPU)
------------------------------
  torchrun --nproc_per_node=NUM_GPUS ../../tools/train_val_ddp.py --config CONFIG.yaml

The script will:
  1. Initialise the NCCL process group via ``torchrun`` environment variables.
  2. Automatically compute an equivalent per-GPU batch size and learning rate
     when the YAML supplies a ``distributed`` section (see below), or keep the
     YAML ``batch_size`` as-is (treated as *per-GPU* batch size).
  3. Wrap the model with ``DistributedDataParallel``.
  4. Use ``DistributedSampler`` for the training set.
  5. Only rank-0 saves checkpoints and runs evaluation.

YAML distributed section (optional)
------------------------------------
  distributed:
    enabled: true
    # If dp_reference is present, the script will compute equivalent DDP
    # settings from the DP config automatically.
    dp_reference:
      total_batch_size: 16
      lr: 0.00125
      num_gpus: 8
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

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.dist_helper import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
    compute_equivalent_config,
)


parser = argparse.ArgumentParser(description='MonoDLE DDP Training')
parser.add_argument('--config', dest='config', required=True,
                    help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False,
                    help='evaluation only (runs on rank 0)')

args = parser.parse_args()


def main():
    # ---- distributed init ---------------------------------------------------
    setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    assert os.path.exists(args.config)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # resolve root_dir relative to the project root when given as a relative path
    root_dir = cfg['dataset'].get('root_dir', 'data/KITTI')
    if not os.path.isabs(root_dir):
        cfg['dataset']['root_dir'] = os.path.normpath(os.path.join(ROOT_DIR, root_dir))

    set_random_seed(cfg.get('random_seed', 444) + rank)

    # ---- logging (rank-0 only) -----------------------------------------------
    log_file = 'train_%s.log' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if is_main_process():
        logger = create_logger(log_file)
    else:
        # non-main ranks get a silent logger
        import logging
        logger = logging.getLogger('silent')
        logger.addHandler(logging.NullHandler())

    # ---- DP → DDP equivalence ------------------------------------------------
    dist_cfg = cfg.get('distributed', {})
    dp_ref = dist_cfg.get('dp_reference', None)

    if dp_ref is not None:
        equiv = compute_equivalent_config(
            dp_total_batch=dp_ref['total_batch_size'],
            dp_lr=dp_ref['lr'],
            dp_num_gpus=dp_ref['num_gpus'],
            ddp_world_size=world_size,
        )
        # Override YAML values to guarantee equivalence
        cfg['dataset']['batch_size'] = equiv['ddp_per_gpu_batch']
        cfg['optimizer']['lr'] = equiv['ddp_lr']

        if is_main_process():
            logger.info('===== DP → DDP Equivalence =====')
            logger.info('DP  : total_batch=%d  per_gpu=%s  lr=%.8f  gpus=%d' % (
                equiv['dp_total_batch'], equiv['dp_per_gpu_batch'],
                equiv['dp_lr'], equiv['dp_num_gpus']))
            logger.info('DDP : total_batch=%d  per_gpu=%d  lr=%.8f  world=%d' % (
                equiv['ddp_total_batch'], equiv['ddp_per_gpu_batch'],
                equiv['ddp_lr'], equiv['ddp_world_size']))
            logger.info('Exact: %s' % equiv['is_exact'])
            for n in equiv['notes']:
                logger.info('  * %s' % n)
            logger.info('================================')
    else:
        if is_main_process():
            logger.info('No dp_reference found – using YAML batch_size as per-GPU batch size.')

    # ---- build dataloader (with DistributedSampler) --------------------------
    train_loader, test_loader, train_sampler = build_dataloader(
        cfg['dataset'], distributed=True)

    # ---- build model ---------------------------------------------------------
    model = build_model(cfg['model'])

    if args.evaluate_only:
        if is_main_process():
            logger.info('###################  Evaluation Only  ##################')
            tester = Tester(cfg=cfg['tester'],
                            model=model,
                            dataloader=test_loader,
                            logger=logger)
            tester.test()
        cleanup_distributed()
        return

    # ---- build optimizer & scheduler -----------------------------------------
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        cfg['lr_scheduler'], optimizer, last_epoch=-1)

    if is_main_process():
        logger.info('###################  DDP Training  ##################')
        logger.info('World Size: %d' % world_size)
        logger.info('Per-GPU Batch Size: %d' % cfg['dataset']['batch_size'])
        logger.info('Effective Total Batch: %d' % (cfg['dataset']['batch_size'] * world_size))
        logger.info('Learning Rate: %f' % cfg['optimizer']['lr'])

    # ---- train ---------------------------------------------------------------
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      distill_cfg=cfg.get('distill', None),
                      distributed=True,
                      train_sampler=train_sampler)
    trainer.train()

    # ---- evaluate (rank-0 only) ----------------------------------------------
    if is_main_process():
        logger.info('###################  Evaluation  ##################')
        # unwrap DDP model for testing
        raw_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        tester = Tester(cfg=cfg['tester'],
                        model=raw_model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()

    cleanup_distributed()


if __name__ == '__main__':
    main()
