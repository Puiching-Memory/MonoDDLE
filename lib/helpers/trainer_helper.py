"""
Trainer for MonoDLE CenterNet3D.

Supports:
  - Single-GPU (DataParallel — legacy) and multi-GPU (DistributedDataParallel)
  - AMP (Automatic Mixed Precision) via ``torch.amp``
  - ``torch.compile`` for graph-mode speedup (PyTorch ≥ 2.0)
  - EMA (Exponential Moving Average) of model weights
  - Deterministic seeding per epoch / worker
"""

import os
import tqdm
import json
import glob

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss


class Trainer(object):
    """Training loop with DDP / AMP / EMA / torch.compile support.

    Args:
        cfg: trainer config dict from the yaml file.
        model: raw ``nn.Module`` (will be wrapped by DP/DDP inside).
        optimizer: torch optimizer.
        train_loader: training DataLoader.
        test_loader: validation DataLoader.
        lr_scheduler: main LR scheduler.
        warmup_lr_scheduler: optional warmup scheduler.
        logger: MonoDDLELogger instance.
        train_sampler: ``DistributedSampler`` (or None).
        rank: process rank (0 for single-GPU / main process).
        world_size: total number of processes.
        use_amp: enable AMP.
        compile_model: enable ``torch.compile``.
        compile_backend: backend for ``torch.compile``.
        ema: ``ModelEMA`` instance (or None).
        scaler: ``torch.amp.GradScaler`` instance (or None, auto-created if
            ``use_amp=True``).
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        train_loader,
        test_loader,
        lr_scheduler,
        warmup_lr_scheduler,
        logger,
        # ── new args ──
        train_sampler=None,
        rank=0,
        world_size=1,
        use_amp=False,
        compile_model=False,
        compile_backend='inductor',
        ema=None,
        scaler=None,
    ):
        self.cfg = cfg
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.train_sampler = train_sampler
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.use_amp = use_amp
        self.ema = ema

        self.device = torch.device("cuda:%d" % rank if torch.cuda.is_available() else "cpu")

        self.output_dir = cfg.get('output_dir', 'runs/default')
        self.ckpt_dir = os.path.join(self.output_dir, 'weights')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_metric = float('-inf')
        self.best_metric_key = cfg.get('best_metric', None)
        if self.best_metric_key is None and hasattr(self.test_loader.dataset, 'writelist'):
            primary_cls = self.test_loader.dataset.writelist[0]
            self.best_metric_key = '%s_3d_moderate_R40' % primary_cls

        # ── AMP GradScaler ────────────────────────────────────────────
        if self.use_amp:
            self.scaler = scaler or torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # ── Load pretrained / resume ──────────────────────────────────
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(
                model=model,
                optimizer=self.optimizer,
                filename=cfg['resume_model'],
                map_location=self.device,
                logger=self.logger,
                scaler=self.scaler,
                ema=self.ema,
            )
            self.lr_scheduler.last_epoch = self.epoch - 1

        # ── torch.compile (before wrapping with DP/DDP) ──────────────
        if compile_model and hasattr(torch, 'compile'):
            self.logger.info('Compiling model with torch.compile (backend=%s)...' % compile_backend)
            model = torch.compile(model, backend=compile_backend)

        # ── Wrap model with DP or DDP ─────────────────────────────────
        model = model.to(self.device)
        if world_size > 1 and torch.distributed.is_initialized():
            # DDP
            self.model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank,
                find_unused_parameters=cfg.get('find_unused_parameters', True),
            )
            self.logger.info('Using DistributedDataParallel on %d GPUs' % world_size)
        else:
            # DataParallel (legacy single-node multi-GPU)
            gpu_ids = list(map(int, str(cfg.get('gpu_ids', '0')).split(',')))
            gpu_ids = [i for i in gpu_ids if i < torch.cuda.device_count()]
            if not gpu_ids:
                gpu_ids = [0]
            if len(gpu_ids) > 1:
                self.model = nn.DataParallel(model, device_ids=gpu_ids)
                self.logger.info('Using DataParallel on GPUs %s' % gpu_ids)
            else:
                self.model = model

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def train(self):
        start_epoch = self.epoch
        is_main = (self.rank == 0)

        progress_bar = tqdm.tqdm(
            range(start_epoch, self.cfg['max_epoch']),
            dynamic_ncols=True, leave=True, desc='epochs',
            disable=(not is_main),
        )

        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # per-epoch seed (for reproducibility with data augmentation)
            np.random.seed(np.random.get_state()[1][0] + epoch)

            # DDP: set sampler epoch for proper shuffling
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # train one epoch
            self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # ── Save & evaluate (main process only) ───────────────────
            if is_main:
                # Save last checkpoint (always)
                last_ckpt = os.path.join(self.ckpt_dir, 'last')
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch,
                                         scaler=self.scaler, ema=self.ema),
                    last_ckpt,
                )

                # Save periodic checkpoints
                if (self.epoch % self.cfg.get('save_frequency', 10)) == 0:
                    ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % self.epoch)
                    save_checkpoint(
                        get_checkpoint_state(self.model, self.optimizer, self.epoch,
                                             scaler=self.scaler, ema=self.ema),
                        ckpt_name,
                    )

                # Evaluate (optionally with EMA weights)
                self.logger.info('==> Evaluating epoch %d ...' % self.epoch)
                from lib.helpers.save_helper import _unwrap_model

                # Unwrap model to avoid DDP synchronization (broadcast) during single-rank evaluation
                eval_model = _unwrap_model(self.model)
                _backed_up = False

                if self.ema is not None:
                    # Swap in EMA weights for evaluation
                    self._model_backup = {k: v.clone() for k, v in eval_model.state_dict().items()}
                    self.ema.apply_shadow(eval_model)
                    _backed_up = True

                from lib.helpers.tester_helper import Tester
                tester = Tester(
                    cfg={'type': 'KITTI', 'mode': 'single',
                         'output_dir': self.output_dir, 'threshold': 0.2},
                    model=eval_model,
                    dataloader=self.test_loader,
                    logger=self.logger,
                    epoch=self.epoch,
                )
                metrics = tester.test()

                # Restore training weights
                if _backed_up:
                    eval_model.load_state_dict(self._model_backup)
                    del self._model_backup

                # Track best model
                if metrics is not None and self.best_metric_key in metrics:
                    current = float(metrics[self.best_metric_key])
                    self.logger.info('Best metric key: %s = %.4f' % (self.best_metric_key, current))
                    if current > self.best_metric:
                        self.best_metric = current
                        for prev_best in glob.glob(os.path.join(self.ckpt_dir, 'best.*')):
                            try:
                                os.remove(prev_best)
                            except Exception:
                                pass
                        best_ckpt = os.path.join(self.ckpt_dir, 'best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch,
                                                 scaler=self.scaler, ema=self.ema),
                            best_ckpt,
                        )
                        self.logger.info('==> Best model saved (epoch %d, %s=%.4f)' %
                                         (self.epoch, self.best_metric_key, current))

            # synchronize after eval
            if self.world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.barrier()

            progress_bar.update()

        # Print best epoch results (main process only)
        if is_main:
            try:
                from lib.helpers.logger_helper import print_best_epoch_results
                csv_path = os.path.join(self.output_dir, 'eval_results.csv')
                print_best_epoch_results(csv_path, metric_key=self.best_metric_key,
                                         logger_obj=self.logger)
            except Exception as e:
                self.logger.warning('Failed to print best epoch results: %s' % str(e))

        return None

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self):
        self.model.train()
        is_main = (self.rank == 0)
        progress_bar = tqdm.tqdm(
            total=len(self.train_loader),
            leave=(self.epoch + 1 == self.cfg['max_epoch']),
            desc='iters',
            disable=(not is_main),
        )

        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward (with optional AMP autocast) ──────────────────
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    total_loss, stats_batch = compute_centernet3d_loss(outputs, targets)
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                total_loss, stats_batch = compute_centernet3d_loss(outputs, targets)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35.0)
                self.optimizer.step()

            # ── EMA update ────────────────────────────────────────────
            if self.ema is not None:
                self.ema.update(self.model)

            # Log every 50 iterations
            if is_main and (batch_idx + 1) % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                stats_str = ', '.join(['%s: %.4f' % (k, v) for k, v in stats_batch.items()])
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d], LR: %.2e, Loss: %.4f (%s)' % (
                        self.epoch + 1, self.cfg['max_epoch'],
                        batch_idx + 1, len(self.train_loader),
                        current_lr, total_loss.item(), stats_str,
                    )
                )

            progress_bar.update()
        progress_bar.close()
