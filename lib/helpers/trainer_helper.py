import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.dist_helper import is_distributed, is_main_process, get_rank, get_local_rank
from lib.losses.centernet_loss import compute_centernet3d_loss


class Trainer(object):
    """Unified trainer supporting both DP and DDP.

    Behaviour
    ---------
    * **DP mode** (default, ``distributed=False``):
      Wraps the model with ``torch.nn.DataParallel`` on the GPUs listed in
      ``cfg['gpu_ids']``.  ``batch_size`` in the YAML is the **total** batch
      size split across GPUs.

    * **DDP mode** (``distributed=True``):
      Wraps the model with ``DistributedDataParallel`` on a **single** local
      GPU determined by ``LOCAL_RANK``.  ``batch_size`` in the YAML is the
      **per-GPU** batch size.  The caller must have already initialised the
      process group (see ``dist_helper.setup_distributed``).
    """

    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 distill_cfg=None,
                 distributed=False,
                 train_sampler=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.distributed = distributed
        self.train_sampler = train_sampler
        self.distill_cfg = distill_cfg or {}
        self.distill_lambda = float(self.distill_cfg.get('lambda', 0.0))

        # ---- device selection ------------------------------------------------
        if distributed:
            local_rank = get_local_rank()
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ---- gpu_ids (only used in DP mode) ----------------------------------
        if not distributed:
            self.gpu_ids = list(map(int, str(cfg['gpu_ids']).split(',')))
            if torch.cuda.is_available():
                self.gpu_ids = [i for i in self.gpu_ids if i < torch.cuda.device_count()]
                if not self.gpu_ids:
                    self.gpu_ids = [0]

        # ---- load pretrain / resume ------------------------------------------
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        # ---- wrap model (DP or DDP) ------------------------------------------
        if distributed:
            self.model = self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=True,
            )
        else:
            self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids).to(self.device)


    def train(self):
        start_epoch = self.epoch

        # Only show progress bar on main process
        show_progress = (not self.distributed) or is_main_process()
        if show_progress:
            progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']),
                                     dynamic_ncols=True, leave=True, desc='epochs')

        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)

            # DDP: set epoch on sampler so each epoch sees a different shuffle
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

            # save trained model (main process only in DDP)
            if (self.epoch % self.cfg['save_frequency']) == 0:
                if (not self.distributed) or is_main_process():
                    os.makedirs('checkpoints', exist_ok=True)
                    ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                    save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            if show_progress:
                progress_bar.update()

        return None


    def train_one_epoch(self):
        self.model.train()
        show_progress = (not self.distributed) or is_main_process()
        if show_progress:
            progress_bar = tqdm.tqdm(total=len(self.train_loader),
                                     leave=(self.epoch+1 == self.cfg['max_epoch']),
                                     desc='iters')
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            total_loss, stats_batch = compute_centernet3d_loss(
                outputs,
                targets,
                distill_lambda=self.distill_lambda,
                distill_cfg=self.distill_cfg)
            total_loss.backward()
            self.optimizer.step()

            if show_progress:
                progress_bar.update()
        if show_progress:
            progress_bar.close()




