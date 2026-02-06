import os
import time
import tempfile
import glob

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss
from lib.helpers.visualization_helper import visualize_results
from lib.helpers.logger_helper import create_progress_bar, console


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0)) if self.is_distributed else 0
        self.is_main = self.rank == 0
        self.device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        self.output_dir = cfg.get('output_dir', 'runs/default')
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        self.best_metric = float('-inf')
        self.best_metric_key = cfg.get('best_metric', None)
        if self.best_metric_key is None and hasattr(self.test_loader.dataset, 'writelist'):
            primary_cls = self.test_loader.dataset.writelist[0]
            self.best_metric_key = '%s_3d_moderate_R40' % primary_cls
        self.report_unused_params = cfg.get('report_unused_params', True)
        self._reported_unused_params = False
        
        self.use_amp = cfg.get('amp', False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.use_compile = cfg.get('compile', False)

        # loading pretrain/resume model
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

        if self.is_distributed:
            self.model = DDP(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        else:
            self.model = model.to(self.device)

        if self.use_compile:
            self.logger.info("Enabling torch.compile...")
            self.model = torch.compile(self.model)

    def train(self):
        start_epoch = self.epoch

        with create_progress_bar(description="Training Epochs") as progress:
            epoch_task = progress.add_task("[magenta]Epochs", total=self.cfg['max_epoch'] - start_epoch)
            
            for epoch in range(start_epoch, self.cfg['max_epoch']):
                # update learning rate
                if self.warmup_lr_scheduler is not None and epoch < 5:
                    self.warmup_lr_scheduler.step()
                else:
                    self.lr_scheduler.step(epoch)

                # reset random seed
                # ref: https://github.com/pytorch/pytorch/issues/5059
                np.random.seed(np.random.get_state()[1][0] + epoch)
                if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)
                # train one epoch
                epoch_loss = self.train_one_epoch()
                self.epoch += 1

                improved = False
                if self.is_main:
                    self.logger.print_section("Evaluating", f"Epoch {self.epoch}")
                    from lib.helpers.tester_helper import Tester
                    
                    # Use a persistent directory to store evaluation results, 
                    # enabling comparison with previous epochs.
                    eval_dir = os.path.join(self.output_dir, 'eval_results')
                    os.makedirs(eval_dir, exist_ok=True)

                    tester = Tester(cfg={'type': 'KITTI', 'mode': 'single', 'output_dir': eval_dir, 'threshold': 0.2},
                                    model=self.model,
                                    dataloader=self.test_loader,
                                    logger=self.logger)
                    metrics = tester.test()

                    if metrics is not None and self.best_metric_key in metrics:
                        current = float(metrics[self.best_metric_key])
                        self.logger.info(f'Best metric key: {self.best_metric_key} = {current:.4f}')
                        if current > self.best_metric:
                            improved = True
                            self.best_metric = current
                            
                            # Clean up previous best checkpoints
                            for prev_best in glob.glob(os.path.join(self.ckpt_dir, 'best_epoch_*.*')):
                                try:
                                    os.remove(prev_best)
                                except Exception:
                                    pass

                            best_ckpt = os.path.join(self.ckpt_dir, 'best_epoch_%d' % self.epoch)
                            save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), best_ckpt)
                            self.logger.print_checkpoint_info(best_ckpt + '.safetensors', action="Saved (Best)")

                    self.logger.info('Visualizing predicted results...')
                    self.visualize()

                # save trained model
                if self.is_main and (self.epoch % self.cfg['save_frequency']) == 0:
                    ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % self.epoch)
                    save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)
                    self.logger.print_checkpoint_info(ckpt_name + '.safetensors')

                progress.update(epoch_task, advance=1)

        return None


    def train_one_epoch(self):
        self.model.train()
        total_loss_sum = 0.0
        total_samples = 0
        total_stats = {}
        
        with create_progress_bar(description="Training Iterations", transient=True) as progress:
            iter_task = progress.add_task(
                f"[cyan]Epoch {self.epoch + 1}", 
                total=len(self.train_loader)
            )
            start_time = time.time()
            end = time.time()
            
            for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
                data_time = time.time() - end
                inputs = inputs.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)

                # train one batch
                self.optimizer.zero_grad()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                iter_start = time.time()

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(inputs)
                    total_loss, stats_batch = compute_centernet3d_loss(outputs, targets)
                
                # accumulation stats
                for k, v in stats_batch.items():
                    total_stats[k] = total_stats.get(k, 0) + v
                
                self.scaler.scale(total_loss).backward()

                if self.is_main and self.report_unused_params and not self._reported_unused_params:
                    model_to_check = self.model.module if hasattr(self.model, 'module') else self.model
                    unused = [name for name, p in model_to_check.named_parameters() if p.requires_grad and p.grad is None]
                    if unused:
                        self.logger.warning(f'Unused parameters (no grad): {", ".join(unused)}')
                    else:
                        self.logger.success('All parameters received gradients.')
                    self._reported_unused_params = True
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                iter_time = time.time() - iter_start

                total_loss_sum += total_loss.item()
                total_samples += 1

                if self.is_main and (batch_idx + 1) % 50 == 0:
                    avg_loss = total_loss_sum / max(1, total_samples)
                    avg_stats = {k: v / max(1, total_samples) for k, v in total_stats.items()}
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.print_training_status(
                        epoch=self.epoch + 1,
                        max_epoch=self.cfg['max_epoch'],
                        batch=batch_idx + 1,
                        total_batches=len(self.train_loader),
                        loss=avg_loss,
                        lr=current_lr,
                        data_time=data_time,
                        iter_time=iter_time,
                        stats_dict=avg_stats,
                    )

                progress.update(iter_task, advance=1)
                end = time.time()

        return total_loss_sum / max(1, total_samples)


    def visualize(self):
        if not self.is_main:
            return
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(self.test_loader):
                if batch_idx > 0: break # only visualize 1 batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Get dataset instance to call get_label and get_calib
                dataset = self.test_loader.dataset
                img_ids = info['img_id'].numpy()
                
                calibs = [dataset.get_calib(index) for index in img_ids]
                # Also get GT labels for visualization
                gt_objects = [dataset.get_label(index) for index in img_ids]
                
                epoch_vis_dir = os.path.join(self.vis_dir, 'epoch_%d' % self.epoch)
                visualize_results(inputs, outputs, targets, info, calibs,
                                  dataset.cls_mean_size,
                                  threshold=0.2, output_dir=epoch_vis_dir,
                                  epoch=self.epoch,
                                  gt_objects=gt_objects,
                                  dataset=dataset)




