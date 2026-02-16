"""
Detection3DTrainer: Ultralytics DetectionTrainer extended for KITTI 3D detection.

Supports multiple YOLO architectures: YOLOv8, YOLO11, YOLO26 (NMS-free).
The architecture is determined by the model config YAML passed to get_model().

Key customizations:
  - Overrides get_dataset() to return KITTI-specific data dict
  - Overrides get_model() to create Detection3DModel with head-swapping
  - Overrides build_dataset/get_dataloader to use KITTI3DDataset
  - Overrides preprocess_batch to handle mono3d_targets
  - Overrides label_loss_items for 7-component loss display
  - Overrides validate() to run MonoDDLE-style KITTI 3D evaluation with
    best metric tracking (Car_3d_moderate_R40) and visualization
"""

import os
import math
import random
import numpy as np
from copy import copy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.cfg import DEFAULT_CFG, DEFAULT_CFG_DICT

from lib.models.yolo3d.model import Detection3DModel
from lib.models.yolo3d.dataset import KITTI3DDataset

# Keys that are valid in DEFAULT_CFG_DICT + known extras allowed by Ultralytics.
# Used to strip non-standard keys (e.g. o2m, sgd_w, topk from pretrained
# yolo26m.pt) that would cause check_dict_alignment to fail in DDP subprocesses.
_VALID_CFG_KEYS = frozenset(DEFAULT_CFG_DICT.keys()) | {"save_dir", "augmentations", "session"}


class Detection3DTrainer(DetectionTrainer):
    """Trainer for YOLO3D monocular 3D object detection."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect3d"
        # Strip non-standard keys that may come from pretrained model checkpoints
        # (e.g. yolo26m.pt trained with a custom Ultralytics fork containing
        # extra hyper-parameters like o2m, sgd_w, topk, cls_w, etc.).
        # Without this, DDP subprocesses fail at check_dict_alignment.
        overrides = {k: v for k, v in overrides.items() if k in _VALID_CFG_KEYS}
        super().__init__(cfg, overrides, _callbacks)

    def check_resume(self, overrides):
        """Check resume and strip non-standard keys that may leak from checkpoint.

        When resuming, the parent ``check_resume`` calls ``load_checkpoint``
        which merges ``DEFAULT_CFG_DICT`` with the checkpoint's ``train_args``.
        If the original training session loaded a pretrained model with
        non-standard keys (e.g. ``yolo26m.pt`` → ``o2m``, ``sgd_w``, …),
        those keys end up in ``self.args`` and later get serialised into the
        DDP temp file, causing ``check_dict_alignment`` to fail in the
        subprocess.  We strip them here right after the parent finishes.
        """
        super().check_resume(overrides)
        # Sanitize: remove any non-standard keys that leaked through
        for k in list(vars(self.args)):
            if k not in _VALID_CFG_KEYS:
                delattr(self.args, k)

    def get_dataset(self):
        """
        Build the data dict for KITTI 3D detection.

        The data YAML should contain:
          - root_dir: path to KITTI data root
          - train: 'train' split name
          - val: 'val' split name
          - nc: number of classes (3)
          - names: {0: 'Pedestrian', 1: 'Car', 2: 'Cyclist'}
          - channels: 3
        """
        data_path = str(self.args.data)

        if data_path.endswith(('.yaml', '.yml')):
            import yaml
            with open(data_path) as f:
                data = yaml.safe_load(f)
        elif isinstance(self.args.data, dict):
            data = self.args.data
        else:
            # Default KITTI config
            data = {
                'root_dir': data_path,
                'train': 'train',
                'val': 'val',
                'nc': 3,
                'names': {0: 'Pedestrian', 1: 'Car', 2: 'Cyclist'},
                'channels': 3,
            }

        # Ensure required keys
        data.setdefault('nc', 3)
        data.setdefault('names', {0: 'Pedestrian', 1: 'Car', 2: 'Cyclist'})
        data.setdefault('channels', 3)
        data.setdefault('train', 'train')
        data.setdefault('val', 'val')

        return data

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build KITTI3DDataset for a given split."""
        split = self.data.get(mode, mode)  # 'train' or 'val'
        cfg = {
            'root_dir': self.data.get('root_dir', 'data/KITTI'),
            'writelist': self.data.get('writelist', ['Car']),
            'use_3d_center': self.data.get('use_3d_center', True),
            'bbox2d_type': self.data.get('bbox2d_type', 'anno'),
            'meanshape': self.data.get('meanshape', False),
            'class_merging': self.data.get('class_merging', False),
            'use_dontcare': self.data.get('use_dontcare', False),
            'resolution': self.data.get('resolution', [1280, 384]),
            'random_flip': self.data.get('random_flip', 0.5) if mode == 'train' else 0.0,
            'random_crop': self.data.get('random_crop', 0.5) if mode == 'train' else 0.0,
            'scale': self.data.get('scale', 0.4),
            'shift': self.data.get('shift', 0.1),
        }
        return KITTI3DDataset(split=split, cfg=cfg)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build DataLoader with custom KITTI3D collate function."""
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and rank == -1,
            num_workers=self.args.workers if mode == "train" else self.args.workers * 2,
            collate_fn=KITTI3DDataset.collate_fn,
            pin_memory=True,
            drop_last=mode == "train",
            sampler=torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
                    if rank >= 0 else None,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Build Detection3DModel: starts with a YOLO detection model and swaps to Detect3D.

        Supports all YOLO versions:
          - yolov8n/s/m/l/x.yaml  → reg_max=16, NMS
          - yolo11n/s/m/l/x.yaml  → reg_max=16, NMS
          - yolo26n/s/m/l/x.yaml  → reg_max=1, end2end (NMS-free)

        Args:
            cfg: Model YAML config (auto-detected from model name).
            weights: Pretrained weights path or loaded model.
            verbose: Print model info.
        """
        nc = self.data.get("nc", 3)
        # Build cls_mean_size from data config
        cls_mean_size = self.data.get("cls_mean_size", None)

        model = Detection3DModel(
            cfg=cfg,
            ch=self.data.get("channels", 3),
            nc=nc,
            num_heading_bin=self.data.get("num_heading_bin", 12),
            cls_mean_size=cls_mean_size,
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Set model attributes from data dict."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def plot_training_labels(self):
        """Skip default plot_training_labels — KITTI3DDataset doesn't use
        the ultralytics `labels` attribute convention."""
        pass

    def auto_batch(self):
        """Return configured batch size — KITTI3DDataset doesn't support
        the ultralytics labels-based auto_batch sizing."""
        return self.args.batch

    def preprocess_batch(self, batch):
        """
        Move batch data to device and normalize images.

        Ultralytics expects images in [0,1] float. Our dataset already
        produces [0,1] images, so we just move to device.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        # Images are already in [0,1] from the dataset
        if batch["img"].dtype != torch.float32:
            batch["img"] = batch["img"].float() / 255.0
        return batch

    def get_validator(self):
        """Return the 3D detection validator."""
        self.loss_names = (
            "box_loss", "cls_loss", "dfl_loss",
            "depth_loss", "off3d_loss", "sz3d_loss", "head_loss",
        )

        from lib.models.yolo3d.val import Detection3DValidator

        return Detection3DValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self):
        """
        Run validation with MonoDDLE-style KITTI 3D evaluation.
        
        Extends the default validation to:
        1. Run ultralytics standard 2D validation metrics  
        2. Run KITTI official 3D evaluation (AP_3D, AP_BEV at easy/moderate/hard)
        3. Track best KITTI 3D metric for model saving
        4. Generate 2D/3D visualization
        """
        # Pass current epoch and vis config to validator
        if hasattr(self, 'validator') and self.validator is not None:
            self.validator._current_epoch = self.epoch + 1
            self.validator._max_vis_batches = self.data.get('max_vis_batches', 2)

        metrics, fitness = super().validate()

        # Check if KITTI 3D metrics improved for best model tracking
        if metrics is not None and hasattr(self, 'validator'):
            kitti_metrics = getattr(self.validator, '_kitti_3d_metrics', {})
            if kitti_metrics:
                # Use primary class 3D moderate R40 as the fitness metric
                writelist = self.data.get('writelist', ['Car'])
                primary_cls = writelist[0] if writelist else 'Car'
                metric_key = f'{primary_cls}_3d_moderate_R40'

                if metric_key in kitti_metrics:
                    kitti_fitness = float(kitti_metrics[metric_key])
                    LOGGER.info(f"KITTI 3D fitness ({metric_key}): {kitti_fitness:.4f}")

                    # Override fitness with KITTI 3D metric for best model selection
                    fitness = kitti_fitness

                    # Store the best KITTI metric for reference
                    if not hasattr(self, '_best_kitti_metric'):
                        self._best_kitti_metric = -1.0
                        self._best_kitti_key = metric_key

                    if kitti_fitness > self._best_kitti_metric:
                        self._best_kitti_metric = kitti_fitness
                        LOGGER.info(
                            f"New best KITTI 3D metric: {metric_key} = {kitti_fitness:.4f}"
                        )

        return metrics, fitness

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Label the loss items with descriptive names for logging."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def progress_string(self):
        """Build a formatted string for training progress display."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size",
        )
