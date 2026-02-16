"""
YOLO3D Training Script for KITTI Monocular 3D Detection.

Uses Ultralytics YOLO as the 2D detection backbone with MonoDDLE-style
3D detection heads (depth, 3D dimensions, heading, 3D center offset).

Supported YOLO versions:
  - YOLOv8 (yolov8n/s/m/l/x) — standard NMS, reg_max=16
  - YOLO11 (yolo11n/s/m/l/x)  — improved backbone, standard NMS
  - YOLO26 (yolo26n/s/m/l/x)  — NMS-free end2end, reg_max=1

Usage:
    # YOLOv8 (standard NMS)
    python tools/train_yolo3d.py --model yolov8n.pt --data experiments/kitti/kitti3d.yaml

    # YOLO11 (standard NMS)
    python tools/train_yolo3d.py --model yolo11n.pt --data experiments/kitti/kitti3d.yaml

    # YOLO26 (NMS-free end2end)
    python tools/train_yolo3d.py --model yolo26n.pt --data experiments/kitti/kitti3d.yaml

    # From scratch (any version)
    python tools/train_yolo3d.py --model yolo26n.yaml --data experiments/kitti/kitti3d.yaml

    # Resume training
    python tools/train_yolo3d.py --model runs/yolo3d/train/weights/last.pt --resume

    # With custom settings — any Ultralytics train() arg is accepted
    python tools/train_yolo3d.py --model yolov8s.pt --data experiments/kitti/kitti3d.yaml \\
        --epochs 200 --batch 8 --device 0,1 --cache ram --optimizer AdamW --cos_lr
"""

import os
import sys
import argparse

# Setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

# Ensure DDP subprocesses (spawned by Ultralytics) can import lib.*
# subprocess.run inherits env vars, but not sys.path modifications
_pythonpath = os.environ.get('PYTHONPATH', '')
if ROOT_DIR not in _pythonpath.split(os.pathsep):
    os.environ['PYTHONPATH'] = ROOT_DIR + os.pathsep + _pythonpath

from lib.models.yolo3d.model import YOLO3D


def _try_cast(value):
    """Try to cast a string value to int, float, or bool; return as-is if no cast fits."""
    if value.lower() in ('true', 'yes'):
        return True
    if value.lower() in ('false', 'no'):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_extra_args(extra):
    """Parse unknown argparse args (e.g. ['--cache', 'ram', '--cos_lr']) into a dict.

    Supports:
      --key value   →  {key: cast(value)}
      --flag        →  {flag: True}           (boolean flag with no value)
    """
    kwargs = {}
    i = 0
    while i < len(extra):
        arg = extra[i]
        if not arg.startswith('--'):
            raise ValueError(f"Unexpected positional argument: {arg}")
        key = arg.lstrip('-').replace('-', '_')
        # Check if next token is a value or another flag / end-of-list
        if i + 1 < len(extra) and not extra[i + 1].startswith('--'):
            kwargs[key] = _try_cast(extra[i + 1])
            i += 2
        else:
            kwargs[key] = True
            i += 1
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO3D Training for KITTI',
        epilog='All extra arguments (e.g. --cache ram, --cos_lr, --optimizer AdamW) '
               'are forwarded directly to Ultralytics model.train().',
    )
    # Only define args that this script handles specially (save_dir logic, etc.)
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model: .pt (pretrained) or .yaml (from scratch)')
    parser.add_argument('--data', type=str, default='experiments/kitti/kitti3d.yaml',
                        help='Dataset YAML config path')
    parser.add_argument('--project', type=str, default='runs/yolo3d',
                        help='Project save directory (e.g. runs/yolo3d)')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name (subfolder under project)')
    # All other YOLO training args (--epochs, --batch, --device, --cache, etc.)
    # are passed through automatically — no need to define them here.
    args, extra = parser.parse_known_args()
    args.extra_train_kwargs = _parse_extra_args(extra)
    return args


def main():
    args = parse_args()

    # Initialize the YOLO3D model
    model = YOLO3D(args.model)

    # Compute save_dir directly to bypass ultralytics' task-based nesting
    # (avoids runs/detect3d/runs/yolo3d/train double-nesting)
    save_dir = os.path.join(args.project, args.name)
    os.makedirs(save_dir, exist_ok=True)

    # Setup log file (matching monodle convention)
    log_file = os.path.join(save_dir, 'train.log')
    try:
        from lib.helpers.logger_helper import MonoDDLELogger
        logger = MonoDDLELogger(log_file)
    except Exception:
        pass

    # Build training kwargs: script defaults + user overrides from CLI
    train_kwargs = dict(
        data=args.data,
        save_dir=save_dir,
        exist_ok=True,
        # Sensible defaults for 3D detection (user can override via CLI)
        epochs=140,
        batch=16,
        imgsz=1280,
        device='0',
        workers=4,
        patience=50,
        lr0=0.01,
        amp=True,
        mosaic=0.0,
        mixup=0.0,
        label_smoothing=0.0,
    )
    # CLI overrides take precedence (e.g. --cache ram, --epochs 200, --device 0,1)
    train_kwargs.update(args.extra_train_kwargs)

    # Train
    results = model.train(**train_kwargs)

    print(f"\nTraining complete. Results saved to {save_dir}")

    # Print best epoch results
    try:
        import yaml
        from lib.helpers.logger_helper import print_best_epoch_results

        # Determine CSV path (same as save_dir)
        csv_path = os.path.join(save_dir, 'eval_results.csv')

        # Determine metric key from data config
        with open(args.data) as f:
            data_cfg = yaml.safe_load(f)
        writelist = data_cfg.get('writelist', ['Car'])
        primary_cls = writelist[0] if writelist else 'Car'
        metric_key = f'{primary_cls}_3d_moderate_R40'

        print_best_epoch_results(csv_path, metric_key=metric_key)
    except Exception as e:
        print(f"Failed to print best epoch results: {e}")

    return results


if __name__ == '__main__':
    main()
