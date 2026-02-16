# Delving into Localization Errors for Monocular 3D Detection

## Introduction

## Usage

### Installation
This repo is tested on our local environment (python=3.13, cuda=12.8, pytorch=2.9.1), and we recommend you to use uv to create a virtual environment:

```bash
uv venv --python=3.13
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```text
#ROOT
└── data
    └── KITTI
        ├── ImageSets [already provided]
        ├── object [create this symlink to self]
        │   ├── training -> ../training
        │   └── testing -> ../testing
        ├── training
        │   ├── calib (unzipped from calib.zip)
        │   ├── image_2 (unzipped from left_color.zip)
        │   └── label_2 (unzipped from label_2.zip)
        └── testing
            ├── calib
            └── image_2
```

### Training & Evaluation

#### MonoDLE (CenterNet-based)

The MonoDLE training pipeline now supports:

- DDP multi-GPU training via `torchrun`
- AMP mixed precision (`--amp`)
- `torch.compile` graph acceleration (`--compile`)
- EMA parameter averaging (`--ema`)
- random seed + deterministic algorithm controls (`--seed`, `--deterministic`)
- timm integration:
    - timm backbone (`model.type: centernet3d_timm`)
    - timm optimizer (`optimizer.type`, e.g. `lamb`)

Run in project root:

```sh
# 1) Default training (single GPU or DataParallel)
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml

# 2) DDP training (example: 2 GPUs)
torchrun --nproc_per_node=2 tools/train_val.py \
    --config experiments/kitti/monodle_kitti.yaml --ddp --amp --ema

# 3) Single GPU with AMP + compile + EMA
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \
    --amp --compile --ema

# 4) Evaluation only
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml -e
```

Output layout is aligned with YOLO3D: `<project>/<name>/` (default: `runs/monodle/train/`).

```sh
# Recommended explicit output layout
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \
    --project runs/monodle --name exp01

# Still supported: explicit output directory override
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \
    -o runs/monodle/exp01
```

Typical artifacts include `args.yaml`, `train.log.*`, `weights/`, `kitti_eval/`, `eval_results.csv`, and `last_eval_result.json`.

See `experiments/kitti/monodle_kitti.yaml` for:

- `project` / `name` output settings
- `model.type: centernet3d | centernet3d_timm`
- `optimizer.type` support for both torch and timm optimizers

| Method            | AP40@Mod. | Note |
| ----------------- | --------- | ---- |
| MonoDLE (Paper)   | 13.66     | Original paper result |
| MonoDLE (Repo)    | 12.69     | Reproduce result (Epoch 140) |

#### YOLO3D (Ultralytics-based)

YOLO3D extends the Ultralytics YOLO detection framework with MonoDDLE-style monocular 3D detection heads. It reuses YOLO's training loop, data augmentation, and post-processing while adding 3D prediction branches for depth, dimensions, heading, and 3D center offset.

Three YOLO backbone versions are supported:

| Version | Config | NMS | Features |
| ------- | ------ | --- | -------- |
| YOLOv8  | `yolo3d_v8n.yaml` | Standard NMS | Anchor-free, reg_max=16 |
| YOLO11  | `yolo3d_11n.yaml` | Standard NMS | Improved backbone (C3k2/C2PSA), reg_max=16 |
| YOLO26  | `yolo3d_26n.yaml` | **NMS-free** (end2end) | Dual one2many/one2one heads, reg_max=1 |

```sh
# YOLOv8 — standard NMS
python tools/train_yolo3d.py --model yolov8n.pt --data experiments/kitti/yolo3d_v8n.yaml

# YOLO11 — standard NMS
python tools/train_yolo3d.py --model yolo11n.pt --data experiments/kitti/yolo3d_11n.yaml

# YOLO26 — NMS-free end2end
python tools/train_yolo3d.py --model yolo26n.pt --data experiments/kitti/yolo3d_26n.yaml

# From scratch (any version)
python tools/train_yolo3d.py --model yolo26n.yaml --data experiments/kitti/yolo3d_26n.yaml

# Multi-GPU
python tools/train_yolo3d.py --model yolov8s.pt --data experiments/kitti/yolo3d_v8n.yaml --device 0,1

# Resume training
python tools/train_yolo3d.py --model runs/yolo3d/yolov8n/weights/last.pt --resume
```

All outputs are saved under `runs/yolo3d/<name>/`. Model scale can also be changed by replacing `n` with `s`/`m`/`l`/`x` in the model name (e.g., `yolov8s.pt`, `yolo26m.yaml`).

**Architecture overview:**

```
YOLO Backbone (v8/11/26) → FPN Neck → Detect3D Head
                                         ├── cv2: BBox regression (2D)
                                         ├── cv3: Classification
                                         └── cv4: Mono3D (depth + 3D center + size + heading)
```

## Citation

If you find our work useful in your research, please consider citing:

```latex
```

## Acknowlegment

This repo benefits from these excellent works. Please also consider citing them.
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [MonoDLE](https://github.com/xinzhuma/monodle/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## License

This project is released under the GNU General Public License v3.0 (GPL-3.0).

## Contact

If you have any questions about this project, please feel free to contact 1138663075@qq.com.
