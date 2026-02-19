# Delving into Localization Errors for Monocular 3D Detection

[中文文档](README.zh-CN.md)

By [Xinzhu Ma](https://scholar.google.com/citations?user=8PuKa_8AAAAJ), Yinmin Zhang, [Dan Xu](https://www.danxurgb.net/), [Dongzhan Zhou](https://scholar.google.com/citations?user=Ox6SxpoAAAAJ), [Shuai Yi](https://scholar.google.com/citations?user=afbbNmwAAAAJ), [Haojie Li](https://scholar.google.com/citations?user=pMnlgVMAAAAJ), [Wanli Ouyang](https://wlouyang.github.io/).


## Introduction

This repository is an official implementation of the paper ['Delving into Localization Errors for Monocular 3D Detection'](https://arxiv.org/abs/2103.16237). In this work, by intensive diagnosis experiments, we quantify the impact introduced by each sub-task and found the ‘localization error’ is the vital factor in restricting monocular 3D detection. Besides, we also investigate the underlying reasons behind localization errors, analyze the issues they might bring, and propose three strategies. 

<img src="resources/example.jpg" alt="vis" style="zoom:50%;" />




## Usage

### Installation
We recommend using `uv` to manage the Python environment and dependencies:

```bash
cd #ROOT
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

To pin a specific Python version:

```bash
uv venv .venv --python 3.10
```

### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |training/
        |calib/
        |image_2/
        |label_2/
      |testing/
        |calib/
        |image_2/
      |DA3_depth_results/   # optional, required for depth distillation
        |000000.npz
        |000001.npz
        |...
```

The default data root is `data/KITTI` (i.e. `dataset.root_dir: 'data/KITTI'` in yaml).

For DA3 depth distillation training, generate metric depth pseudo labels in advance and save them to `data/KITTI/DA3_depth_results`.

Each depth file should be a `.npz` file with key `depth`.

### Training & Evaluation

Move to the workplace and train the network:

> `tools/train_val.py` now resolves `dataset.root_dir` relative to project root.
> You can run commands from repository root directly.

#### Single-process DP mode (DataParallel, default)

```sh
cd #ROOT
python tools/train_val.py --config experiments/configs/monodle/kitti_no_da3.yaml
```

#### Multi-process DDP mode (DistributedDataParallel)

DDP provides better scaling efficiency and avoids the GIL bottleneck of DP.
Use `torchrun` to launch one process per GPU:

```sh
cd #ROOT
# Auto-detect all available GPUs:
bash experiments/scripts/train_ddp.sh experiments/configs/monodle/kitti_da3.yaml
# Or specify the number of GPUs:
bash experiments/scripts/train_ddp.sh experiments/configs/monodle/kitti_da3.yaml 4
# Or use torchrun directly:
torchrun --nproc_per_node=8 tools/train_val_ddp.py --config experiments/configs/monodle/kitti_da3.yaml
```

> **DP ↔ DDP equivalence**: MonoDLE is sensitive to batch size and learning rate.
> When switching between DP and DDP, the hyperparameters must be adjusted to
> produce equivalent training dynamics.  See the
> [DP/DDP Equivalence](#dp--ddp-equivalence) section below for details.

The model will be evaluated automatically if the training completed. If you only want evaluate your trained model (or the provided [pretrained model](https://drive.google.com/file/d/1jaGdvu_XFn5woX0eJ5I2R6wIcBLVMJV6/view?usp=sharing)) , you can modify the test part configuration in the .yaml file and use the following command:

```sh
python tools/train_val.py --config experiments/configs/monodle/kitti_da3.yaml -e
```

### Depth Distillation (DA3)

The training pipeline supports dense depth distillation without changing the model architecture:

\[
L_{total} = L_{base} + \lambda \cdot L_{distill}
\]

where `L_distill` can be configured as weighted `l1` or `silog`, and foreground regions receive larger weights.

Enable the following options in yaml:

```yaml
dataset:
  use_da3_depth: True

distill:
  lambda: 0.5
  loss_type: 'l1'           # 'l1' or 'silog'
  foreground_weight: 5.0
```

Then train as usual (example):

```sh
cd #ROOT
python tools/train_val.py --config experiments/configs/monodle/kitti_da3.yaml
```

To disable distillation, set `distill.lambda: 0.0` or `dataset.use_da3_depth: False`.

### Ultralytics Backbone Integration (YOLO8/11/26)

The project supports replacing only the backbone with Ultralytics YOLO while keeping the original `DLAUp + CenterNet3D heads + loss` unchanged.

Supported backbone keywords in yaml:

- `yolo8`  (default weight: `yolov8n.pt`)
- `yolo11` (default weight: `yolo11n.pt`)
- `yolo26` (default weight: `yolo26n.pt`)

Optional model fields:

```yaml
model:
  type: 'centernet3d'
  backbone: 'yolo11'
  ultralytics_model: 'yolo11n.pt'   # override weight path if needed
  feature_strides: [4, 8, 16, 32]   # multiscale features for DLAUp
  # feature_indices: [2, 15, 18, 21] # optional manual override
  # freeze_backbone: False            # optional
```

### Ablation Configs (YOLO vs YOLO+DA3)

The following configs are provided under `experiments/configs/ablation`:

| Group           | Config                                       |
| --------------- | -------------------------------------------- |
| YOLO8 baseline  | `experiments/configs/ablation/kitti_yolo8.yaml`      |
| YOLO8 + DA3     | `experiments/configs/ablation/kitti_yolo8_da3.yaml`  |
| YOLO11 baseline | `experiments/configs/ablation/kitti_yolo11.yaml`     |
| YOLO11 + DA3    | `experiments/configs/ablation/kitti_yolo11_da3.yaml` |
| YOLO26 baseline | `experiments/configs/ablation/kitti_yolo26.yaml`     |
| YOLO26 + DA3    | `experiments/configs/ablation/kitti_yolo26_da3.yaml` |

Run examples from repo root:

```sh
# baseline
python tools/train_val.py --config experiments/configs/ablation/kitti_yolo8.yaml

# +DA3 distillation
python tools/train_val.py --config experiments/configs/ablation/kitti_yolo8_da3.yaml

# DDP ablation (same equivalent total batch as DP)
bash experiments/scripts/train_ddp.sh experiments/configs/ablation/kitti_yolo8_da3.yaml 2
```

Evaluate only:

```sh
python tools/train_val.py --config experiments/configs/ablation/kitti_yolo11_da3.yaml -e

```

Experiment structure overview:

```text
experiments/
  configs/
    monodle/
    ablation/
  scripts/
  results/
    <config_rel_path>/
      <timestamp>/
        checkpoints/
        outputs/
        logs/
```

> Note: on first use, Ultralytics may download/cache model assets automatically.

### DP / DDP Equivalence

MonoDLE's training is **highly sensitive** to batch size and learning rate.
When migrating from DP to DDP (or changing the number of GPUs), incorrect
hyperparameter settings will cause significant accuracy degradation.
All experiment YAMLs under `experiments/configs/monodle` and `experiments/configs/ablation`
already include `distributed.dp_reference`, so DP/DDP ablation can be run with
equivalent effective total batch size by default.

#### Quick rules

| Scenario                     | `batch_size` (YAML)         | `lr`                                              |
| ---------------------------- | --------------------------- | ------------------------------------------------- |
| **DP** (default)             | Total batch across all GPUs | As configured                                     |
| **DDP** (same total batch)   | `DP_batch / world_size`     | **Same** as DP                                    |
| **DDP** (larger total batch) | Any per-GPU value           | `DP_lr × (DDP_total / DP_total)` (linear scaling) |

#### Why they are equivalent

In DP, the gradient is `g = (1/B) Σ ∇L_i` computed on the full batch *B*.

In DDP with *N* ranks each holding *b = B/N* samples:
```
g_k   = (1/b) Σ_{j∈rank_k} ∇L_j
g_ddp = (1/N) Σ_k g_k = (1/B) Σ ∇L_i = g_dp   ✓
```

So when `b × N = B`, the gradient (and therefore the training) is mathematically identical and no LR change is needed.

When the total batch size changes, the **linear scaling rule** applies:
`lr_ddp = lr_dp × (b × N) / B`.

#### Equivalence calculator

A standalone calculator is provided to compute the exact hyperparameters:

```sh
# Default: DP batch=16, lr=0.00125, 8 GPUs → DDP on 4 GPUs
python tools/dp_ddp_equivalence.py --dp-batch 16 --dp-lr 0.00125 --dp-gpus 8 --ddp-gpus 4

# With detailed math derivation
python tools/dp_ddp_equivalence.py --verbose

# With numerical gradient comparison simulation
python tools/dp_ddp_equivalence.py --simulate
```

#### Auto-equivalence in DDP config

Add a `distributed.dp_reference` section to your YAML and `train_val_ddp.py`
will **automatically override** `batch_size` and `lr` to match the DP reference:

```yaml
distributed:
  enabled: true
  dp_reference:
    total_batch_size: 16
    lr: 0.00125
    num_gpus: 8
```

#### Note on object-level loss normalisation

Some MonoDLE losses (2D/3D offset, size) use `mean(valid_objects)` instead of
`mean(batch_size)`.  DP computes these on the *gathered* full batch; DDP
computes per-process means then averages.  This introduces a tiny gradient
discrepancy when the number of valid objects per GPU is unequal.  On KITTI this
difference is typically < 0.1% and has no measurable impact on AP.

For ease of use, we also provide a pre-trained checkpoint, which can be used for evaluation directly. See the below table to check the performance.

|                   | AP40@Easy | AP40@Mod. | AP40@Hard |
| ----------------- | --------- | --------- | --------- |
| In original paper | 17.45     | 13.66     | 11.68     |
| In this repo      | 17.94     | 13.72     | 12.10     |

## Citation

If you find our work useful in your research, please consider citing:

```latex
@InProceedings{Ma_2021_CVPR,
author = {Ma, Xinzhu and Zhang, Yinmin, and Xu, Dan and Zhou, Dongzhan and Yi, Shuai and Li, Haojie and Ouyang, Wanli},
title = {Delving into Localization Errors for Monocular 3D Object Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}}
```

## Acknowlegment

This repo benefits from the excellent work [CenterNet](https://github.com/xingyizhou/CenterNet). Please also consider citing it.

## License

This project is released under the MIT License.

## Contact

If you have any question about this project, please feel free to contact xinzhu.ma@sydney.edu.au.
