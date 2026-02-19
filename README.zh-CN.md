# 单目 3D 检测定位误差研究（MonoDLE）

[English README](README.md)

作者： [Xinzhu Ma](https://scholar.google.com/citations?user=8PuKa_8AAAAJ), Yinmin Zhang, [Dan Xu](https://www.danxurgb.net/), [Dongzhan Zhou](https://scholar.google.com/citations?user=Ox6SxpoAAAAJ), [Shuai Yi](https://scholar.google.com/citations?user=afbbNmwAAAAJ), [Haojie Li](https://scholar.google.com/citations?user=pMnlgVMAAAAJ), [Wanli Ouyang](https://wlouyang.github.io/)

## 项目简介

本仓库是论文 [Delving into Localization Errors for Monocular 3D Object Detection](https://arxiv.org/abs/2103.16237) 的官方实现。论文通过系统化诊断实验分析单目 3D 检测中的误差来源，指出定位误差是主要瓶颈，并提出了对应改进策略。

<img src="resources/example.jpg" alt="vis" style="zoom:50%;" />

## 使用说明

### 1. 环境安装

当前推荐使用 `uv` 管理 Python 环境与依赖（项目默认使用仓库内 `.venv`）：

```bash
cd #ROOT
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

如需指定 Python 版本，可使用：

```bash
uv venv .venv --python 3.10
```

> `requirements.txt` 已包含 `ultralytics`，用于 YOLO 骨干集成。

### 2. 数据准备

请先下载 [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)，并按以下结构组织：

```text
#ROOT
  |data/
    |KITTI/
      |ImageSets/                 # 仓库已提供划分文件
      |training/
        |calib/
        |image_2/
        |label_2/
      |testing/
        |calib/
        |image_2/
      |DA3_depth_results/         # 可选；DA3 蒸馏训练必需
        |000000.npz
        |000001.npz
        |...
```

默认数据根目录为 `data/KITTI`（即 YAML 中 `dataset.root_dir: 'data/KITTI'`）。

DA3 蒸馏训练需要预先生成深度伪标签并保存到 `data/KITTI/DA3_depth_results`，每个文件是包含 `depth` 键的 `.npz`。

### 3. 训练与评估

`tools/train_val.py` 会将 `dataset.root_dir` 作为相对项目根目录进行解析，因此可直接在仓库根目录执行命令。

#### 单进程 DP（默认，DataParallel）

```sh
cd #ROOT
python tools/train_val.py --config experiments/configs/monodle/kitti_no_da3.yaml
```

#### 多进程 DDP（DistributedDataParallel）

```sh
cd #ROOT
# 自动使用全部可见 GPU
bash experiments/scripts/train_ddp.sh experiments/configs/monodle/kitti_da3.yaml
# 指定 GPU 数量
bash experiments/scripts/train_ddp.sh experiments/configs/monodle/kitti_da3.yaml 4
# 或直接 torchrun
torchrun --nproc_per_node=8 tools/train_val_ddp.py --config experiments/configs/monodle/kitti_da3.yaml
```

如果只评估，不训练：

```sh
python tools/train_val.py --config experiments/configs/monodle/kitti_da3.yaml -e
```

### 4. DA3 深度蒸馏

训练支持在不改变检测网络结构的前提下加入稠密深度蒸馏：

$$
L_{total} = L_{base} + \lambda \cdot L_{distill}
$$

其中 `L_distill` 支持 `l1` 或 `silog`，并可通过前景权重强化目标区域监督。

YAML 示例：

```yaml
dataset:
  use_da3_depth: True

distill:
  lambda: 0.5
  loss_type: 'l1'           # 'l1' 或 'silog'
  foreground_weight: 5.0
```

关闭蒸馏可设置：`distill.lambda: 0.0` 或 `dataset.use_da3_depth: False`。

### 5. Ultralytics 骨干集成（YOLO8/11/26）

本项目已支持仅替换骨干网络为 Ultralytics YOLO，保持原有 `DLAUp + CenterNet3D heads + loss` 不变。

支持的 `model.backbone` 关键字：

- `yolo8`
- `yolo11`
- `yolo26`

支持的常见模型尺寸：`n / s / m / l / x`。

对应默认权重命名：

- YOLO8: `yolov8{size}.pt`（如 `yolov8s.pt`）
- YOLO11: `yolo11{size}.pt`（如 `yolo11m.pt`）
- YOLO26: `yolo26{size}.pt`（如 `yolo26x.pt`）

配置示例：

```yaml
model:
  type: 'centernet3d'
  backbone: 'yolo11'
  ultralytics_model: 'yolo11l.pt'   # 可改为本地权重路径
  feature_strides: [4, 8, 16, 32]   # 提供给 DLAUp 的多尺度特征
  # feature_indices: [2, 15, 18, 21] # 可选：手动指定特征层
  # freeze_backbone: False            # 可选：冻结骨干
```

> 首次使用 Ultralytics 可能会自动下载/缓存模型资源。

### 6. 消融实验配置（YOLO vs YOLO+DA3）

已按 “系列/尺寸” 组织配置目录，并同时提供基线与 DA3：

- 目录：`experiments/configs/ablation/{yolo8|yolo11|yolo26}/{n|s|m|l|x}/`
- 基线：`kitti.yaml`
- DA3：`kitti_da3.yaml`

运行示例（仓库根目录）：

```sh
# YOLO8-n 基线
python tools/train_val.py --config experiments/configs/ablation/yolo8/n/kitti.yaml

# YOLO8-n +DA3
python tools/train_val.py --config experiments/configs/ablation/yolo8/n/kitti_da3.yaml

# YOLO11-l +DA3
python tools/train_val.py --config experiments/configs/ablation/yolo11/l/kitti_da3.yaml

# YOLO26-x 基线
python tools/train_val.py --config experiments/configs/ablation/yolo26/x/kitti.yaml

# DDP 消融（与 DP 保持等效总 batch）
bash experiments/scripts/train_ddp.sh experiments/configs/ablation/yolo11/m/kitti_da3.yaml 2

# 仅评估
python tools/train_val.py --config experiments/configs/ablation/yolo26/s/kitti_da3.yaml -e
```

`experiments` 目录结构：

```text
experiments/
  configs/
    monodle/
    ablation/
      <family>/            # yolo8 / yolo11 / yolo26
        <size>/            # n / s / m / l / x
          kitti.yaml
          kitti_da3.yaml
  scripts/
  results/
    <config_rel_path>/
      <timestamp>/
        checkpoints/
        outputs/
        logs/
```

### 7. DP / DDP 等价性说明

MonoDLE 对批大小和学习率非常敏感。DP 与 DDP 切换时，如果参数不等价，精度会明显下降。
`experiments/configs/monodle` 与 `experiments/configs/ablation` 下所有实验 YAML
均已包含 `distributed.dp_reference`，可直接开展 DP/DDP 消融并默认保持等效总 batch。

#### 快速规则

| 场景                 | YAML 中 `batch_size`    | `lr`           |
| -------------------- | ----------------------- | -------------- |
| DP（默认）           | 所有 GPU 的总 batch     | 配置值         |
| DDP（总 batch 不变） | `DP_batch / world_size` | 与 DP 相同     |
| DDP（总 batch 变大） | 任意每卡 batch          | 按线性规则缩放 |

线性缩放规则：

`lr_ddp = lr_dp × (DDP_total / DP_total)`

#### 等价性计算工具

```sh
python tools/dp_ddp_equivalence.py --dp-batch 16 --dp-lr 0.00125 --dp-gpus 8 --ddp-gpus 4
python tools/dp_ddp_equivalence.py --verbose
python tools/dp_ddp_equivalence.py --simulate
```

#### 在 DDP 配置中自动等价

在 YAML 中加入：

```yaml
distributed:
  enabled: true
  dp_reference:
    total_batch_size: 16
    lr: 0.00125
    num_gpus: 8
```

`train_val_ddp.py` 会自动覆盖 `batch_size` 与 `lr` 以匹配 DP 参考设置。

## 预训练模型结果

|            | AP40@Easy | AP40@Mod. | AP40@Hard |
| ---------- | --------- | --------- | --------- |
| 论文结果   | 17.45     | 13.66     | 11.68     |
| 本仓库复现 | 17.94     | 13.72     | 12.10     |

## 引用

如果本项目对你的研究有帮助，请引用：

```latex
@InProceedings{Ma_2021_CVPR,
author = {Ma, Xinzhu and Zhang, Yinmin, and Xu, Dan and Zhou, Dongzhan and Yi, Shuai and Li, Haojie and Ouyang, Wanli},
title = {Delving into Localization Errors for Monocular 3D Object Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}}
```

## 致谢

本仓库受益于 [CenterNet](https://github.com/xingyizhou/CenterNet) 的优秀实现。

## 许可证

本项目基于 MIT License 开源。

## 联系方式

如有问题，请联系：xinzhu.ma@sydney.edu.au
