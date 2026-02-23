# 基于深度估计和不确定性引导的3D目标检测研究分析

[English README](README.md)

## 项目简介

 **MonoDDLE** (Monocular Dense Depth Distillation for Localization Errors) 是基于 [MonoDLE](https://github.com/XinzhuMa/MonoDLE) 的改进版本。**论文暂未发表。**

3D目标检测的核心难点在于从单张RGB图像中恢复丢失的深度信息。现有的主流方法通常依赖稀疏的 LiDAR 点云真值进行监督训练，存在稀疏性和数据获取成本高的局限性。本项目旨在利用视觉基础模型（如 Depth Anything V3）的绝对度量深度作为“软标签”或“密集监督信号”，通过知识蒸馏的方式，指导轻量级单目检测器学习更鲁棒的深度特征，从而在不增加推理成本的前提下显著提升检测精度。

## 可视化结果

 KITTI 数据集上的部分可视化结果：

|                         2D 边界框                         |                         3D 边界框                         |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| <img src="docs/images/2d.png" alt="2D BBox" width="400"/> | <img src="docs/images/3d.png" alt="3D BBox" width="400"/> |

|                        DA3 深度伪标签                        |                                                            深度不确定性                                                             |
| :----------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |
| <img src="docs/images/da3.png" alt="DA3 Depth" width="400"/> | <img src="docs/images/unc.png" alt="Uncertainty" width="400"/><br><img src="docs/images/img.png" alt="Original Image" width="400"/> |

|                                                             目标中心点热力图                                                              |                         LiDAR BEV 投影                         |
| :---------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------: |
| <img src="docs/images/hm.png" alt="Heatmap" width="400"/><br><img src="docs/images/hm_perclass.png" alt="Heatmap Per-class" width="400"/> | <img src="docs/images/lidar.png" alt="LiDAR BEV" width="400"/> |

**注：以上可视化结果对应的图像编号为 001230。**

## 实验结果

我们在 KITTI 数据集上进行了广泛的实验，以下是部分核心实验结果（详细数据请参考仓库根目录下的 `summary.md`）：

### 1. 核心结果对比 (KITTI Validation Set)

| Method              |  3D@0.7 (Easy/Mod/Hard)   |  BEV@0.7 (Easy/Mod/Hard)  |  3D@0.5 (Easy/Mod/Hard)   |  BEV@0.5 (Easy/Mod/Hard)  |
| :------------------ | :-----------------------: | :-----------------------: | :-----------------------: | :-----------------------: |
| CenterNet           |    0.60 / 0.66 / 0.77     |    3.46 / 3.31 / 3.21     |   20.00 / 17.50 / 15.57   |   34.36 / 27.91 / 24.65   |
| MonoGRNet           |    11.90 / 7.56 / 5.76    |   19.72 / 12.81 / 10.15   |   47.59 / 32.28 / 25.50   |   48.53 / 35.94 / 28.59   |
| MonoDIS             |    11.06 / 7.60 / 6.37    |   18.45 / 12.58 / 10.66   |             -             |             -             |
| M3D-RPN             |   14.53 / 11.07 / 8.65    |   20.85 / 15.62 / 11.88   |   48.53 / 35.94 / 28.59   |   53.35 / 39.60 / 31.76   |
| MonoPair            |   16.28 / 12.30 / 10.42   |   24.12 / 18.17 / 15.76   |   55.38 / 42.39 / 37.99   |   61.06 / 47.63 / 41.92   |
| MonoDLE (Re-impl.)  |   15.17 / 12.10 / 10.82   |   21.10 / 17.20 / 15.10   |   50.70 / 38.91 / 34.82   |   56.94 / 43.74 / 38.41   |
| **MonoDDLE (Ours)** | **18.49 / 14.48 / 12.14** | **26.38 / 20.12 / 17.89** | **59.80 / 43.89 / 39.27** | **65.10 / 48.85 / 42.97** |

### 2. 消融实验：DA3 深度与不确定性

| Method            | DA3 Depth | Uncertainty | 3D AP<sub>R40</sub> (E / M / H) | BEV AP<sub>R40</sub> (E / M / H) |
| :---------------- | :-------: | :---------: | :-----------------------------: | :------------------------------: |
| Baseline          |           |             |      15.17 / 12.10 / 10.82      |      21.10 / 17.20 / 15.10       |
| + DA3             |     ✓     |             |      18.27 / 14.26 / 11.96      |      25.59 / 19.65 / 16.79       |
| **+ Uncertainty** |   **✓**   |    **✓**    |    **18.49 / 14.48 / 12.14**    |    **26.38 / 20.12 / 17.89**     |

### 3. 模型参数量与计算量对比

| Model               | Backbone      | FLOPs (G) | Params (M) |
| :------------------ | :------------ | :-------: | :--------: |
| MonoDLE             | DLA-34        |   79.37   |   20.31    |
| **MonoDDLE (Ours)** | **DLA-34**    | **83.91** | **20.46**  |
|                     | HRNet-W32     |  212.25   |   48.91    |
|                     | ResNet-50     |  439.70   |   91.41    |
|                     | ConvNeXt-Tiny |  129.83   |   38.34    |

### 4. 不同骨干网络的影响 (With DA3)

| Backbone        | 3D AP<sub>R40</sub> (E / M / H) | BEV AP<sub>R40</sub> (E / M / H) |
| :-------------- | :-----------------------------: | :------------------------------: |
| DLA-34          |      17.52 / 13.59 / 12.06      |      25.46 / 19.69 / 17.01       |
| HRNet-W32       |      17.87 / 13.72 / 11.73      |      24.79 / 19.23 / 16.58       |
| ConvNeXtV2-Tiny |      17.17 / 13.25 / 11.69      |      24.97 / 19.42 / 16.74       |
| ResNet-50       |      15.45 / 12.03 / 10.11      |      22.38 / 17.81 / 15.51       |


## 使用说明

### 1. 环境安装

 `uv` 管理 Python 环境与依赖（项目默认使用仓库内 `.venv`）：

```bash
cd #ROOT
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### 2. 数据准备

请先下载 [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)，并按以下结构组织：

```text
MonoDDLE
└── data
    └── KITTI
        ├── ImageSets         # 仓库已提供划分文件
        ├── training
        │   ├── calib
        │   ├── image_2
        │   └── label_2
        ├── testing
        │   ├── calib
        │   └── image_2
        └── DA3_depth_results # DA3 蒸馏训练必需
            ├── 000000.npz
            ├── 000000_vis.jpg
            ├── 000001.npz
            ├── 000001_vis.jpg
            └── ...
```

生成 DA3 深度数据的脚本如下（需确保已安装 `depth_anything_3`）：

```bash
python tools/generate_da3_depth.py --data_path data/KITTI --split training
```

该脚本会为每张图像生成两个文件：
- **`.npz`**：包含 `depth` (H, W)、`intrinsics` (3, 3)、`extrinsics` (3, 4) 三个键，分别为 DA3 预测的度量深度图、相机内参和外参矩阵
- **`_vis.jpg`**：原图与彩色深度图的上下拼接可视化（用于快速检查深度质量）

> 注：训练时仅使用 `.npz` 中的 `depth` 键，`intrinsics` 和 `extrinsics` 为辅助信息。

### 3. 训练与评估

`tools/train_val.py` 会将 `dataset.root_dir` 作为相对项目根目录进行解析，因此可直接在仓库根目录执行命令。

#### 单进程 DP（默认，DataParallel）

```sh
cd #ROOT
# 运行 MonoDDLE (带不确定性蒸馏)
python tools/train_val.py --config experiments/configs/monodle/kitti_da3_uncertainty.yaml
```

#### 多进程 DDP（DistributedDataParallel）

```sh
cd #ROOT
# 自动使用全部可见 GPU
bash experiments/scripts/train_ddp.sh experiments/configs/monodle/kitti_da3_uncertainty.yaml
```

python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml

```sh
python tools/train_val.py --config experiments/configs/monodle/kitti_da3_uncertainty.yaml -e
```

### 4. DA3 深度蒸馏与不确定性

#### DA3 深度蒸馏

本项目使用 [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3) 作为教师模型，预先生成全图稠密度量深度图作为伪标签，并通过蒸馏损失对检测器的深度预测头进行额外监督，无需改变检测器骨干网络结构。

在运行深度蒸馏训练前，需先生成 DA3 深度伪标签（参见第 2 节数据准备）：

```bash
python tools/generate_da3_depth.py --data_path data/KITTI --split training
```

每个 `.npz` 文件包含 `depth`、`intrinsics`、`extrinsics` 三个键，同时生成 `_vis.jpg` 可视化图片。训练时仅读取 `depth` 键。

蒸馏总损失为：

$$
L_{total} = L_{cls} + L_{bbox} + L_{dim} + \lambda \cdot L_{distill}
$$

其中 $L_{distill}$ 为预测深度与 DA3 伪标签之间的 L1 或 SiLog 损失，$\lambda$ 为蒸馏权重。开启深度蒸馏的最小 YAML 配置如下：

```yaml
dataset:
  use_da3_depth: True
```

#### 不确定性引导的自适应蒸馏

在 DA3 深度蒸馏的基础上，进一步引入**逐像素不确定性预测**，使模型对 DA3 伪标签的置信度进行自适应建模。不确定性高的区域（如反光表面、遮挡边界）将自动降低蒸馏损失权重，从而减轻噪声伪标签的负面影响：

$$
L_{distill}^{unc} = \frac{1}{N} \sum_{i} \frac{|d_i - \hat{d}_i|}{\sigma_i} + \log \sigma_i
$$

其中 $\sigma_i$ 为模型预测的深度不确定性，$d_i$ 为 DA3 伪标签，$\hat{d}_i$ 为模型预测深度。开启不确定性引导只需在 YAML 中设置：

```yaml
dataset:
  use_da3_depth: True # 必须开启 DA3 深度蒸馏

distill:
  lambda: 0.5
  loss_type: 'l1'           # 'l1' 或 'silog'
  foreground_weight: 5.0
  use_uncertainty: True     # 开启不确定性引导的自适应深度蒸馏
```

## 致谢

 [MonoDLE](https://github.com/XinzhuMa/MonoDLE) 和 [CenterNet](https://github.com/xingyizhou/CenterNet) 的优秀实现。

## 许可证

 MIT License 开源。
