<h1 align="center">Analysis of Research on 3D Object Detection Guided by Depth Estimation and Uncertainty</h1>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-3D%20Detection-brightgreen.svg" alt="Task">
  <img src="https://img.shields.io/badge/Status-Research-yellow.svg" alt="Status">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs Welcome">
</p>

<p align="center">
  <a href="README.zh-CN.md">Chinese README</a>
</p>

<p align="center">
  <img src="docs/images/combined3.png" width="800"/><br>
  <img src="docs/images/combined2.png" width="800"/><br>
  <img src="docs/images/combined1.png" width="800"/>
</p>

## Introduction

**MonoDDLE** (Monocular Dense Depth Distillation for Localization Errors) is an improved version based on [MonoDLE](https://github.com/XinzhuMa/MonoDLE). **The paper has not been published yet.**

The core challenge of 3D object detection lies in recovering lost depth information from a single RGB image. Existing mainstream methods typically rely on sparse LiDAR point cloud ground truth for supervised training, which is limited by sparsity and high data acquisition costs. This project aims to use the absolute metric depth from visual foundation models (such as Depth Anything V3) as "soft labels" or "dense supervision signals." Through knowledge distillation, we guide lightweight monocular detectors to learn more robust depth features, thereby significantly improving detection accuracy without increasing inference costs.

## Visualization Results

Some visualization results on the KITTI dataset:

|                         2D Bounding Box                         |                         3D Bounding Box                         |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| <img src="docs/images/2d.png" alt="2D BBox" width="400"/> | <img src="docs/images/3d.png" alt="3D BBox" width="400"/> |

|                        DA3 Depth Pseudo-Label                        |                                                            Depth Uncertainty                                                             |
| :----------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |
| <img src="docs/images/da3.png" alt="DA3 Depth" width="400"/> | <img src="docs/images/unc.png" alt="Uncertainty" width="400"/><br><img src="docs/images/img.png" alt="Original Image" width="400"/> |

|                                                             Object Center Heatmap                                                              |                         LiDAR BEV Projection                         |
| :---------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------: |
| <img src="docs/images/hm.png" alt="Heatmap" width="400"/><br><img src="docs/images/hm_perclass.png" alt="Heatmap Per-class" width="400"/> | <img src="docs/images/lidar.png" alt="LiDAR BEV" width="400"/> |

**Note: The image number corresponding to the above visualization results is 001230.**

## Experimental Results

We conducted extensive experiments on the KITTI dataset. The following are some core experimental results (for detailed data, please refer to `summary.md` in the root directory of the repository):

### 1. Core Results Comparison (KITTI Validation Set)

| Method              |  3D@0.7 (Easy/Mod/Hard)   |  BEV@0.7 (Easy/Mod/Hard)  |  3D@0.5 (Easy/Mod/Hard)   |  BEV@0.5 (Easy/Mod/Hard)  |
| :------------------ | :-----------------------: | :-----------------------: | :-----------------------: | :-----------------------: |
| CenterNet           |    0.60 / 0.66 / 0.77     |    3.46 / 3.31 / 3.21     |   20.00 / 17.50 / 15.57   |   34.36 / 27.91 / 24.65   |
| MonoGRNet           |    11.90 / 7.56 / 5.76    |   19.72 / 12.81 / 10.15   |   47.59 / 32.28 / 25.50   |   48.53 / 35.94 / 28.59   |
| MonoDIS             |    11.06 / 7.60 / 6.37    |   18.45 / 12.58 / 10.66   |             -             |             -             |
| M3D-RPN             |   14.53 / 11.07 / 8.65    |   20.85 / 15.62 / 11.88   |   48.53 / 35.94 / 28.59   |   53.35 / 39.60 / 31.76   |
| MonoPair            |   16.28 / 12.30 / 10.42   |   24.12 / 18.17 / 15.76   |   55.38 / 42.39 / 37.99   |   61.06 / 47.63 / 41.92   |
| MonoDLE (Re-impl.)  |   15.17 / 12.10 / 10.82   |   21.10 / 17.20 / 15.10   |   50.70 / 38.91 / 34.82   |   56.94 / 43.74 / 38.41   |
| **MonoDDLE (Ours)** | **18.49 / 14.48 / 12.14** | **26.38 / 20.12 / 17.89** | **59.80 / 43.89 / 39.27** | **65.10 / 48.85 / 42.97** |

### 2. Ablation Study: DA3 Depth and Uncertainty

| Method            | DA3 Depth | Uncertainty | 3D AP<sub>R40</sub> (E / M / H) | BEV AP<sub>R40</sub> (E / M / H) |
| :---------------- | :-------: | :---------: | :-----------------------------: | :------------------------------: |
| Baseline          |           |             |      15.17 / 12.10 / 10.82      |      21.10 / 17.20 / 15.10       |
| + DA3             |     ✓     |             |      18.27 / 14.26 / 11.96      |      25.59 / 19.65 / 16.79       |
| **+ Uncertainty** |   **✓**   |    **✓**    |    **18.49 / 14.48 / 12.14**    |    **26.38 / 20.12 / 17.89**     |

### 3. Comparison of Model Parameters and Computation

| Model               | Backbone      | FLOPs (G) | Params (M) |
| :------------------ | :------------ | :-------: | :--------: |
| MonoDLE             | DLA-34        |   79.37   |   20.31    |
| **MonoDDLE (Ours)** | **DLA-34**    | **83.91** | **20.46**  |
|                     | HRNet-W32     |  212.25   |   48.91    |
|                     | ResNet-50     |  439.70   |   91.41    |
|                     | ConvNeXt-Tiny |  129.83   |   38.34    |

### 4. Impact of Different Backbones (With DA3)

| Backbone        | 3D AP<sub>R40</sub> (E / M / H) | BEV AP<sub>R40</sub> (E / M / H) |
| :-------------- | :-----------------------------: | :------------------------------: |
| DLA-34          |      17.52 / 13.59 / 12.06      |      25.46 / 19.69 / 17.01       |
| HRNet-W32       |      17.87 / 13.72 / 11.73      |      24.79 / 19.23 / 16.58       |
| ConvNeXtV2-Tiny |      17.17 / 13.25 / 11.69      |      24.97 / 19.42 / 16.74       |
| ResNet-50       |      15.45 / 12.03 / 10.11      |      22.38 / 17.81 / 15.51       |

### 5. Oracle Analysis: Depth Bottleneck and Attribute Decoupling

To deeply explore the bottlenecks of model performance and the coupling relationship between various prediction attributes, we conducted an Oracle analysis. By replacing specific prediction heads (such as depth, size, heading, etc.) with real Ground Truth (GT) values during the inference phase, we observed the 3D AP (R40) performance of the model in different depth intervals.

#### Overall AP_R40 (Moderate) Summary

| Method         | baseline | w/ gt proj. center | w/ gt depth | w/ gt location | w/ gt size_3d | w/ gt heading |
| :------------- | :------: | :----------------: | :---------: | :------------: | :-----------: | :-----------: |
| MonoDLE        |  12.09   |       12.10        |    51.92    |     58.20      |     11.97     |     10.60     |
| MonoDDLE       |  14.26   |       10.09        |    21.91    |     21.79      |     5.24      |     4.91      |
| **MonoDDLE+U** |  14.51   |       14.08        |  **48.22**  |   **51.95**    |   **13.50**   |   **11.56**   |

#### Key Conclusions

1.  **Depth estimation is the core bottleneck**: When providing real depth information (`w/ gt depth`), the performance of both MonoDLE and MonoDDLE+U skyrocketed (reaching 51.92 and 48.22 respectively). This indicates that as long as the depth estimation is accurate, the existing network heads already possess very strong 3D box regression capabilities.
2.  **MonoDDLE has severe attribute coupling**: When providing perfect GT size (`w/ gt size_3d`) or heading (`w/ gt heading`), the performance of MonoDDLE plummeted from 14.26 to 5.24 and 4.91 instead. This shows that MonoDDLE generated serious "error compensation" during training, where the model relies on incorrect size or heading to compensate for incorrect depth. Even when providing real depth, its performance only improved to 21.91, far lower than other models.
3.  **MonoDDLE+U successfully achieved decoupling**: After introducing uncertainty guidance, MonoDDLE+U perfectly solved the above coupling problem. When providing real depth, its performance jumped significantly to 48.22 as expected; when providing real size or heading, it also completely eliminated the precipitous drop phenomenon in MonoDDLE, proving that it broke the pathological coupling between attributes and has a higher performance upper bound.

#### AP Curves Varying with Depth

| Baseline | w/ GT Depth |
| :---: | :---: |
| <img src="docs/images/oracle_baseline.png" alt="Oracle Baseline" width="400"/> | <img src="docs/images/oracle_gt_depth.png" alt="Oracle GT Depth" width="400"/> |

| w/ GT Location | w/ GT Proj. Center |
| :---: | :---: |
| <img src="docs/images/oracle_gt_location.png" alt="Oracle GT Location" width="400"/> | <img src="docs/images/oracle_gt_proj_center.png" alt="Oracle GT Proj Center" width="400"/> |

| w/ GT Size 3D | w/ GT Heading |
| :---: | :---: |
| <img src="docs/images/oracle_gt_size_3d.png" alt="Oracle GT Size 3D" width="400"/> | <img src="docs/images/oracle_gt_heading.png" alt="Oracle GT Heading" width="400"/> |

## Usage Instructions

### 1. Environment Installation

`uv` manages the Python environment and dependencies (the project uses the `.venv` in the repository by default):

```bash
cd #ROOT
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### 2. Data Preparation

Please download the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) first, and organize it as follows:

```text
MonoDDLE
└── data
    └── KITTI
        ├── ImageSets         # Split files provided by the repository
        ├── training
        │   ├── calib
        │   ├── image_2
        │   └── label_2
        ├── testing
        │   ├── calib
        │   └── image_2
        └── DA3_depth_results # Required for DA3 distillation training
            ├── 000000.npz
            ├── 000000_vis.jpg
            ├── 000001.npz
            ├── 000001_vis.jpg
            └── ...
```

The script to generate DA3 depth data is as follows (ensure `depth_anything_3` is installed):

```bash
python tools/generate_da3_depth.py --data_path data/KITTI --split training
```

This script will generate two files for each image:
- **`.npz`**: Contains three keys: `depth` (H, W), `intrinsics` (3, 3), and `extrinsics` (3, 4), which are the metric depth map predicted by DA3, camera intrinsics, and extrinsics matrices, respectively.
- **`_vis.jpg`**: Visualization of the original image concatenated vertically with the color depth map (for quick quality check of depth).

> Note: Only the `depth` key in `.npz` is used during training, `intrinsics` and `extrinsics` are auxiliary information.

### 3. Training and Evaluation

`tools/train_val.py` will parse `dataset.root_dir` as relative to the project root directory, so you can execute the command directly in the repository root directory.

#### Single Process DP (Default, DataParallel)

```sh
cd #ROOT
# Run MonoDDLE (with uncertainty distillation)
python tools/train_val.py --config experiments/configs/monodle/kitti_da3_uncertainty.yaml
```

#### Multi-Process DDP (DistributedDataParallel)

```sh
cd #ROOT
# Automatically use all visible GPUs
bash experiments/scripts/train_ddp.sh experiments/configs/monodle/kitti_da3_uncertainty.yaml
```

Run evaluation only:

```sh
python tools/train_val.py --config experiments/configs/monodle/kitti_da3_uncertainty.yaml -e
```

### 4. DA3 Depth Distillation and Uncertainty

#### DA3 Depth Distillation

This project uses [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3) as the teacher model to pre-generate full-image dense metric depth maps as pseudo-labels. Through distillation loss, it provides additional supervision to the depth prediction head of the detector without changing the backbone network structure of the detector.

Before running depth distillation training, you need to generate DA3 depth pseudo-labels (see Section 2 Data Preparation):

```bash
python tools/generate_da3_depth.py --data_path data/KITTI --split training
```

Each `.npz` file contains `depth`, `intrinsics`, and `extrinsics` keys, and a `_vis.jpg` visualization image is generated simultaneously. Only the `depth` key is read during training.

The total distillation loss is:

$$
L_{total} = L_{cls} + L_{bbox} + L_{dim} + \lambda \cdot L_{distill}
$$

Where $L_{distill}$ is the L1 or SiLog loss between the predicted depth and the DA3 pseudo-label, and $\lambda$ is the distillation weight. The minimum YAML configuration to enable depth distillation is as follows:

```yaml
dataset:
  use_da3_depth: True
```

#### Uncertainty-Guided Adaptive Distillation

Based on DA3 depth distillation, **pixel-wise uncertainty prediction** is further introduced to adaptively model the confidence of the model in DA3 pseudo-labels. Regions with high uncertainty (such as reflective surfaces, occlusion boundaries) will automatically reduce the distillation loss weight, thereby mitigating the negative impact of noisy pseudo-labels:

$$
L_{distill}^{unc} = \frac{1}{N} \sum_{i} \frac{|d_i - \hat{d}_i|}{\sigma_i} + \log \sigma_i
$$

Where $\sigma_i$ is the depth uncertainty predicted by the model, $d_i$ is the DA3 pseudo-label, and $\hat{d}_i$ is the model predicted depth. Enabling uncertainty guidance only requires setting in YAML:

```yaml
dataset:
  use_da3_depth: True # Must enable DA3 depth distillation

distill:
  lambda: 0.5
  loss_type: 'l1'           # 'l1' or 'silog'
  foreground_weight: 5.0
  use_uncertainty: True     # Enable uncertainty-guided adaptive depth distillation
```

## Acknowledgements

Excellent implementations of [MonoDLE](https://github.com/XinzhuMa/MonoDLE) and [CenterNet](https://github.com/xingyizhou/CenterNet).

## License

Open sourced under MIT License.
