# Research and Analysis of 3D Object Detection Guided by Depth Estimation and Uncertainty

[中文文档](README.zh-CN.md)

## Introduction

**MonoDDLE** (Monocular Dense Depth Distillation for Localization Errors) is an improved version based on [MonoDLE](https://github.com/XinzhuMa/MonoDLE). **The paper has not been published yet.**

The core challenge of monocular 3D object detection lies in recovering the lost depth information from a single RGB image. Existing mainstream methods typically rely on sparse LiDAR point cloud ground truth for supervised training, suffering from sparsity and high data acquisition costs. This project leverages the absolute metric depth from vision foundation models (e.g., Depth Anything V3) as "soft labels" or "dense supervision signals", guiding lightweight monocular detectors to learn more robust depth features through knowledge distillation, thereby significantly improving detection accuracy without increasing inference cost.

## Visualization Results

Sample visualization results on the KITTI dataset:

|                      2D Bounding Box                      |                      3D Bounding Box                      |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| <img src="docs/images/2d.png" alt="2D BBox" width="400"/> | <img src="docs/images/3d.png" alt="3D BBox" width="400"/> |

|                    DA3 Depth Pseudo-label                    |                                                          Depth Uncertainty                                                          |
| :----------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |
| <img src="docs/images/da3.png" alt="DA3 Depth" width="400"/> | <img src="docs/images/unc.png" alt="Uncertainty" width="400"/><br><img src="docs/images/img.png" alt="Original Image" width="400"/> |

|                                                           Object Center Heatmap                                                           |                      LiDAR BEV Projection                      |
| :---------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------: |
| <img src="docs/images/hm.png" alt="Heatmap" width="400"/><br><img src="docs/images/hm_perclass.png" alt="Heatmap Per-class" width="400"/> | <img src="docs/images/lidar.png" alt="LiDAR BEV" width="400"/> |

**Note: The above visualization results correspond to image ID 001230.**

## Experimental Results

We conducted extensive experiments on the KITTI dataset. Below are some core results (see `summary.md` in the repository root for full details):

### 1. Main Results (KITTI Validation Set)

| Method              |  3D@0.7 (Easy/Mod/Hard)   |  BEV@0.7 (Easy/Mod/Hard)  |  3D@0.5 (Easy/Mod/Hard)   |  BEV@0.5 (Easy/Mod/Hard)  |
| :------------------ | :-----------------------: | :-----------------------: | :-----------------------: | :-----------------------: |
| CenterNet           |    0.60 / 0.66 / 0.77     |    3.46 / 3.31 / 3.21     |   20.00 / 17.50 / 15.57   |   34.36 / 27.91 / 24.65   |
| MonoGRNet           |    11.90 / 7.56 / 5.76    |   19.72 / 12.81 / 10.15   |   47.59 / 32.28 / 25.50   |   48.53 / 35.94 / 28.59   |
| MonoDIS             |    11.06 / 7.60 / 6.37    |   18.45 / 12.58 / 10.66   |             -             |             -             |
| M3D-RPN             |   14.53 / 11.07 / 8.65    |   20.85 / 15.62 / 11.88   |   48.53 / 35.94 / 28.59   |   53.35 / 39.60 / 31.76   |
| MonoPair            |   16.28 / 12.30 / 10.42   |   24.12 / 18.17 / 15.76   |   55.38 / 42.39 / 37.99   |   61.06 / 47.63 / 41.92   |
| MonoDLE (Re-impl.)  |   15.17 / 12.10 / 10.82   |   21.10 / 17.20 / 15.10   |   50.70 / 38.91 / 34.82   |   56.94 / 43.74 / 38.41   |
| **MonoDDLE (Ours)** | **18.49 / 14.48 / 12.14** | **26.38 / 20.12 / 17.89** | **59.80 / 43.89 / 39.27** | **65.10 / 48.85 / 42.97** |

### 2. Ablation Study: DA3 Depth & Uncertainty

| Method            | DA3 Depth | Uncertainty | 3D AP<sub>R40</sub> (E / M / H) | BEV AP<sub>R40</sub> (E / M / H) |
| :---------------- | :-------: | :---------: | :-----------------------------: | :------------------------------: |
| Baseline          |           |             |      15.17 / 12.10 / 10.82      |      21.10 / 17.20 / 15.10       |
| + DA3             |     ✓     |             |      18.27 / 14.26 / 11.96      |      25.59 / 19.65 / 16.79       |
| **+ Uncertainty** |   **✓**   |    **✓**    |    **18.49 / 14.48 / 12.14**    |    **26.38 / 20.12 / 17.89**     |

### 3. Model Parameters & Computation

| Model               | Backbone      | FLOPs (G) | Params (M) |
| :------------------ | :------------ | :-------: | :--------: |
| MonoDLE             | DLA-34        |   79.37   |   20.31    |
| **MonoDDLE (Ours)** | **DLA-34**    | **83.91** | **20.46**  |
|                     | HRNet-W32     |  212.25   |   48.91    |
|                     | ResNet-50     |  439.70   |   91.41    |
|                     | ConvNeXt-Tiny |  129.83   |   38.34    |

### 4. Backbone Comparison (With DA3)

| Backbone        | 3D AP<sub>R40</sub> (E / M / H) | BEV AP<sub>R40</sub> (E / M / H) |
| :-------------- | :-----------------------------: | :------------------------------: |
| DLA-34          |      17.52 / 13.59 / 12.06      |      25.46 / 19.69 / 17.01       |
| HRNet-W32       |      17.87 / 13.72 / 11.73      |      24.79 / 19.23 / 16.58       |
| ConvNeXtV2-Tiny |      17.17 / 13.25 / 11.69      |      24.97 / 19.42 / 16.74       |
| ResNet-50       |      15.45 / 12.03 / 10.11      |      22.38 / 17.81 / 15.51       |


## Usage

### 1. Environment Setup

Use `uv` to manage the Python environment and dependencies (the project uses a local `.venv` by default):

```bash
cd #ROOT
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### 2. Data Preparation

Download the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize it as follows:

```text
MonoDDLE
└── data
    └── KITTI
        ├── ImageSets         # Split files (provided in the repo)
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

Generate DA3 depth data using the following script (requires `depth_anything_3` to be installed):

```bash
python tools/generate_da3_depth.py --data_path data/KITTI --split training
```

The script generates two files per image:
- **`.npz`**: Contains three keys — `depth` (H, W), `intrinsics` (3, 3), and `extrinsics` (3, 4), representing the DA3-predicted metric depth map, camera intrinsics, and extrinsics matrices respectively.
- **`_vis.jpg`**: A vertically stacked visualization of the original image and the colorized depth map (for quick quality inspection).

> Note: Only the `depth` key from `.npz` files is used during training. `intrinsics` and `extrinsics` are saved as auxiliary information.

### 3. Training & Evaluation

`tools/train_val.py` resolves `dataset.root_dir` relative to the project root, so commands can be run directly from the repository root.

#### Single-process DP (default, DataParallel)

```sh
cd #ROOT
# Run MonoDDLE (with uncertainty distillation)
python tools/train_val.py --config experiments/configs/monodle/kitti_da3_uncertainty.yaml
```

#### Multi-process DDP (DistributedDataParallel)

```sh
cd #ROOT
# Automatically uses all visible GPUs
bash experiments/scripts/train_ddp.sh experiments/configs/monodle/kitti_da3_uncertainty.yaml
```

#### Evaluation Only

```sh
python tools/train_val.py --config experiments/configs/monodle/kitti_da3_uncertainty.yaml -e
```

### 4. DA3 Depth Distillation & Uncertainty

#### DA3 Depth Distillation

This project uses [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3) as the teacher model to pre-generate dense metric depth maps as pseudo-labels, providing additional supervision to the detector's depth prediction head via a distillation loss — without modifying the detector backbone architecture.

Before running distillation training, generate the DA3 depth pseudo-labels first (see Section 2 — Data Preparation):

```bash
python tools/generate_da3_depth.py --data_path data/KITTI --split training
```

Each `.npz` file contains `depth`, `intrinsics`, and `extrinsics` keys, along with a `_vis.jpg` visualization image. Only the `depth` key is read during training.

The total distillation loss is:

$$
L_{total} = L_{cls} + L_{bbox} + L_{dim} + \lambda \cdot L_{distill}
$$

where $L_{distill}$ is the L1 or SiLog loss between the predicted depth and the DA3 pseudo-label, and $\lambda$ is the distillation weight. The minimal YAML configuration to enable depth distillation:

```yaml
dataset:
  use_da3_depth: True
```

#### Uncertainty-Guided Adaptive Distillation

Building upon DA3 depth distillation, we further introduce **per-pixel uncertainty prediction**, enabling the model to adaptively estimate confidence for the DA3 pseudo-labels. Regions with high uncertainty (e.g., reflective surfaces, occlusion boundaries) will automatically receive lower distillation loss weights, mitigating the negative impact of noisy pseudo-labels:

$$
L_{distill}^{unc} = \frac{1}{N} \sum_{i} \frac{|d_i - \hat{d}_i|}{\sigma_i} + \log \sigma_i
$$

where $\sigma_i$ is the model-predicted depth uncertainty, $d_i$ is the DA3 pseudo-label, and $\hat{d}_i$ is the model-predicted depth. To enable uncertainty-guided distillation, set the following in the YAML config:

```yaml
dataset:
  use_da3_depth: True # DA3 depth distillation must be enabled

distill:
  lambda: 0.5
  loss_type: 'l1'           # 'l1' or 'silog'
  foreground_weight: 5.0
  use_uncertainty: True     # Enable uncertainty-guided adaptive depth distillation
```

## Acknowledgements

Thanks to the excellent implementations of [MonoDLE](https://github.com/XinzhuMa/MonoDLE) and [CenterNet](https://github.com/xingyizhou/CenterNet).

## License

Released under the MIT License.
