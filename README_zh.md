# 深入研究单目 3D 检测的定位误差


## 简介


## 使用方法

### 安装
本仓库在我们的本地环境（python=3.13, cuda=12.8, pytorch=2.10.0）中进行了测试，建议您使用 uv 创建虚拟环境：

```bash
uv venv --python=3.13
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### 数据准备
请下载 [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 并按以下方式组织数据：

```text
#ROOT
└── data
    └── KITTI
        ├── ImageSets [本仓库已提供]
        ├── object [建议创建此符号链接]
        │   ├── training -> ../training
        │   └── testing -> ../testing
        ├── training
        │   ├── calib (从 calib.zip 解压)
        │   ├── image_2 (从 left_color.zip 解压)
        │   └── label_2 (从 label_2.zip 解压)
        └── testing
            ├── calib
            └── image_2
```

### 训练与评估

#### MonoDLE（CenterNet 架构）

MonoDLE 训练脚本已支持以下能力：

- DDP 多卡训练（`torchrun`）
- AMP 混合精度（`--amp`）
- `torch.compile` 图编译加速（`--compile`）
- EMA 参数滑动平均（`--ema`）
- 随机种子与确定性算法控制（`--seed`、`--deterministic`）
- timm 集成：
    - timm backbone（`model.type: centernet3d_timm`）
    - timm optimizer（`optimizer.type` 可直接使用 timm 优化器名，如 `lamb`）

在项目根目录下运行：

```sh
# 1) 默认训练（单卡或 DataParallel）
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml

# 2) DDP 多卡训练（示例：8 卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=11451 tools/train_val.py \
    --config experiments/kitti/monodle_kitti.yaml --ddp --amp --ema --compile

# 3) 单卡 + AMP + compile + EMA
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \
    --amp --compile --ema

# 4) 仅评估
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml -e
```

输出目录与 YOLO3D 保持一致：`<project>/<name>/`（默认 `runs/monodle/train/`）。

```sh 
# 指定输出目录（推荐）
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \
    --project runs/monodle --name exp01

# 仍支持直接覆盖输出目录
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml \
    -o runs/monodle/exp01
```

典型产出包括：`args.yaml`、`train.log.*`、`weights/`、`kitti_eval/`、`eval_results.csv`、`last_eval_result.json`、`visualizations/`。

训练过程中，每个 epoch 评估时会自动生成前 2 个 batch 的可视化结果（热力图、2D/3D 检测框），保存在 `<output_dir>/visualizations/epoch_<N>/` 目录下。

配置参考：

- `experiments/kitti/monodle_kitti.yaml`
    - `project` / `name`：输出目录
    - `model.type: centernet3d | centernet3d_timm`
    - `optimizer.type`：支持 torch 与 timm 优化器

#### 性能表现 (KITTI 验证集 / Chen-Split)

| 方法                | 3D@0.7 (M) | 说明                         |
| :------------------ | :--------: | :--------------------------- |
| **MonoDLE (Paper)** | **13.66**  | 官方论文结果                 |
| **MonoDLE (Repo)**  | **12.69**  | 当前仓库复现结果 (Epoch 140) |

#### YOLO3D（基于 Ultralytics）

YOLO3D 在 Ultralytics YOLO 检测框架的基础上，添加了 MonoDDLE 风格的单目 3D 检测头，复用 YOLO 的训练循环、数据增强和后处理，同时增加了深度、3D 尺寸、朝向角和 3D 中心偏移的预测分支。

支持三种 YOLO 骨架版本：

| 版本   | 配置文件          | NMS                    | 特性                               |
| ------ | ----------------- | ---------------------- | ---------------------------------- |
| YOLOv8 | `yolo3d_v8n.yaml` | 标准 NMS               | Anchor-free，reg_max=16            |
| YOLO11 | `yolo3d_11n.yaml` | 标准 NMS               | 改进骨架（C3k2/C2PSA），reg_max=16 |
| YOLO26 | `yolo3d_26n.yaml` | **无需 NMS**（端到端） | 双头 one2many/one2one，reg_max=1   |

```sh
# YOLOv8 — 标准 NMS
python tools/train_yolo3d.py --model yolov8n.pt --data experiments/kitti/yolo3d_v8n.yaml

# YOLO11 — 标准 NMS
python tools/train_yolo3d.py --model yolo11n.pt --data experiments/kitti/yolo3d_11n.yaml

# YOLO26 — 无需 NMS（端到端）
python tools/train_yolo3d.py --model yolo26m.pt --data experiments/kitti/yolo3d_26m.yaml

# 从零训练（任意版本）
python tools/train_yolo3d.py --model yolo26m.yaml --data experiments/kitti/yolo3d_26m.yaml

# 恢复训练
python tools/train_yolo3d.py --model runs/yolo3d/yolov8n/weights/last.pt --resume
```

##### DDP 多卡训练

YOLO3D 基于 Ultralytics 训练框架，通过 `--device` 参数指定多张 GPU 即可自动启用 DDP 分布式训练，**无需手动使用 `torchrun`**：

```sh
# 2 卡 DDP
python tools/train_yolo3d.py --model yolov8n.pt --data experiments/kitti/yolo3d_v8n.yaml \
    --device 0,1

# 4 卡 DDP
python tools/train_yolo3d.py --model yolo26m.pt --data experiments/kitti/yolo3d_26m.yaml \
    --device 0,1,2,3

# 8 卡 DDP + 自定义 batch size
python tools/train_yolo3d.py --model yolo11n.pt --data experiments/kitti/yolo3d_11n.yaml \
    --device 0,1,2,3,4,5,6,7 --batch 64

# 使用 CUDA_VISIBLE_DEVICES 选择可见 GPU 后再指定
export CUDA_VISIBLE_DEVICES=2,3,5,7
python tools/train_yolo3d.py --model yolov8s.pt --data experiments/kitti/yolo3d_v8n.yaml \
    --device 0,1,2,3
```

> **注意事项：**
> - `--device 0,1,...` 传入多个 GPU ID 时，Ultralytics 会自动 spawn 子进程并使用 DDP；
> - `--batch` 为**全局** batch size，会被均分到各卡（如 `--batch 16 --device 0,1` → 每卡 8）；
> - DDP 模式下日志仅由 rank 0 输出，各卡梯度自动同步；
> - 若需恢复 DDP 训练，直接 `--resume` 即可，框架会自动重建分布式环境。

所有产出保存到 `runs/yolo3d/<name>/`。模型规模可将名称中的 `n` 替换为 `s`/`m`/`l`/`x`（如 `yolov8s.pt`、`yolo26m.yaml`）。

**架构概览:**

```
YOLO 骨架 (v8/11/26) → FPN Neck → Detect3D Head
                                     ├── cv2: 2D 边界框回归
                                     ├── cv3: 分类
                                     └── cv4: 单目3D（深度 + 3D中心 + 尺寸 + 朝向）
```

## 致谢

本仓库受益于优秀的开源项目, 也请考虑引用它。
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [MonoDLE](https://github.com/xinzhuma/monodle/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## 许可证

本项目在 GNU General Public License v3.0 (GPL-3.0) 许可证下发布。

## 联系方式

如果您对本项目有任何疑问，请随时联系 1138663075@qq.com。
