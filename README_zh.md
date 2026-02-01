# 深入研究单目 3D 检测的定位误差


## 简介


## 使用方法

### 安装
本仓库在我们的本地环境（python=3.13, cuda=12.8, pytorch=2.9.1）中进行了测试，建议您使用 uv 创建虚拟环境：

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
        └── object
            ├── training
            │   ├── calib (从 calib.zip 解压)
            │   ├── image_2 (从 left_color.zip 解压)
            │   └── label_2 (从 label_2.zip 解压)
            └── testing
                ├── calib
                └── image_2
```

### 训练与评估

在项目根目录下运行以下命令进行训练：

```sh
# 训练
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml

# 如果您只想评估模型
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml --e
```

训练完成后，模型将自动进行评估。

为了方便使用，我们还提供了一个预训练权重，可直接用于评估。参考下表查看性能。

|          | AP40@Easy | AP40@Mod. | AP40@Hard |
| -------- | --------- | --------- | --------- |
| 原始论文 | 17.45     | 13.66     | 11.68     |
| 本仓库   | 17.94     | 13.72     | 12.10     |

## 引用

如果您发现我们的工作对您的研究有用，请考虑引用：

```latex
```

## 致谢

本仓库受益于优秀的开源项目 [CenterNet](https://github.com/xingyizhou/CenterNet)。也请考虑引用它。

## 许可证

本项目在 GNU General Public License v3.0 (GPL-3.0) 许可证下发布。

## 联系方式

如果您对本项目有任何疑问，请随时联系 1138663075@qq.com。
