# 深入研究单目 3D 检测的定位误差

作者：[Xinzhu Ma](https://scholar.google.com/citations?user=8PuKa_8AAAAJ), Yinmin Zhang, [Dan Xu](https://www.danxurgb.net/), [Dongzhan Zhou](https://scholar.google.com/citations?user=Ox6SxpoAAAAJ), [Shuai Yi](https://scholar.google.com/citations?user=afbbNmwAAAAJ), [Haojie Li](https://scholar.google.com/citations?user=pMnlgVMAAAAJ), [Wanli Ouyang](https://wlouyang.github.io/)。


## 简介

本仓库是论文 ['Delving into Localization Errors for Monocular 3D Detection'](https://arxiv.org/abs/2103.16237) 的官方实现。在这项工作中，通过深入的诊断实验，我们量化了每个子任务引入的影响，并发现“定位误差”是限制单目 3D 检测的关键因素。此外，我们还调查了定位误差背后的潜在原因，分析了它们可能带来的问题，并提出了三种策略。 

<img src="resources/example.jpg" alt="vis" style="zoom:50%;" />




## 使用方法

### 安装
本仓库在我们的本地环境（python=3.6, cuda=9.0, pytorch=1.1）中进行了测试，建议您使用 anaconda 创建虚拟环境：

```bash
conda create -n monodle python=3.6
```
然后，激活环境：
```bash
conda activate monodle
```

安装 PyTorch：

```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```

以及其他依赖项：
```bash
pip install -r requirements.txt
```

### 数据准备
请下载 [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 并按以下方式组织数据：

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [本仓库已提供]
      |object/			
        |training/
          |calib/
          |image_2/
          |label/
        |testing/
          |calib/
          |image_2/
```

### 训练与评估

切换到工作目录并训练网络：

```sh
 cd #ROOT
 cd experiments/example
 python ../../tools/train_val.py --config kitti_example.yaml
```
训练完成后，模型将自动进行评估。如果您只想评估训练好的模型（或提供的 [预训练模型](https://drive.google.com/file/d/1jaGdvu_XFn5woX0eJ5I2R6wIcBLVMJV6/view?usp=sharing)），您可以修改 .yaml 文件中的测试部分配置，并使用以下命令：

```sh
python ../../tools/train_val.py --config kitti_example.yaml --e
```

为了方便使用，我们还提供了一个预训练权重，可直接用于评估。参考下表查看性能。

|                   | AP40@Easy | AP40@Mod. | AP40@Hard |
| ----------------- | --------- | --------- | --------- |
| 原始论文           | 17.45     | 13.66     | 11.68     |
| 本仓库             | 17.94     | 13.72     | 12.10     |

## 引用

如果您发现我们的工作对您的研究有用，请考虑引用：

```latex
@InProceedings{Ma_2021_CVPR,
author = {Ma, Xinzhu and Zhang, Yinmin, and Xu, Dan and Zhou, Dongzhan and Yi, Shuai and Li, Haojie and Ouyang, Wanli},
title = {Delving into Localization Errors for Monocular 3D Object Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}}
```

## 致谢

本仓库受益于优秀的开源项目 [CenterNet](https://github.com/xingyizhou/CenterNet)。也请考虑引用它。

## 许可证

本项目在 MIT 许可证下发布。

## 联系方式

如果您对本项目有任何疑问，请随时联系 xinzhu.ma@sydney.edu.au。
