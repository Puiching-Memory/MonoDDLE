"""
MonoDDLE 数据集框架
=================

提供统一的数据集注册、构建和评估接口。
支持 KITTI 等主流 3D 目标检测数据集，内置 C++/CUDA 加速评估。

用法示例::

    from lib.datasets import build_dataset, DATASET_REGISTRY

    dataset = build_dataset('kitti', split='train', cfg=my_cfg)

"""

from lib.datasets.registry import DATASET_REGISTRY, register_dataset, build_dataset
from lib.datasets.utils import (
    angle2class, class2angle,
    gaussian_radius, gaussian2D,
    draw_umich_gaussian, draw_msra_gaussian,
    draw_projected_box3d,
)

__all__ = [
    # 注册系统
    'DATASET_REGISTRY', 'register_dataset', 'build_dataset',
    # 通用工具
    'angle2class', 'class2angle',
    'gaussian_radius', 'gaussian2D',
    'draw_umich_gaussian', 'draw_msra_gaussian',
    'draw_projected_box3d',
]
