"""
KITTI 3D 目标检测数据集
======================

提供标准 KITTI 数据集的加载、增强和评估支持。
评估模块内置 C++/CUDA 加速，保证结果与官方 kitti_eval_python 完全一致。

公开 API::

    from lib.datasets.kitti import KITTI_Dataset, Calibration, Object3d
    from lib.datasets.kitti import get_official_eval_result, get_distance_eval_result
"""

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_utils import (
    Object3d, Calibration,
    get_objects_from_label, get_calib_from_file,
)
from lib.datasets.kitti.kitti_eval import (
    get_official_eval_result,
    get_distance_eval_result,
)

__all__ = [
    'KITTI_Dataset',
    'Object3d', 'Calibration',
    'get_objects_from_label', 'get_calib_from_file',
    'get_official_eval_result', 'get_distance_eval_result',
]
