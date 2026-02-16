"""
可组合数据变换 (Composable Transforms)
=====================================

为 3D 目标检测数据集提供标准化的数据增强变换管道。

用法::

    from lib.datasets.transforms import Compose, RandomHorizontalFlip, RandomCrop

    transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomCrop(scale=0.4, shift=0.1, p=0.5),
        AffineResize(resolution=(1280, 384)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

__all__ = [
    'Compose', 'RandomHorizontalFlip', 'RandomCrop',
    'AffineResize', 'Normalize', 'ToTensor',
    'get_affine_transform', 'affine_transform',
]


# ────────────────────── 仿射变换辅助 ──────────────────────

def _get_dir(src_point: Sequence[float], rot_rad: float) -> List[float]:
    """旋转一个向量。"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ]


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """由两点求仿射变换所需的第三个参考点。"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray | float,
    rot: float,
    output_size: np.ndarray | Sequence[int],
    shift: np.ndarray = np.array([0, 0], dtype=np.float32),
    inv: int = 0,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """计算仿射变换矩阵。

    Args:
        center: 输入图像中心 (W, H)。
        scale: 缩放因子或数组。
        rot: 旋转角度 (度)。
        output_size: 输出尺寸 (W, H)。
        shift: 平移偏移。
        inv: 若 >0 同时返回逆变换。

    Returns:
        trans 或 (trans, trans_inv)。
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    output_size = np.asarray(output_size)

    src_w = scale[0]
    dst_w, dst_h = float(output_size[0]), float(output_size[1])

    rot_rad = np.pi * rot / 180.0
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans


def affine_transform(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """对 2D 点施加仿射变换。

    Args:
        pt: 形状 (2,) 的点坐标。
        t: 形状 (2, 3) 的仿射矩阵。
    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


# ────────────────────── 变换基类 ──────────────────────

class BaseTransform:
    """所有变换的基类。"""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(BaseTransform):
    """组合多个变换。"""

    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + "\n".join(lines) + "\n])"


class RandomHorizontalFlip(BaseTransform):
    """随机水平翻转。"""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() < self.p:
            data['random_flip'] = True
        else:
            data['random_flip'] = False
        return data

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomCrop(BaseTransform):
    """随机缩放裁剪。"""

    def __init__(self, scale: float = 0.4, shift: float = 0.1, p: float = 0.5) -> None:
        self.scale = scale
        self.shift = shift
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() < self.p:
            data['random_crop'] = True
            data['aug_scale'] = np.clip(
                np.random.randn() * self.scale + 1,
                1 - self.scale, 1 + self.scale,
            )
            img_size = data['img_size']
            center = data['center']
            center[0] += img_size[0] * np.clip(
                np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift
            )
            center[1] += img_size[1] * np.clip(
                np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift
            )
            data['center'] = center
        else:
            data['random_crop'] = False
        return data

    def __repr__(self) -> str:
        return f"RandomCrop(scale={self.scale}, shift={self.shift}, p={self.p})"


class AffineResize(BaseTransform):
    """基于仿射变换的尺寸归一化。"""

    def __init__(self, resolution: Tuple[int, int] = (1280, 384)) -> None:
        self.resolution = np.array(resolution)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        center = data['center']
        crop_size = data.get('crop_size', data['img_size'])

        trans, trans_inv = get_affine_transform(
            center, crop_size, 0, self.resolution, inv=1
        )
        data['trans'] = trans
        data['trans_inv'] = trans_inv
        return data


class Normalize(BaseTransform):
    """图像标准化。"""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data['image']
        if isinstance(img, Image.Image):
            img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        data['image'] = img
        return data

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()})"
