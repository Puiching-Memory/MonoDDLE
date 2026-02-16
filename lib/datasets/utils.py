"""
通用数据集工具 (Dataset Utilities)
=================================

提供高斯热图、角度编码、3D 框绘制等通用功能。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

__all__ = [
    'NUM_HEADING_BIN',
    'angle2class', 'class2angle',
    'gaussian_radius', 'gaussian2D',
    'draw_umich_gaussian', 'draw_msra_gaussian',
    'draw_projected_box3d',
]

# ────────────────────── 常量 ──────────────────────

NUM_HEADING_BIN: int = 12
"""航向角离散化的 bin 数量。"""


# ────────────────────── 角度编码 ──────────────────────

def angle2class(angle: float) -> Tuple[int, float]:
    """将连续角度编码为离散类别 + 残差。

    Args:
        angle: 弧度制角度。

    Returns:
        (class_id, residual_angle) 元组。
    """
    angle = angle % (2 * np.pi)
    assert 0 <= angle <= 2 * np.pi
    angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(cls: int, residual: float, to_label_format: bool = False) -> float:
    """将离散类别 + 残差解码为连续角度 (angle2class 的逆函数)。

    Args:
        cls: 类别索引。
        residual: 残差角度。
        to_label_format: 若为 True，将结果限制到 [-π, π]。

    Returns:
        解码后的弧度角度。
    """
    angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


# ────────────────────── 高斯热图 ──────────────────────

def gaussian_radius(bbox_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    """根据边界框尺寸计算高斯核半径 (CornerNet 方法)。

    Args:
        bbox_size: (height, width) 的元组。
        min_overlap: 最小 IoU 阈值。

    Returns:
        三种情况的最小高斯半径。
    """
    height, width = bbox_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """生成 2D 高斯核。

    Args:
        shape: (height, width) 核尺寸。
        sigma: 标准差。

    Returns:
        归一化的二维高斯核。
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(
    heatmap: np.ndarray,
    center: np.ndarray | Tuple[int, int],
    radius: int,
    k: float = 1.0,
) -> np.ndarray:
    """在热图上绘制 UMich 风格的高斯分布。

    Args:
        heatmap: 目标热图 (H, W)。
        center: 高斯中心 (x, y)。
        radius: 高斯半径。
        k: 缩放因子。

    Returns:
        更新后的热图 (原地修改)。
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_msra_gaussian(
    heatmap: np.ndarray,
    center: np.ndarray | Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """在热图上绘制 MSRA 风格的高斯分布。

    Args:
        heatmap: 目标热图 (H, W)。
        center: 高斯中心 (x, y)。
        sigma: 标准差。

    Returns:
        更新后的热图 (原地修改)。
    """
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
    )
    return heatmap


# ────────────────────── 可视化 ──────────────────────

def draw_projected_box3d(
    image: np.ndarray,
    corners3d: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """在图像上绘制投影后的 3D 边界框。

    Args:
        image: RGB 图像 (H, W, 3)。
        corners3d: 投影到图像平面的 8 个顶点 (8, 3)。
            顶点顺序::

                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7

        color: 线条颜色 (B, G, R)。
        thickness: 线宽。

    Returns:
        绘制后的图像。
    """
    corners3d = corners3d.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(
            image,
            (corners3d[i, 0], corners3d[i, 1]),
            (corners3d[j, 0], corners3d[j, 1]),
            color, thickness, lineType=cv2.LINE_AA,
        )
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(
            image,
            (corners3d[i, 0], corners3d[i, 1]),
            (corners3d[j, 0], corners3d[j, 1]),
            color, thickness, lineType=cv2.LINE_AA,
        )
        i, j = k, k + 4
        cv2.line(
            image,
            (corners3d[i, 0], corners3d[i, 1]),
            (corners3d[j, 0], corners3d[j, 1]),
            color, thickness, lineType=cv2.LINE_AA,
        )
    return image
