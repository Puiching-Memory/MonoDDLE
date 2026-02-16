"""
KITTI 评估 IO 工具
==================

提供标签文件的读取、解析和过滤功能。

函数
----
- ``get_label_anno``   : 读取单个标签文件为 annotation 字典
- ``get_label_annos``  : 批量读取标签文件
- ``filter_annos_low_score`` : 按置信度阈值过滤检测结果

annotation 字典格式
-------------------
.. code-block:: python

    {
        'name':       np.ndarray[str],      # (N,) 类别名
        'truncated':  np.ndarray[float64],   # (N,)
        'occluded':   np.ndarray[float64],   # (N,)
        'alpha':      np.ndarray[float64],   # (N,) 观察角
        'bbox':       np.ndarray[float32],   # (N,4) [x1,y1,x2,y2]
        'dimensions': np.ndarray[float64],   # (N,3) [l,h,w]
        'location':   np.ndarray[float64],   # (N,3) [x,y,z]
        'rotation_y': np.ndarray[float64],   # (N,)
        'score':      np.ndarray[float64],   # (N,) (-1 if absent)
    }
"""

from __future__ import annotations

import pathlib
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "get_label_anno",
    "get_label_annos",
    "filter_annos_low_score",
]


# ────────────────────────────────────────────────────────────────────
# 内部: 轻量级 Object3d 解析 (仅在本模块使用)
# ────────────────────────────────────────────────────────────────────

class _Object3d:
    """KITTI 标签行的最小化解析器（仅供 IO 使用）。"""
    __slots__ = (
        "cls_type", "truncation", "occlusion", "alpha",
        "box2d", "h", "w", "l", "pos", "ry", "score",
    )

    def __init__(self, line: str) -> None:
        label = line.strip().split()
        self.cls_type: str = label[0]
        self.truncation: float = float(label[1])
        self.occlusion: float = float(label[2])
        self.alpha: float = float(label[3])
        self.box2d = np.array(
            [float(label[4]), float(label[5]),
             float(label[6]), float(label[7])],
            dtype=np.float32,
        )
        self.h: float = float(label[8])
        self.w: float = float(label[9])
        self.l: float = float(label[10])
        self.pos = np.array(
            [float(label[11]), float(label[12]), float(label[13])],
            dtype=np.float64,
        )
        self.ry: float = float(label[14])
        self.score: float = float(label[15]) if len(label) >= 16 else -1.0


# ────────────────────────────────────────────────────────────────────
# 空 annotation 模板
# ────────────────────────────────────────────────────────────────────

def _empty_annotation() -> dict:
    """返回一个元素数为 0 的 annotation 字典。"""
    return {
        "name": np.array([]),
        "truncated": np.array([]),
        "occluded": np.array([]),
        "alpha": np.array([]),
        "bbox": np.zeros((0, 4), dtype=np.float32),
        "dimensions": np.zeros((0, 3), dtype=np.float64),
        "location": np.zeros((0, 3), dtype=np.float64),
        "rotation_y": np.array([]),
        "score": np.array([]),
    }


# ────────────────────────────────────────────────────────────────────
# 公共 API
# ────────────────────────────────────────────────────────────────────

def get_image_index_str(img_idx: int) -> str:
    """将图像序号转为 6 位零填充字符串。"""
    return f"{img_idx:06d}"


def get_label_anno(label_path: Union[str, pathlib.Path]) -> dict:
    """读取单个 KITTI 标签文件并返回 annotation 字典。

    Parameters
    ----------
    label_path : str or Path
        标签 ``.txt`` 文件的完整路径。

    Returns
    -------
    dict
        标准 annotation 字典。
    """
    label_path = pathlib.Path(label_path)
    try:
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception as e:                              # noqa: BLE001
        print(f"Error reading {label_path}: {e}")
        return _empty_annotation()

    if not lines:
        return _empty_annotation()

    objects = [_Object3d(ln) for ln in lines]

    return {
        "name": np.array([o.cls_type for o in objects]),
        "truncated": np.array([o.truncation for o in objects]),
        "occluded": np.array([o.occlusion for o in objects]),
        "alpha": np.array([o.alpha for o in objects]),
        "bbox": np.array([o.box2d for o in objects]),
        # dimensions: [l, h, w] — 与 kitti_eval_python/kitti_common.py 保持一致
        "dimensions": np.array([[o.l, o.h, o.w] for o in objects]),
        "location": np.array([o.pos for o in objects]),
        "rotation_y": np.array([o.ry for o in objects]),
        "score": np.array([o.score for o in objects]),
    }


def get_label_annos(
    label_folder: Union[str, pathlib.Path],
    image_ids: Optional[Union[Sequence[int], int]] = None,
) -> List[dict]:
    """批量读取标签文件。

    Parameters
    ----------
    label_folder : str or Path
        包含 ``000000.txt`` 格式标签的文件夹。
    image_ids : list[int] | int | None
        * ``None``  — 自动扫描文件夹中所有 ``\\d{6}.txt``。
        * ``int``   — 等价于 ``range(image_ids)``。
        * ``list``  — 明确指定图像序号列表。

    Returns
    -------
    list[dict]
        每张图片一个 annotation 字典。
    """
    label_folder = pathlib.Path(label_folder)

    if image_ids is None:
        prog = re.compile(r"^\d{6}\.txt$")
        image_ids = sorted(
            int(p.stem)
            for p in label_folder.iterdir()
            if prog.match(p.name)
        )
    elif isinstance(image_ids, int):
        image_ids = list(range(image_ids))

    paths = [label_folder / f"{get_image_index_str(idx)}.txt" for idx in image_ids]

    # 并行 I/O: 大量文件时使用线程池加速
    if len(paths) >= 100:
        with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
            return list(pool.map(get_label_anno, paths))

    return [get_label_anno(p) for p in paths]


def filter_annos_low_score(
    image_annos: List[dict],
    thresh: float,
) -> List[dict]:
    """过滤掉 score < thresh 的检测结果。

    Parameters
    ----------
    image_annos : list[dict]
        由 :func:`get_label_annos` 返回的 annotation 列表。
    thresh : float
        最低置信度阈值。

    Returns
    -------
    list[dict]
        过滤后的 annotation 列表。
    """
    filtered = []
    for anno in image_annos:
        keep = np.array([i for i, s in enumerate(anno["score"]) if s >= thresh])
        new_anno = {}
        for key, val in anno.items():
            new_anno[key] = val[keep] if len(keep) > 0 else val[:0]
        filtered.append(new_anno)
    return filtered
