"""
KITTI 评估 CLI 入口
===================

用法::

    python -m lib.datasets.kitti.kitti_eval.launch \\
        --label_path /path/to/gt \\
        --result_path /path/to/predictions \\
        --label_split_file /path/to/val.txt

也可作为模块直接执行::

    python -m lib.datasets.kitti.kitti_eval.launch evaluate \\
        /path/to/gt /path/to/predictions /path/to/val.txt
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

try:
    import fire
except ImportError:
    fire = None  # type: ignore[assignment]

from . import utils as kitti
from .core import get_official_eval_result


def _read_imageset_file(path: str) -> list[int]:
    """读取 ImageSets 分割文件，返回图像序号列表。"""
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def evaluate(
    label_path: str,
    result_path: str,
    label_split_file: str,
    current_class: int = 0,
    score_thresh: float = -1.0,
) -> str:
    """执行 KITTI 目标检测评估。

    Parameters
    ----------
    label_path : str
        GT 标签文件夹路径。
    result_path : str
        预测结果文件夹路径（包含 ``XXXXXX.txt``）。
    label_split_file : str
        验证集图像 ID 文件（如 ``val.txt``）。
    current_class : int
        要评估的类别索引 (0=Car, 1=Ped, 2=Cyc)。
    score_thresh : float
        置信度过滤阈值，-1 表示不过滤。

    Returns
    -------
    str
        评估结果文本。
    """
    t0 = time.perf_counter()

    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)

    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)

    results_str, results_dict, rich_data = get_official_eval_result(
        gt_annos, dt_annos, current_class,
    )

    # ── 历史结果比较 ──
    prev_rich_data = None
    last_result_path: Optional[str] = None
    try:
        parent_dir = os.path.dirname(os.path.normpath(result_path))
        if os.path.basename(parent_dir) == "data":
            parent_dir = os.path.dirname(parent_dir)
        base_dir = os.path.dirname(parent_dir)
        last_result_path = os.path.join(base_dir, "last_eval_result.json")

        if os.path.exists(last_result_path):
            with open(last_result_path, "r") as f:
                prev_rich_data = json.load(f)

        with open(last_result_path, "w") as f:
            json.dump(
                rich_data, f,
                default=lambda o: o.tolist() if isinstance(o, np.ndarray) else None,
            )
    except Exception:
        pass

    # ── Rich 打印 ──
    try:
        from lib.helpers.logger_helper import print_kitti_eval_results
        print_kitti_eval_results(rich_data, prev_rich_data)
    except ImportError:
        print(results_str)

    elapsed = time.perf_counter() - t0
    print(f"\n[kitti_eval] Evaluation completed in {elapsed:.2f}s")

    return results_str


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(evaluate)
    else:
        raise RuntimeError(
            "python-fire is required for CLI usage. "
            "Install it with: pip install fire"
        )
