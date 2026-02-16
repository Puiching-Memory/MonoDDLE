"""
KITTI 评估包
============

公共 API:
    - :func:`get_official_eval_result`
    - :func:`get_distance_eval_result`
    - :mod:`.utils` — 标签文件 IO
"""

from .core import get_official_eval_result, get_distance_eval_result
from . import utils

__all__ = [
    "get_official_eval_result",
    "get_distance_eval_result",
    "utils",
]
