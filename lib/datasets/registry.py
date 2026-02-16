"""
数据集注册表 (Dataset Registry)
==============================

提供类似 detectron2 / mmdet 风格的数据集注册机制，
允许通过名称字符串动态构建数据集实例。

用法::

    from lib.datasets.registry import DATASET_REGISTRY, register_dataset

    @register_dataset('my_dataset')
    class MyDataset(torch.utils.data.Dataset):
        ...

    dataset = DATASET_REGISTRY.build('my_dataset', split='train', cfg={})
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Type

import torch.utils.data as data

logger = logging.getLogger(__name__)

__all__ = ['DatasetRegistry', 'DATASET_REGISTRY', 'register_dataset', 'build_dataset']


class DatasetRegistry:
    """线程安全的数据集注册表。"""

    def __init__(self, name: str = 'datasets') -> None:
        self._name = name
        self._registry: Dict[str, Type[data.Dataset]] = {}

    def register(self, name: str, dataset_cls: Optional[Type[data.Dataset]] = None) -> Callable:
        """注册数据集类。可作为装饰器或直接调用。

        Args:
            name: 注册名称（不区分大小写、去除首尾空白）。
            dataset_cls: 数据集类。若为 None 则返回装饰器。

        Returns:
            注册后的原始类，或装饰器函数。
        """
        name = name.strip().lower()

        def _register(cls: Type[data.Dataset]) -> Type[data.Dataset]:
            if name in self._registry:
                existing = self._registry[name]
                logger.warning(
                    f"[{self._name}] 覆盖已注册数据集 '{name}': "
                    f"{existing.__name__} -> {cls.__name__}"
                )
            self._registry[name] = cls
            logger.debug(f"[{self._name}] 注册数据集 '{name}' -> {cls.__name__}")
            return cls

        if dataset_cls is not None:
            return _register(dataset_cls)
        return _register

    def build(self, name: str, *args: Any, **kwargs: Any) -> data.Dataset:
        """根据名称构建数据集实例。

        Args:
            name: 注册名。
            *args, **kwargs: 传给数据集构造函数的参数。

        Raises:
            KeyError: 未注册的数据集名称。
        """
        name = name.strip().lower()
        if name not in self._registry:
            available = ', '.join(sorted(self._registry.keys()))
            raise KeyError(
                f"数据集 '{name}' 未注册。"
                f" 可用数据集: [{available}]"
            )
        cls = self._registry[name]
        logger.info(f"构建数据集 '{name}' ({cls.__name__})")
        return cls(*args, **kwargs)

    def get(self, name: str) -> Type[data.Dataset]:
        """获取已注册的数据集类。"""
        name = name.strip().lower()
        if name not in self._registry:
            raise KeyError(f"数据集 '{name}' 未注册。")
        return self._registry[name]

    @property
    def registered_names(self) -> list[str]:
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._registry

    def __repr__(self) -> str:
        names = ', '.join(self.registered_names)
        return f"DatasetRegistry(name={self._name!r}, datasets=[{names}])"


# 全局单例
DATASET_REGISTRY = DatasetRegistry('MonoDDLE')


def register_dataset(name: str) -> Callable:
    """装饰器快捷方式。

    用法::

        @register_dataset('kitti')
        class KITTIDataset(data.Dataset):
            ...
    """
    return DATASET_REGISTRY.register(name)


def build_dataset(name: str, *args: Any, **kwargs: Any) -> data.Dataset:
    """通过名称构建数据集 (快捷方式 for DATASET_REGISTRY.build)。"""
    return DATASET_REGISTRY.build(name, *args, **kwargs)
