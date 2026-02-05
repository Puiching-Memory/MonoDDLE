# KITTI 评估模块 (C++/CUDA)

本模块实现了基于 CUDA 核心和 C++ 绑定的高性能 KITTI 评估（IoU 计算与统计），旨在替换旧版基于 Python/Numba/Triton 的实现，以获得显著的性能提升并消除 JIT 编译开销。

## 核心逻辑
- **IoU 计算**:
    - 2D (图像) IoU: CUDA Kernel
    - BEV (旋转框) IoU: CUDA Kernel (Sutherland-Hodgman 多边形裁剪)
    - 3D IoU: CUDA Kernel (BEV 面积 * 高度重叠)
- **统计**:
    - TP/FP/FN 累积与阈值匹配完全在 C++ 中实现，避免 Python 循环开销。

## 安装
在该目录下运行以下命令以编译扩展：
```bash
python3 setup.py build_ext --inplace
```

## 依赖
- PyTorch
- CUDA Toolkit (需与 PyTorch 版本兼容)

## 使用
接口相较于旧版本进行了简化。
可以使用以下命令运行评估：
```bash
# 请确保当前路径为项目根目录
python3 -m lib.datasets.kitti.kitti_eval.launch --label_path=... --result_path=...
```

## 文件结构说明
- `__init__.py`: 包入口，暴露核心接口。
- `core.py`: 核心逻辑入口 (Python 包装器)，调用 C++ 扩展。
- `utils.py`: (原 kitti_common.py) 通用文件路径与标签解析工具。
- `launch.py`: (原 evaluate.py) 命令行启动脚本。
- `src/iou3d.cpp`: PyTorch C++ 绑定与主机端统计逻辑。
- `src/iou3d_kernel.cu`: CUDA 核心代码 (IoU 计算)。
- `setup.py`: 编译脚本。
