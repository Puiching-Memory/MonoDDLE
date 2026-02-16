# KITTI 评估模块 (C++/CUDA)

本模块实现了基于 CUDA 核心和 C++ 绑定的高性能 KITTI 评估（IoU 计算与统计），旨在替换旧版基于 Python/Numba/Triton 的实现，以获得显著的性能提升。

## 核心逻辑
- **IoU 计算**:
    - 2D (图像) IoU: CUDA Kernel
    - BEV (旋转框) IoU: CUDA Kernel (Sutherland-Hodgman 多边形裁剪)
    - 3D IoU: CUDA Kernel (BEV 面积 * 高度重叠)
- **统计**:
    - TP/FP/FN 累积与阈值匹配完全在 C++ 中实现，避免 Python 循环开销。
    - `batch_collect_thresholds`: 批量收集所有图片的 TP 阈值（第一阶段）。
    - `batch_compute_pr`: 批量计算所有阈值的 PR（第二阶段）。

## 安装
无需手动编译。C++/CUDA 扩展通过 `torch.utils.cpp_extension.load()` 在首次 `import` 时自动 JIT 编译，编译产物会被缓存，后续调用无需重新编译。

## 依赖
- PyTorch (含 CUDA 支持)
- CUDA Toolkit (需与 PyTorch 版本兼容)
- `python-fire` (仅 CLI 使用需要)

## 公共 API
```python
from lib.datasets.kitti.kitti_eval import get_official_eval_result, get_distance_eval_result, utils
```
- `get_official_eval_result(gt_annos, dt_annos, current_class)` — 标准 KITTI 评估
- `get_distance_eval_result(gt_annos, dt_annos, current_class)` — 按距离区间评估
- `utils` — 标签文件 IO 工具 (`get_label_anno`, `get_label_annos`, `filter_annos_low_score`)

## CLI 使用
```bash
# 请确保当前路径为项目根目录
python3 -m lib.datasets.kitti.kitti_eval.launch \
    --label_path=/path/to/gt \
    --result_path=/path/to/predictions \
    --label_split_file=/path/to/val.txt \
    --current_class=0 \
    --score_thresh=-1
```
- `label_path`: GT 标签文件夹路径。
- `result_path`: 预测结果文件夹路径（包含 `XXXXXX.txt`）。
- `label_split_file`: 验证集图像 ID 文件（如 `val.txt`）。
- `current_class`: 要评估的类别索引 (0=Car, 1=Ped, 2=Cyc)，默认 0。
- `score_thresh`: 置信度过滤阈值，-1 表示不过滤。

也可使用位置参数：
```bash
python3 -m lib.datasets.kitti.kitti_eval.launch evaluate \
    /path/to/gt /path/to/predictions /path/to/val.txt
```

## 文件结构说明
- `__init__.py`: 包入口，暴露 `get_official_eval_result`、`get_distance_eval_result` 和 `utils`。
- `core.py`: 核心逻辑入口 (Python 包装器)，通过 JIT 编译加载 C++ 扩展。
- `utils.py`: (原 kitti_common.py) 标签文件读取、解析和过滤工具。
- `launch.py`: (原 evaluate.py) 命令行启动脚本，基于 `python-fire`。
- `src/iou3d.cpp`: PyTorch C++ 绑定、批量统计逻辑与主机端计算。
- `src/iou3d_kernel.cu`: CUDA 核心代码 (2D/BEV/3D IoU 计算)。
