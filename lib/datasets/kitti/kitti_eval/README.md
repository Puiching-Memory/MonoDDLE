# KITTI Evaluation Module (C++/CUDA)

This module implements high-performance IoU calculations and statistic accumulation for KITTI evaluation using CUDA kernels and C++ bindings.
It replaces the previous Numba/Triton implementation for significantly better performance.

## Logic Overview
- **IoU Calculation**:
    - 2D (Image) IoU: CUDA Kernel
    - BEV (Rotated) IoU: CUDA Kernel (Sutherland-Hodgman Polygon Clipping)
    - 3D IoU: CUDA Kernel (BEV Area * Height Overlap)
- **Statistics**:
    - TP/FP/FN accumulating and threshold matching are implemented in C++ to avoid Python overhead.
    - `batch_collect_thresholds`: Batch-collect TP thresholds for all images (phase 1).
    - `batch_compute_pr`: Batch-compute PR for all thresholds (phase 2).

## Installation
No manual build step required. The C++/CUDA extension is automatically JIT-compiled via `torch.utils.cpp_extension.load()` on first import. Build artifacts are cached for subsequent runs.

## Dependencies
- PyTorch (with CUDA support)
- CUDA Toolkit (compatible with your PyTorch version)
- `python-fire` (required for CLI usage only)

## Public API
```python
from lib.datasets.kitti.kitti_eval import get_official_eval_result, get_distance_eval_result, utils
```
- `get_official_eval_result(gt_annos, dt_annos, current_class)` — Standard KITTI evaluation
- `get_distance_eval_result(gt_annos, dt_annos, current_class)` — Distance-based evaluation
- `utils` — Label file IO (`get_label_anno`, `get_label_annos`, `filter_annos_low_score`)

## CLI Usage
```bash
# Ensure you are at the project root
python3 -m lib.datasets.kitti.kitti_eval.launch \
    --label_path=/path/to/gt \
    --result_path=/path/to/predictions \
    --label_split_file=/path/to/val.txt \
    --current_class=0 \
    --score_thresh=-1
```
- `label_path`: Path to ground truth label directory.
- `result_path`: Path to prediction directory (containing `XXXXXX.txt` files).
- `label_split_file`: Validation image ID file (e.g., `val.txt`).
- `current_class`: Class index to evaluate (0=Car, 1=Ped, 2=Cyc), default 0.
- `score_thresh`: Confidence filter threshold, -1 for no filtering.

Positional arguments are also supported:
```bash
python3 -m lib.datasets.kitti.kitti_eval.launch evaluate \
    /path/to/gt /path/to/predictions /path/to/val.txt
```

## Files
- `__init__.py`: Package initialization, exposing `get_official_eval_result`, `get_distance_eval_result`, and `utils`.
- `core.py`: Main entry point (Python wrapper), JIT-loads the C++ extension.
- `utils.py`: Label file reading, parsing and filtering (formerly kitti_common.py).
- `launch.py`: Command-line interface (formerly evaluate.py), powered by `python-fire`.
- `src/iou3d.cpp`: C++ bindings, batch statistics logic, and host-side computation.
- `src/iou3d_kernel.cu`: CUDA kernels (2D/BEV/3D IoU).
