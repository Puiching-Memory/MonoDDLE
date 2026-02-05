# KITTI Evaluation Module (C++/CUDA)

This module implements high-performance IoU calculations and statistic accumulation for KITTI evaluation using CUDA kernels and C++ bindings.
It replaces the previous Numba/Triton implementation for significantly better performance and removal of JIT overhead.

## Logic Overview
- **IoU Calculation**:
    - 2D (Image) IoU: CUDA Kernel
    - BEV (Rotated) IoU: CUDA Kernel (Sutherland-Hodgman Polygon Clipping)
    - 3D IoU: CUDA Kernel (BEV Area * Height Overlap)
- **Statistics**:
    - TP/FP/FN accumulating and threshold matching are implemented in C++ to avoid Python overhead.

## Installation
Run the following command in this directory to compile the extension:
```bash
python3 setup.py build_ext --inplace
```

## Dependencies
- PyTorch
- CUDA Toolkit (compatible with your PyTorch version)

## Usage
The interface is now simplified compared to the previous version.
```bash
# Ensure you are at the project root
python3 -m lib.datasets.kitti.kitti_eval.launch --label_path=... --result_path=...
```

## Files
- `__init__.py`: Package initialization.
- `core.py`: Main entry point (Python wrapper).
- `utils.py`: File & label helpers (formerly kitti_common.py).
- `launch.py`: Command-line interface (formerly evaluate.py).
- `src/iou3d.cpp`: C++ bindings and CPU-side logic.
- `src/iou3d_kernel.cu`: CUDA kernels.
- `setup.py`: Build script.
