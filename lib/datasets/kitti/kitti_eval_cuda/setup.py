"""Build script for KITTI eval CUDA extension."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name="kitti_eval_cuda_ops",
    ext_modules=[
        CUDAExtension(
            name="kitti_eval_cuda_ops",
            sources=[
                os.path.join("csrc", "eval_ops.cpp"),
                os.path.join("csrc", "rotate_iou_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
                "nvcc": ["-O3", "--fmad=false"],
            },
            extra_link_args=["-fopenmp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
