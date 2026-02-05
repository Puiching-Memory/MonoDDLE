from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sources = [os.path.join(src_dir, 'iou3d.cpp'), os.path.join(src_dir, 'iou3d_kernel.cu')]

setup(
    name='iou3d_cuda',
    ext_modules=[
        CUDAExtension('iou3d_cuda', sources)
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
