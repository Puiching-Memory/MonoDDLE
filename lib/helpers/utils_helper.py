import os
import torch
import numpy as np
import random

from lib.helpers.logger_helper import MonoDDLELogger


def create_logger(log_file, rank=0):
    """Create logger (uses MonoDDLELogger with Rich output)."""
    return MonoDDLELogger(log_file if rank == 0 else None, rank=rank)


def set_random_seed(seed, deterministic=True):
    """Set random seed for full reproducibility.

    Args:
        seed: integer seed value.
        deterministic: if True, enable CUDA deterministic algorithms and
            disable cuDNN benchmarking for bitwise reproducibility (may be
            slower).  Set False to keep cuDNN auto-tuning for speed.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # older PyTorch without warn_only
                torch.use_deterministic_algorithms(True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def setup_ddp(rank, world_size, backend='nccl', port=None):
    """Initialise the distributed process group for DDP.

    Args:
        rank: process rank (0-based).
        world_size: total number of processes.
        backend: 'nccl' (GPU) or 'gloo' (CPU).
        port: master port (default picked from env or '29500').
    """
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', str(port or 29500))
    torch.distributed.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Destroy the distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process():
    """Return True on the main process (rank 0) or when not using DDP."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0