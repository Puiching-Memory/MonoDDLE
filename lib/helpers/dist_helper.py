"""
Distributed training utilities for DP and DDP dual-path support.

Provides helper functions for:
  - Setting up / tearing down DDP process groups
  - Querying rank, world_size, and local_rank
  - Reducing / broadcasting tensors across ranks
  - Scaling learning rate for DP ↔ DDP equivalence
"""

import os
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Process-group helpers
# ---------------------------------------------------------------------------

def setup_distributed(backend='nccl', init_method='env://'):
    """Initialise the distributed process group.

    Expected environment variables (set by ``torchrun`` / ``torch.distributed.launch``):
        RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    if dist.is_initialized():
        return

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed():
    """Destroy the distributed process group (if initialised)."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Rank / device queries
# ---------------------------------------------------------------------------

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Collective helpers
# ---------------------------------------------------------------------------

def reduce_mean(tensor):
    """All-reduce (mean) a tensor across all ranks.  No-op if non-distributed."""
    if not is_distributed():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def broadcast(tensor, src=0):
    """Broadcast a tensor from *src* to all ranks.  No-op if non-distributed."""
    if not is_distributed():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


# ---------------------------------------------------------------------------
# DP ↔ DDP equivalence utilities
# ---------------------------------------------------------------------------

def compute_ddp_lr(dp_lr, dp_total_batch, dp_num_gpus,
                   ddp_per_gpu_batch, ddp_world_size):
    """Compute the DDP learning-rate that is *gradient-equivalent* to a DP run.

    Explanation
    -----------
    * **DP**: the dataloader produces ``dp_total_batch`` samples.  They are
      scattered to ``dp_num_gpus`` GPUs; the outputs are gathered back to
      GPU 0 where the loss is computed on the **full** batch.
      ``gradient = ∇ mean_B(L)``   where *B = dp_total_batch*.

    * **DDP**: every rank loads ``ddp_per_gpu_batch`` samples independently.
      Each rank computes ``gradient_k = ∇ mean_{b_k}(L)`` and the gradients
      are **all-reduced** (averaged).
      ``effective_gradient = (1/N) Σ_k gradient_k``

    For the two to be mathematically *identical* we need:

        (i)  ddp_per_gpu_batch == dp_total_batch / dp_num_gpus
             (same per-GPU sample count → same effective total batch)
        (ii) learning rate stays the same

    When ``ddp_per_gpu_batch × ddp_world_size ≠ dp_total_batch`` (user wants
    a different total batch), the **linear scaling rule** applies::

        lr_ddp = dp_lr × (ddp_per_gpu_batch × ddp_world_size) / dp_total_batch

    Returns
    -------
    float
        The equivalent learning rate for DDP.
    """
    dp_eff = dp_total_batch
    ddp_eff = ddp_per_gpu_batch * ddp_world_size
    return dp_lr * (ddp_eff / dp_eff)


def compute_dp_lr(ddp_lr, ddp_per_gpu_batch, ddp_world_size,
                  dp_total_batch, dp_num_gpus):
    """Inverse of :func:`compute_ddp_lr`."""
    ddp_eff = ddp_per_gpu_batch * ddp_world_size
    dp_eff = dp_total_batch
    return ddp_lr * (dp_eff / ddp_eff)


def compute_equivalent_config(dp_total_batch, dp_lr, dp_num_gpus,
                              ddp_world_size, ddp_per_gpu_batch=None):
    """Return an equivalent DDP config dict given a DP config.

    Parameters
    ----------
    dp_total_batch : int
        Total batch size used by DP (the value in the YAML).
    dp_lr : float
        Learning rate used in DP.
    dp_num_gpus : int
        Number of GPUs used in DP.
    ddp_world_size : int
        Number of DDP processes (normally == number of GPUs).
    ddp_per_gpu_batch : int or None
        If ``None``, defaults to ``dp_total_batch // ddp_world_size`` (exact
        equivalence).

    Returns
    -------
    dict
        ``{ 'ddp_per_gpu_batch', 'ddp_total_batch', 'ddp_lr',
            'dp_per_gpu_batch', 'is_exact', 'notes' }``
    """
    dp_per_gpu = dp_total_batch / dp_num_gpus

    if ddp_per_gpu_batch is None:
        # exact equivalence
        ddp_per_gpu_batch = int(dp_total_batch // ddp_world_size)
        if ddp_per_gpu_batch == 0:
            ddp_per_gpu_batch = 1

    ddp_total_batch = ddp_per_gpu_batch * ddp_world_size
    ddp_lr = compute_ddp_lr(dp_lr, dp_total_batch, dp_num_gpus,
                            ddp_per_gpu_batch, ddp_world_size)

    is_exact = (ddp_total_batch == dp_total_batch)

    notes = []
    if is_exact:
        notes.append('Exact equivalence: same total batch & same LR.')
    else:
        notes.append(
            f'Total batch changed {dp_total_batch} → {ddp_total_batch}; '
            f'LR scaled {dp_lr} → {ddp_lr:.8f} (linear scaling rule).'
        )

    # warn about object-level mean reduction
    notes.append(
        'Note: losses normalised by #valid_objects (not batch size) may '
        'introduce a tiny gradient discrepancy between DP and DDP.  '
        'In practice this is negligible for typical KITTI distributions.'
    )

    return {
        'dp_total_batch': dp_total_batch,
        'dp_per_gpu_batch': dp_per_gpu,
        'dp_lr': dp_lr,
        'dp_num_gpus': dp_num_gpus,
        'ddp_per_gpu_batch': ddp_per_gpu_batch,
        'ddp_total_batch': ddp_total_batch,
        'ddp_lr': ddp_lr,
        'ddp_world_size': ddp_world_size,
        'is_exact': is_exact,
        'notes': notes,
    }
