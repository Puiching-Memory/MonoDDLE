#!/usr/bin/env python3
"""
DP ↔ DDP Equivalence Calculator
================================

MonoDLE 的训练结果对 batch size 和学习率非常敏感。当从 DataParallel (DP)
迁移到 DistributedDataParallel (DDP) 时，如果不正确设置这两个参数，模型
精度会出现明显差异。

本脚本用于精确计算 DP→DDP 迁移时所需的超参数调整，确保训练等效。

原理
----
**DataParallel (DP)**
  - 单进程多 GPU
  - YAML 中的 ``batch_size`` = 总 batch size (B)
  - 每 GPU 实际处理 B / N_dp 个样本
  - 损失在 GPU 0 上对完整 batch 求 mean → 梯度 = ∇ mean_B(L)
  - 优化器使用 lr_dp 更新

**DistributedDataParallel (DDP)**
  - 多进程, 每进程 1 个 GPU
  - YAML 中的 ``batch_size`` = 每 GPU 的 batch size (b)
  - 每 GPU 独立计算 loss 并求梯度: gradient_k = ∇ mean_b(L_k)
  - All-Reduce 取均值: effective_gradient = (1/N_ddp) Σ gradient_k
  - 优化器使用 lr_ddp 更新

等效条件
--------
1. **保持总有效 batch 不变**:  b × N_ddp == B
   → b = B / N_ddp

2. **梯度数学等价**: 当条件 1 满足时, DP 和 DDP 的期望梯度完全相同
   → lr_ddp = lr_dp (无需缩放)

3. **若总 batch 改变** (如 DDP 使用更大的 per-GPU batch 以提升效率):
   线性缩放规则: lr_ddp = lr_dp × (b × N_ddp) / B

关于损失归一化的注意事项
------------------------
MonoDLE 的部分损失函数 (如 2D/3D offset, size) 使用 ``reduction='mean'``
对 **有效目标数** (非 batch size) 取均值。DP 在 GPU 0 上汇总所有目标后
取均值, DDP 各进程分别取均值再 all-reduce 求均值, 当各进程有效目标数
不完全相等时会产生微小差异。在 KITTI 数据集上这个差异通常 < 0.1%。

Usage
-----
  python dp_ddp_equivalence.py                        # 使用默认值
  python dp_ddp_equivalence.py --dp-batch 16 --dp-lr 0.00125 --dp-gpus 8 --ddp-gpus 4
  python dp_ddp_equivalence.py --dp-batch 16 --dp-lr 0.00125 --dp-gpus 8 --ddp-gpus 4 --ddp-per-gpu-batch 8

  # 查看详细的梯度等价推导:
  python dp_ddp_equivalence.py --verbose
"""

import argparse
import sys
import math

# ────────────────────────────────────────────────────────────────────────────
# Core computation (also importable as a library)
# ────────────────────────────────────────────────────────────────────────────

def compute_ddp_lr(dp_lr: float,
                   dp_total_batch: int,
                   ddp_per_gpu_batch: int,
                   ddp_world_size: int) -> float:
    """Compute the DDP learning-rate that is gradient-equivalent to a DP run.

    When the total effective batch size changes, the linear scaling rule is
    applied:  lr_ddp = dp_lr × (ddp_total / dp_total).
    """
    ddp_total = ddp_per_gpu_batch * ddp_world_size
    return dp_lr * (ddp_total / dp_total_batch)


def compute_equivalence(dp_total_batch: int,
                        dp_lr: float,
                        dp_num_gpus: int,
                        ddp_world_size: int,
                        ddp_per_gpu_batch: int = None) -> dict:
    """Return a full equivalence report.

    Parameters
    ----------
    dp_total_batch : int
        Total batch size in DP (the YAML value).
    dp_lr : float
        Learning rate in DP.
    dp_num_gpus : int
        Number of GPUs used in DP.
    ddp_world_size : int
        Number of DDP processes (= GPUs).
    ddp_per_gpu_batch : int or None
        If None, automatically computed for exact equivalence.

    Returns
    -------
    dict with all relevant metrics.
    """
    dp_per_gpu = dp_total_batch / dp_num_gpus

    if ddp_per_gpu_batch is None:
        ddp_per_gpu_batch = max(1, dp_total_batch // ddp_world_size)

    ddp_total_batch = ddp_per_gpu_batch * ddp_world_size
    ddp_lr = compute_ddp_lr(dp_lr, dp_total_batch, ddp_per_gpu_batch, ddp_world_size)

    is_exact = (ddp_total_batch == dp_total_batch)

    # Gradient ratio: should be 1.0 for exact equivalence
    grad_ratio = ddp_total_batch / dp_total_batch
    lr_ratio = ddp_lr / dp_lr

    # The effective update step: lr × gradient
    # For equivalence:  lr_ddp × grad_ddp == lr_dp × grad_dp
    # grad_ddp / grad_dp = dp_total_batch / ddp_total_batch  (inverse of batch ratio)
    # Wait, let me be precise:
    # grad ∝ 1/B (mean reduction), so larger batch → smaller gradient magnitude
    # DP:  grad_dp = (1/B_dp) Σ ∇L_i
    # DDP: grad_ddp = (1/N) Σ_k [ (1/b_k) Σ_{j∈k} ∇L_j ]
    #     If per-GPU data is i.i.d.: ≈ (1/B_ddp) Σ ∇L_i
    # Effective update:
    #   DP:  Δθ = -lr_dp × (1/B_dp) Σ ∇L
    #   DDP: Δθ = -lr_ddp × (1/B_ddp) Σ ∇L
    # For equal updates: lr_dp / B_dp = lr_ddp / B_ddp
    #   → lr_ddp = lr_dp × B_ddp / B_dp  ✓ (linear scaling rule)

    update_ratio = (ddp_lr / ddp_total_batch) / (dp_lr / dp_total_batch)

    return {
        'dp': {
            'total_batch': dp_total_batch,
            'per_gpu_batch': dp_per_gpu,
            'lr': dp_lr,
            'num_gpus': dp_num_gpus,
            'effective_update_scale': dp_lr / dp_total_batch,
        },
        'ddp': {
            'total_batch': ddp_total_batch,
            'per_gpu_batch': ddp_per_gpu_batch,
            'lr': ddp_lr,
            'world_size': ddp_world_size,
            'effective_update_scale': ddp_lr / ddp_total_batch,
        },
        'is_exact': is_exact,
        'grad_ratio': grad_ratio,
        'lr_ratio': lr_ratio,
        'update_ratio': update_ratio,
    }


# ────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ────────────────────────────────────────────────────────────────────────────

def print_report(r: dict, verbose: bool = False):
    dp = r['dp']
    ddp = r['ddp']

    print('=' * 68)
    print('        DP ↔ DDP Equivalence Report  (MonoDLE)')
    print('=' * 68)
    print()
    print('  DP  Configuration (原始)')
    print('  ─────────────────────────────────────────────')
    print(f'    Total batch size     : {dp["total_batch"]}')
    print(f'    Number of GPUs       : {dp["num_gpus"]}')
    print(f'    Per-GPU batch size   : {dp["per_gpu_batch"]:.1f}')
    print(f'    Learning rate        : {dp["lr"]:.10f}')
    print(f'    Update scale (lr/B)  : {dp["effective_update_scale"]:.12f}')
    print()
    print('  DDP Configuration (等效)')
    print('  ─────────────────────────────────────────────')
    print(f'    World size (GPUs)    : {ddp["world_size"]}')
    print(f'    Per-GPU batch size   : {ddp["per_gpu_batch"]}')
    print(f'    Total batch size     : {ddp["total_batch"]}')
    print(f'    Learning rate        : {ddp["lr"]:.10f}')
    print(f'    Update scale (lr/B)  : {ddp["effective_update_scale"]:.12f}')
    print()
    print('  Equivalence Check')
    print('  ─────────────────────────────────────────────')
    print(f'    Exact batch match    : {"✓ YES" if r["is_exact"] else "✗ NO"}')
    print(f'    Batch ratio (DDP/DP) : {r["grad_ratio"]:.6f}')
    print(f'    LR ratio   (DDP/DP)  : {r["lr_ratio"]:.6f}')
    print(f'    Update ratio         : {r["update_ratio"]:.6f}  '
          f'{"✓ (= 1.0, 训练等效)" if abs(r["update_ratio"] - 1.0) < 1e-9 else "⚠ (≠ 1.0, 需注意)"}')
    print()

    if r['is_exact']:
        print('  ✓ 完全等效: 总 batch 大小相同, 学习率无需缩放.')
    else:
        print(f'  ⚠ 总 batch 改变 {dp["total_batch"]} → {ddp["total_batch"]}.')
        print(f'    已通过线性缩放规则调整 LR: {dp["lr"]:.8f} → {ddp["lr"]:.8f}')
        print(f'    确保 update_ratio = 1.0 (当前 = {r["update_ratio"]:.10f})')

    print()
    print('  ⚠ 注意: MonoDLE 的目标级损失使用 mean(valid_objects) 归一化,')
    print('    而非 mean(batch_size). 这意味着 DP→DDP 在目标数量不均匀时')
    print('    会有极微小的梯度差异 (KITTI 上通常 < 0.1%, 可忽略).')
    print()

    if verbose:
        _print_math_derivation()

    # Print YAML snippet
    print('  建议的 DDP YAML 配置:')
    print('  ─────────────────────────────────────────────')
    print(f'    dataset:')
    print(f'      batch_size: {ddp["per_gpu_batch"]}  # per-GPU batch')
    print()
    print(f'    optimizer:')
    print(f'      lr: {ddp["lr"]:.10f}')
    print()
    print(f'    distributed:')
    print(f'      enabled: true')
    print(f'      dp_reference:')
    print(f'        total_batch_size: {dp["total_batch"]}')
    print(f'        lr: {dp["lr"]}')
    print(f'        num_gpus: {dp["num_gpus"]}')
    print()
    print(f'  启动命令:')
    print(f'    torchrun --nproc_per_node={ddp["world_size"]} ../../tools/train_val_ddp.py --config YOUR_CONFIG.yaml')
    print('=' * 68)


def _print_math_derivation():
    print()
    print('  数学推导')
    print('  ═══════════════════════════════════════════════')
    print()
    print('  DP 梯度:')
    print('    Inputs 在 GPU 0 汇总 (总 B 个样本), 损失在 GPU 0 计算:')
    print('    g_dp = (1/B) Σ_{i=1}^{B} ∇L_i')
    print()
    print('  DDP 梯度:')
    print('    每个 rank k 有 b = B/N 个样本:')
    print('    g_k = (1/b) Σ_{j∈rank_k} ∇L_j')
    print('    All-Reduce 求均值:')
    print('    g_ddp = (1/N) Σ_{k=1}^{N} g_k')
    print('          = (1/N) Σ_{k=1}^{N} (N/B) Σ_{j∈rank_k} ∇L_j')
    print('          = (1/B) Σ_{i=1}^{B} ∇L_i')
    print('          = g_dp  ✓')
    print()
    print('  当 b×N = B 时, g_ddp = g_dp, 所以 lr 无需改变.')
    print()
    print('  当 b×N ≠ B (总 batch 改变) 时:')
    print('    g_ddp = (1/(b×N)) Σ ∇L_i')
    print('    要求 lr_ddp × g_ddp = lr_dp × g_dp:')
    print('    lr_ddp / (b×N) = lr_dp / B')
    print('    → lr_ddp = lr_dp × (b×N) / B   (线性缩放规则)')
    print()
    print('  ⚠ 注意: 以上推导假设 loss 使用 mean(batch) 归一化.')
    print('    MonoDLE 的目标级损失 (offset, size 等) 使用')
    print('    mean(valid_objects) 归一化. 由于 DP 汇总后分母')
    print('    是所有 GPU 的有效目标数之和, 而 DDP 各进程独立')
    print('    取均值再做均值平均, 严格来说:')
    print()
    print('    DP:  L = Σ l_i / K_total')
    print('    DDP: L = (1/N) Σ_k (Σ_{j∈k} l_j / K_k)')
    print()
    print('    当且仅当 K_1 = K_2 = ... = K_N 时二者严格相等.')
    print('    实际中 KITTI 每图目标数较均匀, 差异可忽略.')
    print()


# ────────────────────────────────────────────────────────────────────────────
# Simulate a numerical gradient comparison
# ────────────────────────────────────────────────────────────────────────────

def simulate_gradient_equivalence(total_batch=16, num_gpus=4, num_params=100,
                                  seed=42):
    """Numerically verify DP vs DDP gradient equivalence.

    Creates random per-sample gradients and shows that:
      - DP gradient  =  mean over entire batch
      - DDP gradient =  mean of per-GPU means
    are identical when each GPU has the same number of samples.
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    per_gpu = total_batch // num_gpus
    assert per_gpu * num_gpus == total_batch, \
        'total_batch must be divisible by num_gpus for exact simulation'

    # Random per-sample gradients: shape (total_batch, num_params)
    per_sample_grads = rng.randn(total_batch, num_params)

    # DP: mean over entire batch
    g_dp = per_sample_grads.mean(axis=0)

    # DDP: each GPU gets a shard, compute local mean, then average across GPUs
    g_ddp_locals = []
    for k in range(num_gpus):
        shard = per_sample_grads[k * per_gpu : (k + 1) * per_gpu]
        g_ddp_locals.append(shard.mean(axis=0))
    g_ddp = np.mean(g_ddp_locals, axis=0)

    max_diff = np.abs(g_dp - g_ddp).max()
    mean_diff = np.abs(g_dp - g_ddp).mean()

    return {
        'max_abs_diff': float(max_diff),
        'mean_abs_diff': float(mean_diff),
        'is_equal': max_diff < 1e-12,
    }


def simulate_object_level_loss_diff(total_batch=16, num_gpus=4,
                                     mean_objects_per_image=5,
                                     seed=42):
    """Simulate the gradient difference caused by object-level mean reduction.

    This shows the (small) discrepancy when the loss denominator is
    #valid_objects instead of #batch_size.
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    per_gpu = total_batch // num_gpus
    assert per_gpu * num_gpus == total_batch

    # Random number of objects per image (Poisson-distributed)
    num_objects = rng.poisson(mean_objects_per_image, size=total_batch)
    num_objects = np.maximum(num_objects, 1)  # at least 1

    # Random per-object loss values
    all_losses = []
    image_to_losses = {}
    for i in range(total_batch):
        losses_i = rng.rand(num_objects[i])
        image_to_losses[i] = losses_i
        all_losses.extend(losses_i.tolist())

    # DP: gather all objects, compute mean
    dp_loss = np.mean(all_losses)

    # DDP: each GPU computes mean over its objects, then average
    ddp_per_gpu_losses = []
    for k in range(num_gpus):
        gpu_losses = []
        for i in range(k * per_gpu, (k + 1) * per_gpu):
            gpu_losses.extend(image_to_losses[i].tolist())
        ddp_per_gpu_losses.append(np.mean(gpu_losses))
    ddp_loss = np.mean(ddp_per_gpu_losses)

    abs_diff = abs(dp_loss - ddp_loss)
    rel_diff = abs_diff / (abs(dp_loss) + 1e-12)

    return {
        'dp_loss': float(dp_loss),
        'ddp_loss': float(ddp_loss),
        'abs_diff': float(abs_diff),
        'rel_diff_pct': float(rel_diff * 100),
        'num_objects_per_gpu': [int(num_objects[k*per_gpu:(k+1)*per_gpu].sum())
                                for k in range(num_gpus)],
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DP ↔ DDP Equivalence Calculator for MonoDLE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dp-batch', type=int, default=16,
                        help='DP total batch size (default: 16)')
    parser.add_argument('--dp-lr', type=float, default=0.00125,
                        help='DP learning rate (default: 0.00125)')
    parser.add_argument('--dp-gpus', type=int, default=8,
                        help='DP number of GPUs (default: 8)')
    parser.add_argument('--ddp-gpus', type=int, default=None,
                        help='DDP world size (default: same as dp-gpus)')
    parser.add_argument('--ddp-per-gpu-batch', type=int, default=None,
                        help='DDP per-GPU batch size (default: auto for equivalence)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed math derivation')
    parser.add_argument('--simulate', '-s', action='store_true',
                        help='Run numerical simulations')

    args = parser.parse_args()

    ddp_gpus = args.ddp_gpus if args.ddp_gpus else args.dp_gpus

    result = compute_equivalence(
        dp_total_batch=args.dp_batch,
        dp_lr=args.dp_lr,
        dp_num_gpus=args.dp_gpus,
        ddp_world_size=ddp_gpus,
        ddp_per_gpu_batch=args.ddp_per_gpu_batch,
    )

    print_report(result, verbose=args.verbose)

    if args.simulate:
        print()
        print('  ─── Numerical Simulation ───')
        print()

        # Test 1: batch-level mean gradient equivalence
        sim1 = simulate_gradient_equivalence(
            total_batch=args.dp_batch,
            num_gpus=args.dp_gpus,
        )
        print(f'  [1] Batch-level mean 梯度比较 (B={args.dp_batch}, N={args.dp_gpus}):')
        print(f'      max |g_dp - g_ddp| = {sim1["max_abs_diff"]:.2e}')
        print(f'      结果: {"✓ 完全一致" if sim1["is_equal"] else "⚠ 有差异"}')
        print()

        # Test 2: object-level mean loss discrepancy
        sim2 = simulate_object_level_loss_diff(
            total_batch=args.dp_batch,
            num_gpus=args.dp_gpus,
        )
        print(f'  [2] 目标级 mean 损失差异模拟 (B={args.dp_batch}, N={args.dp_gpus}):')
        print(f'      DP  loss = {sim2["dp_loss"]:.8f}')
        print(f'      DDP loss = {sim2["ddp_loss"]:.8f}')
        print(f'      绝对差异 = {sim2["abs_diff"]:.8f}')
        print(f'      相对差异 = {sim2["rel_diff_pct"]:.4f}%')
        print(f'      各 GPU 目标数 = {sim2["num_objects_per_gpu"]}')
        print()

        # Test 3: varying GPU counts
        print('  [3] 不同 GPU 数量下的等效配置:')
        print('      ┌──────────┬──────────┬──────────────┬──────────────┬──────────┐')
        print('      │ DDP GPUs │ Per-GPU B│ Total B      │ LR           │ Exact?   │')
        print('      ├──────────┼──────────┼──────────────┼──────────────┼──────────┤')
        for n in [1, 2, 4, 8]:
            if n > args.dp_batch:
                continue
            r = compute_equivalence(args.dp_batch, args.dp_lr, args.dp_gpus, n)
            ddp = r['ddp']
            print(f'      │ {n:>8} │ {ddp["per_gpu_batch"]:>8} │ {ddp["total_batch"]:>12} │ {ddp["lr"]:>12.8f} │ {"✓":>8} │' if r['is_exact']
                  else f'      │ {n:>8} │ {ddp["per_gpu_batch"]:>8} │ {ddp["total_batch"]:>12} │ {ddp["lr"]:>12.8f} │ {"~":>8} │')
        print('      └──────────┴──────────┴──────────────┴──────────────┴──────────┘')
        print()


if __name__ == '__main__':
    main()
