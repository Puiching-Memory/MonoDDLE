#!/usr/bin/env bash
# ============================================================================
# DDP Training Launch Script for MonoDLE
# ============================================================================
#
# Usage:
#   ./train_ddp.sh kitti_da3_ddp.yaml          # auto-detect GPUs
#   ./train_ddp.sh kitti_da3_ddp.yaml  4       # use 4 GPUs
#
# The script uses torchrun (PyTorch >= 1.10) to launch one process per GPU.
# Each process initialises NCCL via environment variables set by torchrun.
# ============================================================================

set -euo pipefail

CONFIG=${1:?'Usage: train_ddp.sh <config.yaml> [NUM_GPUS]'}
NUM_GPUS=${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}

if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi

echo "============================================"
echo " MonoDLE DDP Training"
echo " Config   : ${CONFIG}"
echo " GPUs     : ${NUM_GPUS}"
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    ../../tools/train_val_ddp.py \
    --config "${CONFIG}"
