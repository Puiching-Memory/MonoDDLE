#!/usr/bin/env bash
# ============================================================================
# DDP Training Launch Script for MonoDLE
# ============================================================================
#
# Usage:
#   ./train_ddp.sh experiments/configs/monodle/kitti_da3.yaml
#   ./train_ddp.sh kitti_da3.yaml 4         # use 4 GPUs
#
# The script uses torchrun (PyTorch >= 1.10) to launch one process per GPU.
# Each process initialises NCCL via environment variables set by torchrun.
# ============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

CONFIG=${1:?'Usage: train_ddp.sh <config.yaml> [NUM_GPUS]'}
NUM_GPUS=${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}

if [[ ! -f "${CONFIG}" ]]; then
    if [[ -f "${ROOT_DIR}/${CONFIG}" ]]; then
        CONFIG="${ROOT_DIR}/${CONFIG}"
    elif [[ -f "${ROOT_DIR}/experiments/configs/monodle/${CONFIG}" ]]; then
        CONFIG="${ROOT_DIR}/experiments/configs/monodle/${CONFIG}"
    elif [[ -f "${ROOT_DIR}/experiments/configs/ablation/${CONFIG}" ]]; then
        CONFIG="${ROOT_DIR}/experiments/configs/ablation/${CONFIG}"
    fi
fi

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
    "${ROOT_DIR}/tools/train_val_ddp.py" \
    --config "${CONFIG}"
