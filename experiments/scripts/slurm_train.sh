#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

PARTITION="spring_scheduler"
JOB_NAME=$1
GPUS=$2
CONFIG=$3

if [[ ! -f "${CONFIG}" ]]; then
    if [[ -f "${ROOT_DIR}/${CONFIG}" ]]; then
        CONFIG="${ROOT_DIR}/${CONFIG}"
    elif [[ -f "${ROOT_DIR}/experiments/configs/monodle/${CONFIG}" ]]; then
        CONFIG="${ROOT_DIR}/experiments/configs/monodle/${CONFIG}"
    elif [[ -f "${ROOT_DIR}/experiments/configs/ablation/${CONFIG}" ]]; then
        CONFIG="${ROOT_DIR}/experiments/configs/ablation/${CONFIG}"
    fi
fi

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=5

PROT=18888

srun -p ${PARTITION} \
    --comment=spring-submit \
    --job-name=${JOB_NAME} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
python "${ROOT_DIR}/tools/train_val.py" \
    --config ${CONFIG} \
    # --e \
