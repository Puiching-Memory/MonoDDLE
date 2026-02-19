#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

CONFIG=${1:-experiments/configs/monodle/kitti_no_da3.yaml}
if [[ ! -f "${CONFIG}" ]]; then
	if [[ -f "${ROOT_DIR}/${CONFIG}" ]]; then
		CONFIG="${ROOT_DIR}/${CONFIG}"
	elif [[ -f "${ROOT_DIR}/experiments/configs/monodle/${CONFIG}" ]]; then
		CONFIG="${ROOT_DIR}/experiments/configs/monodle/${CONFIG}"
	elif [[ -f "${ROOT_DIR}/experiments/configs/ablation/${CONFIG}" ]]; then
		CONFIG="${ROOT_DIR}/experiments/configs/ablation/${CONFIG}"
	fi
fi

python "${ROOT_DIR}/tools/train_val.py" --config "${CONFIG}"