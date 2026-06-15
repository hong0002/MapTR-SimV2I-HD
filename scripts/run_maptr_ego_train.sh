#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_ego_r18_stronger.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/ego_r18_stronger}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${MAPTR_GPUS:-1}"
PORT="${PORT:-28509}"
mkdir -p "${LOG_DIR}"

maptr_validate_data
maptr_check_runtime

LOG_FILE="${LOG_DIR}/train_$(maptr_timestamp).log"
if [[ "${GPUS}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  "${PYTHON_BIN}" tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
fi
