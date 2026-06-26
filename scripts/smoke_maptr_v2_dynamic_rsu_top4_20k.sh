#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_v2_dynamic_rsu_top4_r18_20k_smoke.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/v2_dynamic_rsu_top4_r18_20k_smoke}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${GPUS:-${MAPTR_GPUS:-1}}"
PORT="${PORT:-28754}"
mkdir -p "${LOG_DIR}"

maptr_check_runtime

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${GPUS}" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

LOG_FILE="${LOG_DIR}/smoke_train_$(maptr_timestamp).log"
ROOT_LOG_FILE="${WORK_DIR}/smoke_train.log"
if [[ "${GPUS}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
else
  "${PYTHON_BIN}" tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
fi
