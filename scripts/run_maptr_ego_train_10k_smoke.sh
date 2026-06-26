#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_ego_r18_stronger_10k_smoke.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/ego_r18_stronger_10k_smoke}"
VALIDATION_DIR="${MAPTR_10K_VALIDATION_DIR:-outputs/maptr/ego_r18_stronger_10k_validation}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${MAPTR_GPUS:-4}"
PORT="${PORT:-28610}"
mkdir -p "${LOG_DIR}" "${VALIDATION_DIR}"

"${PYTHON_BIN}" integrations/maptr/tools/validate_simv2i_10k_image_paths.py \
  --root "${MAPTR_ROOT}" \
  --output-dir "${VALIDATION_DIR}"
maptr_check_runtime

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${GPUS}" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

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
  "${PYTHON_BIN}" tools/train.py "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
fi
