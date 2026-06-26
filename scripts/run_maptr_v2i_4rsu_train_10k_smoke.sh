#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_v2i_4rsu_r18_stronger_10k_smoke.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/v2i_4rsu_r18_stronger_10k_smoke}"
VALIDATION_DIR="${MAPTR_V2I_VALIDATION_DIR:-outputs/maptr/v2i_fusion_precheck}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${MAPTR_GPUS:-4}"
PORT="${PORT:-28714}"
mkdir -p "${LOG_DIR}" "${VALIDATION_DIR}"

"${PYTHON_BIN}" integrations/maptr/tools/validate_simv2i_v2i_views.py \
  --root "${MAPTR_ROOT}" \
  --output-dir "${VALIDATION_DIR}" \
  --view-mode v2i_4rsu
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
