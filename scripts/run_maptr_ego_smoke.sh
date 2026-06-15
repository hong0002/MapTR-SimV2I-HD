#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_ego_r18_compact.py"
WORK_DIR="${MAPTR_SMOKE_WORK_DIR:-outputs/maptr/ego_smoke}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

maptr_validate_data
maptr_check_runtime

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_FILE="${LOG_DIR}/train_$(maptr_timestamp).log"
"${PYTHON_BIN}" tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
