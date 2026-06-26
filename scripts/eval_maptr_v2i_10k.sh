#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="${1:?Usage: bash scripts/eval_maptr_v2i_10k.sh CONFIG CHECKPOINT [extra tools/test.py args...]}"
CHECKPOINT="${2:?Usage: bash scripts/eval_maptr_v2i_10k.sh CONFIG CHECKPOINT [extra tools/test.py args...]}"
CONFIG_STEM="$(basename "${CONFIG}" .py)"
WORK_DIR="${MAPTR_EVAL_WORK_DIR:-outputs/maptr/${CONFIG_STEM}/eval}"
VALIDATION_DIR="${MAPTR_V2I_VALIDATION_DIR:-outputs/maptr/v2i_fusion_precheck}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${MAPTR_EVAL_GPUS:-${MAPTR_GPUS:-4}}"
PORT="${PORT:-29720}"
mkdir -p "${LOG_DIR}" "${VALIDATION_DIR}"

"${PYTHON_BIN}" integrations/maptr/tools/validate_simv2i_v2i_views.py \
  --root "${MAPTR_ROOT}" \
  --output-dir "${VALIDATION_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

maptr_check_runtime

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${GPUS}" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

LOG_FILE="${LOG_DIR}/eval_$(maptr_timestamp).log"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  tools/test.py "${CONFIG}" "${CHECKPOINT}" \
  --launcher pytorch \
  --eval chamfer \
  --eval-options "jsonfile_prefix=${WORK_DIR}/results" \
  "${@:3}" 2>&1 | tee "${LOG_FILE}"
