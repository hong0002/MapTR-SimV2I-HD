#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_v2_dynamic_rsu_top4_r18_20k.py"
CHECKPOINT="${1:?Usage: bash scripts/eval_maptr_v2_dynamic_rsu_top4_20k.sh path/to/checkpoint.pth}"
WORK_DIR="${MAPTR_EVAL_WORK_DIR:-outputs/maptr/v2_dynamic_rsu_top4_r18_20k}"
RESULT_DIR="${WORK_DIR}/eval/results"
LOG_DIR="${WORK_DIR}/eval/logs"
GPUS="${MAPTR_EVAL_GPUS:-${GPUS:-${MAPTR_GPUS:-1}}}"
PORT="${PORT:-29754}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

maptr_check_runtime

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${GPUS}" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

LOG_FILE="${LOG_DIR}/test_eval_$(basename "${CHECKPOINT}" .pth)_$(maptr_timestamp).log"
ROOT_LOG_FILE="${WORK_DIR}/test_eval_$(basename "${CHECKPOINT}" .pth).log"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  tools/test.py "${CONFIG}" "${CHECKPOINT}" \
  --launcher pytorch \
  --eval chamfer \
  --eval-options "jsonfile_prefix=${RESULT_DIR}" \
  "${@:2}" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
