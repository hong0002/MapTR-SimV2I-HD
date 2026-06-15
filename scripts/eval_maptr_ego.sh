#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="${MAPTR_EVAL_CONFIG:-integrations/maptr/configs/simv2i_maptr_ego_r18_stronger.py}"
CHECKPOINT="${1:-outputs/maptr/ego_r18_stronger/latest.pth}"
WORK_DIR="${MAPTR_EVAL_WORK_DIR:-outputs/maptr/ego_r18_stronger/eval}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${MAPTR_EVAL_GPUS:-1}"
PORT="${PORT:-29503}"
mkdir -p "${LOG_DIR}"

maptr_validate_data

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

maptr_check_runtime

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_FILE="${LOG_DIR}/eval_$(maptr_timestamp).log"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  tools/test.py "${CONFIG}" "${CHECKPOINT}" \
  --launcher pytorch \
  --eval chamfer \
  --eval-options "jsonfile_prefix=${WORK_DIR}/results" \
  "${@:2}" 2>&1 | tee "${LOG_FILE}"
