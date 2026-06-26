#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="${MAPTR_EVAL_CONFIG:-integrations/maptr/configs/simv2i_maptr_ego_r18_stronger_10k.py}"
CHECKPOINT="${1:-outputs/maptr/ego_r18_stronger_10k/latest.pth}"
WORK_DIR="${MAPTR_EVAL_WORK_DIR:-outputs/maptr/ego_r18_stronger_10k/eval}"
VALIDATION_DIR="${MAPTR_10K_VALIDATION_DIR:-outputs/maptr/ego_r18_stronger_10k_validation}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${MAPTR_EVAL_GPUS:-${MAPTR_GPUS:-4}}"
PORT="${PORT:-29610}"
mkdir -p "${LOG_DIR}" "${VALIDATION_DIR}"

"${PYTHON_BIN}" integrations/maptr/tools/validate_simv2i_10k_image_paths.py \
  --root "${MAPTR_ROOT}" \
  --output-dir "${VALIDATION_DIR}"

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
  "${@:2}" 2>&1 | tee "${LOG_FILE}"
