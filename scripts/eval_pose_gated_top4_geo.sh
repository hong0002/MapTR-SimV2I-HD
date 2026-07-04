#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init
maptr_require_simv2i_geo_data

CONFIG="integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_geo_top4_b4.py"
CHECKPOINT="${1:?Usage: bash scripts/eval_pose_gated_top4_geo.sh path/to/checkpoint.pth}"
WORK_DIR="${MAPTR_EVAL_WORK_DIR:-outputs/maptr_geo/pose_gated_v2i_r18_20k_geo_top4_b4}"
RESULT_DIR="${WORK_DIR}/eval/results"
LOG_DIR="${WORK_DIR}/eval/logs"
GPUS="${GPUS:-${MAPTR_GPUS:-4}}"
PORT="${PORT:-29913}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

maptr_check_runtime

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS - 1))")"
  export CUDA_VISIBLE_DEVICES
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
