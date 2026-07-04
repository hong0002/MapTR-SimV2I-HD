#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init
maptr_require_simv2i_geo_data

CONFIG="integrations/maptr/configs/simv2i_maptr_pose_gated_v2i_r18_20k_geo_top2_b4.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr_geo/pose_gated_v2i_r18_20k_geo_top2_b4}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${GPUS:-${MAPTR_GPUS:-4}}"
PORT="${PORT:-28912}"
mkdir -p "${LOG_DIR}"

maptr_check_runtime

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS - 1))")"
  export CUDA_VISIBLE_DEVICES
fi

LOG_FILE="${LOG_DIR}/train_$(maptr_timestamp).log"
ROOT_LOG_FILE="${WORK_DIR}/train.log"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  tools/train.py "${CONFIG}" \
  --launcher pytorch \
  --work-dir "${WORK_DIR}" \
  "$@" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
