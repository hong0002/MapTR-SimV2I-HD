#!/usr/bin/env bash
set -euo pipefail

# Run after: conda activate maptr_simv2i
# Resume only from this No-pose Gate work_dir, for example:
#   GPUS=4 bash scripts/run_maptr_gate_nopose_v2i_train_20k_b4_full.sh \
#     --resume-from outputs/maptr/gate_nopose_v2i_r18_20k_b4/epoch_XX.pth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

CONFIG="integrations/maptr/configs/simv2i_maptr_gate_nopose_v2i_r18_20k_b4.py"
WORK_DIR="${MAPTR_TRAIN_WORK_DIR:-outputs/maptr/gate_nopose_v2i_r18_20k_b4}"
LOG_DIR="${WORK_DIR}/logs"
GPUS="${GPUS:-4}"
PORT="${PORT:-28833}"
mkdir -p "${LOG_DIR}"

maptr_check_runtime

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS - 1))")"
  export CUDA_VISIBLE_DEVICES
fi

echo "Using GPUS=${GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

LOG_FILE="${LOG_DIR}/train_$(maptr_timestamp).log"
ROOT_LOG_FILE="${WORK_DIR}/train.log"
"${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  tools/train.py "${CONFIG}" \
  --launcher pytorch \
  --work-dir "${WORK_DIR}" \
  "$@" 2>&1 | tee "${LOG_FILE}" "${ROOT_LOG_FILE}"
