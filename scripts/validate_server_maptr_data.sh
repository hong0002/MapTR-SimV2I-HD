#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPTR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MAPTR_PYTHON:-python}"

cd "${MAPTR_ROOT}"
ARGS=("$@")
if [[ -n "${SIMV2I_HD_ROOT:-}" ]]; then
  HAS_DATASET_DIR=0
  for arg in "${ARGS[@]}"; do
    if [[ "${arg}" == "--dataset-dir" || "${arg}" == --dataset-dir=* ]]; then
      HAS_DATASET_DIR=1
      break
    fi
  done
  if [[ "${HAS_DATASET_DIR}" -eq 0 ]]; then
    ARGS=(--dataset-dir "${SIMV2I_HD_ROOT}" "${ARGS[@]}")
  fi
fi
exec "${PYTHON_BIN}" \
  integrations/maptr/tools/validate_simv2i_maptr_data.py \
  --root "${MAPTR_ROOT}" \
  "${ARGS[@]}"
