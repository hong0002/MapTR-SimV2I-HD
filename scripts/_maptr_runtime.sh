#!/usr/bin/env bash

maptr_init() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  MAPTR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  PYTHON_BIN="${MAPTR_PYTHON:-python}"
  export PYTHONPATH="${MAPTR_ROOT}:${MAPTR_ROOT}/mmdetection3d:${PYTHONPATH:-}"
  cd "${MAPTR_ROOT}"
}

maptr_validate_data() {
  "${PYTHON_BIN}" \
    integrations/maptr/tools/validate_simv2i_maptr_data.py \
    --root "${MAPTR_ROOT}" \
    --require-data
}

maptr_check_runtime() {
  "${PYTHON_BIN}" -c \
    "import mmcv, mmdet, mmdet3d, torch; print('runtime:', torch.__version__, mmcv.__version__, mmdet.__version__, mmdet3d.__version__)"
}

maptr_timestamp() {
  date '+%Y%m%d_%H%M%S'
}
