#!/usr/bin/env bash

maptr_init() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  MAPTR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  PYTHON_BIN="${MAPTR_PYTHON:-python}"
  export PYTHONPATH="${MAPTR_ROOT}:${MAPTR_ROOT}/mmdetection3d:${PYTHONPATH:-}"
  cd "${MAPTR_ROOT}"
}

maptr_simv2i_dataset_dir() {
  if [[ -n "${SIMV2I_HD_ROOT:-}" ]]; then
    printf '%s\n' "${SIMV2I_HD_ROOT}"
  else
    printf '%s\n' "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k"
  fi
}

maptr_require_simv2i_data() {
  local dataset_dir
  dataset_dir="$(maptr_simv2i_dataset_dir)"
  local missing=0
  for split in train val test; do
    if [[ ! -f "${dataset_dir}/simv2i_maptr_infos_${split}.pkl" ]]; then
      missing=1
      break
    fi
    if [[ ! -f "${dataset_dir}/simv2i_maptr_map_gt_${split}.json" ]]; then
      missing=1
      break
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    return 0
  fi

  {
    echo "SimV2I-HD MapTR dataset not found at: ${dataset_dir}"
    echo
    echo "Set SIMV2I_HD_ROOT to the prepared external dataset directory:"
    echo "  export SIMV2I_HD_ROOT=/path/to/simv2i_hd_benchmark_v2_dynamic_rsu_20k"
    echo
    echo "Expected files:"
    echo "  simv2i_maptr_infos_train.pkl"
    echo "  simv2i_maptr_infos_val.pkl"
    echo "  simv2i_maptr_infos_test.pkl"
    echo "  simv2i_maptr_map_gt_train.json"
    echo "  simv2i_maptr_map_gt_val.json"
    echo "  simv2i_maptr_map_gt_test.json"
    echo
    echo "Backward-compatible fallback is also supported with a symlink at:"
    echo "  data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k"
  } >&2
  return 2
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
