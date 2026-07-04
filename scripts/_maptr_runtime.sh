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

maptr_simv2i_geo_dataset_dir() {
  if [[ -n "${SIMV2I_HD_GEO_ROOT:-}" ]]; then
    printf '%s\n' "${SIMV2I_HD_GEO_ROOT}"
  else
    printf '%s\n' "data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset"
  fi
}

maptr_simv2i_image_root() {
  if [[ -n "${SIMV2I_HD_IMAGE_ROOT:-}" ]]; then
    printf '%s\n' "${SIMV2I_HD_IMAGE_ROOT}"
  else
    printf '%s\n' "."
  fi
}

maptr_require_simv2i_geo_data() {
  local dataset_dir
  dataset_dir="$(maptr_simv2i_geo_dataset_dir)"
  local image_root
  image_root="$(maptr_simv2i_image_root)"
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
  if [[ "${missing}" -ne 0 ]]; then
    {
      echo "SimV2I-HD geo subset annotation files not found at: ${dataset_dir}"
      echo
      echo "Set SIMV2I_HD_GEO_ROOT to the generated geo subset annotation directory:"
      echo "  export SIMV2I_HD_GEO_ROOT=/path/to/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset"
      echo
      echo "Expected files:"
      echo "  simv2i_maptr_infos_train.pkl"
      echo "  simv2i_maptr_infos_val.pkl"
      echo "  simv2i_maptr_infos_test.pkl"
      echo "  simv2i_maptr_map_gt_train.json"
      echo "  simv2i_maptr_map_gt_val.json"
      echo "  simv2i_maptr_map_gt_test.json"
      echo
      echo "Generate them with:"
      echo "  python tools/create_geo_subset_split.py --input-root \"\${SIMV2I_HD_ROOT}\" --output-root \"\${SIMV2I_HD_GEO_ROOT}\""
      echo
      echo "Backward-compatible annotation fallback:"
      echo "  data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k_geo_subset"
    } >&2
    return 2
  fi

  if "${PYTHON_BIN}" - "${dataset_dir}" "${image_root}" <<'PY'
import pickle
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
image_root = Path(sys.argv[2])
checked = 0
missing = []

for split in ("train", "val", "test"):
    ann_file = dataset_dir / f"simv2i_maptr_infos_{split}.pkl"
    with ann_file.open("rb") as f:
        data = pickle.load(f)
    infos = data.get("infos", data) if isinstance(data, dict) else data
    for info in infos[:2]:
        for group_name in ("cams", "rsu_cams"):
            group = info.get(group_name) or {}
            if not isinstance(group, dict):
                continue
            for cam in group.values():
                if not isinstance(cam, dict):
                    continue
                data_path = cam.get("data_path")
                if not data_path:
                    continue
                path = Path(data_path)
                resolved = path if path.is_absolute() else image_root / path
                checked += 1
                if not resolved.is_file():
                    missing.append(
                        (split, info.get("token", "<unknown>"), data_path, str(resolved))
                    )
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
        if len(missing) >= 10:
            break

if missing:
    print("SimV2I-HD geo image path check failed.", file=sys.stderr)
    print(f"Annotation root: {dataset_dir}", file=sys.stderr)
    print(f"Image root: {image_root}", file=sys.stderr)
    for split, token, data_path, resolved in missing:
        print(
            f"  [{split}] {token}: {data_path} -> {resolved}",
            file=sys.stderr,
        )
    sys.exit(2)

print(
    f"SimV2I-HD geo image path check passed: {checked} camera images under {image_root}",
    file=sys.stderr,
)
PY
  then
    return 0
  fi

  {
    echo
    echo "Geo subset annotations and image paths are intentionally separate."
    echo "The geo directory should contain only PKL/JSON annotation files."
    echo "Image paths stored in the PKLs are resolved from SIMV2I_HD_IMAGE_ROOT,"
    echo "or from the repository root '.' when SIMV2I_HD_IMAGE_ROOT is unset."
    echo
    echo "For PKL paths such as data/raw/..., either run from MAPTR_ROOT with"
    echo "data/raw available there, or set:"
    echo "  export SIMV2I_HD_IMAGE_ROOT=/path/whose/child/is/data/raw"
    echo
    echo "Optional compatibility workaround: create a symlink so the repo-relative"
    echo "data/raw/... paths exist under MAPTR_ROOT."
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
