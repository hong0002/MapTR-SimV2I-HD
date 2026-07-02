#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

NUM_SAMPLES="${NUM_SAMPLES:-20}"
SCORE_THR="${SCORE_THR:-0.3}"
PRED_DIR="outputs/maptr/visualization_predictions"
OUT_DIR="outputs/maptr/visualizations/bev_map_comparison"
REPORT_PATH="outputs/maptr/visualizations/VISUALIZATION_REPORT.md"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maptr}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/maptr-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${XDG_CACHE_HOME}/fontconfig"
mkdir -p "${PRED_DIR}" "${OUT_DIR}"

convert_json_to_pkl() {
  local json_path="$1"
  local pkl_path="$2"
  if [[ -f "${pkl_path}" ]]; then
    return 0
  fi
  if [[ ! -f "${json_path}" ]]; then
    return 1
  fi
  echo "Creating ${pkl_path} from ${json_path}"
  "${PYTHON_BIN}" - "${json_path}" "${pkl_path}" <<'PY'
import json
import pickle
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r") as f:
    obj = json.load(f)
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("wb") as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"wrote {dst}")
PY
}

ensure_prediction() {
  local model_name="$1"
  local pkl_path="$2"
  local json_path="$3"
  local eval_script="$4"
  local checkpoint="$5"
  if [[ -f "${pkl_path}" ]]; then
    echo "Found ${model_name}: ${pkl_path}"
    return 0
  fi
  if convert_json_to_pkl "${json_path}" "${pkl_path}"; then
    return 0
  fi
  cat >&2 <<EOF
Missing prediction for ${model_name}.
Expected pkl: ${pkl_path}
Expected existing eval JSON: ${json_path}

Generate eval results first, then rerun this script:
  bash ${eval_script} ${checkpoint}

Optional raw --out support is available through tools/test.py, but this
visualizer expects the formatted nuscmap_results.json-derived pkl above.
EOF
  return 2
}

ensure_prediction \
  "ego_only" \
  "${PRED_DIR}/ego_only_test_epoch18.pkl" \
  "outputs/maptr/ego_only_r18_20k_b4/eval/results/pts_bbox/nuscmap_results.json" \
  "scripts/eval_maptr_ego_only_20k_b4.sh" \
  "outputs/maptr/ego_only_r18_20k_b4/epoch_18.pth"

ensure_prediction \
  "dynamic_top4" \
  "${PRED_DIR}/dynamic_top4_test_epoch18.pkl" \
  "outputs/maptr/v2_dynamic_rsu_top4_r18_20k/eval/results/pts_bbox/nuscmap_results.json" \
  "scripts/eval_maptr_v2_dynamic_rsu_top4_20k.sh" \
  "outputs/maptr/v2_dynamic_rsu_top4_r18_20k/epoch_18.pth"

ensure_prediction \
  "constant_gate" \
  "${PRED_DIR}/constant_gate_test_epoch18.pkl" \
  "outputs/maptr/gate_constant_v2i_r18_20k_b4/eval/results/pts_bbox/nuscmap_results.json" \
  "scripts/eval_maptr_gate_constant_v2i_20k_b4.sh" \
  "outputs/maptr/gate_constant_v2i_r18_20k_b4/epoch_18.pth"

ensure_prediction \
  "pose_gated_top4" \
  "${PRED_DIR}/pose_gated_top4_test_epoch18.pkl" \
  "outputs/maptr/pose_gated_v2i_r18_20k_b4/eval/results/pts_bbox/nuscmap_results.json" \
  "scripts/eval_maptr_pose_gated_v2i_20k_b4.sh" \
  "outputs/maptr/pose_gated_v2i_r18_20k_b4/epoch_18.pth"

"${PYTHON_BIN}" tools/visualize_simv2i_maptr_predictions.py \
  --pred-pkl "ego_only=${PRED_DIR}/ego_only_test_epoch18.pkl" \
  --pred-pkl "dynamic_top4=${PRED_DIR}/dynamic_top4_test_epoch18.pkl" \
  --pred-pkl "constant_gate=${PRED_DIR}/constant_gate_test_epoch18.pkl" \
  --pred-pkl "pose_gated_top4=${PRED_DIR}/pose_gated_top4_test_epoch18.pkl" \
  --num-samples "${NUM_SAMPLES}" \
  --score-thr "${SCORE_THR}" \
  --output-dir "${OUT_DIR}" \
  --report-path "${REPORT_PATH}" \
  "$@"
