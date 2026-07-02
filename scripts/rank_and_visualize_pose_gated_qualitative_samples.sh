#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

TOP_K="${TOP_K:-30}"
SCORE_THR="${SCORE_THR:-0.4}"
DIST_THR="${DIST_THR:-1.5}"
OUT_DIR="outputs/maptr/visualizations/qualitative_ranking"
FIG_DIR="${OUT_DIR}/figures"
REPORT_PATH="${OUT_DIR}/QUALITATIVE_RANKING_REPORT.md"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maptr}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/maptr-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${XDG_CACHE_HOME}/fontconfig"
mkdir -p "${OUT_DIR}" "${FIG_DIR}"

"${PYTHON_BIN}" tools/rank_qualitative_pose_gated_samples.py \
  --top-k "${TOP_K}" \
  --score-thr "${SCORE_THR}" \
  --dist-thr "${DIST_THR}" \
  --out-dir "${OUT_DIR}" \
  "$@"

INDICES="$("${PYTHON_BIN}" - "${OUT_DIR}/top_indices_by_category.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r") as f:
    data = json.load(f)

limits = {
    "clean_qualitative_candidates": 10,
    "ped_crossing_recovery": 5,
    "divider_recovery": 5,
}
indices = []
for category, limit in limits.items():
    for index in data.get(category, [])[:limit]:
        if index not in indices:
            indices.append(index)
print(" ".join(str(index) for index in indices))
PY
)"

if [[ -n "${INDICES}" ]]; then
  echo "Generating overlay paper figures for sample indices: ${INDICES}"
  "${PYTHON_BIN}" tools/visualize_simv2i_maptr_predictions.py \
    --paper-4panel \
    --overlay-gt \
    --score-thr "${SCORE_THR}" \
    --sample-indices ${INDICES} \
    --filename-prefix "ranked_" \
    --output-dir "${FIG_DIR}" \
    --report-path "${OUT_DIR}/FIGURE_VISUALIZATION_REPORT.md"
else
  echo "No ranked sample indices found; skipping figure generation."
fi

"${PYTHON_BIN}" - "${REPORT_PATH}" "${FIG_DIR}" <<'PY'
import sys
from pathlib import Path

report = Path(sys.argv[1])
fig_dir = Path(sys.argv[2])
figures = sorted(fig_dir.glob("*.png"))

with report.open("a") as f:
    f.write("\n## Generated PNG Files\n\n")
    if not figures:
        f.write("- None\n")
    else:
        for path in figures:
            f.write(f"- {path}\n")
PY

echo "Qualitative ranking report: ${REPORT_PATH}"
echo "Figures directory: ${FIG_DIR}"
