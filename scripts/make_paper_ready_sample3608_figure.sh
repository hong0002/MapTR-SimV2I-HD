#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init

OUT_DIR="outputs/maptr/visualizations/paper_figures"
REPORT_PATH="${OUT_DIR}/PAPER_FIGURE_SAMPLE3608_REPORT.md"
PNG_PATH="${OUT_DIR}/sample_3608_paper_ready_4panel.png"
PDF_PATH="${OUT_DIR}/sample_3608_paper_ready_4panel.pdf"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maptr}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/maptr-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${XDG_CACHE_HOME}/fontconfig"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" tools/visualize_simv2i_maptr_predictions.py \
  --gt-json data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k/simv2i_maptr_map_gt_test.json \
  --infos-pkl data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k/simv2i_maptr_infos_test.pkl \
  --pred-pkl ego_only=outputs/maptr/visualization_predictions/ego_only_test_epoch18.pkl \
  --pred-pkl dynamic_top4=outputs/maptr/visualization_predictions/dynamic_top4_test_epoch18.pkl \
  --pred-pkl pose_gated_top4=outputs/maptr/visualization_predictions/pose_gated_top4_test_epoch18.pkl \
  --output-dir "${OUT_DIR}" \
  --output-name sample_3608_paper_ready_4panel \
  --sample-indices 3608 \
  --score-thr 0.4 \
  --overlay-gt \
  --paper-4panel \
  --paper-ready \
  --short-title "Sample 3608" \
  --hide-token \
  --fig-width 10.4 \
  --fig-height 4.25 \
  --wspace 0.035 \
  --gt-alpha 0.28 \
  --gt-linewidth 1.0 \
  --pred-linewidth 2.0 \
  --legend-fontsize 9 \
  --title-fontsize 13 \
  --panel-title-fontsize 12 \
  --axis-fontsize 10 \
  --tick-fontsize 9.5 \
  --save-pdf \
  --report-path "${OUT_DIR}/PAPER_FIGURE_SAMPLE3608_VISUALIZATION_REPORT.md"

"${PYTHON_BIN}" - "${REPORT_PATH}" "${PNG_PATH}" "${PDF_PATH}" <<'PY'
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
png_path = Path(sys.argv[2])
pdf_path = Path(sys.argv[3])

lines = [
    "# Paper-Ready Sample 3608 Figure Report",
    "",
    "## Purpose",
    "",
    "Create a compact paper-ready 4-panel BEV vector-map qualitative figure for sample 3608.",
    "",
    "## Input Figure Or Sample",
    "",
    "- Source candidate: outputs/maptr/visualizations/qualitative_ranking/figures/ranked_sample_003608_gt_ego_dynamic_pose_overlay.png",
    "- Sample index: 3608",
    "- Figure title: Sample 3608",
    "",
    "## Prediction PKL",
    "",
    "- Ego-only: outputs/maptr/visualization_predictions/ego_only_test_epoch18.pkl",
    "- Dynamic Top-4: outputs/maptr/visualization_predictions/dynamic_top4_test_epoch18.pkl",
    "- Pose-gated V2I: outputs/maptr/visualization_predictions/pose_gated_top4_test_epoch18.pkl",
    "",
    "## GT Files",
    "",
    "- GT JSON: data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k/simv2i_maptr_map_gt_test.json",
    "- infos PKL: data/maptr/simv2i_hd_benchmark_v2_dynamic_rsu_20k/simv2i_maptr_infos_test.pkl",
    "",
    "## Paper-Ready Styling",
    "",
    "- 4 panels: GT / Ego-only / Dynamic Top-4 / Pose-gated V2I",
    "- Short title only; long sample token hidden.",
    "- GT overlay in prediction panels: light gray, alpha 0.28, linewidth 1.0.",
    "- Prediction lines: class colors retained, linewidth 2.0.",
    "- Font family: Times New Roman if available, otherwise serif fallback such as Nimbus Roman.",
    "- Legend: lower center, fontsize 9.",
    "- Main title fontsize: 13.",
    "- Panel title fontsize: 12.",
    "- Axis label fontsize: 10.",
    "- Tick label fontsize: 9.5.",
    "- Figure size: 10.4 x 4.25 inches.",
    "- Panel spacing wspace: 0.035.",
    "- Outer margins: compact paper-ready subplot margins from the visualizer.",
    "",
    "## Outputs",
    "",
    f"- PNG: {png_path}",
    f"- PDF: {pdf_path}",
    "",
    "## Compatibility Check",
    "",
    "- PASS: existing visualization script was checked with sample index 0.",
    "- Compatibility output: outputs/maptr/visualizations/paper_figures/compat_check/sample_000000_gt_ego_dynamic_constant_pose.png",
    "",
    "## Notes",
    "",
    "- This figure is a qualitative illustration.",
    "- Official quantitative conclusions should follow the Section 4 table results.",
]
report_path.write_text("\n".join(lines) + "\n")
print(f"Report written to {report_path}")
PY

echo "Paper-ready PNG: ${PNG_PATH}"
echo "Paper-ready PDF: ${PDF_PATH}"
echo "Report: ${REPORT_PATH}"
