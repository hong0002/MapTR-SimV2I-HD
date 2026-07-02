#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval_pose_gated_top4.sh path/to/checkpoint.pth [extra test.py args...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/eval_maptr_pose_gated_v2i_20k_b4.sh" "$@"
