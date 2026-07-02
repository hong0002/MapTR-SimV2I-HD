#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval_ego_only.sh path/to/checkpoint.pth [extra test.py args...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init
maptr_require_simv2i_data
exec bash "${SCRIPT_DIR}/eval_maptr_ego_only_20k_b4.sh" "$@"
