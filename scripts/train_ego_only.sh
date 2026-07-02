#!/usr/bin/env bash
set -euo pipefail

# Public wrapper for the SimV2I-HD ego-only 20k experiment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_maptr_runtime.sh"
maptr_init
maptr_require_simv2i_data
exec bash "${SCRIPT_DIR}/run_maptr_ego_only_train_20k_b4_full.sh" "$@"
