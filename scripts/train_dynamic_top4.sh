#!/usr/bin/env bash
set -euo pipefail

# Public wrapper for the SimV2I-HD Dynamic Top-4 RSU 20k experiment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_maptr_v2_dynamic_rsu_top4_train_20k_full.sh" "$@"
