#!/usr/bin/env bash
set -euo pipefail

# Public wrapper for the SimV2I-HD Pose-gated Top-4 V2I 20k experiment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_maptr_pose_gated_v2i_train_20k_b4_full.sh" "$@"
