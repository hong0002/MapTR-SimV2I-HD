#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPTR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MAPTR_PYTHON:-python}"

cd "${MAPTR_ROOT}"
exec "${PYTHON_BIN}" \
  integrations/maptr/tools/validate_simv2i_maptr_data.py \
  --root "${MAPTR_ROOT}" \
  "$@"
