#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_root="${1:-${repo_root}/../IBAM_KD_H200_V2}"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

cd "${repo_root}"
PYTHONPATH=src "${python_bin}" -m ibkd_seg.phase05.run_frozen_probe \
  --source-root "${source_root}" \
  --output phase0.5/reports/full.local.json
