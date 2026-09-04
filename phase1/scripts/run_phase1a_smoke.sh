#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_root="${1:-${repo_root}/../IBAM_KD_H200_V2}"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

cd "${repo_root}"
PYTHONPATH=src "${python_bin}" -m ibkd_seg.phase1.run_frozen_probe \
  --source-root "${source_root}" \
  --smoke \
  --output phase1/reports/phase1a_smoke.local.json
