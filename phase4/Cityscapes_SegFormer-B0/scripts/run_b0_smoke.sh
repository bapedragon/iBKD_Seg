#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
output="${B0_SMOKE_OUTPUT:-/app/output/cityscapes_b0_smoke_v1}"
mkdir -p "${output}"
if [[ -e "${output}/run.log" ]]; then
  echo "Existing run.log: choose a fresh B0_SMOKE_OUTPUT" >&2
  exit 1
fi
python -u phase4/Cityscapes_SegFormer-B0/scripts/run_b0_smoke.py 2>&1 | tee "${output}/run.log"
