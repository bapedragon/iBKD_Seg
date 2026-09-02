#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source_root="${1:-${repo_root}/../IBAM_KD_H200_V2}"
dataset_root="${2:-${repo_root}/data/flowers102}"
python_bin="${IBKD_SEG_PYTHON:-python}"

cd "${repo_root}"

PYTHONPATH=src "${python_bin}" -m unittest discover -s tests -v
PYTHONPATH=src "${python_bin}" -m ibkd_seg.phase0.checkpoints \
  --manifest manifests/checkpoints.json \
  --source-root "${source_root}" \
  --output phase0_sanity/reports/checkpoint_audit.local.json
if PYTHONPATH=src "${python_bin}" -m ibkd_seg.phase0.flowers_data \
  --data-root "${dataset_root}" \
  --manifest manifests/flowers102.json \
  --output phase0_sanity/reports/dataset_audit.local.json; then
  echo "Phase 0 audits passed. Review phase0_sanity/reports/*.local.json."
else
  dataset_status="$?"
  if [[ "${dataset_status}" -eq 1 ]]; then
    echo "Phase 0 data-quality gate returned HOLD. Review phase0_sanity/DECISION.md."
  else
    echo "Phase 0 dataset audit could not complete (status ${dataset_status})."
  fi
  exit "${dataset_status}"
fi
