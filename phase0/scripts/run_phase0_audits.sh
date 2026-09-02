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
  --output phase0/reports/checkpoint_audit.local.json
if PYTHONPATH=src "${python_bin}" -m ibkd_seg.phase0.flowers_data \
  --data-root "${dataset_root}" \
  --manifest manifests/flowers102.json \
  --output phase0/reports/dataset_audit.local.json; then
  echo "Phase 0 감사 통과: phase0/reports/*.local.json을 확인하세요."
else
  dataset_status="$?"
  if [[ "${dataset_status}" -eq 1 ]]; then
    echo "Phase 0 데이터 품질 gate는 보류(HOLD)입니다. phase0/DECISION.md를 확인하세요."
  else
    echo "Phase 0 데이터셋 감사를 완료하지 못했습니다(종료 코드 ${dataset_status})."
  fi
  exit "${dataset_status}"
fi
