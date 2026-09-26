#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CITYSCAPES_VAL_TIMING_STARTED="$(date +%s)"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
resume="${CITYSCAPES_TI16_EVAL_RESUME:-/app/output/cityscapes_ti16_beta_grid500_shared_lg_8betas_v6/run_20260925T182606Z_36/artifacts/lg_b1/resume.json}"
output_root="${CITYSCAPES_TI16_VAL_OUTPUT:-/app/output/cityscapes_ti16_val500_timing_v1/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
config="${repo_root}/phase4/Cityscapes_Segmenter-Ti16/configs/val500_timing_v1.json"
mkdir -p "${output_root}"
eval_args=(--cache-root "${cache_root}" --data-dir "${data_dir}" --resume "${resume}"
           --output-dir "${output_root}" --config "${config}")
report_failure() {
  local code="$1"
  if [[ "${code}" -ne 0 ]]; then
    python -m ibkd_seg.cityscapes.tiny_val_timing "${eval_args[@]}" --failure-code "${code}"
  fi
}
trap 'report_failure "$?"' EXIT
run_evaluation() {
  # Fail quickly on a missing old result; never start new training as a fallback.
  python -m ibkd_seg.cityscapes.tiny_val_timing "${eval_args[@]}" --preflight-only
  python -m pip install --disable-pip-version-check -e .
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" --student tiny
  python -u -m ibkd_seg.cityscapes.tiny_val_timing "${eval_args[@]}"
}
run_evaluation 2>&1 | tee "${output_root}/run.log"
