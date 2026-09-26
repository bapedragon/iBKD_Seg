#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CITYSCAPES_TI16_JOB_STARTED="${CITYSCAPES_TI16_JOB_STARTED:-$(date +%s)}"
zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_TI16_OUTPUT:-/app/output/cityscapes_ti16_ibkd_l025_grid2000_v2/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
config="${repo_root}/phase4/Cityscapes_Segmenter-Ti16/configs/beta_grid2000_ibkd_l025_v2.json"
start_candidate="${CITYSCAPES_TI16_START_CANDIDATE:-1}"
mkdir -p "${output_root}"
args=(--config "${config}" --cache-root "${cache_root}" --data-dir "${data_dir}"
      --zip-dir "${zip_dir}" --start-candidate "${start_candidate}")
if [[ -n "${CITYSCAPES_TI16_RESUME:-}" ]]; then args+=(--resume "${CITYSCAPES_TI16_RESUME}"); fi
if [[ -n "${CITYSCAPES_TI16_JOB_DEADLINE:-}" ]]; then args+=(--deadline "${CITYSCAPES_TI16_JOB_DEADLINE}"); fi
report_failure() {
  local code="$1"
  if [[ "${code}" -ne 0 ]]; then
    python -m ibkd_seg.cityscapes.tiny_screen2000_report --output-root "${output_root}" \
      --config "${config}" --start-candidate "${start_candidate}" --exit-code "${code}"
  fi
}
trap 'report_failure "$?"' EXIT
run_suite() {
  python -m ibkd_seg.cityscapes.tiny_screen2000 "${args[@]}" --output-dir "${output_root}/setup" --preflight-only
  python -m pip install --disable-pip-version-check -e .
  python -u -m ibkd_seg.cityscapes.tiny_screen2000 "${args[@]}" --output-dir "${output_root}/setup" --prepare-data-only
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" --student tiny
  echo "[TI16_GRID2000] pack=ibkd_l025 candidates=8 steps_each=2000 seed=1 full_val=500 budget_seconds=36000 save_reserve_seconds=120 stop_after_seconds=35880 output=${output_root}"
  python -u -m ibkd_seg.cityscapes.tiny_screen2000 "${args[@]}" \
    --manifest "${output_root}/manifest.json" --output-dir "${output_root}/artifacts"
}
run_suite 2>&1 | tee "${output_root}/run.log"
