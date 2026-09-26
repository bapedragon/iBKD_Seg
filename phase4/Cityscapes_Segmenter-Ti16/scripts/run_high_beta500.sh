#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
if [[ "$#" -ne 0 ]]; then
  echo 'This fixed two-run 500-step probe accepts no arguments' >&2
  exit 2
fi
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CITYSCAPES_TI16_JOB_STARTED="${CITYSCAPES_TI16_JOB_STARTED:-$(date +%s)}"
zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_TI16_OUTPUT:-/app/output/cityscapes_ti16_high_beta500_v1/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
config="${repo_root}/phase4/Cityscapes_Segmenter-Ti16/configs/high_beta500_v1.json"
mkdir -p "${output_root}"
report_failure() {
  local code="$?"
  if [[ "${code}" -ne 0 ]]; then
    python -m ibkd_seg.cityscapes.tiny_grid_report --output-root "${output_root}" \
      --config "${config}" --exit-code "${code}"
  fi
}
trap report_failure EXIT
timeout --signal=TERM --kill-after=90s 35880s bash -c '
  set -euo pipefail
  zip_dir="$1"; data_dir="$2"; cache_root="$3"; output_root="$4"; config="$5"
  echo "[TI16_HIGH_BETA500_START] runs=2 steps_each=500 seed=1 beta=2.5 methods=lg,ibkd_l025 fresh_start=true output=${output_root}"
  python -m pip install --disable-pip-version-check -e .
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" --student tiny
  python -u -m ibkd_seg.cityscapes.tiny_grid --cache-root "${cache_root}" \
    --data-dir "${data_dir}" --manifest "${output_root}/manifest.json" \
    --output-dir "${output_root}/artifacts" --config "${config}"
' high-beta "${zip_dir}" "${data_dir}" "${cache_root}" "${output_root}" "${config}" 2>&1 | tee "${output_root}/run.log"
