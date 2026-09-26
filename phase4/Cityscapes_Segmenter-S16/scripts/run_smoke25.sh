#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_S16_CACHE:-/app/scratch/cityscapes_s16_v1/upstream}"
output_root="${CITYSCAPES_S16_OUTPUT:-/app/output/cityscapes_s16_crop512_smoke25_v1/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
config="${repo_root}/phase4/Cityscapes_Segmenter-S16/configs/s16_smoke25_fskd_c2vkd_v1.json"
if [[ -e "${output_root}" ]]; then
  echo "Use a new Small output directory: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}"

finish_report() {
  local exit_code="$?"
  trap - EXIT
  python -m ibkd_seg.cityscapes.small_smoke_report --config "${config}" \
    --output-root "${output_root}" --exit-code "${exit_code}" \
    2>&1 | tee -a "${output_root}/run.log" || exit 1
  exit "${exit_code}"
}
trap finish_report EXIT

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  export MAX_JOBS=2
  export TORCH_CUDA_ARCH_LIST=9.0
  python -m pip install --disable-pip-version-check ninja==1.13.0
  python -m pip install --disable-pip-version-check --no-build-isolation torchsort==0.1.10
  echo "[S16_PIPELINE] stage=verify_uploaded_zip"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  echo "[S16_PIPELINE] stage=prepare_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  echo "[S16_PIPELINE] stage=verify_small_and_openmmlab_teacher"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" --student small
  echo "[S16_PIPELINE] stage=verify_c2vkd_clip_pool"
  python -m ibkd_seg.cityscapes.tiny_c2vkd --cache-root "${cache_root}"
  echo "[S16_PIPELINE] stage=calibration_and_training_smoke output=${output_root}"
  python -u -m ibkd_seg.cityscapes.tiny_smoke \
    --cache-root "${cache_root}" --data-dir "${data_dir}" \
    --manifest "${output_root}/manifest.json" --output-dir "${output_root}/artifacts" \
    --config "${config}"
}

run_suite 2>&1 | tee "${output_root}/run.log"
