#!/usr/bin/env bash
set -euo pipefail

zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_CROP512_STABILITY_OUTPUT:-/app/output/cityscapes_l16_crop512_stability_v1}"
config="${PWD}/phase4/phase4_cityscapes/configs/paper_l16_crop512_stability_v1.json"
mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  echo "[CROP512_L16_STABILITY] stage=verify_zip"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  echo "[CROP512_L16_STABILITY] stage=prepare_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  echo "[CROP512_L16_STABILITY] stage=official_sources_and_pretrained_weights"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}"
  echo "[CROP512_L16_STABILITY] stage=500_step_diagnostic"
  python -u -m ibkd_seg.cityscapes.official_stability --device cuda \
    --cache-root "${cache_root}" --data-dir "${data_dir}" \
    --manifest "${output_root}/manifest.json" --output-dir "${output_root}/artifacts" \
    --config "${config}"
}

run_suite 2>&1 | tee "${output_root}/run.log"
