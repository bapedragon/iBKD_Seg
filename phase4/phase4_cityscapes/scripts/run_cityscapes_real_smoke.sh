#!/usr/bin/env bash
set -euo pipefail

zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_REAL_DATA_DIR:-/app/scratch/cityscapes_real_smoke_v1}"
output_root="${CITYSCAPES_REAL_SMOKE_OUTPUT_DIR:-/app/output/cityscapes_deeplabv3_segmenter_real_smoke_v1}"
mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  echo "[CITYSCAPES_REAL_PIPELINE] stage=verify_uploaded_zips"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  echo "[CITYSCAPES_REAL_PIPELINE] stage=extract_and_audit_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  echo "[CITYSCAPES_REAL_PIPELINE] stage=real_data_model_smoke"
  python -m ibkd_seg.cityscapes.real_smoke --device cuda \
    --data-dir "${data_dir}" --manifest "${output_root}/manifest.json" \
    --output-dir "${output_root}/artifacts"
}

run_suite 2>&1 | tee "${output_root}/run.log"
