#!/usr/bin/env bash
set -euo pipefail

zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_CROP512_REPRO_SMOKE_OUTPUT:-/app/output/cityscapes_l16_crop512_repro_smoke25_v7}"
config="${PWD}/phase4/phase4_cityscapes/configs/paper_l16_crop512_repro_smoke25_v7.json"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1
mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  echo "[CROP512_L16_REPRO_SMOKE] stage=verify_zip"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  echo "[CROP512_L16_REPRO_SMOKE] stage=prepare_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  echo "[CROP512_L16_REPRO_SMOKE] stage=official_sources_and_pretrained_weights"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}"
  echo "[CROP512_L16_REPRO_SMOKE] stage=lg_a_lg_b_alg_beta_0p02_25_steps"
  python -u -m ibkd_seg.cityscapes.reproducibility_audit \
    --cache-root "${cache_root}" --data-dir "${data_dir}" \
    --manifest "${output_root}/manifest.json" --output-dir "${output_root}/artifacts" \
    --config "${config}"
}

run_suite 2>&1 | tee "${output_root}/run.log"
