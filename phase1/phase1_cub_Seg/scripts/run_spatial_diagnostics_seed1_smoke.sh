#!/usr/bin/env bash
set -euo pipefail

data_dir="${CUB_SEGMENTATION_DATA_DIR:-/app/scratch/cub_segmentation_spatial_smoke_v1/data}"
release_dir="${CUB_SEGMENTATION_SEED1_RELEASE_DIR:-/app/scratch/cub_segmentation_window30_seed1_issue783}"
download_dir="${CUB_SEGMENTATION_SEED1_DOWNLOAD_DIR:-/app/scratch/cub_segmentation_window30_seed1_download}"
output_root="${CUB_SEGMENTATION_SPATIAL_SMOKE_OUTPUT_DIR:-/app/output/cub_segmentation_spatial_diagnostics_seed1_smoke_v1}"
manifest="phase1/phase1_cub_Seg/reports/window30_seed1_v1/checkpoint_release.json"
config="phase1/phase1_cub_Seg/configs/spatial_diagnostics_seed1_smoke_v1.json"
mkdir -p "${output_root}/setup"
log_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"

run_suite() {
  echo "[CUB_SEG_SPATIAL_SMOKE_SETUP] stage=environment"
  if ! python -m pip install --disable-pip-version-check -e . > "${output_root}/setup/install_${log_id}.log" 2>&1; then
    tail -n 80 "${output_root}/setup/install_${log_id}.log"
    return 1
  fi
  python -u -m ibkd_seg.phase1.release_asset \
    --manifest "${manifest}" \
    --destination "${release_dir}" \
    --download-dir "${download_dir}"
  python -u -m ibkd_seg.cub_segmentation.spatial_diagnostics \
    --config "${config}" \
    --release-dir "${release_dir}" \
    --data-dir "${data_dir}" \
    --output-dir "${output_root}/artifacts" \
    --device cuda \
    --feature-batch-size 16 \
    --cka-batch-size 8 \
    --attention-batch-size 16 \
    --num-workers 4
}

run_suite 2>&1 | tee "${output_root}/run_${log_id}.log"
