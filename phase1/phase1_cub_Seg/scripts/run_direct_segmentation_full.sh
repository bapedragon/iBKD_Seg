#!/usr/bin/env bash
set -euo pipefail

data_dir="${CUB_SEGMENTATION_DATA_DIR:-/app/scratch/cub_direct_segmentation_full_v1/data}"
output_root="${CUB_SEGMENTATION_FULL_OUTPUT_DIR:-/app/output/cub_direct_segmentation_full_v1}"
mkdir -p "${output_root}/setup"
log_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"

run_suite() {
  echo "[CUB_DIRECT_SEGMENTATION_FULL_SETUP] stage=environment"
  if ! python -m pip install --disable-pip-version-check -e . > "${output_root}/setup/install_${log_id}.log" 2>&1; then
    tail -n 80 "${output_root}/setup/install_${log_id}.log"
    return 1
  fi
  python -u -m ibkd_seg.cub_segmentation.full \
    --device cuda \
    --resume \
    --config phase1/phase1_cub_Seg/configs/direct_segmentation_full_v1.json \
    --data-dir "${data_dir}" \
    --output-dir "${output_root}/artifacts"
}

run_suite 2>&1 | tee "${output_root}/run_${log_id}.log"
