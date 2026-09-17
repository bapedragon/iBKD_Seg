#!/usr/bin/env bash
set -euo pipefail

data_dir="${CUB_SEGMENTATION_DATA_DIR:-/app/scratch/cub_direct_segmentation_smoke_v1/data}"
output_root="${CUB_SEGMENTATION_WINDOW30_SMOKE_OUTPUT_DIR:-/app/output/cub_direct_segmentation_window30_smoke_v1}"
mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  echo "[CUB_DIRECT_SEGMENTATION_PIPELINE] stage=real_data_four_method_window30_smoke"
  python -m ibkd_seg.cub_segmentation.smoke \
    --device cuda \
    --config phase1/phase1_cub_Seg/configs/direct_segmentation_window30_smoke_v1.json \
    --data-dir "${data_dir}" \
    --output-dir "${output_root}/artifacts"
}

run_suite 2>&1 | tee "${output_root}/run.log"
