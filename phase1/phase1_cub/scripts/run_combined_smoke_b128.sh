#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_dir="${PHASE1_CUB_SMOKE_OUTPUT_DIR:-/app/output/phase1_cub_b128_combined_smoke_v1}"
cache_dir="${PHASE1_CUB_SMOKE_CACHE_DIR:-/app/scratch/phase1_cub_b128_combined_smoke_v1_cache}"

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_cub_combined_smoke \
  --smoke \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --cache-dir "${cache_dir}" \
  --config phase1/phase1_cub/configs/cub200_b128_combined_smoke_v1.json \
  --device cuda \
  --feature-batch-size 32 \
  --eval-batch-size 200 \
  --num-workers 4
