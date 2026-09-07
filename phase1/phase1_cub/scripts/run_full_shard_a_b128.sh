#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_dir="${PHASE1_CUB_OUTPUT_DIR:-/app/output/phase1_cub_b128_full_v1_shard_a}"
cache_dir="${PHASE1_CUB_CACHE_DIR:-/app/scratch/phase1_cub_b128_full_v1_shard_a_cache}"

mkdir -p "${output_dir}"
python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_cub_full_shard \
  --full-shard \
  --shard a \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --cache-dir "${cache_dir}" \
  --config phase1/phase1_cub/configs/cub200_b128_full_v1.json \
  --device cuda \
  --feature-batch-size 32 \
  --eval-batch-size 200 \
  --num-workers 4 2>&1 | tee "${output_dir}/run.log"
