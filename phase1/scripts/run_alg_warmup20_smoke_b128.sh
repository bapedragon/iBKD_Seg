#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_PET_DATA_DIR:-/app/scratch/phase1_pet_data}"
output_dir="${PHASE1_ALG_W20_OUTPUT_DIR:-/app/output/phase1_pet_alg_warmup20_b128_combined_smoke_v1}"
cache_dir="${PHASE1_ALG_W20_CACHE_DIR:-/app/scratch/phase1_pet_alg_warmup20_b128_combined_smoke_v1_cache}"

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_alg_warmup20_smoke \
  --smoke \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --cache-dir "${cache_dir}" \
  --device cuda \
  --feature-batch-size 32 \
  --eval-batch-size 200 \
  --num-workers 4
