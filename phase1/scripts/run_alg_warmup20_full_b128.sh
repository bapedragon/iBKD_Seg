#!/usr/bin/env bash
set -euo pipefail

reference_root="${PHASE1_B128_REFERENCE_ROOT:-/app/scratch/phase1_pet_full_b128_v1_reference}"
data_dir="${PHASE1_PET_DATA_DIR:-/app/scratch/phase1_pet_data}"
output_dir="${PHASE1_ALG_W20_OUTPUT_DIR:-/app/output/phase1_pet_alg_warmup20_b128_full_v1}"
cache_dir="${PHASE1_ALG_W20_CACHE_DIR:-/app/scratch/phase1_pet_alg_warmup20_b128_full_v1_cache}"

python -m pip install --disable-pip-version-check -e .
if [[ ! -f "${reference_root}/classification_summary.json" ]]; then
  echo "[INPUT_FETCH] downloading audited batch-128 classification reference"
  python -m ibkd_seg.phase1.release_asset \
    --manifest phase1/reports/classification/batch128/checkpoint_release.json \
    --destination "${reference_root}" \
    --download-dir "${cache_dir}/download"
fi

python -m ibkd_seg.phase1.run_alg_warmup20_full \
  --full-diagnostic \
  --reference-classification-root "${reference_root}" \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --cache-dir "${cache_dir}" \
  --device cuda \
  --feature-batch-size 32 \
  --eval-batch-size 200 \
  --num-workers 4
