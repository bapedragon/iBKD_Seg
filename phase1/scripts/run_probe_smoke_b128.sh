#!/usr/bin/env bash
set -euo pipefail

classification_root="${PHASE1_B128_CLASSIFICATION_ROOT:-/app/scratch/phase1_pet_full_b128_v1_input}"
data_dir="${PHASE1_PET_DATA_DIR:-/app/scratch/phase1_pet_data}"
output_dir="${PHASE1_PROBE_OUTPUT_DIR:-/app/output/phase1_pet_probe_b128_smoke_v1}"
cache_dir="${PHASE1_PROBE_CACHE_DIR:-/app/scratch/phase1_pet_probe_b128_smoke_v1_cache}"

python -m pip install --disable-pip-version-check -e .
if [[ ! -f "${classification_root}/classification_summary.json" ]]; then
  echo "[INPUT_FETCH] batch-128 classification output is absent; downloading audited GitHub Release asset"
  python -m ibkd_seg.phase1.release_asset \
    --manifest phase1/reports/classification/batch128/checkpoint_release.json \
    --destination "${classification_root}" \
    --download-dir "${cache_dir}"
fi

python -m ibkd_seg.phase1.run_probe_smoke \
  --smoke \
  --classification-root "${classification_root}" \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --cache-dir "${cache_dir}" \
  --smoke-config phase1/configs/oxford_iiit_pet_probe_smoke_b128_v1.json \
  --device cuda \
  --feature-batch-size 32 \
  --num-workers 4
