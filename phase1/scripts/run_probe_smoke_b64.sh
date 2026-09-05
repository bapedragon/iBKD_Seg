#!/usr/bin/env bash
set -euo pipefail

classification_root="${PHASE1_B64_CLASSIFICATION_ROOT:-/app/output/phase1_pet_full_b64_v1}"
data_dir="${PHASE1_PET_DATA_DIR:-${classification_root}/data}"
output_dir="${PHASE1_PROBE_OUTPUT_DIR:-/app/output/phase1_pet_probe_b64_smoke_v1}"
cache_dir="${PHASE1_PROBE_CACHE_DIR:-/app/scratch/phase1_pet_probe_b64_smoke_v1_cache}"

if [[ ! -f "${classification_root}/classification_summary.json" ]]; then
  echo "[INPUT_MISSING] batch-64 classification output is not mounted: ${classification_root}" >&2
  echo "[INPUT_REQUIRED] retain or mount H200 issue #390/build 700 output at that path" >&2
  exit 2
fi

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_probe_smoke \
  --smoke \
  --classification-root "${classification_root}" \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --cache-dir "${cache_dir}" \
  --device cuda \
  --feature-batch-size 32 \
  --num-workers 4
