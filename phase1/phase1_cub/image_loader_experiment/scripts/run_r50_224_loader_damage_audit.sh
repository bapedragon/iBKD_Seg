#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_dir="${PHASE1_CUB_LOADER_AUDIT_OUTPUT_DIR:-/app/output/phase1_cub_r50_224_loader_damage_audit_v1}"
config="phase1/phase1_cub/image_loader_experiment/configs/cub200_r50_224_loader_damage_audit_v1.json"

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_cub_loader_audit \
  --audit \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --config "${config}"
