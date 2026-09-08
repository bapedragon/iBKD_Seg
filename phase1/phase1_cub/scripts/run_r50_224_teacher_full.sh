#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_dir="${PHASE1_CUB_R50_TEACHER_OUTPUT_DIR:-/app/output/phase1_cub_r50_224_teacher_full_v3}"

mkdir -p "${output_dir}"
python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_cub_r50_teacher_full \
  --full-teacher \
  --data-dir "${data_dir}" \
  --output-dir "${output_dir}" \
  --config phase1/phase1_cub/configs/cub200_r50_224_b128_full_v3.json \
  --device cuda \
  --num-workers 4 2>&1 | tee "${output_dir}/run.log"
