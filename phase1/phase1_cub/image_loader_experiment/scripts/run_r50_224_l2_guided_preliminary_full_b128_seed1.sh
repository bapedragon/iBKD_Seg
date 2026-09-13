#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_root="${PHASE1_CUB_L2_GUIDED_FULL_OUTPUT_DIR:-/app/output/phase1_cub_r50_224_b128_seed1_l2_guided_preliminary_full_v1}"
cache_root="${PHASE1_CUB_L2_GUIDED_FULL_CACHE_DIR:-/app/scratch/phase1_cub_r50_224_b128_seed1_l2_guided_preliminary_full_v1_cache}"
teacher_dir="${PHASE1_CUB_R50_TEACHER_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_issue722}"
teacher_download_dir="${PHASE1_CUB_R50_TEACHER_DOWNLOAD_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_download}"
teacher_manifest="phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3/checkpoint_release.json"
full_config="phase1/phase1_cub/image_loader_experiment/configs/cub200_r50_224_b128_seed1_l2_guided_preliminary_full_execution_v1.json"

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.release_asset \
  --manifest "${teacher_manifest}" \
  --destination "${teacher_dir}" \
  --download-dir "${teacher_download_dir}"

mkdir -p "${output_root}"
python -m ibkd_seg.phase1.run_cub_r50_batch_profile_full \
  --loader-followup-full \
  --data-dir "${data_dir}" \
  --output-dir "${output_root}" \
  --cache-dir "${cache_root}" \
  --config "${full_config}" \
  --teacher-checkpoint "${teacher_dir}/teacher_best_validation.pt" \
  --device cuda \
  --feature-batch-size 32 \
  --eval-batch-size 200 \
  --num-workers 4 2>&1 | tee "${output_root}/run.log"
