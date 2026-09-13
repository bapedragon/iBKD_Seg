#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
combined_root="${PHASE1_CUB_LOADER_PILOT_FULL_L1_L2_OUTPUT_DIR:-/app/output/phase1_cub_r50_224_b128_seed1_loader_pilot_full_l1_l2_v1}"
cache_root="${PHASE1_CUB_LOADER_PILOT_FULL_L1_L2_CACHE_DIR:-/app/scratch/phase1_cub_r50_224_b128_seed1_loader_pilot_full_l1_l2_v1_cache}"
teacher_dir="${PHASE1_CUB_R50_TEACHER_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_issue722}"
teacher_download_dir="${PHASE1_CUB_R50_TEACHER_DOWNLOAD_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_download}"
teacher_manifest="phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3/checkpoint_release.json"
full_config="phase1/phase1_cub/image_loader_experiment/configs/cub200_r50_224_b128_seed1_loader_pilot_full_v1.json"
job_config="phase1/phase1_cub/image_loader_experiment/configs/cub200_r50_224_b128_seed1_loader_pilot_l1_l2_combined_job_v1.json"

mkdir -p "${combined_root}"

run_profile() {
  local profile="$1"
  local profile_output="${combined_root}/${profile}"
  local profile_cache="${cache_root}/${profile}"
  mkdir -p "${profile_output}" "${profile_cache}"
  python -m ibkd_seg.phase1.run_cub_loader_pilot_full \
    --full \
    --profile "${profile}" \
    --data-dir "${data_dir}" \
    --output-dir "${profile_output}" \
    --cache-dir "${profile_cache}" \
    --teacher-checkpoint "${teacher_dir}/teacher_best_validation.pt" \
    --config "${full_config}" \
    --device cuda \
    --feature-batch-size 16 \
    --cka-batch-size 8 \
    --attention-batch-size 16 \
    --eval-batch-size 200 \
    --num-workers 4 2>&1 | tee "${profile_output}/run.log"
}

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  python -m ibkd_seg.phase1.release_asset \
    --manifest "${teacher_manifest}" \
    --destination "${teacher_dir}" \
    --download-dir "${teacher_download_dir}"

  run_profile l1_matched_weak
  run_profile l2_conservative_spatial

  python -m ibkd_seg.phase1.summarize_cub_loader_pilot_l1_l2 \
    --combined-root "${combined_root}" \
    --job-config "${job_config}"
}

run_suite 2>&1 | tee "${combined_root}/run.log"
