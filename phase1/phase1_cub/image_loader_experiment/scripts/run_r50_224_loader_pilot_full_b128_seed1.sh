#!/usr/bin/env bash
set -euo pipefail

profile="${1:?usage: $0 l0_current_strong|l1_matched_weak|l2_conservative_spatial}"
case "${profile}" in
  l0_current_strong|l1_matched_weak|l2_conservative_spatial) ;;
  *)
    echo "unsupported loader profile: ${profile}" >&2
    exit 2
    ;;
esac

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_root="${PHASE1_CUB_LOADER_PILOT_FULL_OUTPUT_DIR:-/app/output/phase1_cub_r50_224_b128_seed1_loader_pilot_full_v1_${profile}}"
cache_root="${PHASE1_CUB_LOADER_PILOT_FULL_CACHE_DIR:-/app/scratch/phase1_cub_r50_224_b128_seed1_loader_pilot_full_v1_${profile}_cache}"
teacher_dir="${PHASE1_CUB_R50_TEACHER_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_issue722}"
teacher_download_dir="${PHASE1_CUB_R50_TEACHER_DOWNLOAD_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_download}"
teacher_manifest="phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3/checkpoint_release.json"
full_config="phase1/phase1_cub/image_loader_experiment/configs/cub200_r50_224_b128_seed1_loader_pilot_full_v1.json"

mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  python -m ibkd_seg.phase1.release_asset \
    --manifest "${teacher_manifest}" \
    --destination "${teacher_dir}" \
    --download-dir "${teacher_download_dir}"
  python -m ibkd_seg.phase1.run_cub_loader_pilot_full \
    --full \
    --profile "${profile}" \
    --data-dir "${data_dir}" \
    --output-dir "${output_root}" \
    --cache-dir "${cache_root}" \
    --teacher-checkpoint "${teacher_dir}/teacher_best_validation.pt" \
    --config "${full_config}" \
    --device cuda \
    --feature-batch-size 16 \
    --cka-batch-size 8 \
    --attention-batch-size 16 \
    --eval-batch-size 200 \
    --num-workers 4
}

run_suite 2>&1 | tee "${output_root}/run.log"
