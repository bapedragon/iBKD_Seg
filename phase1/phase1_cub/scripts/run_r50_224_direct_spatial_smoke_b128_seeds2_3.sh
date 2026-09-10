#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_root="${PHASE1_CUB_DIRECT_SPATIAL_SEED23_SMOKE_OUTPUT_DIR:-/app/output/phase1_cub_r50_224_b128_seed2_3_direct_spatial_smoke_v2}"
student_dir="${PHASE1_CUB_GUIDED_SEED23_RELEASE_DIR:-/app/scratch/phase1_cub_r50_224_guided_seed23_v5_issue730}"
student_download_dir="${PHASE1_CUB_GUIDED_SEED23_DOWNLOAD_DIR:-/app/scratch/phase1_cub_r50_224_guided_seed23_v5_download}"
teacher_dir="${PHASE1_CUB_R50_TEACHER_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_issue722}"
teacher_download_dir="${PHASE1_CUB_R50_TEACHER_DOWNLOAD_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_download}"
student_manifest="phase1/phase1_cub/reports/frozen_probe/resnet50_224_b128_guided_3seed_v5/artifact_release.json"
teacher_manifest="phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3/checkpoint_release.json"
smoke_config="phase1/phase1_cub/configs/cub200_r50_224_b128_seed2_3_direct_spatial_smoke_v2.json"

mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  python -m ibkd_seg.phase1.release_asset \
    --manifest "${teacher_manifest}" \
    --destination "${teacher_dir}" \
    --download-dir "${teacher_download_dir}"
  python -m ibkd_seg.phase1.release_asset \
    --manifest "${student_manifest}" \
    --destination "${student_dir}" \
    --download-dir "${student_download_dir}"
  python -m ibkd_seg.phase1.run_cub_direct_spatial_smoke \
    --smoke \
    --data-dir "${data_dir}" \
    --output-dir "${output_root}" \
    --student-release-dir "${student_dir}" \
    --teacher-checkpoint "${teacher_dir}/teacher_best_validation.pt" \
    --config "${smoke_config}" \
    --device cuda \
    --feature-batch-size 16 \
    --cka-batch-size 8 \
    --attention-batch-size 16 \
    --num-workers 4
}

run_suite 2>&1 | tee "${output_root}/run.log"
