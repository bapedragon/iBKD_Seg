#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_root="${PHASE1_CUB_BATCH_PROFILE_OUTPUT_DIR:-/app/output/phase1_cub_r50_224_b128_b64_guided_probe_smoke_v4}"
cache_root="${PHASE1_CUB_BATCH_PROFILE_CACHE_DIR:-/app/scratch/phase1_cub_r50_224_b128_b64_guided_probe_smoke_v4_cache}"
teacher_dir="${PHASE1_CUB_R50_TEACHER_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_issue722}"
teacher_download_dir="${PHASE1_CUB_R50_TEACHER_DOWNLOAD_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_download}"
teacher_manifest="phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3/checkpoint_release.json"
smoke_config="phase1/phase1_cub/configs/cub200_r50_224_b64_b128_guided_smoke_v4.json"

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.release_asset \
  --manifest "${teacher_manifest}" \
  --destination "${teacher_dir}" \
  --download-dir "${teacher_download_dir}"

teacher_checkpoint="${teacher_dir}/teacher_best_validation.pt"
for student_batch_size in 128 64; do
  python -m ibkd_seg.phase1.run_cub_r50_guided_smoke \
    --smoke \
    --data-dir "${data_dir}" \
    --output-dir "${output_root}/batch${student_batch_size}" \
    --cache-dir "${cache_root}" \
    --config "${smoke_config}" \
    --student-batch-size "${student_batch_size}" \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --device cuda \
    --feature-batch-size 32 \
    --eval-batch-size 200 \
    --num-workers 4
done

python -m ibkd_seg.phase1.summarize_cub_r50_batch_profile_smoke \
  --root "${output_root}"
