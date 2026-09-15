#!/usr/bin/env bash
set -euo pipefail

data_dir="${PHASE1_CUB_DATA_DIR:-/app/scratch/phase1_cub_data}"
output_root="${PHASE1_CUB_DETERMINISTIC_AA_SMOKE_OUTPUT_DIR:-/app/output/phase1_cub_main_l0_ibkd_deterministic_aa_smoke_v1}"
teacher_dir="${PHASE1_CUB_R50_TEACHER_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_issue722}"
teacher_download_dir="${PHASE1_CUB_R50_TEACHER_DOWNLOAD_DIR:-/app/scratch/phase1_cub_r50_224_teacher_v3_download}"
teacher_manifest="phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3/checkpoint_release.json"
protocol_config="phase1/phase1_cub/reproducibility/configs/cub200_r50_224_b128_main_l0_ibkd_deterministic_aa_smoke_v1.json"

export PYTHONHASHSEED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0

mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  python -m ibkd_seg.phase1.release_asset \
    --manifest "${teacher_manifest}" \
    --destination "${teacher_dir}" \
    --download-dir "${teacher_download_dir}"
  python -m ibkd_seg.phase1.run_cub_ibkd_deterministic_aa_smoke \
    --smoke-run \
    --data-dir "${data_dir}" \
    --output-dir "${output_root}" \
    --teacher-checkpoint "${teacher_dir}/teacher_best_validation.pt" \
    --config "${protocol_config}"
}

run_suite 2>&1 | tee "${output_root}/run.log"
