#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {pack1|pack2|pack3|pack4}" >&2
  exit 2
fi

group="$1"
case "${group}" in
  pack1)
    run_ids=(lg_beta_0p05 ibkd_lambda_0p25_beta_0p25)
    ;;
  pack2)
    run_ids=(lg_beta_0p02 ibkd_lambda_0p25_beta_0p5)
    ;;
  pack3)
    run_ids=(alg_beta_0p05 ibkd_lambda_0p5_beta_0p1)
    ;;
  pack4)
    run_ids=(alg_beta_0p02 ibkd_lambda_0p5_beta_0p25)
    ;;
  *)
    echo "unknown group: ${group}" >&2
    exit 2
    ;;
esac

zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_base="${CITYSCAPES_TOP2_GRID10000_OUTPUT_BASE:-/app/output/cityscapes_l16_crop512_candidate_top2_grid10000_v18}"
output_root="${output_base}_${group}"
config="${PWD}/phase4/phase4_cityscapes/configs/paper_l16_crop512_candidate_top2_grid10000_v18.json"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1
mkdir -p "${output_root}"

suite_args=()
for run_id in "${run_ids[@]}"; do
  suite_args+=(--suite-run-id "${run_id}")
done

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  echo "[CROP512_L16_TOP2_GRID10000_GROUP] group=${group} stage=verify_zip"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  echo "[CROP512_L16_TOP2_GRID10000_GROUP] group=${group} stage=prepare_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  echo "[CROP512_L16_TOP2_GRID10000_GROUP] group=${group} stage=official_sources_and_pretrained_weights"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}"
  echo "[CROP512_L16_TOP2_GRID10000_GROUP] group=${group} stage=selected_10000_step_runs_val500"
  python -u -m ibkd_seg.cityscapes.official_stability --device cuda \
    --cache-root "${cache_root}" --data-dir "${data_dir}" \
    --manifest "${output_root}/manifest.json" --output-dir "${output_root}/artifacts" \
    --config "${config}" "${suite_args[@]}"
}

run_suite 2>&1 | tee "${output_root}/run.log"
