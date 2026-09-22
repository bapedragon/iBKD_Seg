#!/usr/bin/env bash
set -euo pipefail

method="${1:-}"
case "${method}" in
  vanilla|lg|alg|ibkd) shift ;;
  *) echo "Usage: bash $0 {vanilla|lg|alg|ibkd} [--max-hours N]" >&2; exit 2 ;;
esac

zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_FINAL80000_OUTPUT:-/app/output/cityscapes_l16_crop512_final80000_v19}/${method}_seed1"
config="${PWD}/phase4/phase4_cityscapes/configs/paper_l16_crop512_final80000_v19.json"
export PIP_CONSTRAINT="${PWD}/phase4/phase4_cityscapes/configs/h200_full_constraints.txt"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1
mkdir -p "${output_root}/setup"
log_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"

resume_args=()
if [[ -f "${output_root}/artifacts/resume.json" ]]; then
  resume_args=(--resume "${output_root}/artifacts/resume.json")
elif [[ -d "${output_root}/artifacts" ]] && [[ -n "$(find "${output_root}/artifacts" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: nonempty artifact directory has no resume.json: ${output_root}/artifacts" >&2
  exit 1
fi

run_training() {
  echo "[CROP512_L16_FINAL80000_SETUP] stage=environment method=${method}"
  if ! python -m pip install --disable-pip-version-check -e . > "${output_root}/setup/install_${log_id}.log" 2>&1; then
    tail -n 60 "${output_root}/setup/install_${log_id}.log"
    return 1
  fi
  echo "[CROP512_L16_FINAL80000_SETUP] stage=verify_zip"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/setup/upload_check.json"
  echo "[CROP512_L16_FINAL80000_SETUP] stage=prepare_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/setup/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/setup/preparation.json"
  echo "[CROP512_L16_FINAL80000_SETUP] stage=official_assets"
  if ! python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" > "${output_root}/setup/assets_${log_id}.log" 2>&1; then
    tail -n 60 "${output_root}/setup/assets_${log_id}.log"
    return 1
  fi
  echo "[CROP512_L16_FINAL80000_SETUP] stage=train method=${method} steps=80000 crop=512 batch=8 precision=fp32"
  python -u -m ibkd_seg.cityscapes.official_full --device cuda --method "${method}" \
    --cache-root "${cache_root}" --data-dir "${data_dir}" \
    --manifest "${output_root}/setup/manifest.json" --output-dir "${output_root}/artifacts" \
    --config "${config}" "${resume_args[@]}" "$@"
}

run_training "$@" 2>&1 | tee "${output_root}/run_${log_id}.log"
