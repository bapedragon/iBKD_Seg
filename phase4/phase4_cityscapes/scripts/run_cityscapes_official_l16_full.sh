#!/usr/bin/env bash
set -euo pipefail

method="${1:-}"
case "${method}" in
  vanilla|lg|alg|ibkd) shift ;;
  *) echo "Usage: bash $0 {vanilla|lg|alg|ibkd} [--max-hours N] [--resume /path/to/resume.json]" >&2; exit 2 ;;
esac

zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_OFFICIAL_DATA_DIR:-/app/scratch/cityscapes_official_l16_v2/cityscapes}"
cache_root="${CITYSCAPES_OFFICIAL_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_FULL_OUTPUT:-/app/output/cityscapes_official_l16_full_v1/${method}_seed1}"
export PIP_CONSTRAINT="${PWD}/phase4/phase4_cityscapes/configs/h200_full_constraints.txt"
mkdir -p "${output_root}/setup"
log_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"

run_training() {
  # Keep installation chatter out of the issue's 65k-character log limit.
  echo "[L16_FULL_SETUP] stage=environment method=${method}"
  if ! python -m pip install --disable-pip-version-check -e . > "${output_root}/setup/install_${log_id}.log" 2>&1; then
    tail -n 60 "${output_root}/setup/install_${log_id}.log"
    return 1
  fi
  echo "[L16_FULL_SETUP] stage=verify_zip"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/setup/upload_check.json"
  echo "[L16_FULL_SETUP] stage=prepare_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/setup/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/setup/preparation.json"
  echo "[L16_FULL_SETUP] stage=official_assets"
  if ! python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" > "${output_root}/setup/assets_${log_id}.log" 2>&1; then
    tail -n 60 "${output_root}/setup/assets_${log_id}.log"
    return 1
  fi
  echo "[L16_FULL_SETUP] stage=train epochs=216 crop=768 batch=8 precision=fp32"
  python -u -m ibkd_seg.cityscapes.official_full --device cuda --method "${method}" \
    --cache-root "${cache_root}" --data-dir "${data_dir}" \
    --manifest "${output_root}/setup/manifest.json" --output-dir "${output_root}/artifacts" "$@"
}

run_training "$@" 2>&1 | tee "${output_root}/run_${log_id}.log"
