#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_TI16_OUTPUT:-/app/output/cityscapes_ti16_crop512_smoke25_v1/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
config="${repo_root}/phase4/Cityscapes_Segmenter-Ti16/configs/smoke25_v1.json"
mkdir -p "${output_root}"

report_failure() {
  local exit_code="$1"
  if [[ "${exit_code}" -eq 0 ]]; then return; fi
  python - "${output_root}" "${exit_code}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / "artifacts/smoke_summary.json"
report = json.loads(path.read_text()) if path.exists() else {
    "status": "failed", "scientific_result": False, "test_used": False,
    "runs": [], "error": "Pipeline setup failed; see run.log",
    "selected_step": None, "selected_epoch": None, "metrics": None,
}
report.update(status="failed", pipeline_exit_code=int(sys.argv[2]), output_root=str(root))
print("[CITYSCAPES_TI16_SMOKE_FINAL] " + json.dumps(report, allow_nan=False), flush=True)
PY
}
trap 'report_failure "$?"' EXIT

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  echo "[TI16_PIPELINE] stage=verify_uploaded_zip"
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  echo "[TI16_PIPELINE] stage=prepare_train_val"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  echo "[TI16_PIPELINE] stage=verify_tiny_and_openmmlab_teacher"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" --student tiny
  echo "[TI16_PIPELINE] stage=calibration_and_five_path_smoke output=${output_root}"
  python -u -m ibkd_seg.cityscapes.tiny_smoke \
    --cache-root "${cache_root}" --data-dir "${data_dir}" \
    --manifest "${output_root}/manifest.json" --output-dir "${output_root}/artifacts" \
    --config "${config}"
}

run_suite 2>&1 | tee "${output_root}/run.log"
