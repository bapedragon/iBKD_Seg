#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
if [[ "$#" -ne 0 ]]; then
  echo "This fixed 16-run grid always includes iBKD lambda0.25 and lambda0.5; no arguments accepted" >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
zip_dir="${CITYSCAPES_ZIP_DIR:-/app/data/chaoyang}"
data_dir="${CITYSCAPES_CROP512_DATA_DIR:-/app/scratch/cityscapes_l16_crop512_v3/cityscapes}"
cache_root="${CITYSCAPES_CROP512_CACHE:-/app/scratch/cityscapes_official_l16_v2/upstream}"
output_root="${CITYSCAPES_TI16_OUTPUT:-/app/output/cityscapes_ti16_beta_grid500_both_lambdas_v5/run_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
config="${repo_root}/phase4/Cityscapes_Segmenter-Ti16/configs/beta_grid500_both_lambdas_v5.json"
mkdir -p "${output_root}"
report_failure() {
  local code="$1"
  if [[ "${code}" -eq 0 ]]; then return; fi
  python - "${output_root}" "${code}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); path=root/'artifacts/grid_summary.json'
report=json.loads(path.read_text()) if path.exists() else dict(
    status='runtime_failure',error='Pipeline setup failed; see run.log',runs=[],
    selected_step=None,selected_epoch=None,metrics=None,scientific_result=False,test_used=False)
report.update(pipeline_exit_code=int(sys.argv[2]),output_root=str(root))
print('[CITYSCAPES_TI16_GRID500_FINAL] '+json.dumps(report,allow_nan=False),flush=True)
PY
}
trap 'report_failure "$?"' EXIT
run_suite() {
  python -m pip install --disable-pip-version-check -e .
  python phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py \
    --search-root "${zip_dir}" --output "${output_root}/upload_check.json"
  python -m ibkd_seg.cityscapes.prepare --zip-dir "${zip_dir}" --data-dir "${data_dir}"
  cp "${data_dir}/manifest.json" "${output_root}/manifest.json"
  cp "${data_dir}/preparation.json" "${output_root}/preparation.json"
  python -m ibkd_seg.cityscapes.official_assets --cache-root "${cache_root}" --student tiny
  echo "[TI16_GRID500] runs=16 steps_each=500 ibkd_lambdas=0.25,0.5 output=${output_root}"
  python -u -m ibkd_seg.cityscapes.tiny_grid --cache-root "${cache_root}" \
    --data-dir "${data_dir}" --manifest "${output_root}/manifest.json" \
    --output-dir "${output_root}/artifacts" --config "${config}"
}
run_suite 2>&1 | tee "${output_root}/run.log"
