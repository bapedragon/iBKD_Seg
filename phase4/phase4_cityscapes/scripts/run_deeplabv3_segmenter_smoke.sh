#!/usr/bin/env bash
set -euo pipefail

output_root="${CITYSCAPES_ARCH_SMOKE_OUTPUT_DIR:-/app/output/cityscapes_deeplabv3_segmenter_smoke_v1}"
mkdir -p "${output_root}"

run_suite() {
  python -m pip install --disable-pip-version-check -e .
  python -m ibkd_seg.cityscapes.public_smoke \
    --device cuda --output-dir "${output_root}/artifacts"
}

run_suite 2>&1 | tee "${output_root}/run.log"
