#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 {lambda025|lambda050}" >&2
  exit 2
fi
case "$1" in lambda025|lambda050) ;; *) echo "Unknown pack: $1" >&2; exit 2 ;; esac
cd "$(dirname "$0")/../../.."
output="${B0_W20_OUTPUT_BASE:-/app/output/cityscapes_b0_ibkd_warmup20_v1}/$1"
mkdir -p "${output}"
if [[ -e "${output}/run.log" && -z "${B0_W20_RESUME_FROM:-}" ]]; then
  echo "Existing run.log: choose a new output or explicitly set B0_W20_RESUME_FROM" >&2
  exit 1
fi
python -u phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.py "$1" 2>&1 | tee -a "${output}/run.log"
