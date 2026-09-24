#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {lambda025|lambda050} [smoke|2000|10000]" >&2
  exit 2
fi
case "$1" in lambda025|lambda050) ;; *) echo "Unknown pack: $1" >&2; exit 2 ;; esac
mode="${2:-10000}"
case "${mode}" in
  smoke)
    if [[ "$1" != lambda025 ]]; then echo 'Smoke is lambda025 only' >&2; exit 2; fi
    suffix='_smoke32'; args=(--smoke) ;;
  2000) suffix='_check2000'; args=(--target-steps 2000) ;;
  10000) suffix=''; args=(--target-steps 10000) ;;
  *) echo "Unknown mode: ${mode}" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../../.."
export B0_W20_OUTPUT_BASE="${B0_W20_OUTPUT_BASE:-/app/output/cityscapes_b0_ibkd_warmup20_v1${suffix}}"
output="${B0_W20_OUTPUT_BASE}/$1"
mkdir -p "${output}"
if [[ -e "${output}/run.log" && -z "${B0_W20_RESUME_FROM:-}" ]]; then
  echo "Existing run.log: choose a new output or explicitly set B0_W20_RESUME_FROM" >&2
  exit 1
fi
python -u phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.py "$1" "${args[@]}" 2>&1 | tee -a "${output}/run.log"
