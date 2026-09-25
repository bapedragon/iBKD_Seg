#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 ]]; then echo "usage: $0 [fresh|resume]" >&2; exit 2; fi
mode="${1:-fresh}"
case "${mode}" in fresh|resume) ;; *) echo "Unknown mode: ${mode}" >&2; exit 2 ;; esac
cd "$(dirname "$0")/../../.."
export B0_10K_OUTPUT="${B0_10K_OUTPUT:-/app/output/cityscapes_b0_top1_10k_v1}"
if [[ -d "${B0_10K_OUTPUT}" && -n "$(ls -A "${B0_10K_OUTPUT}")" && "${B0_10K_RESUME_FROM:-}" != "${B0_10K_OUTPUT}" ]]; then
  echo 'Nonempty output: choose a new B0_10K_OUTPUT or explicitly resume this group in place' >&2
  exit 1
fi
mkdir -p "${B0_10K_OUTPUT}"
python -u phase4/Cityscapes_SegFormer-B0/scripts/run_b0_top1_10k.py --start "${mode}" 2>&1 | tee -a "${B0_10K_OUTPUT}/run.log"
