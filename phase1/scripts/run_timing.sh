#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_timing \
  --timing-run \
  --data-dir /app/output/phase1_pet_12way_timing_v1/data \
  --output-dir /app/output/phase1_pet_12way_timing_v1 \
  --num-workers 4 \
  --eval-batch-size 200
