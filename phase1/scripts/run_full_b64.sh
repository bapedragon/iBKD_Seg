#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check -e .
python -m ibkd_seg.phase1.run_full \
  --full-suite \
  --batch-size 64 \
  --data-dir /app/output/phase1_pet_full_b64_v1/data \
  --output-dir /app/output/phase1_pet_full_b64_v1 \
  --num-workers 4 \
  --eval-batch-size 200
