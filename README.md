# iBKD Segmentation Extension

This repository tests whether the spatial representations learned by iBKD
extend from data-scarce image classification to dense prediction.

The project uses explicit phase gates. A later phase starts only after the
current phase has produced its required artifacts and a written decision.

## Current status

**Phase 0 audited — HOLD on using Flowers masks as scientific ground truth.**

The code, checkpoints, official files, splits, shapes, and metrics are valid.
The data audit nevertheless found 220 all-background and 22 effectively
all-foreground automatic Flowers masks. Phase 1A may use these masks only as a
labelled pipeline diagnostic; Phase 1B must use genuine pixel ground truth for
the scientific frozen spatial probe.

| Phase | Question | Status |
|---|---|---|
| 0 | Are the inputs and evaluation contracts trustworthy? | Audited / Hold |
| 1 | Is a dense mask more decodable from frozen iBKD features? | Pending |
| 2 | Is any gain robust to spatial controls and encoder seeds? | Pending |
| 3 | Does the signal survive a stronger shared decoder and fine-tuning? | Pending |
| 4 | Does it generalize to multi-class semantic segmentation? | Pending |
| 5 | Is a dense-task-specific iBKD extension justified? | Pending |

See [ROADMAP.md](ROADMAP.md) for the complete gates.

## Repository policy

- `src/` contains implementation shared by multiple phases.
- Each `phase*_*/` directory owns its protocol, commands, reports, and gate.
- Dataset archives, extracted data, checkpoints, and raw runs are not tracked.
- Every consumed checkpoint is identified by SHA-256 and expected metadata.
- Paper table values and checkpoint reproduction values are never mixed.

## Phase 0 quick start

The reference model environment is Python 3.10+, PyTorch 2.11.0,
torchvision 0.26.0, and timm 1.0.27.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run repository unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Audit the locally preserved Flowers-102 checkpoints:

```bash
PYTHONPATH=src python -m ibkd_seg.phase0.checkpoints \
  --manifest manifests/checkpoints.json \
  --source-root /path/to/IBAM_KD_H200_V2 \
  --output phase0_sanity/reports/checkpoint_audit.local.json
```

Dataset preparation and audit commands are documented in
[phase0_sanity/README.md](phase0_sanity/README.md).
The local-versus-H200 split is recorded in
[phase0_sanity/COMPUTE_PLAN.md](phase0_sanity/COMPUTE_PLAN.md).
The evidence-backed gate decision is in
[phase0_sanity/DECISION.md](phase0_sanity/DECISION.md).
