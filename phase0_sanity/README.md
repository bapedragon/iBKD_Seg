# Phase 0 — Sanity and Protocol Lock

## Objective

Establish that the official dataset, preserved classification checkpoints,
feature extraction path, and evaluation metrics are trustworthy before any
segmentation head is trained.

See [COMPUTE_PLAN.md](COMPUTE_PLAN.md) for the local/H200 execution boundary.
The required ground-truth choice is compared in
[GROUND_TRUTH_OPTIONS.md](GROUND_TRUTH_OPTIONS.md).

## Inputs

- Official Oxford Flowers-102 images
- Official Oxford Flowers-102 image segmentations
- `imagelabels.mat` and `setid.mat`
- Hash-identified Ours/ALG matched checkpoints
- Hash-identified KD exploratory checkpoint

## Step 0.1 — Environment

The reference dependency contract is recorded in `requirements.txt`. A
dependency-compatible environment must import PyTorch, torchvision, timm,
NumPy, Pillow, and SciPy.

## Step 0.2 — Checkpoint audit

Run with the directory that contains `results/Ours`, `results/ALG`, and
`results/KD`:

```bash
PYTHONPATH=src python -m ibkd_seg.phase0.checkpoints \
  --manifest manifests/checkpoints.json \
  --source-root /path/to/IBAM_KD_H200_V2 \
  --output phase0_sanity/reports/checkpoint_audit.local.json
```

The deep audit checks bytes, metadata, strict `state_dict` loading, twelve
NCHW intermediate features, a final `[1, 192, 14, 14]` grid, and a completely
frozen encoder. It also backpropagates through the 386-parameter Phase 1 probe
contract and verifies that only the probe receives gradients.

## Step 0.3 — Download and extract official data

The download is approximately 548 MB before extraction.

```bash
bash phase0_sanity/scripts/download_flowers102.sh
```

The default local destination is `data/flowers102/`, which is ignored by Git.
The official server can be slow, so the downloader resumes an existing prefix
and uses eight verified HTTP byte ranges by default. Override with
`IBKD_SEG_DOWNLOAD_CONNECTIONS` if necessary.

## Step 0.4 — Dataset audit

```bash
PYTHONPATH=src python -m ibkd_seg.phase0.flowers_data \
  --data-root data/flowers102 \
  --manifest manifests/flowers102.json \
  --output phase0_sanity/reports/dataset_audit.local.json
```

This verifies archive sizes and records their local SHA-256 values, counts,
one-to-one image-mask pairing, canonical IDs, official split disjointness and
coverage, label/class IDs, and image-mask dimensions.

The official segmentations are JPEG blue-screen composites, not class-index
PNG files. More importantly, the source paper describes them as outputs of its
automatic flower-segmentation scheme, so they must not be presented as complete
human-annotated semantic-segmentation ground truth. The locked conversion
estimates alpha from `composite = alpha * original + (1 - alpha) *
RGB(0, 0, 255)` and marks `alpha >= 0.5` as flower. The audit reports
transition-pixel and foreground statistics from an evenly spaced sample so the
rule remains inspectable.

The complete audit also screens every mask at native resolution and lists
empty, near-empty, and near-full outputs. Any empty automatic segmentation
blocks use of the unfiltered set as segmentation ground truth.

On the official files the command intentionally exits with status 1: all
structural checks pass, while 220 masks contain no foreground and 22 contain
less than 0.5% background. This is a research-gate result, not a download or
decoder failure. The compact committed evidence is in
`reports/dataset_audit.summary.json`; the full ID lists remain in the ignored
local report.

## Step 0.5 — Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

After data preparation, all Phase 0 checks can be rerun with one command:

```bash
IBKD_SEG_PYTHON=/path/to/python \
  bash phase0_sanity/scripts/run_phase0_audits.sh \
  /path/to/IBAM_KD_H200_V2 \
  data/flowers102
```

## Completion checklist

- [x] Dependency contract recorded and importable
- [x] All checkpoint hashes and sizes match
- [x] All checkpoint metadata matches the manifest
- [x] All three DeiT-Ti state dictionaries strict-load
- [x] Final frozen feature shape is `B x 192 x 14 x 14`
- [x] Official assets downloaded and local hashes recorded
- [x] Exactly 8,189 image-mask pairs found
- [x] Official train/val/test IDs are disjoint and cover all images
- [x] Every image and mask has matching dimensions
- [x] Binary mask conversion rule documented from actual mask inspection
- [x] Automatic/pseudo-mask status disclosed and genuine-ground-truth options recorded
- [x] Segmentation metric unit tests pass
- [x] Phase 1A input paths and known leakage caveat documented

Phase 0 has produced a **HOLD / protocol-revision** decision. Before scientific
Phase 1B starts, lock one option from `GROUND_TRUTH_OPTIONS.md`; Oxford-IIIT Pet
trimaps are recommended.
