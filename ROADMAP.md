# Research Roadmap

## Phase 0 — Sanity and protocol lock

Validate official Flowers-102 assets, image-mask pairing, official splits,
checkpoint provenance, strict model loading, the frozen feature contract, and
segmentation metrics.

**Recorded outcome (2026-09-02):** the implementation and input contracts pass,
but the Flowers automatic masks fail the ground-truth quality gate. Phase 1A is
limited to a pseudo-mask pipeline diagnostic; scientific Phase 1B remains on
hold until a genuine-ground-truth route is locked. See
`phase0_sanity/DECISION.md`.

## Phase 1 — Frozen spatial probe

Freeze each DeiT-Ti encoder and train the same `Conv2d(192, 2, 1)` head on the
final `14 x 14` feature grid. The matched Ours/ALG pair is primary; KD is an
exploratory, protocol-mismatched baseline. Run three probe-head seeds and report
flower IoU, background IoU, two-class mIoU, and flower Dice.

The distributed Flowers-102 segmentations are outputs of an automatic
segmentation method used by the original classification pipeline, not complete
human ground truth. Consequently Phase 1 is split into two internal steps:

- **Phase 1A — Flowers pseudo-mask diagnostic:** reuse the preserved Flowers
  checkpoints to validate the end-to-end probe pipeline and obtain only a
  historical automatic-mask decodability result.
- **Phase 1B — genuine-ground-truth probe:** use a selected ground-truth route
  (recommended: Oxford-IIIT Pet trimaps; alternatives are a manually verified
  Flowers subset or a standard semantic-segmentation benchmark) for the actual
  scientific feasibility result.

**Exit:** a written Go/Hold/No-Go decision based on matched Ours versus ALG,
non-image baselines, qualitative masks, and evidence from genuine pixel-level
ground truth.

## Phase 2 — Spatial controls

Add mean-mask/center-prior, translation, fixed grid permutation, layer-wise
probes, paired bootstrap confidence intervals, and multiple encoder seeds.

**Exit:** determine whether the Phase 1 signal is spatially meaningful and
reproducible rather than a classifier-quality or dataset-position effect.

## Phase 3 — Shared decoder

Use an identical lightweight decoder for every encoder. Separate frozen,
partial-fine-tuning, and full-fine-tuning regimes. Add boundary evaluation only
after the output resolution is sufficiently high.

## Phase 4 — Standard semantic segmentation

Move to a multi-class benchmark, starting with PASCAL VOC under explicit
data-scarce fractions. Compare matched Vanilla, KD, LG, ALG, iBKD, and selected
segmentation-specific KD baselines.

## Phase 5 — Dense iBKD

Only if earlier phases justify it, design a dense-task-specific method such as
multi-scale grid alignment, boundary-aware guidance, or encoder-decoder feature
transfer. This phase is a new method contribution, not merely an analysis.
