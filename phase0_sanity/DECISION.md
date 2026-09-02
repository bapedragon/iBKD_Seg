# Phase 0 Decision

Status: **COMPLETE — HOLD / PROTOCOL REVISION REQUIRED**

Decision date: 2026-09-02

## Verdict

The repository, environment, preserved checkpoints, official asset contract,
feature-extraction path, and metric implementations are ready. The official
Flowers-102 automatic segmentations are not suitable as the sole scientific
ground truth for a semantic-segmentation claim.

- **Phase 1A is allowed only as a pseudo-mask pipeline diagnostic.** It may
  reuse the preserved Flowers Ours/ALG pair, with KD labelled unmatched and
  exploratory. Results must say “automatic-mask decodability,” not semantic
  segmentation.
- **Scientific Phase 1B remains on hold** until a genuine pixel-ground-truth
  route is locked. The current recommendation is Oxford-IIIT Pet trimaps,
  followed by matched Vanilla/ALG/iBKD encoder training on H200.
- No automatic-mask failures may be silently filtered after method results are
  inspected.

## Verified technical contract

- The local reference environment imports Python 3.13.1, PyTorch 2.11.0,
  torchvision 0.26.0, and timm 1.0.27.
- Ours, ALG, and exploratory KD checkpoint byte sizes and SHA-256 values match
  the manifest; metadata and strict DeiT-Ti loading pass for all three.
- Every encoder emits twelve NCHW feature grids ending at
  `[1, 192, 14, 14]` with zero trainable encoder parameters.
- Backpropagation through the 386-parameter `Conv2d(192, 2, 1)` probe produces
  two probe-gradient tensors and zero encoder-gradient tensors.
- Fourteen unit tests pass locally.
- All four official assets match their locked sizes and SHA-256 values.
- Exactly 8,189 images, 8,189 masks, and 8,189 labels are present. The official
  1,020/1,020/6,149 train/validation/test splits are disjoint and cover every
  canonical ID; all paired files decode and have matching dimensions.

## Ground-truth quality gate

The source paper describes the distributed segmentations as outputs of an
automatic iterative colour/shape procedure and explicitly warns that the
segmentation may be imperfect. They are JPEG blue-screen composites, not
human-annotated class-index masks.

The locked conversion recovers alpha from
`composite = alpha * original + (1 - alpha) * RGB(0, 0, 255)` and defines
flower as `alpha >= 0.5`. Across all 8,189 native-resolution pairs:

- 220 masks (2.687%) contain zero foreground: 28 train, 36 validation, and
  156 test;
- 22 masks (0.269%) contain less than 0.5% background: 2 train, 5 validation,
  and 15 test;
- foreground-fraction quantiles at 0/1/5/50/95/99/100% are
  0.000/0.000/0.112/0.342/0.655/0.816/1.000;
- direct original-mask inspection in every split confirms both failure modes.

The first two examples of the all-background failure are image IDs 36 (train)
and 38 (validation); ID 45 demonstrates it in test. IDs 1270 (train), 4246
(validation), and 568 (test) demonstrate the effectively all-foreground mode.

## Next gate

Choose the Phase 1B ground-truth route documented in
`GROUND_TRUTH_OPTIONS.md`. The recommended sequence is:

1. optionally run Flowers Phase 1A locally as an explicitly labelled smoke
   test;
2. use Oxford-IIIT Pet trimaps for the primary frozen spatial probe;
3. train the matched Pet encoders and multi-seed runs on H200;
4. move to PASCAL VOC only if the ground-truth probe shows a stable signal.

Compact committed evidence lives in `reports/`. Full local reports and all
dataset/checkpoint bytes remain Git-ignored.
