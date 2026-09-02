# Ground-Truth Route Decision

Status: **DECISION REQUIRED — choose before scientific Phase 1B runs**

## Finding that forces this decision

Section 2 of the original Flowers-102 classification paper states that the
authors *automatically segment each image* using an iterative colour/shape
method and explicitly notes that an initial segmentation may be imperfect:

- https://www.robots.ox.ac.uk/~men/papers/nilsback_icvgip08.pdf

The distributed `segmim_*.jpg` files are therefore algorithmic blue-screen
outputs, not exhaustive human semantic-segmentation ground truth. The complete
native-resolution audit found 220 all-background masks and 22 with less than
0.5% background; failures occur in every official split.

## Option A — Oxford-IIIT Pet ground-truth trimaps (recommended)

Train matched data-scarce classification encoders on the 37 pet breeds, then
apply the same frozen spatial probe to the official pixel-level trimaps.

- Official dataset page: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Strength: breed labels and genuine pixel-level foreground/background trimaps
  are associated with every image.
- Strength: dataset scale and fine-grained classification are close to the
  original iBKD setting.
- Cost: matched Vanilla/ALG/iBKD encoders must be trained, preferably on H200.

## Option B — Manually verified Flowers subset

Retain the existing Flowers Ours/ALG checkpoints but create a held-out,
human-reviewed mask subset.

- Strength: directly reuses the current matched checkpoints.
- Cost: annotation and quality-control effort; subset size may be too small for
  a strong paper claim.
- Requirement: automatic masks may initialize annotation, but every included
  mask must be corrected and independently reviewed.

## Option C — Flowers automatic-mask diagnostic only

Run the existing frozen probe against the automatic masks solely to validate
the code path and compare spatial decodability with respect to that historical
preprocessing output.

- Strength: cheapest and immediately reuses Ours/ALG/KD checkpoints.
- Limitation: cannot be called ground-truth semantic-segmentation evaluation.
- Limitation: empty and poor masks require explicit reporting, not silent
  filtering chosen after looking at method results.

## Option D — Direct standard semantic segmentation

Move immediately to a benchmark such as PASCAL VOC and train segmentation
models with matched distillation protocols.

- Strength: strongest external validity.
- Cost: skips the cheap representation probe and needs substantially more H200
  experimentation and method integration.

## Recommended sequence

1. Preserve the complete Flowers asset/mask-quality audit as Phase 0 evidence.
2. Use Flowers automatic masks only for a pipeline smoke test, clearly labelled
   as pseudo-mask evaluation.
3. Make Oxford-IIIT Pet the primary frozen-representation feasibility test.
4. Move to PASCAL VOC only after the ground-truth probe yields a stable signal.
