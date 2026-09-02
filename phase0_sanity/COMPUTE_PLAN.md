# Local and H200 Compute Plan

## Local workstation

Use the local machine for:

- all Phase 0 integrity, metadata, mask, and metric audits;
- repository tests and smoke runs;
- qualitative inspection and report generation;
- Phase 1 feature caching and small frozen-probe experiments.

Observed on the current Apple M2 Max / 32 GB machine:

- all fourteen repository tests complete in less than one second;
- deep audit of all three 22 MB checkpoints completes in a few seconds;
- native-resolution audit of all 8,189 image-mask pairs completes locally in
  roughly two minutes;
- a synthetic batch-8 DeiT-Ti twelve-block feature-extraction benchmark runs
  at roughly 272 images/second on CPU.

The synthetic throughput excludes JPEG decoding and transforms, but it confirms
that Phase 0 is comfortably local and that a cached-feature Phase 1 pilot is
also practical. PyTorch is MPS-enabled but MPS is unavailable inside the
current Codex execution session, so the plan must remain valid on CPU.

## H200

Use the H200 for:

- retraining Vanilla/ALG/iBKD classification encoders on a dataset with genuine
  pixel ground truth;
- multiple encoder-training seeds;
- repeated end-to-end decoder fine-tuning;
- standard multi-class semantic-segmentation benchmarks;
- direct dense teacher-student distillation.

H200 runs must consume the same manifests and configs validated locally and
must write a run summary containing the Git commit, config, seeds, dependency
versions, checkpoint hashes, and dataset hashes.
