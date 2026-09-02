# Local Phase 0 Reports

Audit commands write machine-local JSON files here. Files ending in
`.local.json` are intentionally ignored because they contain local paths and
timestamps.

Stable evidence needed for reproducibility is summarized in `../DECISION.md`.
The committed `checkpoint_audit.summary.json` and
`dataset_audit.summary.json` retain the stable results without local paths or
large per-image ID lists. Official asset SHA-256 values are also locked in
`../../manifests/flowers102.json`.
