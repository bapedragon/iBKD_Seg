"""Load and validate the result-independent Phase 1 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


class ProtocolError(ValueError):
    """Raised when a Phase 1 config violates the locked contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate_protocol(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == 1, "unsupported protocol schema")
    _require(
        str(config.get("status", "")).startswith("locked_before_results_"),
        "protocol must be locked before results",
    )
    _require(
        config.get("interpretation") == "pseudo_mask_pipeline_diagnostic_only",
        "Phase 1A must remain a pseudo-mask diagnostic",
    )

    dataset = config.get("dataset", {})
    _require(dataset.get("name") == "flowers102", "Phase 1A dataset must be flowers102")
    mask = dataset.get("mask", {})
    _require(mask.get("alpha_threshold") == 0.5, "alpha threshold must remain 0.5")
    _require(mask.get("posthoc_exclusion") == "none", "post-hoc exclusions are forbidden")
    _require(mask.get("postprocessing") == "none", "mask postprocessing is forbidden")
    input_config = dataset.get("input", {})
    _require(input_config.get("size") == 224, "encoder input must be 224 x 224")
    _require(input_config.get("resize") == "direct_square", "resize contract changed")
    _require(input_config.get("random_augmentation") is False, "Phase 1A must be deterministic")

    encoder = config.get("encoder", {})
    _require(encoder.get("architecture") == "deit_tiny_patch16_224", "unexpected encoder")
    _require(encoder.get("strict_load") is True, "strict checkpoint loading is required")
    _require(encoder.get("frozen") is True, "encoder must be frozen")
    _require(encoder.get("eval_mode") is True, "encoder must remain in eval mode")
    feature = encoder.get("feature", {})
    _require(feature.get("block_index") == 11, "primary feature must use block 11")
    _require(feature.get("norm") is False, "Phase 0 locked norm=False")
    _require(
        [feature.get("channels"), feature.get("height"), feature.get("width")]
        == [192, 14, 14],
        "unexpected feature shape",
    )
    _require(feature.get("cache_dtype") == "float32", "feature cache must be float32")

    probe = config.get("probe", {})
    _require(probe.get("parameter_count") == 386, "probe must contain 386 parameters")
    _require(probe.get("loss", {}).get("name") == "cross_entropy", "loss must be CE")
    _require(
        probe.get("loss", {}).get("class_weighting") == "none",
        "primary protocol cannot use class weighting",
    )
    optimizer = probe.get("optimizer", {})
    _require(optimizer.get("name") == "sgd", "probe optimizer must be SGD")
    _require(optimizer.get("weight_decay") == 0.0, "linear probe weight decay must be zero")
    _require(int(probe.get("epochs", 0)) > 0, "epochs must be positive")
    _require(int(probe.get("batch_size", 0)) > 0, "batch size must be positive")
    learning_rates = probe.get("learning_rates", [])
    _require(learning_rates and all(float(value) > 0 for value in learning_rates), "invalid LR grid")
    _require(len(set(learning_rates)) == len(learning_rates), "LR grid contains duplicates")
    seeds = probe.get("seeds", [])
    _require(seeds and all(int(value) >= 0 for value in seeds), "invalid probe seeds")
    _require(len(set(seeds)) == len(seeds), "probe seeds contain duplicates")

    selection = config.get("selection", {})
    _require(selection.get("split") == "validation", "selection must use validation")
    _require(selection.get("metric") == "grid_mean_iou", "selection metric changed")
    _require(
        selection.get("test_policy") == "once_after_validation_selection",
        "test-once policy is required",
    )


def load_protocol(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ProtocolError("protocol root must be a JSON object")
    validate_protocol(config)
    return config


def protocol_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def effective_protocol(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """Return an execution copy; smoke overrides are explicitly non-comparable."""

    effective = copy.deepcopy(config)
    effective["execution_mode"] = "smoke" if smoke else "full"
    if smoke:
        smoke_config = effective["smoke"]
        effective["probe"]["epochs"] = smoke_config["epochs"]
        effective["probe"]["learning_rates"] = smoke_config["learning_rates"]
        effective["probe"]["seeds"] = smoke_config["seeds"]
        effective["non_comparable_smoke_override"] = True
    return effective
