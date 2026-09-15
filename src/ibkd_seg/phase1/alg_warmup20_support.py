"""Validation support for the ALG controller warm-up-20 full diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .train_timing import file_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DIAGNOSTIC_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_pet/configs/oxford_iiit_pet_alg_warmup20_diagnostic_v1.json"
)

LOCKED_PROTOCOL_SHA256 = (
    "38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143"
)

DIAGNOSTIC_ID = "oxford_iiit_pet_alg_controller_warmup20_posthoc_v1"

def _validate_config(
    diagnostic: dict[str, Any],
    protocol: dict[str, Any],
    protocol_path: Path,
) -> None:
    checks = {
        "diagnostic_id": diagnostic.get("diagnostic_id") == DIAGNOSTIC_ID,
        "posthoc_status": diagnostic.get("status")
        == "posthoc_diagnostic_does_not_replace_locked_phase1_v1",
        "source_protocol_id": diagnostic.get("source_protocol")
        == protocol.get("protocol_id"),
        "source_protocol_hash_recorded": diagnostic.get(
            "source_protocol_config_sha256"
        )
        == LOCKED_PROTOCOL_SHA256,
        "source_protocol_hash_actual": file_sha256(protocol_path)
        == LOCKED_PROTOCOL_SHA256,
        "canonical_result_retained": diagnostic.get("interpretation", {}).get(
            "canonical_alg_result_is_retained"
        )
        is True,
        "replacement_forbidden": diagnostic.get("interpretation", {}).get(
            "replacement_of_locked_phase1_result_forbidden"
        )
        is True,
    }
    classification = diagnostic.get("classification", {})
    controller = classification.get("controller", {})
    checks.update(
        {
            "method_alg": classification.get("method") == "alg",
            "batch_128": classification.get("batch_size") == 128,
            "full_encoder_seeds": classification.get("encoder_seeds_full")
            == [1, 2, 3],
            "optimizer_lr_warmup_unchanged": classification.get(
                "optimizer_lr_warmup_epochs"
            )
            == 20,
            "controller_kind_alg": controller.get("kind") == "alg",
            "controller_warmup_20": controller.get("warmup_epochs") == 20,
            "controller_beta_unchanged": controller.get("beta_on") == 2.5,
            "controller_threshold_unchanged": controller.get("threshold") == -0.02,
            "controller_windows_unchanged": (
                controller.get("loss_smoothing_window") == 50
                and controller.get("derivative_smoothing_window") == 50
            ),
            "controller_boundary_unchanged": controller.get("stop_boundary")
            == "smoothed_derivative_greater_than_or_equal_to_threshold",
            "controller_equations_unchanged": controller.get("derivative_mode")
            == "alg_paper_equations",
        }
    )
    smoke = diagnostic.get("smoke", {})
    smoke_classification = smoke.get("classification", {})
    smoke_probe = smoke.get("probe", {})
    checks.update(
        {
            "smoke_non_scientific": smoke.get("scientific_result") is False,
            "smoke_selection_forbidden": smoke.get(
                "selection_from_smoke_metrics_forbidden"
            )
            is True,
            "smoke_test_forbidden": smoke.get("official_test_accessed") is False,
            "smoke_classification_seed_1": smoke_classification.get("encoder_seed")
            == 1,
            "smoke_classification_2_epochs": smoke_classification.get(
                "actual_epochs"
            )
            == 2,
            "smoke_probe_seed_1": smoke_probe.get("probe_seeds") == [1],
            "smoke_probe_lr_grid": smoke_probe.get("learning_rates")
            == [0.01, 0.03, 0.1],
            "smoke_probe_2_epochs": smoke_probe.get("epochs_per_candidate") == 2,
            "smoke_probe_test_zero": smoke_probe.get("test_samples") == 0,
        }
    )
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "ALG warm-up-20 diagnostic config failed: " + ", ".join(failures)
        )
