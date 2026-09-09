#!/usr/bin/env python3
"""Run a locked CUB ResNet-50/224 guided timing smoke end to end.

This is deliberately non-scientific: a two-epoch scratch ResNet-50 teacher and
four two-epoch guided DeiT-Tiny students are used only to measure runtime,
memory, and artifact plumbing.  The user precommitted the complete six-method
by three-seed v3 matrix, so this smoke also exercises official-test inference.
No smoke metric may change any v3 setting or decide whether a run is performed.
The batch-profile v4 mode reuses the audited scientific teacher and measures
the otherwise identical batch-128 and batch-64 student/probe paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .cub_data import (
    ARCHIVE_MD5,
    DATASET_NAME,
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
    NUM_CLASSES,
    OFFICIAL_TEST_COUNT,
    OFFICIAL_TRAIN_COUNT,
)
from .cub_probe_data import (
    SEGMENTATION_ARCHIVE_MD5,
    CubProbeRecord,
    load_official_test_records,
    load_train_validation_records,
)
from .models import create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .run_cub_combined_smoke import (
    _atomic_json_save,
    _atomic_torch_save,
    _device,
    _feature_cache,
    _finite_metrics,
    _load_json,
    _runtime,
    _target_cache,
)
from .train_timing import file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/cub200_r50_224_b128_guided_smoke_v3.json"
)
BATCH_PROFILE_SMOKE_ID = (
    "cub200_phase1_r50_224_b64_b128_guided_classification_to_probe_smoke_v4"
)
EXPECTED_BATCH_PROFILE_CONFIG_SHA256 = (
    "dd8e61c94f085096fed18615af217dd9b9e35bda0d79bdb19154d168c92ef211"
)
SEED_EXTENSION_SMOKE_ID = (
    "cub200_phase1_r50_224_b128_s23_b64_s2_guided_"
    "classification_to_probe_smoke_v5"
)
EXPECTED_SEED_EXTENSION_CONFIG_SHA256 = (
    "6160cdcb19f2225e574bf9f397b1c99be27c95c006ba7dfc88d9e41344042162"
)
EXPECTED_SEED_EXTENSION_FULL_CONFIG_SHA256 = (
    "f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9"
)
EXPECTED_SCIENTIFIC_TEACHER_SHA256 = (
    "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
)
EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256 = (
    "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
)
EXPECTED_VARIANTS = (
    "lg",
    "alg_warmup20",
    "ibkd_lambda_0.25",
    "ibkd_lambda_0.5",
)
VARIANT_ARGUMENTS: dict[str, tuple[str, float | None, int]] = {
    "lg": ("lg", None, 0),
    "alg_warmup20": ("alg", None, 20),
    "ibkd_lambda_0.25": ("ibkd", 0.25, 20),
    "ibkd_lambda_0.5": ("ibkd", 0.5, 20),
}


def log(message: str = "") -> None:
    print(message, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_full_config(smoke_config: dict[str, Any]) -> Path:
    value = Path(str(smoke_config["full_protocol_config"]))
    path = value if value.is_absolute() else REPOSITORY_ROOT / value
    if not path.is_file():
        raise RuntimeError(f"locked v3 full config is missing: {path}")
    return path


def _validate_configs(
    smoke: dict[str, Any], full: dict[str, Any]
) -> None:
    checks = {
        "smoke_id": smoke.get("smoke_id")
        == "cub200_phase1_r50_224_b128_guided_classification_to_probe_smoke_v3",
        "smoke_non_scientific": smoke.get("scientific_result") is False,
        "selection_forbidden": smoke.get("selection_from_smoke_metrics_forbidden")
        is True,
        "smoke_test_enabled": smoke.get("official_test_accessed") is True,
        "full_locked": smoke.get("full_protocol_status")
        == "locked_unconditional_6_variants_x_3_seeds_v3",
        "dataset": smoke.get("dataset", {}).get("name") == DATASET_NAME,
        "classes": smoke.get("dataset", {}).get("num_classes") == NUM_CLASSES,
        "official_counts": smoke.get("dataset", {}).get("official_counts")
        == {
            "train": OFFICIAL_TRAIN_COUNT,
            "test": OFFICIAL_TEST_COUNT,
            "total": OFFICIAL_TRAIN_COUNT + OFFICIAL_TEST_COUNT,
        },
        "derived_counts": smoke.get("dataset", {}).get("split", {}).get("counts")
        == {
            "train": DERIVED_TRAIN_COUNT,
            "validation": DERIVED_VALIDATION_COUNT,
            "test": OFFICIAL_TEST_COUNT,
        },
        "archive_md5s": smoke.get("dataset", {}).get("image_archive", {}).get(
            "md5"
        )
        == ARCHIVE_MD5
        and smoke.get("dataset", {}).get("segmentation_archive", {}).get("md5")
        == SEGMENTATION_ARCHIVE_MD5,
        "split": smoke.get("dataset", {}).get("split", {}).get(
            "validation_per_class"
        )
        == 3
        and smoke.get("dataset", {}).get("split", {}).get("split_seed") == 2027,
        "teacher": smoke.get("classification", {}).get("teacher", {}).get(
            "architecture"
        )
        == "torchvision_resnet50"
        and smoke.get("classification", {}).get("teacher", {}).get(
            "initialization"
        )
        == "scratch"
        and smoke.get("classification", {}).get("teacher", {}).get("input_size")
        == 224
        and smoke.get("classification", {}).get("teacher", {}).get("batch_size")
        == 128
        and smoke.get("classification", {}).get("teacher", {}).get(
            "planned_epochs"
        )
        == 200,
        "teacher_features": smoke.get("classification", {}).get(
            "teacher", {}
        ).get("feature_channels")
        == [512, 1024, 2048]
        and smoke.get("classification", {}).get("teacher", {}).get(
            "feature_spatial_sizes"
        )
        == [28, 14, 7],
        "teacher_optimizer": smoke.get("classification", {}).get(
            "teacher", {}
        ).get("optimizer")
        == {
            "name": "sgd",
            "learning_rate": 0.05,
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 0.0001,
            "weight_decay_exclusions": ["bias", "normalization_parameters"],
        }
        and smoke.get("classification", {}).get("teacher", {}).get(
            "scheduler", {}
        ).get("warmup_epochs")
        == 5,
        "student": smoke.get("classification", {}).get("student", {}).get(
            "batch_size"
        )
        == 128
        and smoke.get("classification", {}).get("student", {}).get(
            "planned_epochs"
        )
        == 300,
        "variants": tuple(smoke.get("classification", {}).get("variants", ()))
        == EXPECTED_VARIANTS,
        "shared_guided_view": smoke.get("classification", {}).get(
            "guided_teacher_view", {}
        ).get("additional_resize")
        is False
        and smoke.get("classification", {}).get("guided_teacher_view", {}).get(
            "shared_random_geometry_between_student_and_teacher"
        )
        is True,
        "probe_lrs": smoke.get("frozen_probe", {}).get("probe", {}).get(
            "learning_rates"
        )
        == [0.01, 0.03, 0.1],
        "probe_epochs": smoke.get("frozen_probe", {}).get("probe", {}).get(
            "epochs"
        )
        == 2,
        "probe_seed": smoke.get("frozen_probe", {}).get("probe", {}).get(
            "probe_seeds"
        )
        == [1],
        "task_count": smoke.get("task_count")
        == {
            "teacher": 1,
            "classification_students": 4,
            "probe_lr_candidates": 12,
            "selected_smoke_probes": 4,
            "logical_total": 17,
        },
        "full_id": full.get("protocol_id")
        == "cub200_phase1_resnet50_224_scratch_b128_full_v3",
        "full_locked_before_results": str(full.get("status", "")).startswith(
            "locked_before_v3_smoke_and_full_results"
        ),
        "full_unconditional": full.get("matrix_commitment", {}).get(
            "unconditional"
        )
        is True,
        "full_no_changes": full.get("matrix_commitment", {}).get(
            "configuration_changes_after_smoke"
        )
        is False,
        "full_six_by_three": len(
            full.get("matrix_commitment", {}).get("variants", [])
        )
        == 6
        and full.get("matrix_commitment", {}).get("encoder_seeds") == [1, 2, 3],
        "full_teacher_match": full.get("classification", {}).get(
            "teacher", {}
        ).get("architecture")
        == "torchvision_resnet50"
        and full.get("classification", {}).get("teacher", {}).get(
            "initialization"
        )
        == "scratch"
        and full.get("classification", {}).get("teacher", {}).get("epochs")
        == 200,
        "full_student_match": full.get("classification", {}).get(
            "student", {}
        ).get("architecture")
        == "deit_tiny_patch16_224"
        and full.get("classification", {}).get("student", {}).get(
            "train_batch_size"
        )
        == 128
        and full.get("classification", {}).get("student", {}).get("epochs")
        == 300
        and full.get("classification", {}).get("student", {}).get(
            "optimizer", {}
        ).get("learning_rate")
        == 0.0005,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid CUB v3 smoke/full contract: " + ", ".join(failures))


def _validate_batch_profile_configs(
    smoke: dict[str, Any], full: dict[str, Any], *, batch_size: int
) -> None:
    teacher = smoke.get("classification", {}).get("teacher", {})
    student = smoke.get("classification", {}).get("student", {})
    frozen_probe = smoke.get("frozen_probe", {})
    probe = frozen_probe.get("probe", {})
    feature = frozen_probe.get("encoder", {}).get("feature", {})
    target = probe.get("target", {})
    checks = {
        "smoke_id": smoke.get("smoke_id") == BATCH_PROFILE_SMOKE_ID,
        "smoke_non_scientific": smoke.get("scientific_result") is False,
        "selection_forbidden": smoke.get("selection_from_smoke_metrics_forbidden")
        is True
        and smoke.get("batch_or_method_selection_from_smoke_forbidden") is True,
        "official_test_enabled": smoke.get("official_test_accessed") is True,
        "profile_status": smoke.get("full_protocol_status")
        == "batch128_v3_locked_batch64_sensitivity_not_confirmatory",
        "dataset": smoke.get("dataset", {}).get("name") == DATASET_NAME,
        "classes": smoke.get("dataset", {}).get("num_classes") == NUM_CLASSES,
        "official_counts": smoke.get("dataset", {}).get("official_counts")
        == {
            "train": OFFICIAL_TRAIN_COUNT,
            "test": OFFICIAL_TEST_COUNT,
            "total": OFFICIAL_TRAIN_COUNT + OFFICIAL_TEST_COUNT,
        },
        "derived_counts": smoke.get("dataset", {}).get("split", {}).get(
            "counts"
        )
        == {
            "train": DERIVED_TRAIN_COUNT,
            "validation": DERIVED_VALIDATION_COUNT,
            "test": OFFICIAL_TEST_COUNT,
        },
        "archives": smoke.get("dataset", {}).get("image_archive", {}).get("md5")
        == ARCHIVE_MD5
        and smoke.get("dataset", {}).get("segmentation_archive", {}).get("md5")
        == SEGMENTATION_ARCHIVE_MD5,
        "split": smoke.get("dataset", {}).get("split", {}).get(
            "validation_per_class"
        )
        == 3
        and smoke.get("dataset", {}).get("split", {}).get("split_seed") == 2027,
        "teacher_reused": teacher.get("train_in_smoke") is False
        and teacher.get("reuse_h200_issue") == 722
        and teacher.get("architecture") == "torchvision_resnet50"
        and teacher.get("initialization") == "scratch"
        and teacher.get("external_pretraining") is False
        and teacher.get("input_size") == 224
        and teacher.get("training_epochs") == 200
        and teacher.get("checkpoint_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_SHA256
        and teacher.get("model_state_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
        "teacher_features": teacher.get("feature_channels") == [512, 1024, 2048]
        and teacher.get("feature_spatial_sizes") == [28, 14, 7],
        "student": student.get("architecture") == "deit_tiny_patch16_224"
        and student.get("actual_epochs") == 2
        and student.get("planned_epochs") == 300
        and student.get("batch_sizes") == [128, 64]
        and batch_size in student.get("batch_sizes", [])
        and student.get("encoder_seed") == 1,
        "variants": tuple(smoke.get("classification", {}).get("variants", ()))
        == EXPECTED_VARIANTS,
        "shared_guided_view": smoke.get("classification", {}).get(
            "guided_teacher_view", {}
        ).get("additional_resize")
        is False
        and smoke.get("classification", {}).get("guided_teacher_view", {}).get(
            "shared_random_geometry_between_student_and_teacher"
        )
        is True,
        "probe": probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 2
        and probe.get("planned_epochs") == 100
        and probe.get("probe_seeds") == [1]
        and probe.get("planned_probe_seeds") == [1, 2, 3, 4, 5],
        "probe_runtime_schema": feature
        == {
            "block_index": 11,
            "norm": False,
            "exclude_cls_token": True,
            "output_format": "NCHW",
            "channels": 192,
            "height": 14,
            "width": 14,
            "dtype": "float32",
        }
        and target
        == {
            "grid_height": 14,
            "grid_width": 14,
            "foreground_occupancy_threshold": 0.5,
        }
        and probe.get("initialization")
        == {"weight": "normal", "weight_std": 0.01, "bias": 0.0}
        and probe.get("optimizer")
        == {
            "name": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "nesterov": False,
        }
        and probe.get("scheduler")
        == {"name": "cosine", "minimum_learning_rate": 0.0},
        "task_count": smoke.get("task_count")
        == {
            "teacher_download_and_audit": 1,
            "batch_profiles": 2,
            "classification_students_per_batch": 4,
            "probe_lr_candidates_per_batch": 12,
            "logical_gpu_tasks_per_batch": 16,
            "logical_gpu_tasks_total": 32,
        },
        "full_id": full.get("protocol_id")
        == "cub200_phase1_resnet50_224_scratch_b128_full_v3",
        "full_hash_bound": smoke.get("full_protocol_config_sha256")
        == "e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc",
        "full_teacher_match": full.get("classification", {}).get(
            "teacher", {}
        ).get("architecture")
        == "torchvision_resnet50"
        and full.get("classification", {}).get("teacher", {}).get("epochs")
        == 200,
        "full_batch128_remains_primary": full.get("classification", {}).get(
            "student", {}
        ).get("train_batch_size")
        == 128
        and student.get("batch_profile_roles", {}).get("64")
        == "non_scientific_sensitivity_profile_capacity_and_timing_only",
        "full_h200": smoke.get("runtime", {}).get("requested_mig_slices") == 7,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB v4 batch-profile smoke contract: " + ", ".join(failures)
        )


def _validate_seed_extension_configs(
    smoke: dict[str, Any],
    full: dict[str, Any],
    *,
    batch_size: int,
    encoder_seed: int,
) -> None:
    teacher = smoke.get("classification", {}).get("teacher", {})
    student = smoke.get("classification", {}).get("student", {})
    frozen_probe = smoke.get("frozen_probe", {})
    probe = frozen_probe.get("probe", {})
    feature = frozen_probe.get("encoder", {}).get("feature", {})
    target = probe.get("target", {})
    profiles = student.get("batch_seed_profiles", [])
    expected_profiles = [
        {
            "batch_size": 128,
            "encoder_seeds": [2, 3],
            "role": "locked_v3_confirmatory_continuation",
        },
        {
            "batch_size": 64,
            "encoder_seeds": [2],
            "role": "posthoc_exploratory_batch_sensitivity",
        },
    ]
    allowed_pairs = {(128, 2), (128, 3), (64, 2)}
    scope = full.get("result_scope", {})
    provenance = full.get("protocol_provenance", {})
    full_teacher = full.get("teacher", {})
    full_dataset = full.get("dataset", {})
    full_classification = full.get("classification", {})
    full_probe = full.get("frozen_probe", {})
    checks = {
        "requested_profile": (batch_size, encoder_seed) in allowed_pairs,
        "smoke_id": smoke.get("smoke_id") == SEED_EXTENSION_SMOKE_ID,
        "smoke_non_scientific": smoke.get("scientific_result") is False,
        "selection_forbidden": smoke.get("selection_from_smoke_metrics_forbidden")
        is True
        and smoke.get("method_lambda_batch_or_seed_selection_from_smoke_forbidden")
        is True,
        "official_test_enabled": smoke.get("official_test_accessed") is True,
        "profile_status": smoke.get("full_protocol_status")
        == "batch128_confirmatory_seed_extension_and_batch64_posthoc_sensitivity_v5",
        "dataset": smoke.get("dataset", {}).get("name") == DATASET_NAME
        and smoke.get("dataset", {}).get("num_classes") == NUM_CLASSES
        and smoke.get("dataset", {}).get("counts")
        == {
            "train": DERIVED_TRAIN_COUNT,
            "validation": DERIVED_VALIDATION_COUNT,
            "test": OFFICIAL_TEST_COUNT,
        },
        "archives": smoke.get("dataset", {}).get("image_archive", {}).get("md5")
        == ARCHIVE_MD5
        and smoke.get("dataset", {}).get("segmentation_archive", {}).get("md5")
        == SEGMENTATION_ARCHIVE_MD5,
        "split": smoke.get("dataset", {}).get("split", {}).get(
            "validation_per_class"
        )
        == 3
        and smoke.get("dataset", {}).get("split", {}).get("split_seed") == 2027,
        "teacher_reused": teacher.get("train_in_smoke") is False
        and teacher.get("reuse_h200_issue") == 722
        and teacher.get("architecture") == "torchvision_resnet50"
        and teacher.get("initialization") == "scratch"
        and teacher.get("external_pretraining") is False
        and teacher.get("input_size") == 224
        and teacher.get("training_epochs") == 200
        and teacher.get("checkpoint_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_SHA256
        and teacher.get("model_state_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
        "teacher_features": teacher.get("feature_channels") == [512, 1024, 2048]
        and teacher.get("feature_spatial_sizes") == [28, 14, 7],
        "student": student.get("architecture") == "deit_tiny_patch16_224"
        and student.get("actual_epochs") == 2
        and student.get("planned_epochs") == 300
        and profiles == expected_profiles,
        "variants": tuple(smoke.get("classification", {}).get("variants", ()))
        == EXPECTED_VARIANTS,
        "controller": smoke.get("classification", {}).get("controller")
        == {
            "lg": "all_epochs",
            "alg_warmup_epochs": 20,
            "canonical_alg_warmup0_included": False,
            "ibkd_warmup_epochs": 20,
        },
        "shared_guided_view": smoke.get("classification", {}).get(
            "guided_teacher_view", {}
        ).get("additional_resize")
        is False
        and smoke.get("classification", {}).get("guided_teacher_view", {}).get(
            "shared_random_geometry_between_student_and_teacher"
        )
        is True,
        "probe": probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 2
        and probe.get("planned_epochs") == 100
        and probe.get("probe_seeds") == [1]
        and probe.get("planned_probe_seeds") == [1, 2, 3, 4, 5],
        "probe_runtime_schema": feature
        == {
            "block_index": 11,
            "norm": False,
            "exclude_cls_token": True,
            "output_format": "NCHW",
            "channels": 192,
            "height": 14,
            "width": 14,
            "dtype": "float32",
        }
        and target
        == {
            "grid_height": 14,
            "grid_width": 14,
            "foreground_occupancy_threshold": 0.5,
        }
        and probe.get("initialization")
        == {"weight": "normal", "weight_std": 0.01, "bias": 0.0}
        and probe.get("optimizer")
        == {
            "name": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "nesterov": False,
        }
        and probe.get("scheduler")
        == {"name": "cosine", "minimum_learning_rate": 0.0},
        "task_count": smoke.get("task_count")
        == {
            "teacher_download_and_audit": 1,
            "batch_encoder_profiles": 3,
            "classification_students_per_profile": 4,
            "classification_students_total": 12,
            "probe_lr_candidates_per_profile": 12,
            "probe_lr_candidates_total": 36,
            "selected_probes_total": 12,
            "logical_gpu_tasks_per_profile": 16,
            "logical_gpu_tasks_total": 48,
        },
        "full_id": full.get("protocol_id")
        == "cub200_phase1_r50_224_guided_b128_s23_b64_s2_full_v5",
        "full_locked": full.get("status")
        == "locked_after_seed1_results_before_seed_extension_smoke_2026-09-09",
        "full_hash_bound": smoke.get("full_protocol_config_sha256")
        == EXPECTED_SEED_EXTENSION_FULL_CONFIG_SHA256,
        "full_scope": tuple(scope.get("variants", ())) == EXPECTED_VARIANTS
        and scope.get("batch_seed_order")
        == [
            {"batch_size": 128, "encoder_seeds": [2, 3]},
            {"batch_size": 64, "encoder_seeds": [2]},
        ]
        and scope.get("batch64_extension_decided_after_seed1_and_labeled_exploratory")
        is True
        and scope.get("final_six_variant_three_seed_matrix_complete") is False,
        "full_provenance": provenance.get("base_v3", {}).get("sha256")
        == "e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc"
        and provenance.get("seed1_batch_profile_v4", {}).get("sha256")
        == "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633"
        and provenance.get("inherit_training_and_probe_settings_without_change")
        is True,
        "full_teacher": full_teacher.get("checkpoint_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_SHA256
        and full_teacher.get("model_state_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
        "full_dataset": full_dataset.get("name") == DATASET_NAME
        and full_dataset.get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": (
                "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
            ),
        },
        "full_classification": full_classification.get("architecture")
        == "deit_tiny_patch16_224"
        and full_classification.get("epochs") == 300
        and full_classification.get("alg_controller_warmup_epochs") == 20
        and full_classification.get("ibkd_controller_warmup_epochs") == 20
        and full_classification.get("ibkd_fusion_ratio_lambdas") == [0.25, 0.5],
        "full_probe": full_probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and full_probe.get("epochs") == 100
        and full_probe.get("probe_seeds") == [1, 2, 3, 4, 5]
        and full_probe.get("encoder_frozen_eval_strict_load") is True,
        "full_task_count": full.get("task_count")
        == {
            "teacher_download_and_audit": 1,
            "batch_encoder_profiles": 3,
            "classification_students_total": 12,
            "probe_lr_candidates_total": 180,
            "selected_probes_total": 60,
            "classification_official_test_evaluations": 12,
            "probe_official_test_evaluations": 60,
            "retained_new_checkpoints": 72,
        },
        "full_h200": smoke.get("runtime", {}).get("requested_mig_slices") == 7
        and full.get("execution", {}).get("requested_mig_slices") == 7,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB v5 seed-extension smoke contract: "
            + ", ".join(failures)
        )


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    classification_complete: int,
    probe_candidates_complete: int,
    selected_probes_complete: int,
    student_batch_size: int | None = None,
    encoder_seed: int | None = None,
    active_variant: str | None = None,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "classification_complete": classification_complete,
            "classification_expected": 4,
            "probe_candidates_complete": probe_candidates_complete,
            "probe_candidates_expected": 12,
            "selected_probes_complete": selected_probes_complete,
            "selected_probes_expected": 4,
            "student_batch_size": student_batch_size,
            "encoder_seed": encoder_seed,
            "active_variant": active_variant,
            "scientific_result": False,
            "official_test_accessed": True,
            "failure": failure,
        },
        output_dir / "sequence_status.json",
    )


def _run_command(command: list[str], *, label: str) -> None:
    log(f"[CUB_R50_SMOKE_TASK_START] {label} command={' '.join(command)}")
    subprocess.run(command, check=True)
    log(f"[CUB_R50_SMOKE_TASK_DONE] {label}")


def _complete_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete v3 timing task: {path}")
    if payload.get("scientific_result") is not False:
        raise RuntimeError(f"v3 smoke task was marked scientific: {path}")
    if payload.get("official_test_accessed") is not True:
        raise RuntimeError(f"v3 smoke task skipped official test: {path}")
    return payload


def _load_encoder(
    checkpoint_path: Path,
    summary: dict[str, Any],
    *,
    variant: str,
    validation_hash: str,
    batch_size: int = 128,
    encoder_seed: int = 1,
    scientific_teacher: bool = False,
    seed_extension_smoke: bool = False,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    method, _, controller_warmup = VARIANT_ARGUMENTS[variant]
    expected = {
        "purpose": "phase1_cub_r50_224_guided_smoke_student_v3",
        "scientific_result": False,
        "official_test_accessed": True,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": method,
        "batch_size": batch_size,
        "seed": encoder_seed,
        "actual_epochs": 2,
        "planned_epochs": 300,
        "teacher_architecture": "resnet50_224_scratch",
        "controller_warmup_epochs": 20 if method == "alg" else 0,
        "guidance_controller_warmup_epochs": controller_warmup,
        "validation_image_ids_sha256": validation_hash,
        "official_test_evaluations_at_checkpoint_write": 1,
    }
    if scientific_teacher:
        expected.update(
            {
                "teacher_checkpoint_kind": "cub_r50_v3_scientific",
                "teacher_checkpoint_sha256": EXPECTED_SCIENTIFIC_TEACHER_SHA256,
                "teacher_model_state_sha256": (
                    EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256
                ),
            }
        )
    if seed_extension_smoke:
        expected["seed_extension_smoke"] = True
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"CUB v3 student checkpoint mismatch {variant}:{key}: "
                f"expected={value!r} got={metadata.get(key)!r}"
            )
    checkpoint_hash = file_sha256(checkpoint_path)
    if summary.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"CUB v3 checkpoint SHA mismatch: {variant}")
    student = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = student.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict encoder load failed: {variant}")
    state_hash = state_dict_sha256(student)
    if state_hash != metadata.get("student_state_sha256"):
        raise RuntimeError(f"CUB v3 student state hash mismatch: {variant}")
    student.to(device).eval().requires_grad_(False)
    audit = {
        "strict_load": True,
        "eval_mode": not student.training,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in student.parameters()
            if parameter.requires_grad
        ),
        "checkpoint_sha256": checkpoint_hash,
        "student_state_sha256": state_hash,
    }
    if not audit["eval_mode"] or audit["trainable_parameter_count"] != 0:
        raise RuntimeError(f"CUB v3 frozen encoder contract failed: {variant}")
    return student, audit


def _write_csv(
    rows: list[dict[str, Any]], path: Path, fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.num_workers < 0 or args.feature_batch_size <= 0:
        raise ValueError("invalid CUB v3 smoke loader settings")
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUB v3 timing smoke requires CUDA")
    smoke_config = _load_json(args.config)
    full_config_path = _resolve_full_config(smoke_config)
    full_config = _load_json(full_config_path)
    smoke_config_sha256 = file_sha256(args.config)
    full_config_sha256 = file_sha256(full_config_path)
    smoke_id = smoke_config.get("smoke_id")
    batch_profile_mode = smoke_id == BATCH_PROFILE_SMOKE_ID
    seed_extension_mode = smoke_id == SEED_EXTENSION_SMOKE_ID
    reuse_scientific_teacher = batch_profile_mode or seed_extension_mode
    student_batch_size = int(getattr(args, "student_batch_size", 128))
    encoder_seed = int(getattr(args, "encoder_seed", 1))
    supplied_teacher_checkpoint = getattr(args, "teacher_checkpoint", None)
    if reuse_scientific_teacher:
        if student_batch_size not in {64, 128}:
            raise ValueError("CUB reusable-teacher smoke requires batch 64 or 128")
        if supplied_teacher_checkpoint is None:
            raise ValueError("CUB reusable-teacher smoke requires --teacher-checkpoint")
        if seed_extension_mode:
            _validate_seed_extension_configs(
                smoke_config,
                full_config,
                batch_size=student_batch_size,
                encoder_seed=encoder_seed,
            )
            if smoke_config_sha256 != EXPECTED_SEED_EXTENSION_CONFIG_SHA256:
                raise RuntimeError("locked CUB v5 smoke-config SHA-256 mismatch")
            for label in ("base_v3", "seed1_batch_profile_v4"):
                reference = full_config["protocol_provenance"][label]
                reference_value = Path(reference["path"])
                reference_path = (
                    reference_value
                    if reference_value.is_absolute()
                    else REPOSITORY_ROOT / reference_value
                )
                if (
                    not reference_path.is_file()
                    or file_sha256(reference_path) != reference["sha256"]
                ):
                    raise RuntimeError(
                        f"locked CUB v5 referenced protocol mismatch: {label}"
                    )
        else:
            if encoder_seed != 1:
                raise ValueError("CUB v4 batch-profile smoke is fixed to encoder seed 1")
            _validate_batch_profile_configs(
                smoke_config, full_config, batch_size=student_batch_size
            )
            if smoke_config_sha256 != EXPECTED_BATCH_PROFILE_CONFIG_SHA256:
                raise RuntimeError("locked CUB v4 smoke-config SHA-256 mismatch")
        if full_config_sha256 != smoke_config["full_protocol_config_sha256"]:
            raise RuntimeError("locked CUB full-config SHA-256 mismatch")
        release_manifest_value = Path(
            smoke_config["classification"]["teacher"]["release_manifest"]
        )
        release_manifest_path = (
            release_manifest_value
            if release_manifest_value.is_absolute()
            else REPOSITORY_ROOT / release_manifest_value
        )
        if (
            not release_manifest_path.is_file()
            or file_sha256(release_manifest_path)
            != smoke_config["classification"]["teacher"][
                "release_manifest_sha256"
            ]
        ):
            raise RuntimeError("audited CUB v3 teacher release-manifest mismatch")
    else:
        if student_batch_size != 128:
            raise ValueError("legacy CUB v3 smoke is fixed to batch 128")
        if encoder_seed != 1:
            raise ValueError("legacy CUB v3 smoke is fixed to encoder seed 1")
        if supplied_teacher_checkpoint is not None:
            raise ValueError("legacy CUB v3 smoke trains its own timing teacher")
        _validate_configs(smoke_config, full_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        args.output_dir,
        status="running",
        phase="dataset_setup",
        classification_complete=0,
        probe_candidates_complete=0,
        selected_probes_complete=0,
        student_batch_size=student_batch_size,
        encoder_seed=encoder_seed,
    )

    log("=" * 96)
    log("CUB PHASE 1 — RESNET-50/224 GUIDED CLASSIFICATION -> PROBE SMOKE")
    log("=" * 96)
    log(
        "[CUB_R50_SMOKE_POLICY] scientific_result=false "
        "official_test_accessed=true smoke_selection_forbidden=true "
        f"student_batch_size={student_batch_size} "
        f"encoder_seed={encoder_seed} "
        f"teacher_reused={str(reuse_scientific_teacher).lower()} "
        "full_matrix_precommitted=6variants_x3seeds"
    )

    partitions, split_manifest, source = load_train_validation_records(
        args.data_dir, download=True
    )
    official_test, test_source = load_official_test_records(
        args.data_dir, download=True
    )
    records: dict[str, list[CubProbeRecord]] = {
        **partitions,
        "test": official_test,
    }
    counts = {split: len(values) for split, values in records.items()}
    if counts != {"train": 5394, "validation": 600, "test": 5794}:
        raise RuntimeError(f"unexpected CUB v3 counts: {counts}")
    validation_hash = split_manifest["validation_image_ids_sha256"]
    _atomic_json_save(
        {
            "status": "pass",
            "source": source,
            "official_test_source": test_source,
            "split_manifest": split_manifest,
            "counts": counts,
            "image_mask_pairs_present": sum(counts.values()),
            "official_test_images_and_masks_opened": True,
            "scientific_result": False,
        },
        args.output_dir / "data_smoke_audit.json",
    )
    log(
        "[CUB_R50_SMOKE_DATA] train=5394 validation=600 official_test=5794 "
        f"split_sha256={validation_hash}"
    )

    classification_root = args.output_dir / "classification"
    if reuse_scientific_teacher:
        from .run_cub_r50_teacher_full import load_scientific_teacher

        teacher_checkpoint = Path(supplied_teacher_checkpoint)
        teacher_model, teacher_metadata, teacher_hash, teacher_state_hash = (
            load_scientific_teacher(
                teacher_checkpoint,
                device=device,
                validation_hash=validation_hash,
            )
        )
        del teacher_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (
            teacher_hash != EXPECTED_SCIENTIFIC_TEACHER_SHA256
            or teacher_state_hash != EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256
        ):
            raise RuntimeError("audited CUB v3 teacher checkpoint hash mismatch")
        teacher_summary = {
            "status": "complete",
            "source_h200_issue": 722,
            "reused": True,
            "checkpoint_path": str(teacher_checkpoint.resolve()),
            "checkpoint_sha256": teacher_hash,
            "model_state_sha256": teacher_state_hash,
            "metadata": teacher_metadata,
            "strict_loaded_frozen_eval": True,
            "trained_in_smoke": False,
        }
        log(
            "[CUB_R50_TEACHER_REUSED] issue=722 strict_load=true frozen=true "
            f"checkpoint_sha256={teacher_hash} model_state_sha256={teacher_state_hash}"
        )
    else:
        teacher_root = classification_root / "teacher"
        teacher_name = "cub_teacher_resnet50_224_scratch_b128_smoke_2ep_seed1"
        teacher_dir = teacher_root / teacher_name
        teacher_checkpoint = teacher_dir / "timing_teacher_latest.pt"
        teacher_command = [
            sys.executable,
            "-m",
            "ibkd_seg.phase1.train_timing",
            "--timing-run",
            "--dataset",
            "cub",
            "--kind",
            "teacher",
            "--teacher-architecture",
            "resnet50_224_scratch",
            "--access-official-test",
            "--batch-size",
            "128",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(teacher_root),
            "--run-name",
            teacher_name,
            "--num-workers",
            str(args.num_workers),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--seed",
            "1",
        ]
        _run_command(teacher_command, label="teacher_resnet50_224_scratch_seed1")
        teacher_summary = _complete_summary(teacher_dir / "summary.json")
        teacher_contract = teacher_summary.get("teacher_contract", {})
        if (
            teacher_summary.get("dataset") != DATASET_NAME
            or teacher_summary.get("planned_epochs") != 200
            or teacher_contract.get("feature_channels") != [512, 1024, 2048]
            or teacher_summary.get("split_manifest", {}).get(
                "validation_image_ids_sha256"
            )
            != validation_hash
            or not teacher_checkpoint.is_file()
        ):
            raise RuntimeError("CUB v3 teacher contract failed")

    student_root = classification_root / "students"
    classification_rows: list[dict[str, Any]] = []
    for variant in EXPECTED_VARIANTS:
        _write_status(
            args.output_dir,
            status="running",
            phase="classification_students",
            classification_complete=len(classification_rows),
            probe_candidates_complete=0,
            selected_probes_complete=0,
            student_batch_size=student_batch_size,
            encoder_seed=encoder_seed,
            active_variant=variant,
        )
        method, fusion_ratio, controller_warmup = VARIANT_ARGUMENTS[variant]
        run_name = (
            f"cub_r50_224_{variant}_deit_tiny_b{student_batch_size}_"
            f"smoke_2ep_seed{encoder_seed}"
        )
        run_dir = student_root / run_name
        command = [
            sys.executable,
            "-m",
            "ibkd_seg.phase1.train_timing",
            "--timing-run",
            "--dataset",
            "cub",
            "--kind",
            "student",
            "--method",
            method,
            "--teacher-architecture",
            "resnet50_224_scratch",
            "--access-official-test",
            "--batch-size",
            str(student_batch_size),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(student_root),
            "--run-name",
            run_name,
            "--teacher-checkpoint",
            str(teacher_checkpoint),
            "--num-workers",
            str(args.num_workers),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--seed",
            str(encoder_seed),
            "--save-student-checkpoint",
        ]
        if reuse_scientific_teacher:
            command.append("--scientific-cub-r50-teacher")
        if seed_extension_mode:
            command.append("--seed-extension-smoke")
        if fusion_ratio is not None:
            command.extend(["--fusion-ratio", str(fusion_ratio)])
        if method == "alg":
            command.extend(
                ["--alg-controller-warmup-epochs", str(controller_warmup)]
            )
        _run_command(command, label=f"classification_{variant}")
        summary_path = run_dir / "summary.json"
        summary = _complete_summary(summary_path)
        checkpoint_path = run_dir / "timing_student_latest.pt"
        expected = {
            "dataset": DATASET_NAME,
            "method": method,
            "batch_size": student_batch_size,
            "seed": encoder_seed,
            "fusion_ratio_lambda": fusion_ratio,
            "actual_epochs": 2,
            "alg_controller_warmup_epochs": 20 if method == "alg" else 0,
            "guidance_controller_warmup_epochs": controller_warmup,
        }
        if reuse_scientific_teacher:
            expected.update(
                {
                    "teacher_checkpoint_kind": "cub_r50_v3_scientific",
                    "teacher_checkpoint_sha256": (
                        EXPECTED_SCIENTIFIC_TEACHER_SHA256
                    ),
                    "teacher_model_state_sha256": (
                        EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256
                    ),
                }
            )
        if seed_extension_mode:
            expected["seed_extension_smoke"] = True
        for key, value in expected.items():
            if summary.get(key) != value:
                raise RuntimeError(f"CUB v3 classification mismatch {variant}:{key}")
        if (
            summary["split_manifest"]["validation_image_ids_sha256"]
            != validation_hash
            or not checkpoint_path.is_file()
            or summary.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        ):
            raise RuntimeError(f"CUB v3 student artifact failed: {variant}")
        classification_rows.append(
            {
                "variant": variant,
                "method": method,
                "fusion_ratio_lambda": fusion_ratio,
                "controller_warmup_epochs": controller_warmup,
                "summary_path": str(summary_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "summary": summary,
            }
        )

    initial_hashes = {
        row["summary"]["initial_student_state_sha256"]
        for row in classification_rows
    }
    teacher_hashes = {
        row["summary"]["teacher_checkpoint_sha256"]
        for row in classification_rows
    }
    if len(initial_hashes) != 1 or teacher_hashes != {
        teacher_summary["checkpoint_sha256"]
    }:
        raise RuntimeError("CUB v3 paired initialization/shared teacher failed")
    classification_csv_rows: list[dict[str, Any]] = []
    for row in classification_rows:
        final_epoch = row["summary"]["epochs"][-1]
        official = row["summary"]["official_test"]
        classification_csv_rows.append(
            {
                "variant": row["variant"],
                "controller_warmup_epochs": row["controller_warmup_epochs"],
                "validation_macro_top1_epoch2": final_epoch["validation"][
                    "macro_top1"
                ],
                "official_test_macro_top1_epoch2": official["macro_top1"],
                "official_test_overall_top1_epoch2": official["overall_top1"],
                "avg_epoch_seconds": row["summary"]["avg_epoch_seconds"],
                "peak_cuda_memory_bytes": max(
                    int(epoch.get("peak_cuda_memory_bytes") or 0)
                    for epoch in row["summary"]["epochs"]
                ),
                "checkpoint_sha256": row["summary"]["checkpoint_sha256"],
                "scientific_result": False,
            }
        )
    _write_csv(
        classification_csv_rows,
        args.output_dir / "classification_smoke_results.csv",
        tuple(classification_csv_rows[0]),
    )

    target_started = time.monotonic()
    targets: dict[str, dict[str, Any]] = {}
    target_reload_checks: list[bool] = []
    for split in ("train", "validation", "test"):
        targets[split], reloaded = _target_cache(
            records[split],
            split=split,
            config=smoke_config,
            config_sha256=smoke_config_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
        target_reload_checks.append(reloaded)
    target_seconds = time.monotonic() - target_started
    target_values = {
        int(value)
        for payload in targets.values()
        for key in ("input_targets", "grid_targets")
        for value in torch.unique(payload[key])
    }
    if target_values != {0, 1}:
        raise RuntimeError(f"CUB v3 binary target values changed: {target_values}")

    probe_config = smoke_config["frozen_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    probe_seed = int(probe_config["probe_seeds"][0])
    probe_epochs = int(probe_config["epochs"])
    probe_rows: list[dict[str, Any]] = []
    feature_reload_checks: list[bool] = []
    initial_probe_hashes: set[str] = set()
    batch_orders_by_epoch = {
        epoch: set() for epoch in range(1, probe_epochs + 1)
    }
    log(
        "[CUB_R50_PROBE_TASK_COUNT] encoders=4 lr_candidates=12 "
        "epochs_per_candidate=2 probe_seed=1 official_test=true"
    )
    for classification in classification_rows:
        variant = classification["variant"]
        _write_status(
            args.output_dir,
            status="running",
            phase="frozen_probe",
            classification_complete=4,
            probe_candidates_complete=len(probe_rows) * len(learning_rates),
            selected_probes_complete=len(probe_rows),
            student_batch_size=student_batch_size,
            encoder_seed=encoder_seed,
            active_variant=variant,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, encoder_audit = _load_encoder(
            Path(classification["checkpoint_path"]),
            classification["summary"],
            variant=variant,
            validation_hash=validation_hash,
            batch_size=student_batch_size,
            encoder_seed=encoder_seed,
            scientific_teacher=reuse_scientific_teacher,
            seed_extension_smoke=seed_extension_mode,
            device=device,
        )
        feature_started = time.monotonic()
        features: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation", "test"):
            features[split], reloaded = _feature_cache(
                model,
                records[split],
                split=split,
                variant=variant,
                encoder_seed=encoder_seed,
                checkpoint_sha256=encoder_audit["checkpoint_sha256"],
                state_sha256=encoder_audit["student_state_sha256"],
                config=smoke_config,
                config_sha256=smoke_config_sha256,
                cache_path=(
                    args.cache_dir
                    / "features"
                    / f"batch{student_batch_size}_seed{encoder_seed}"
                    / variant
                    / f"{split}.pt"
                ),
                device=device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
            feature_reload_checks.append(reloaded)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        feature_seconds = time.monotonic() - feature_started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        probe_started = time.monotonic()
        candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        for learning_rate in learning_rates:
            state, candidate = train_candidate(
                features["train"]["features"],
                targets["train"]["grid_targets"],
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                probe_config=probe_config,
                learning_rate=learning_rate,
                seed=probe_seed,
                device=device,
                epochs=probe_epochs,
            )
            candidates.append((state, candidate))
            initial_probe_hashes.add(candidate["initial_probe_state_sha256"])
            for epoch, digest in enumerate(
                candidate["batch_order_sha256_by_epoch"], start=1
            ):
                batch_orders_by_epoch[epoch].add(digest)
            log(
                f"[CUB_R50_PROBE_CANDIDATE] variant={variant} "
                f"lr={learning_rate:g} best_epoch={candidate['best_epoch']} "
                "validation_grid_miou="
                f"{candidate['best_validation_grid_mean_iou']:.6f} "
                "scientific_result=false"
            )
        selected_index = max(
            range(len(candidates)),
            key=lambda index: candidates[index][1][
                "best_validation_grid_mean_iou"
            ],
        )
        selected_state, selected = candidates[selected_index]
        probe = probe_from_state(probe_config, probe_seed, selected_state, device)
        validation_metrics, _ = evaluate_probe_both_resolutions(
            probe,
            features["validation"]["features"],
            targets["validation"]["grid_targets"],
            targets["validation"]["input_targets"],
            batch_size=int(probe_config["batch_size"]),
            device=device,
            input_size=int(smoke_config["frozen_probe"]["image_input"]["size"]),
            ignore_index=int(probe_config["loss"]["ignore_index"]),
        )
        test_eval_started = time.monotonic()
        test_metrics, _ = evaluate_probe_both_resolutions(
            probe,
            features["test"]["features"],
            targets["test"]["grid_targets"],
            targets["test"]["input_targets"],
            batch_size=int(probe_config["batch_size"]),
            device=device,
            input_size=int(smoke_config["frozen_probe"]["image_input"]["size"]),
            ignore_index=int(probe_config["loss"]["ignore_index"]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        test_eval_seconds = time.monotonic() - test_eval_started
        probe_seconds = time.monotonic() - probe_started
        probe_path = (
            args.output_dir
            / "probes"
            / f"{variant}_encoder_seed{encoder_seed}_probe_seed1_smoke.pt"
        )
        _atomic_torch_save(
            {
                "purpose": (
                    "phase1_cub_r50_224_seed_extension_frozen_probe_smoke_v5"
                    if seed_extension_mode
                    else (
                        "phase1_cub_r50_224_batch_profile_frozen_probe_smoke_v4"
                        if batch_profile_mode
                        else "phase1_cub_r50_224_frozen_probe_smoke_v3"
                    )
                ),
                "scientific_result": False,
                "official_test_accessed": True,
                "config_sha256": smoke_config_sha256,
                "full_protocol_config_sha256": full_config_sha256,
                "variant": variant,
                "student_batch_size": student_batch_size,
                "encoder_seed": encoder_seed,
                "encoder_checkpoint_sha256": encoder_audit[
                    "checkpoint_sha256"
                ],
                "probe_seed": probe_seed,
                "selection": {
                    "split": "validation",
                    "learning_rate": selected["learning_rate"],
                    "epoch": selected["best_epoch"],
                    "validation_grid_mean_iou": selected[
                        "best_validation_grid_mean_iou"
                    ],
                    "smoke_plumbing_only": True,
                },
                "official_test_evaluations": 1,
                "model": selected_state,
            },
            probe_path,
        )
        saved_probe = torch.load(probe_path, map_location="cpu", weights_only=True)
        strict_probe = probe_from_state(
            probe_config, probe_seed, saved_probe["model"], device
        )
        del strict_probe, probe
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        result = {
            "variant": variant,
            "method": classification["method"],
            "fusion_ratio_lambda": classification["fusion_ratio_lambda"],
            "encoder_seed": encoder_seed,
            "encoder_audit": encoder_audit,
            "probe_seed": probe_seed,
            "candidates": [candidate for _, candidate in candidates],
            "selection": {
                "split": "validation",
                "learning_rate": selected["learning_rate"],
                "epoch": selected["best_epoch"],
                "validation_grid_mean_iou": selected[
                    "best_validation_grid_mean_iou"
                ],
                "smoke_plumbing_only": True,
            },
            "validation": validation_metrics,
            "official_test": test_metrics,
            "official_test_evaluations": 1,
            "probe_artifact": str(probe_path.resolve()),
            "probe_artifact_sha256": file_sha256(probe_path),
            "selected_probe_strict_reloaded": True,
            "timing": {
                "feature_cache_seconds": feature_seconds,
                "probe_training_and_validation_seconds": probe_seconds
                - test_eval_seconds,
                "official_test_evaluation_seconds": test_eval_seconds,
            },
            "peak_cuda_memory_bytes": peak_memory,
            "scientific_result": False,
        }
        if not all(
            _finite_metrics(metrics)
            for metrics in (*validation_metrics.values(), *test_metrics.values())
        ):
            raise RuntimeError(f"non-finite CUB v3 probe metrics: {variant}")
        probe_rows.append(result)
        del features, candidates
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probe_csv_rows: list[dict[str, Any]] = []
    for row in probe_rows:
        probe_csv_rows.append(
            {
                "variant": row["variant"],
                "selected_learning_rate": row["selection"]["learning_rate"],
                "selected_epoch": row["selection"]["epoch"],
                "validation_input_224_mean_iou": row["validation"]["input_224"][
                    "mean_iou"
                ],
                "official_test_input_224_mean_iou": row["official_test"][
                    "input_224"
                ]["mean_iou"],
                "official_test_input_224_foreground_iou": row["official_test"][
                    "input_224"
                ]["foreground_iou"],
                "feature_cache_seconds": row["timing"]["feature_cache_seconds"],
                "probe_training_and_validation_seconds": row["timing"][
                    "probe_training_and_validation_seconds"
                ],
                "official_test_evaluation_seconds": row["timing"][
                    "official_test_evaluation_seconds"
                ],
                "peak_cuda_memory_bytes": row["peak_cuda_memory_bytes"],
                "scientific_result": False,
            }
        )
    _write_csv(
        probe_csv_rows,
        args.output_dir / "probe_smoke_results.csv",
        tuple(probe_csv_rows[0]),
    )

    classification_estimate_seconds = sum(
        float(row["summary"]["avg_epoch_seconds"]) * 300 * 3
        + float(row["summary"]["official_test_seconds"]) * 3
        for row in classification_rows
    )
    teacher_estimate_seconds = (
        0.0
        if reuse_scientific_teacher
        else float(teacher_summary["avg_epoch_seconds"]) * 200
        + float(teacher_summary["official_test_seconds"])
    )
    probe_training_estimate_seconds = sum(
        float(row["timing"]["probe_training_and_validation_seconds"])
        * (int(probe_config["planned_epochs"]) / probe_epochs)
        * 15
        for row in probe_rows
    )
    probe_feature_estimate_seconds = sum(
        float(row["timing"]["feature_cache_seconds"]) * 3
        for row in probe_rows
    )
    probe_test_estimate_seconds = sum(
        float(row["timing"]["official_test_evaluation_seconds"]) * 15
        for row in probe_rows
    )
    probe_feature_and_test_estimate_seconds = (
        probe_feature_estimate_seconds + probe_test_estimate_seconds
    )
    guided_full_estimate_seconds = (
        teacher_estimate_seconds
        + classification_estimate_seconds
        + probe_training_estimate_seconds
        + probe_feature_and_test_estimate_seconds
    )
    one_seed_classification_estimate_seconds = sum(
        float(row["summary"]["avg_epoch_seconds"]) * 300
        + float(row["summary"]["official_test_seconds"])
        for row in classification_rows
    )
    one_encoder_probe_estimate_seconds = sum(
        float(row["timing"]["probe_training_and_validation_seconds"])
        * (int(probe_config["planned_epochs"]) / probe_epochs)
        * 5
        + float(row["timing"]["feature_cache_seconds"])
        + float(row["timing"]["official_test_evaluation_seconds"]) * 5
        for row in probe_rows
    )
    one_seed_four_variant_estimate_seconds = (
        one_seed_classification_estimate_seconds
        + one_encoder_probe_estimate_seconds
    )
    peak_by_variant = {
        row["variant"]: max(
            int(epoch.get("peak_cuda_memory_bytes") or 0)
            for epoch in row["summary"]["epochs"]
        )
        for row in classification_rows
    }
    reserved_by_variant = {
        row["variant"]: max(
            int(epoch.get("peak_cuda_memory_reserved_bytes") or 0)
            for epoch in row["summary"]["epochs"]
        )
        for row in classification_rows
    }
    teacher_peak = (
        0
        if reuse_scientific_teacher
        else max(
            int(epoch.get("peak_cuda_memory_bytes") or 0)
            for epoch in teacher_summary["epochs"]
        )
    )
    teacher_reserved = (
        0
        if reuse_scientific_teacher
        else max(
            int(epoch.get("peak_cuda_memory_reserved_bytes") or 0)
            for epoch in teacher_summary["epochs"]
        )
    )
    contracts = {
        "config_validated": True,
        "non_scientific": True,
        "official_test_accessed_by_explicit_precommitment": True,
        "full_matrix_locked_six_by_three": True,
        "classification_and_probe_split_match": all(
            row["summary"]["split_manifest"]["validation_image_ids_sha256"]
            == validation_hash
            for row in classification_rows
        ),
        "teacher_completed": teacher_summary["status"] == "complete",
        "scientific_teacher_reuse_contract": (
            teacher_summary.get("checkpoint_sha256")
            == EXPECTED_SCIENTIFIC_TEACHER_SHA256
            and teacher_summary.get("model_state_sha256")
            == EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256
            if reuse_scientific_teacher
            else True
        ),
        "four_guided_students_completed": len(classification_rows) == 4,
        "same_initial_student_state_across_variants": len(initial_hashes) == 1,
        "one_teacher_shared_by_all_guided_students": len(teacher_hashes) == 1,
        "binary_masks_only": target_values == {0, 1},
        "target_cache_safe_reload": all(target_reload_checks),
        "encoder_strict_loaded_frozen_eval": all(
            row["encoder_audit"]["strict_load"]
            and row["encoder_audit"]["eval_mode"]
            and row["encoder_audit"]["trainable_parameter_count"] == 0
            for row in probe_rows
        ),
        "feature_cache_safe_reload": all(feature_reload_checks),
        "twelve_probe_lr_candidates_completed": len(probe_rows) * 3 == 12,
        "same_probe_initial_state_across_all_candidates": len(
            initial_probe_hashes
        )
        == 1,
        "same_probe_batch_order_across_all_candidates": all(
            len(values) == 1 for values in batch_orders_by_epoch.values()
        ),
        "four_selected_probes_strict_reloaded": len(probe_rows) == 4
        and all(row["selected_probe_strict_reloaded"] for row in probe_rows),
        "one_official_test_eval_per_selected_probe": all(
            row["official_test_evaluations"] == 1 for row in probe_rows
        ),
        "finite_validation_and_test_metrics": all(
            all(
                _finite_metrics(metrics)
                for metrics in (
                    *row["validation"].values(),
                    *row["official_test"].values(),
                )
            )
            for row in probe_rows
        ),
    }
    status = "pass" if all(contracts.values()) else "fail"
    elapsed_seconds = time.monotonic() - started
    summary = {
        "status": status,
        "completed_at_utc": _utc_now(),
        "smoke_id": smoke_config["smoke_id"],
        "batch_profile_mode": batch_profile_mode,
        "seed_extension_mode": seed_extension_mode,
        "student_batch_size": student_batch_size,
        "encoder_seed": encoder_seed,
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": True,
        "full_protocol_status": smoke_config["full_protocol_status"],
        "config": {"path": str(args.config), "sha256": smoke_config_sha256},
        "full_protocol_config": {
            "path": str(full_config_path),
            "sha256": full_config_sha256,
        },
        "data": {
            "source": source,
            "official_test_source": test_source,
            "counts": counts,
            "validation_image_ids_sha256": validation_hash,
            "target_preparation_seconds": target_seconds,
        },
        "teacher": teacher_summary,
        "classification": classification_rows,
        "frozen_probe": probe_rows,
        "contracts": {"all_passed": status == "pass", **contracts},
        "capacity": {
            "teacher_peak_cuda_memory_bytes": teacher_peak,
            "teacher_peak_cuda_memory_reserved_bytes": teacher_reserved,
            "peak_cuda_memory_bytes_by_student": peak_by_variant,
            "peak_cuda_memory_reserved_bytes_by_student": reserved_by_variant,
            "largest_peak_cuda_memory_bytes": max(
                teacher_peak, *peak_by_variant.values()
            ),
            "largest_peak_cuda_memory_reserved_bytes": max(
                teacher_reserved, *reserved_by_variant.values()
            ),
            "requested_batch_size": student_batch_size,
            "precision": "float32",
        },
        "timing": {
            "smoke_suite_seconds": elapsed_seconds,
            "rough_guided_block_full_estimate": {
                "teacher_1x200ep_seconds": teacher_estimate_seconds,
                "classification_4variants_x3seeds_x300ep_seconds": (
                    classification_estimate_seconds
                ),
                "probe_training_4variants_x3encoders_x5probe_seeds_x3lr_x100ep_seconds": (
                    probe_training_estimate_seconds
                ),
                "probe_feature_cache_seconds": probe_feature_estimate_seconds,
                "probe_official_test_seconds": probe_test_estimate_seconds,
                "total_seconds": guided_full_estimate_seconds,
                "warning": "linear_extrapolation_for_job_partitioning_only",
            },
            "rough_one_encoder_seed_four_variant_estimate": {
                "teacher_seconds": (
                    0.0 if reuse_scientific_teacher else teacher_estimate_seconds
                ),
                "classification_4variants_x1seed_x300ep_seconds": (
                    one_seed_classification_estimate_seconds
                ),
                "probe_4encoders_x5probe_seeds_x3lr_x100ep_seconds": (
                    one_encoder_probe_estimate_seconds
                ),
                "total_seconds": one_seed_four_variant_estimate_seconds
                + (0.0 if reuse_scientific_teacher else teacher_estimate_seconds),
                "warning": "linear_extrapolation_for_job_partitioning_only",
            },
        },
        "runtime": _runtime(device),
    }
    summary_path = args.output_dir / "combined_smoke_summary.json"
    _atomic_json_save(summary, summary_path)
    _write_status(
        args.output_dir,
        status=status,
        phase="complete",
        classification_complete=len(classification_rows),
        probe_candidates_complete=len(probe_rows) * 3,
        selected_probes_complete=len(probe_rows),
        student_batch_size=student_batch_size,
        encoder_seed=encoder_seed,
    )

    log("[CUB_R50_SMOKE_FINAL_CLASSIFICATION_RESULTS]")
    for row in classification_rows:
        final_epoch = row["summary"]["epochs"][-1]
        official = row["summary"]["official_test"]
        log(
            f"[CUB_R50_CLASSIFICATION_SMOKE_RESULT] variant={row['variant']} "
            f"batch={student_batch_size} "
            f"encoder_seed={encoder_seed} "
            f"epoch2_val_macro_top1={final_epoch['validation']['macro_top1']:.4f} "
            f"epoch2_test_macro_top1={official['macro_top1']:.4f} "
            f"avg_epoch_seconds={row['summary']['avg_epoch_seconds']:.3f} "
            f"peak_cuda_bytes={peak_by_variant[row['variant']]} "
            "peak_cuda_reserved_bytes="
            f"{reserved_by_variant[row['variant']]} "
            "scientific_result=false"
        )
    log("[CUB_R50_SMOKE_FINAL_PROBE_RESULTS]")
    for row in probe_rows:
        log(
            f"[CUB_R50_PROBE_SMOKE_RESULT] variant={row['variant']} "
            f"encoder_batch={student_batch_size} "
            f"encoder_seed={encoder_seed} "
            f"selected_lr={row['selection']['learning_rate']:g} "
            f"selected_epoch={row['selection']['epoch']} "
            "validation_input_miou="
            f"{row['validation']['input_224']['mean_iou']:.6f} "
            "official_test_input_miou="
            f"{row['official_test']['input_224']['mean_iou']:.6f} "
            "scientific_result=false"
        )
    log(
        "[CUB_R50_ROUGH_GUIDED_FULL_ESTIMATE] total="
        f"{format_duration(guided_full_estimate_seconds)} "
        f"teacher={format_duration(teacher_estimate_seconds)} "
        f"classification={format_duration(classification_estimate_seconds)} "
        "probe="
        f"{format_duration(probe_training_estimate_seconds + probe_feature_and_test_estimate_seconds)} "
        "linear_extrapolation_only=true"
    )
    log(
        "[CUB_R50_ROUGH_ONE_SEED_FOUR_VARIANT_ESTIMATE] "
        f"batch={student_batch_size} "
        f"encoder_seed={encoder_seed} "
        f"total={format_duration(one_seed_four_variant_estimate_seconds)} "
        f"classification={format_duration(one_seed_classification_estimate_seconds)} "
        f"probe={format_duration(one_encoder_probe_estimate_seconds)} "
        "teacher_reused=true linear_extrapolation_only=true"
        if reuse_scientific_teacher
        else "[CUB_R50_ROUGH_ONE_SEED_FOUR_VARIANT_ESTIMATE] not_applicable=true"
    )
    log(
        f"[CUB_R50_GUIDED_SMOKE_DONE] status={status} "
        f"batch={student_batch_size} encoder_seed={encoder_seed} "
        f"teacher={'reused' if reuse_scientific_teacher else '1/1'} "
        f"classification={len(classification_rows)}/4 "
        f"probe_candidates={len(probe_rows) * 3}/12 "
        f"selected_probes={len(probe_rows)}/4 "
        f"tasks={'16/16' if reuse_scientific_teacher else '17/17'} "
        "scientific_result=false official_test_accessed=true "
        f"seconds={elapsed_seconds:.2f} summary={summary_path.resolve()}"
    )
    if status != "pass":
        raise RuntimeError("CUB guided smoke failed one or more contracts")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--student-batch-size",
        type=int,
        choices=(64, 128),
        default=128,
        help="Student batch for the v4 batch-profile smoke; v3 remains fixed to 128.",
    )
    parser.add_argument(
        "--encoder-seed",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Student encoder seed; v3/v4 are fixed to 1 and v5 validates its batch/seed plan.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Audited issue-722 teacher checkpoint required by v4 batch-profile smoke.",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            previous = _load_json(args.output_dir / "sequence_status.json")
        except Exception:
            previous = {}
        _write_status(
            args.output_dir,
            status="failed",
            phase="failed",
            classification_complete=int(
                previous.get("classification_complete", 0)
            ),
            probe_candidates_complete=int(
                previous.get("probe_candidates_complete", 0)
            ),
            selected_probes_complete=int(
                previous.get("selected_probes_complete", 0)
            ),
            student_batch_size=getattr(args, "student_batch_size", None),
            encoder_seed=getattr(args, "encoder_seed", None),
            active_variant=previous.get("active_variant"),
            failure=f"{type(error).__name__}: {error}",
        )
        log(f"[CUB_R50_GUIDED_SMOKE_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
