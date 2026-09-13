#!/usr/bin/env python3
"""Run locked CUB R50/224 guided batch/seed profiles through frozen probes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .cub_data import (
    ARCHIVE_BYTES,
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    DATASET_NAME,
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
    NUM_CLASSES,
    OFFICIAL_TEST_COUNT,
    build_official_test_loader,
    file_digest,
)
from .cub_probe_data import (
    SEGMENTATION_ARCHIVE_BYTES,
    SEGMENTATION_ARCHIVE_MD5,
    SEGMENTATION_ARCHIVE_NAME,
    SEGMENTATION_ARCHIVE_SHA256,
    CubProbeRecord,
    load_official_test_records,
    load_train_validation_records,
)
from .cub_loader_profiles import L2_CONSERVATIVE_SPATIAL, loader_profile_contract
from .models import create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .run_cub_combined_smoke import (
    _atomic_json_save,
    _atomic_torch_save,
    _feature_cache,
    _finite_metrics,
    _load_json,
    _runtime,
    _target_cache,
)
from .run_cub_r50_teacher_full import load_scientific_teacher
from .train_timing import evaluate, file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_b64_guided_seed1_full_v4.json"
)
SEED_EXTENSION_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_s23_b64_s2_guided_full_v5.json"
)
LOADER_FOLLOWUP_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/image_loader_experiment/configs/"
    "cub200_r50_224_b128_seed1_l2_guided_preliminary_full_execution_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633"
)
EXPECTED_SEED_EXTENSION_CONFIG_SHA256 = (
    "f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9"
)
EXPECTED_LOADER_FOLLOWUP_CONFIG_SHA256 = (
    "d1138ac889fd6bd7ec6503776be9444d9e7915ec37d0eb0c4ac1692a63c24907"
)
EXPECTED_LOADER_FOLLOWUP_PARENT_SHA256 = (
    "1ecef8999e7233f1f2dc5c960394fbcb2f8892ae85df6f5de98a36bacd51a7b4"
)
EXPECTED_BASE_CONFIG_SHA256 = (
    "e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc"
)
EXPECTED_RELEASE_MANIFEST_SHA256 = (
    "589082b6526e31ac97c08943f61a53d2799eff07f949e947b37d8870912b6ad6"
)
EXPECTED_TEACHER_SHA256 = (
    "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
)
EXPECTED_TEACHER_STATE_SHA256 = (
    "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
)
EXPECTED_VALIDATION_HASH = (
    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
)
EXPECTED_VARIANTS = (
    "lg",
    "alg_warmup20",
    "ibkd_lambda_0.25",
    "ibkd_lambda_0.5",
)
EXPECTED_BATCHES = (128, 64)
PROBE_SEEDS = (1, 2, 3, 4, 5)
VARIANT_ARGUMENTS: dict[str, tuple[str, float | None, int]] = {
    "lg": ("lg", None, 0),
    "alg_warmup20": ("alg", None, 20),
    "ibkd_lambda_0.25": ("ibkd", 0.25, 20),
    "ibkd_lambda_0.5": ("ibkd", 0.5, 20),
}


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch-profile-full", action="store_true")
    mode.add_argument("--seed-extension-full", action="store_true")
    mode.add_argument("--loader-followup-full", action="store_true")
    parser.add_argument(
        "--partition",
        choices=("batch128_encoder_seeds_2_3", "batch64_encoder_seed_2"),
        help="Required exact v5 H200 partition; not used by the v4 seed-1 run.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _execution_profiles(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    if bool(getattr(args, "loader_followup_full", False)):
        if args.partition is not None:
            raise ValueError("The selected-loader follow-up does not accept --partition")
        return ((128, 1),)
    if args.batch_profile_full:
        if args.partition is not None:
            raise ValueError("The v4 batch-profile run does not accept --partition")
        return ((128, 1), (64, 1))
    if args.partition == "batch128_encoder_seeds_2_3":
        return ((128, 2), (128, 3))
    if args.partition == "batch64_encoder_seed_2":
        return ((64, 2),)
    raise ValueError("The v5 seed-extension run requires an exact --partition")


def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    scope = config.get("result_scope", {})
    student = config.get("classification", {}).get("student", {})
    probe = config.get("frozen_probe", {}).get("probe", {})
    split = config.get("dataset", {}).get("split", {})
    teacher = config.get("teacher", {})
    counts = config.get("task_count", {})
    checks = {
        "config_sha256": file_sha256(config_path) == EXPECTED_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_guided_b128_b64_seed1_full_v4",
        "locked_before_results": str(config.get("status", "")).startswith(
            "locked_before_batch_profile_full_results"
        ),
        "scientific": config.get("scientific_result") is True,
        "scope": tuple(scope.get("variants", ())) == EXPECTED_VARIANTS
        and scope.get("encoder_seeds") == [1]
        and scope.get("batch_order") == [128, 64]
        and scope.get("final_six_variant_three_seed_matrix_complete") is False
        and scope.get("selection_of_batch_method_or_lambda_from_smoke_or_test")
        is False,
        "batch_roles": scope.get("batch_profiles", {}).get("128", {}).get("role")
        == "locked_v3_partial_cell"
        and scope.get("batch_profiles", {}).get("64", {}).get("role")
        == "batch64_sensitivity",
        "base": config.get("base_protocol", {}).get("sha256")
        == EXPECTED_BASE_CONFIG_SHA256,
        "teacher": teacher.get("train_in_this_run") is False
        and teacher.get("source_h200_issue") == 722
        and teacher.get("architecture") == "torchvision_resnet50"
        and teacher.get("initialization") == "scratch"
        and teacher.get("input_size") == 224
        and teacher.get("training_epochs") == 200
        and teacher.get("checkpoint_sha256") == EXPECTED_TEACHER_SHA256
        and teacher.get("model_state_sha256") == EXPECTED_TEACHER_STATE_SHA256,
        "dataset": config.get("dataset", {}).get("name") == DATASET_NAME
        and config.get("dataset", {}).get("num_classes") == NUM_CLASSES,
        "split": split.get("expected_counts")
        == {"train": 5394, "validation": 600, "test": 5794}
        and split.get("validation_per_class") == 3
        and split.get("split_seed") == 2027
        and split.get("validation_image_ids_sha256")
        == EXPECTED_VALIDATION_HASH,
        "student": student.get("architecture") == "deit_tiny_patch16_224"
        and student.get("initialization") == "scratch"
        and student.get("train_batch_sizes") == [128, 64]
        and student.get("encoder_seeds") == [1]
        and student.get("epochs") == 300
        and student.get("optimizer", {}).get("learning_rate") == 0.0005
        and student.get("scheduler", {}).get("warmup_epochs") == 20,
        "probe": probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 100
        and probe.get("batch_size") == 64
        and probe.get("probe_seeds") == [1, 2, 3, 4, 5],
        "counts": counts.get("classification_students_total") == 8
        and counts.get("probe_lr_candidates_total") == 120
        and counts.get("selected_probes_total") == 40
        and counts.get("retained_new_checkpoints") == 48,
        "runtime": config.get("execution", {}).get("requested_mig_slices") == 7
        and config.get("execution", {}).get("batch_order") == [128, 64],
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid CUB R50 batch-profile full config: " + ", ".join(failures))

    base_path = _resolve_repository_path(config["base_protocol"]["path"])
    release_path = _resolve_repository_path(teacher["release_manifest"])
    if not base_path.is_file() or file_sha256(base_path) != EXPECTED_BASE_CONFIG_SHA256:
        raise RuntimeError("locked CUB v3 base protocol is missing or changed")
    if (
        not release_path.is_file()
        or file_sha256(release_path) != EXPECTED_RELEASE_MANIFEST_SHA256
        or teacher.get("release_manifest_sha256")
        != EXPECTED_RELEASE_MANIFEST_SHA256
    ):
        raise RuntimeError("audited issue-722 teacher release manifest changed")


def _validate_seed_extension_config(
    config: dict[str, Any], config_path: Path, *, partition: str
) -> None:
    scope = config.get("result_scope", {})
    provenance = config.get("protocol_provenance", {})
    teacher = config.get("teacher", {})
    dataset = config.get("dataset", {})
    classification = config.get("classification", {})
    probe = config.get("frozen_probe", {})
    statistics_config = config.get("statistics", {})
    counts = config.get("task_count", {})
    execution = config.get("execution", {})
    checks = {
        "config_sha256": file_sha256(config_path)
        == EXPECTED_SEED_EXTENSION_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_guided_b128_s23_b64_s2_full_v5",
        "locked": config.get("status")
        == "locked_after_seed1_results_before_seed_extension_smoke_2026-09-09",
        "scientific": config.get("scientific_result") is True,
        "partition": partition
        in execution.get("recommended_full_job_partition", []),
        "scope": tuple(scope.get("variants", ())) == EXPECTED_VARIANTS
        and scope.get("batch_seed_order")
        == [
            {"batch_size": 128, "encoder_seeds": [2, 3]},
            {"batch_size": 64, "encoder_seeds": [2]},
        ]
        and scope.get("guided_four_variant_matrix_only") is True
        and scope.get("vanilla_and_kd_not_in_this_run") is True
        and scope.get("configuration_changes_after_smoke") is False
        and scope.get("direct_spatial_diagnostics_in_this_run") is False,
        "batch_roles": scope.get("batch_profiles", {}).get("128", {}).get("role")
        == "locked_v3_confirmatory_continuation"
        and scope.get("batch_profiles", {}).get("64", {}).get("role")
        == "posthoc_exploratory_batch_sensitivity",
        "provenance": provenance.get("base_v3", {}).get("sha256")
        == EXPECTED_BASE_CONFIG_SHA256
        and provenance.get("seed1_batch_profile_v4", {}).get("sha256")
        == EXPECTED_CONFIG_SHA256
        and provenance.get("inherit_training_and_probe_settings_without_change")
        is True,
        "teacher": teacher.get("train_in_this_run") is False
        and teacher.get("source_h200_issue") == 722
        and teacher.get("architecture") == "torchvision_resnet50"
        and teacher.get("initialization") == "scratch"
        and teacher.get("external_pretraining") is False
        and teacher.get("input_size") == 224
        and teacher.get("training_epochs") == 200
        and teacher.get("checkpoint_sha256") == EXPECTED_TEACHER_SHA256
        and teacher.get("model_state_sha256") == EXPECTED_TEACHER_STATE_SHA256,
        "dataset": dataset.get("name") == DATASET_NAME
        and dataset.get("num_classes") == NUM_CLASSES
        and dataset.get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_HASH,
        },
        "classification": classification.get("architecture")
        == "deit_tiny_patch16_224"
        and classification.get("initialization") == "scratch"
        and classification.get("input_size") == 224
        and classification.get("epochs") == 300
        and classification.get("precision") == "float32"
        and classification.get("alg_controller_warmup_epochs") == 20
        and classification.get("ibkd_controller_warmup_epochs") == 20
        and classification.get("ibkd_fusion_ratio_lambdas") == [0.25, 0.5]
        and classification.get("selection_metric") == "validation_macro_top1",
        "probe": probe.get("encoder_frozen_eval_strict_load") is True
        and probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 100
        and probe.get("probe_seeds") == [1, 2, 3, 4, 5]
        and probe.get("selection_split") == "validation"
        and probe.get("primary_metric") == "input_224_two_class_mean_iou",
        "statistics": statistics_config.get("probe_seeds_are_nested_within_encoder_seed")
        is True
        and statistics_config.get("do_not_flatten_probe_seeds_as_independent_encoder_runs")
        is True,
        "counts": counts
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
        "runtime": execution.get("requested_mig_slices") == 7
        and execution.get("full_h200_required_for_comparable_timing") is True
        and execution.get("retain_classification_best_checkpoints") is True
        and execution.get("retain_selected_probe_checkpoints") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB R50 seed-extension full config: " + ", ".join(failures)
        )

    base_path = _resolve_repository_path(provenance["base_v3"]["path"])
    batch_profile_path = _resolve_repository_path(
        provenance["seed1_batch_profile_v4"]["path"]
    )
    release_path = _resolve_repository_path(teacher["release_manifest"])
    if not base_path.is_file() or file_sha256(base_path) != EXPECTED_BASE_CONFIG_SHA256:
        raise RuntimeError("locked CUB v3 base protocol is missing or changed")
    if (
        not batch_profile_path.is_file()
        or file_sha256(batch_profile_path) != EXPECTED_CONFIG_SHA256
    ):
        raise RuntimeError("locked CUB v4 execution protocol is missing or changed")
    if (
        not release_path.is_file()
        or file_sha256(release_path) != EXPECTED_RELEASE_MANIFEST_SHA256
        or teacher.get("release_manifest_sha256")
        != EXPECTED_RELEASE_MANIFEST_SHA256
    ):
        raise RuntimeError("audited issue-722 teacher release manifest changed")


def _validate_loader_followup_config(
    config: dict[str, Any], config_path: Path
) -> None:
    provenance = config.get("protocol_provenance", {})
    teacher = config.get("teacher", {})
    dataset = config.get("dataset", {})
    classification = config.get("classification", {})
    probe = config.get("frozen_probe", {}).get("probe", {})
    counts = config.get("task_count", {})
    execution = config.get("execution", {})
    loader = config.get("loader", {})
    expected_loader = loader_profile_contract(L2_CONSERVATIVE_SPATIAL)
    checks = {
        "config_sha256": file_sha256(config_path)
        == EXPECTED_LOADER_FOLLOWUP_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == (
            "cub200_phase1_r50_224_b128_seed1_l2_guided_"
            "preliminary_full_execution_v1"
        ),
        "released": config.get("status")
        == "released_after_successful_smoke_before_preliminary_full_2026-09-13"
        and config.get("execution_gate", {}).get("current_state")
        == "ready_for_preliminary_full",
        "result_role": config.get("scientific_result") is True
        and config.get("exploratory_selected_loader_followup") is True
        and config.get("confirmatory_main_result") is False,
        "parent": provenance.get("pre_smoke_locked_full", {}).get("sha256")
        == EXPECTED_LOADER_FOLLOWUP_PARENT_SHA256
        and provenance.get("pre_smoke_locked_full", {}).get(
            "scientific_parameters_changed_after_smoke"
        )
        is False,
        "smoke": config.get("smoke_evidence", {}).get("status") == "pass"
        and config.get("smoke_evidence", {}).get("scientific_result") is False
        and config.get("smoke_evidence", {}).get("metrics_used_to_change_protocol")
        is False
        and config.get("smoke_evidence", {}).get("logical_tasks_completed")
        == "16/16",
        "teacher": teacher.get("train_in_this_run") is False
        and teacher.get("source_h200_issue") == 722
        and teacher.get("architecture") == "torchvision_resnet50"
        and teacher.get("initialization") == "scratch"
        and teacher.get("external_pretraining") is False
        and teacher.get("input_size") == 224
        and teacher.get("training_epochs") == 200
        and teacher.get("checkpoint_sha256") == EXPECTED_TEACHER_SHA256
        and teacher.get("model_state_sha256") == EXPECTED_TEACHER_STATE_SHA256,
        "dataset": dataset.get("name") == DATASET_NAME
        and dataset.get("num_classes") == NUM_CLASSES
        and dataset.get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_HASH,
        },
        "loader": loader.get("profile") == L2_CONSERVATIVE_SPATIAL
        and loader.get("random_resized_crop_scale")
        == expected_loader["random_resized_crop"]["scale"]
        and loader.get("random_resized_crop_ratio")
        == expected_loader["random_resized_crop"]["ratio"]
        and loader.get("horizontal_flip_probability")
        == expected_loader["horizontal_flip_probability"]
        and loader.get("color_jitter_argument")
        == expected_loader["color_jitter_argument"]
        and loader.get("auto_augment") == expected_loader["auto_augment"]
        and loader.get("random_erasing_probability")
        == expected_loader["random_erasing_probability"]
        and loader.get("teacher_and_student_receive_same_augmented_tensor") is True
        and loader.get("spatial_annotations_used_for_student_training") is False,
        "classification": classification.get("student_architecture")
        == "deit_tiny_patch16_224"
        and classification.get("initialization") == "scratch"
        and classification.get("batch_size") == 128
        and classification.get("encoder_seeds") == [1]
        and classification.get("epochs") == 300
        and tuple(classification.get("variants", ())) == EXPECTED_VARIANTS
        and classification.get("optimizer", {}).get("learning_rate") == 0.0005
        and classification.get("scheduler", {}).get("warmup_epochs") == 20
        and classification.get("controller_warmup_epochs")
        == {"lg": 0, "alg_warmup20": 20, "ibkd": 20}
        and classification.get("ibkd_fusion_ratio_lambdas") == [0.25, 0.5]
        and classification.get("selection_split") == "validation"
        and classification.get("selection_metric") == "macro_top1",
        "probe": probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 100
        and probe.get("batch_size") == 64
        and probe.get("probe_seeds") == [1, 2, 3, 4, 5]
        and probe.get("selection_split") == "validation",
        "test_policy": config.get("official_test_policy")
        == {
            "classification_open_after_all_four_validation_checkpoint_selections": True,
            "segmentation_open_after_all_twenty_validation_probe_selections": True,
            "one_evaluation_per_selected_checkpoint": True,
            "test_used_for_model_hyperparameter_or_loader_selection": False,
        },
        "counts": counts
        == {
            "teacher_download_and_audit": 1,
            "classification_students": 4,
            "classification_official_test_evaluations": 4,
            "segmentation_probe_lr_candidates": 60,
            "segmentation_probe_validation_selections": 20,
            "segmentation_probe_official_test_evaluations": 20,
            "retained_new_checkpoints": 24,
        },
        "runtime": execution.get("requested_mig_slices") == 7
        and execution.get("precision") == "float32"
        and execution.get("feature_batch_size") == 32
        and execution.get("classification_eval_batch_size") == 200
        and execution.get("classification_num_workers") == 4,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB selected-loader follow-up full config: "
            + ", ".join(failures)
        )

    parent_path = _resolve_repository_path(
        provenance["pre_smoke_locked_full"]["path"]
    )
    release_path = _resolve_repository_path(teacher["release_manifest"])
    if (
        not parent_path.is_file()
        or file_sha256(parent_path) != EXPECTED_LOADER_FOLLOWUP_PARENT_SHA256
    ):
        raise RuntimeError("selected-loader pre-smoke full protocol changed")
    if (
        not release_path.is_file()
        or file_sha256(release_path) != EXPECTED_RELEASE_MANIFEST_SHA256
        or teacher.get("release_manifest_sha256")
        != EXPECTED_RELEASE_MANIFEST_SHA256
    ):
        raise RuntimeError("audited issue-722 teacher release manifest changed")


def _write_status(
    path: Path,
    *,
    status: str,
    phase: str,
    batch_size: int | None,
    classification_complete: int,
    probe_candidates_complete: int,
    probe_selections_complete: int,
    test_evaluations_complete: int,
    profile_total: int = 2,
    active: str | None = None,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "active_batch_size": batch_size,
            "classification_complete": classification_complete,
            "classification_expected": profile_total * 4,
            "probe_candidates_complete": probe_candidates_complete,
            "probe_candidates_expected": profile_total * 60,
            "probe_selections_complete": probe_selections_complete,
            "probe_selections_expected": profile_total * 20,
            "official_test_evaluations_complete": test_evaluations_complete,
            "official_test_evaluations_expected": profile_total * 20,
            "active": active,
            "failure": failure,
        },
        path,
    )


def _run_or_resume(command: list[str], summary_path: Path, *, label: str) -> dict[str, Any]:
    if summary_path.is_file():
        payload = _load_json(summary_path)
        if payload.get("status") == "complete":
            log(f"[CUB_R50_FULL_RESUME] {label} summary={summary_path}")
            return payload
    log(f"[CUB_R50_FULL_TASK_START] {label} command={' '.join(command)}")
    subprocess.run(command, check=True)
    payload = _load_json(summary_path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"full student task produced incomplete summary: {label}")
    log(f"[CUB_R50_FULL_TASK_DONE] {label}")
    return payload


def _archive_audit(data_dir: Path) -> dict[str, Any]:
    identities = (
        ("image", data_dir / ARCHIVE_NAME, ARCHIVE_BYTES, ARCHIVE_MD5, ARCHIVE_SHA256),
        (
            "segmentation",
            data_dir / SEGMENTATION_ARCHIVE_NAME,
            SEGMENTATION_ARCHIVE_BYTES,
            SEGMENTATION_ARCHIVE_MD5,
            SEGMENTATION_ARCHIVE_SHA256,
        ),
    )
    result: dict[str, Any] = {}
    for name, path, expected_bytes, expected_md5, expected_sha256 in identities:
        if not path.is_file():
            raise RuntimeError(f"CUB {name} archive missing after setup: {path}")
        actual = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "md5": file_digest(path, "md5"),
            "sha256": file_digest(path),
        }
        if actual != {
            "path": str(path.resolve()),
            "bytes": expected_bytes,
            "md5": expected_md5,
            "sha256": expected_sha256,
        }:
            raise RuntimeError(f"CUB {name} archive identity mismatch")
        result[name] = actual
    return result


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _batch_role(
    batch_size: int, *, seed_extension: bool, loader_followup: bool = False
) -> str:
    if loader_followup:
        if batch_size != 128:
            raise ValueError("selected-loader follow-up is fixed to batch 128")
        return "exploratory_selected_loader_followup"
    if seed_extension:
        if batch_size == 128:
            return "locked_v3_confirmatory_continuation"
        if batch_size == 64:
            return "posthoc_exploratory_batch_sensitivity"
        raise ValueError(f"unsupported seed-extension batch profile: {batch_size}")
    if batch_size == 128:
        return "locked_v3_partial_cell"
    if batch_size == 64:
        return "batch64_sensitivity"
    raise ValueError(f"unsupported batch profile: {batch_size}")


def _profile_dir(args: argparse.Namespace, batch_size: int, encoder_seed: int) -> Path:
    if args.seed_extension_full or args.loader_followup_full:
        return args.output_dir / f"batch{batch_size}_seed{encoder_seed}"
    return args.output_dir / f"batch{batch_size}"


def _variant_run_name(variant: str, batch_size: int, encoder_seed: int) -> str:
    return (
        f"cub_r50_224_{variant}_deit_tiny_b{batch_size}_"
        f"full_300ep_seed{encoder_seed}"
    )


def _checkpoint_purpose(
    seed_extension: bool, *, loader_followup: bool = False
) -> str:
    return (
        "phase1_cub_r50_224_l2_guided_preliminary_full_student_v1"
        if loader_followup
        else "phase1_cub_r50_224_seed_extension_full_student_v5"
        if seed_extension
        else "phase1_cub_r50_224_batch_profile_full_student_v4"
    )


def _probe_checkpoint_purpose(
    seed_extension: bool, *, loader_followup: bool = False
) -> str:
    return (
        "phase1_cub_r50_224_l2_guided_preliminary_frozen_probe_v1"
        if loader_followup
        else "phase1_cub_r50_224_seed_extension_frozen_probe_v5"
        if seed_extension
        else "phase1_cub_r50_224_batch_profile_frozen_probe_v4"
    )


def _classification_rows(
    args: argparse.Namespace,
    *,
    config_sha256: str,
    validation_hash: str,
    batch_size: int,
    encoder_seed: int,
    prior_profile_count: int,
) -> list[dict[str, Any]]:
    profile_dir = _profile_dir(args, batch_size, encoder_seed)
    student_root = profile_dir / "classification" / "students"
    rows: list[dict[str, Any]] = []
    for variant in EXPECTED_VARIANTS:
        method, fusion_ratio, controller_warmup = VARIANT_ARGUMENTS[variant]
        run_name = _variant_run_name(variant, batch_size, encoder_seed)
        run_dir = student_root / run_name
        checkpoint_path = run_dir / "student_best_validation.pt"
        command = [
            sys.executable,
            "-m",
            "ibkd_seg.phase1.train_full",
            "--full-run",
            "--dataset",
            "cub",
            "--kind",
            "student",
            "--method",
            method,
            "--teacher-architecture",
            "resnet50_224_scratch",
            "--scientific-cub-r50-teacher",
            "--batch-profile-role",
            _batch_role(
                batch_size,
                seed_extension=args.seed_extension_full,
                loader_followup=args.loader_followup_full,
            ),
            "--protocol-config",
            str(args.config),
            "--batch-size",
            str(batch_size),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(student_root),
            "--run-name",
            run_name,
            "--teacher-checkpoint",
            str(args.teacher_checkpoint),
            "--num-workers",
            str(args.num_workers),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--seed",
            str(encoder_seed),
        ]
        if args.loader_followup_full:
            command.extend(
                [
                    "--loader-followup-full",
                    "--defer-official-test",
                    "--cub-loader-profile",
                    L2_CONSERVATIVE_SPATIAL,
                ]
            )
        elif args.seed_extension_full:
            command.append("--seed-extension-full")
        if fusion_ratio is not None:
            command.extend(["--fusion-ratio", str(fusion_ratio)])
        if method == "alg":
            command.extend(["--alg-controller-warmup-epochs", "20"])
        summary = _run_or_resume(
            command,
            run_dir / "summary.json",
            label=f"batch{batch_size}_{variant}_seed{encoder_seed}",
        )
        expected_confirmatory = batch_size == 128 and not args.loader_followup_full
        expected = {
            "status": "complete",
            "scientific_result": True,
            "confirmatory_main_result": expected_confirmatory,
            "kind": "student",
            "dataset": DATASET_NAME,
            "num_classes": NUM_CLASSES,
            "method": method,
            "fusion_ratio_lambda": fusion_ratio,
            "batch_size": batch_size,
            "epochs": 300,
            "seed": encoder_seed,
            "official_test_evaluations": 0 if args.loader_followup_full else 1,
            "official_test_accessed": not args.loader_followup_full,
            "official_test_used_for_training_or_selection": False,
            "selected_checkpoint_strict_reloaded": True,
            "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
            "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
            "teacher_architecture": "resnet50_224_scratch",
            "protocol_config_sha256": config_sha256,
            "batch_profile_role": _batch_role(
                batch_size,
                seed_extension=args.seed_extension_full,
                loader_followup=args.loader_followup_full,
            ),
            "eligible_locked_v3_matrix_cell": expected_confirmatory,
            "final_confirmatory_matrix_complete": False,
            "guidance_controller_warmup_epochs": controller_warmup,
        }
        if args.seed_extension_full:
            expected["seed_extension_full"] = True
        if args.loader_followup_full:
            expected.update(
                {
                    "loader_followup_full": True,
                    "cub_loader_profile": L2_CONSERVATIVE_SPATIAL,
                    "exploratory_selected_loader_followup": True,
                    "official_test_policy": (
                        "deferred_to_orchestrator_after_all_four_"
                        "validation_selections"
                    ),
                }
            )
        failures = [key for key, value in expected.items() if summary.get(key) != value]
        if (
            failures
            or summary.get("split_manifest", {}).get("validation_image_ids_sha256")
            != validation_hash
            or not checkpoint_path.is_file()
            or summary.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        ):
            raise RuntimeError(
                f"CUB R50 full student contract failed batch={batch_size} "
                f"variant={variant}: {','.join(failures)}"
            )
        rows.append(
            {
                "batch_size": batch_size,
                "batch_profile_role": _batch_role(
                    batch_size,
                    seed_extension=args.seed_extension_full,
                    loader_followup=args.loader_followup_full,
                ),
                "eligible_locked_v3_matrix_cell": expected_confirmatory,
                "cub_loader_profile": (
                    L2_CONSERVATIVE_SPATIAL
                    if args.loader_followup_full
                    else "l0_current_strong"
                ),
                "variant": variant,
                "method": method,
                "fusion_ratio_lambda": fusion_ratio,
                "controller_warmup_epochs": controller_warmup,
                "encoder_seed": encoder_seed,
                "checkpoint_purpose": _checkpoint_purpose(
                    args.seed_extension_full,
                    loader_followup=args.loader_followup_full,
                ),
                "summary_path": str((run_dir / "summary.json").resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "summary": summary,
            }
        )
        _write_status(
            args.output_dir / "sequence_status.json",
            status="running",
            phase="classification_students",
            batch_size=batch_size,
            classification_complete=prior_profile_count * 4 + len(rows),
            probe_candidates_complete=prior_profile_count * 60,
            probe_selections_complete=prior_profile_count * 20,
            test_evaluations_complete=prior_profile_count * 20,
            profile_total=len(_execution_profiles(args)),
            active=f"batch{batch_size}/{variant}/encoder_seed{encoder_seed}",
        )
    return rows


def _evaluate_deferred_classification_tests(
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
    validation_hash: str,
    batch_size: int,
    encoder_seed: int,
    device: torch.device,
) -> None:
    if not args.loader_followup_full:
        return
    if len(rows) != 4 or {row["variant"] for row in rows} != set(EXPECTED_VARIANTS):
        raise RuntimeError(
            "all four validation-selected encoders are required before official test"
        )
    profile_dir = _profile_dir(args, batch_size, encoder_seed)
    marker_path = (
        profile_dir / "classification" / "selection_complete_before_test.json"
    )
    journal_path = profile_dir / "classification" / "official_test_results.json"
    if not marker_path.is_file():
        _atomic_json_save(
            {
                "status": "complete",
                "config_sha256": config_sha256,
                "batch_size": batch_size,
                "encoder_seed": encoder_seed,
                "completed_validation_selections": 4,
                "expected_validation_selections": 4,
                "official_test_accessed": False,
            },
            marker_path,
        )
    marker = _load_json(marker_path)
    if marker != {
        "status": "complete",
        "config_sha256": config_sha256,
        "batch_size": batch_size,
        "encoder_seed": encoder_seed,
        "completed_validation_selections": 4,
        "expected_validation_selections": 4,
        "official_test_accessed": False,
    }:
        raise RuntimeError("classification pre-test selection marker changed")

    entries: list[dict[str, Any]] = []
    if journal_path.is_file():
        journal = _load_json(journal_path)
        if (
            journal.get("config_sha256") != config_sha256
            or journal.get("batch_size") != batch_size
            or journal.get("encoder_seed") != encoder_seed
            or not isinstance(journal.get("entries"), list)
        ):
            raise RuntimeError("classification official-test journal changed")
        entries = journal["entries"]
    by_variant = {entry.get("variant"): entry for entry in entries}
    if len(by_variant) != len(entries) or any(
        variant not in EXPECTED_VARIANTS for variant in by_variant
    ):
        raise RuntimeError("classification official-test journal has invalid variants")

    test_loader = None
    for row in rows:
        variant = row["variant"]
        existing = by_variant.get(variant)
        if existing is None:
            if test_loader is None:
                test_loader = build_official_test_loader(
                    args.data_dir,
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    device=device,
                )
            model, encoder_audit = _load_encoder(
                row,
                config_sha256=config_sha256,
                validation_hash=validation_hash,
                device=device,
            )
            metrics = evaluate(
                model,
                test_loader,
                device,
                teacher=False,
                num_classes=NUM_CLASSES,
            )
            del model
            if not all(_finite_metrics(value) for value in metrics.values()):
                raise RuntimeError("non-finite classification official-test metric")
            existing = {
                "variant": variant,
                "encoder_checkpoint_sha256": encoder_audit["checkpoint_sha256"],
                "metrics": metrics,
                "official_test_evaluations": 1,
            }
            entries.append(existing)
            by_variant[variant] = existing
            _atomic_json_save(
                {
                    "status": "complete" if len(entries) == 4 else "in_progress",
                    "config_sha256": config_sha256,
                    "batch_size": batch_size,
                    "encoder_seed": encoder_seed,
                    "completed": len(entries),
                    "expected": 4,
                    "settings_locked_before_first_test": True,
                    "entries": entries,
                },
                journal_path,
            )
            log(
                f"[CUB_R50_FULL_CLASSIFICATION_TEST] batch={batch_size} "
                f"encoder_seed={encoder_seed} variant={variant} "
                f"test_macro_top1={metrics['macro_top1']:.4f}"
            )
        if (
            existing.get("encoder_checkpoint_sha256")
            != row["summary"]["checkpoint_sha256"]
            or existing.get("official_test_evaluations") != 1
            or not isinstance(existing.get("metrics"), dict)
        ):
            raise RuntimeError("classification official-test result failed audit")
        row["official_test"] = existing["metrics"]

    if len(entries) != 4:
        raise RuntimeError("classification official-test completion gate failed")


def _classification_flat_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        summary = row["summary"]
        official_test = row.get("official_test", summary.get("official_test"))
        if not isinstance(official_test, dict):
            raise RuntimeError("classification official-test metrics are missing")
        flattened.append(
            {
                "batch_size": row["batch_size"],
                "batch_profile_role": row["batch_profile_role"],
                "cub_loader_profile": row["cub_loader_profile"],
                "variant": row["variant"],
                "method": row["method"],
                "fusion_ratio_lambda": row["fusion_ratio_lambda"],
                "controller_warmup_epochs": row["controller_warmup_epochs"],
                "encoder_seed": row["encoder_seed"],
                "selected_epoch": summary["selected_epoch"],
                "validation_macro_top1": summary["selected_validation"]["macro_top1"],
                "test_macro_top1": official_test["macro_top1"],
                "test_overall_top1": official_test["overall_top1"],
                "test_top5": official_test["top5"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "confirmatory_main_result": summary["confirmatory_main_result"],
            }
        )
    return flattened


def _classification_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    _write_csv(_classification_flat_rows(rows), path)


def _load_encoder(
    row: dict[str, Any],
    *,
    config_sha256: str,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(row["checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": row["checkpoint_purpose"],
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": row["method"],
        "fusion_ratio_lambda": row["fusion_ratio_lambda"],
        "batch_size": row["batch_size"],
        "epochs": 300,
        "seed": row["encoder_seed"],
        "validation_image_ids_sha256": validation_hash,
        "guidance_controller_warmup_epochs": row["controller_warmup_epochs"],
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
        "teacher_architecture": "resnet50_224_scratch",
        "protocol_config_sha256": config_sha256,
        "batch_profile_role": row["batch_profile_role"],
        "eligible_locked_v3_matrix_cell": row["eligible_locked_v3_matrix_cell"],
        "final_confirmatory_matrix_complete": False,
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    if row["checkpoint_purpose"].endswith("_v5"):
        expected["seed_extension_full"] = True
    if row["checkpoint_purpose"].endswith("_v1"):
        expected.update(
            {
                "loader_followup_full": True,
                "cub_loader_profile": L2_CONSERVATIVE_SPATIAL,
                "exploratory_selected_loader_followup": True,
            }
        )
    failures = [key for key, value in expected.items() if metadata.get(key) != value]
    if failures:
        raise RuntimeError(
            f"encoder metadata mismatch {row['variant']}/b{row['batch_size']}: "
            + ",".join(failures)
        )
    checkpoint_hash = file_sha256(checkpoint_path)
    if row["summary"].get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("encoder checkpoint SHA-256 changed")
    model = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict encoder load returned incompatible keys")
    state_hash = state_dict_sha256(model)
    if state_hash != metadata.get("student_state_sha256"):
        raise RuntimeError("encoder state SHA-256 changed")
    model.to(device).eval().requires_grad_(False)
    audit = {
        "strict_load": True,
        "eval_mode": not model.training,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "checkpoint_sha256": checkpoint_hash,
        "student_state_sha256": state_hash,
    }
    if not audit["eval_mode"] or audit["trainable_parameter_count"] != 0:
        raise RuntimeError("frozen encoder contract failed")
    return model, audit


def _selection_path(batch_dir: Path) -> Path:
    return batch_dir / "probe" / "validation_selections.json"


def _load_existing_selections(
    profile_dir: Path,
    *,
    config_sha256: str,
    batch_size: int,
    encoder_seed: int,
) -> list[dict[str, Any]]:
    path = _selection_path(profile_dir)
    if not path.is_file():
        return []
    payload = _load_json(path)
    if (
        payload.get("config_sha256") != config_sha256
        or payload.get("batch_size") != batch_size
        or payload.get("encoder_seed", 1) != encoder_seed
        or not isinstance(payload.get("selections"), list)
    ):
        raise RuntimeError("existing probe selection journal does not match this run")
    selections = payload["selections"]
    keys = [(row.get("variant"), row.get("probe_seed")) for row in selections]
    if len(keys) != len(set(keys)) or len(keys) > 20:
        raise RuntimeError("existing probe selection journal has invalid keys")
    for row in selections:
        checkpoint = Path(row["probe_checkpoint_path"])
        if (
            row.get("batch_size") != batch_size
            or row.get("encoder_seed") != encoder_seed
            or row.get("variant") not in EXPECTED_VARIANTS
            or row.get("probe_seed") not in PROBE_SEEDS
            or not checkpoint.is_file()
            or file_sha256(checkpoint) != row.get("probe_checkpoint_sha256")
            or row.get("official_test_evaluations") not in {0, 1}
        ):
            raise RuntimeError("existing selected probe artifact failed validation")
    log(
        f"[CUB_R50_FULL_RESUME] batch={batch_size} encoder_seed={encoder_seed} "
        f"selected_probes={len(selections)}/20"
    )
    return selections


def _save_selections(
    profile_dir: Path,
    *,
    config_sha256: str,
    batch_size: int,
    encoder_seed: int,
    selections: Sequence[dict[str, Any]],
    official_test_masks_accessed: bool,
) -> None:
    _atomic_json_save(
        {
            "status": "complete" if len(selections) == 20 else "selection_in_progress",
            "config_sha256": config_sha256,
            "batch_size": batch_size,
            "encoder_seed": encoder_seed,
            "completed": len(selections),
            "expected": 20,
            "official_test_masks_accessed": official_test_masks_accessed,
            "selections": list(selections),
        },
        _selection_path(profile_dir),
    )


def _selection_contracts(selections: Sequence[dict[str, Any]]) -> dict[str, bool]:
    initial_hashes: dict[int, set[str]] = defaultdict(set)
    batch_orders: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in selections:
        candidates = row.get("candidates", [])
        for candidate in candidates:
            seed = int(row["probe_seed"])
            initial_hashes[seed].add(candidate["initial_probe_state_sha256"])
            for epoch, digest in enumerate(
                candidate["batch_order_sha256_by_epoch"], start=1
            ):
                batch_orders[(seed, epoch)].add(digest)
    return {
        "selection_count": len(selections) == 20,
        "three_lr_candidates_per_selection": all(
            len(row.get("candidates", [])) == 3 for row in selections
        ),
        "same_initial_probe_state_per_probe_seed": set(initial_hashes) == set(PROBE_SEEDS)
        and all(len(initial_hashes[seed]) == 1 for seed in PROBE_SEEDS),
        "same_probe_batch_order_per_seed_and_epoch": len(batch_orders) == 500
        and all(len(values) == 1 for values in batch_orders.values()),
        "all_selected_probes_strict_reloaded": all(
            row.get("selected_probe_strict_reloaded") is True for row in selections
        ),
    }


def _cleanup_feature_files(paths: Sequence[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _select_probes(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    config_sha256: str,
    records: dict[str, list[CubProbeRecord]],
    targets: dict[str, dict[str, Any]],
    classification_rows: Sequence[dict[str, Any]],
    validation_hash: str,
    batch_size: int,
    encoder_seed: int,
    prior_profile_count: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    profile_dir = _profile_dir(args, batch_size, encoder_seed)
    selections = _load_existing_selections(
        profile_dir,
        config_sha256=config_sha256,
        batch_size=batch_size,
        encoder_seed=encoder_seed,
    )
    completed_keys = {
        (row["variant"], int(row["probe_seed"])) for row in selections
    }
    if len(selections) == 20:
        contracts = _selection_contracts(selections)
        if not all(contracts.values()):
            raise RuntimeError("resumed probe selection contracts failed")
        return selections, contracts

    probe_config = config["frozen_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    log(
        f"[CUB_R50_FULL_PROBE_PLAN] batch={batch_size} encoder_seed={encoder_seed} "
        "encoders=4 probe_seeds=5 "
        "lr_candidates=60 selections=20 epochs_per_candidate=100"
    )
    for classification in classification_rows:
        variant = classification["variant"]
        missing_seeds = [
            seed for seed in PROBE_SEEDS if (variant, seed) not in completed_keys
        ]
        if not missing_seeds:
            continue
        model, encoder_audit = _load_encoder(
            classification,
            config_sha256=config_sha256,
            validation_hash=validation_hash,
            device=device,
        )
        features: dict[str, dict[str, Any]] = {}
        cache_paths: list[Path] = []
        for split_name in ("train", "validation"):
            cache_path = (
                args.cache_dir
                / f"batch{batch_size}"
                / f"encoder_seed{encoder_seed}"
                / "features"
                / variant
                / f"{split_name}.pt"
            )
            cache_paths.append(cache_path)
            features[split_name], _ = _feature_cache(
                model,
                records[split_name],
                split=split_name,
                variant=f"b{batch_size}_s{encoder_seed}_{variant}",
                encoder_seed=encoder_seed,
                checkpoint_sha256=encoder_audit["checkpoint_sha256"],
                state_sha256=encoder_audit["student_state_sha256"],
                config=config,
                config_sha256=config_sha256,
                cache_path=cache_path,
                device=device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for probe_seed in missing_seeds:
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
                    epochs=100,
                )
                candidates.append((state, candidate))
                log(
                    f"[CUB_R50_FULL_PROBE_CANDIDATE] batch={batch_size} "
                    f"encoder_seed={encoder_seed} "
                    f"variant={variant} probe_seed={probe_seed} lr={learning_rate:g} "
                    f"best_epoch={candidate['best_epoch']} validation_grid_miou="
                    f"{candidate['best_validation_grid_mean_iou']:.6f}"
                )
            selected_state, selected = min(
                candidates,
                key=lambda item: (
                    -float(item[1]["best_validation_grid_mean_iou"]),
                    float(item[1]["learning_rate"]),
                    int(item[1]["best_epoch"]),
                ),
            )
            probe = probe_from_state(probe_config, probe_seed, selected_state, device)
            validation_metrics, _ = evaluate_probe_both_resolutions(
                probe,
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                targets["validation"]["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=int(config["frozen_probe"]["image_input"]["size"]),
                ignore_index=int(probe_config["loss"]["ignore_index"]),
            )
            if not all(_finite_metrics(value) for value in validation_metrics.values()):
                raise RuntimeError("non-finite validation probe metrics")
            probe_path = (
                profile_dir
                / "probe"
                / "checkpoints"
                / variant
                / f"probe_seed{probe_seed}_best_validation.pt"
            )
            _atomic_torch_save(
                {
                    "metadata": {
                        "purpose": _probe_checkpoint_purpose(
                            args.seed_extension_full,
                            loader_followup=args.loader_followup_full,
                        ),
                        "scientific_result": True,
                        "confirmatory_main_result": (
                            batch_size == 128 and not args.loader_followup_full
                        ),
                        "config_sha256": config_sha256,
                        "batch_size": batch_size,
                        "batch_profile_role": _batch_role(
                            batch_size,
                            seed_extension=args.seed_extension_full,
                            loader_followup=args.loader_followup_full,
                        ),
                        "cub_loader_profile": (
                            L2_CONSERVATIVE_SPATIAL
                            if args.loader_followup_full
                            else "l0_current_strong"
                        ),
                        "variant": variant,
                        "encoder_seed": encoder_seed,
                        "encoder_checkpoint_sha256": encoder_audit[
                            "checkpoint_sha256"
                        ],
                        "probe_seed": probe_seed,
                        "selection_split": "validation",
                        "selected_learning_rate": selected["learning_rate"],
                        "selected_epoch": selected["best_epoch"],
                        "validation_grid_mean_iou": selected[
                            "best_validation_grid_mean_iou"
                        ],
                        "official_test_evaluations_at_checkpoint_write": 0,
                    },
                    "model": selected_state,
                },
                probe_path,
            )
            saved = torch.load(probe_path, map_location="cpu", weights_only=True)
            strict_probe = probe_from_state(probe_config, probe_seed, saved["model"], device)
            del strict_probe, probe
            row = {
                "batch_size": batch_size,
                "batch_profile_role": _batch_role(
                    batch_size,
                    seed_extension=args.seed_extension_full,
                    loader_followup=args.loader_followup_full,
                ),
                "cub_loader_profile": (
                    L2_CONSERVATIVE_SPATIAL
                    if args.loader_followup_full
                    else "l0_current_strong"
                ),
                "confirmatory_main_result": (
                    batch_size == 128 and not args.loader_followup_full
                ),
                "variant": variant,
                "method": classification["method"],
                "fusion_ratio_lambda": classification["fusion_ratio_lambda"],
                "controller_warmup_epochs": classification[
                    "controller_warmup_epochs"
                ],
                "encoder_seed": encoder_seed,
                "encoder_checkpoint_path": classification["checkpoint_path"],
                "encoder_checkpoint_sha256": encoder_audit["checkpoint_sha256"],
                "encoder_state_sha256": encoder_audit["student_state_sha256"],
                "probe_seed": probe_seed,
                "candidates": [candidate for _, candidate in candidates],
                "selected_learning_rate": selected["learning_rate"],
                "selected_epoch": selected["best_epoch"],
                "validation": validation_metrics,
                "probe_checkpoint_path": str(probe_path.resolve()),
                "probe_checkpoint_sha256": file_sha256(probe_path),
                "selected_probe_strict_reloaded": True,
                "official_test_evaluations": 0,
            }
            selections.append(row)
            completed_keys.add((variant, probe_seed))
            _save_selections(
                profile_dir,
                config_sha256=config_sha256,
                batch_size=batch_size,
                encoder_seed=encoder_seed,
                selections=selections,
                official_test_masks_accessed=False,
            )
            _write_status(
                args.output_dir / "sequence_status.json",
                status="running",
                phase="probe_validation_selection",
                batch_size=batch_size,
                classification_complete=prior_profile_count * 4 + 4,
                probe_candidates_complete=prior_profile_count * 60
                + len(selections) * 3,
                probe_selections_complete=prior_profile_count * 20
                + len(selections),
                test_evaluations_complete=prior_profile_count * 20,
                profile_total=len(_execution_profiles(args)),
                active=(
                    f"batch{batch_size}/encoder_seed{encoder_seed}/"
                    f"{variant}/probe_seed{probe_seed}"
                ),
            )
            del candidates
        del features
        _cleanup_feature_files(cache_paths)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contracts = _selection_contracts(selections)
    if not all(contracts.values()):
        raise RuntimeError("CUB R50 batch-profile probe selection contract failed")
    return selections, contracts


def _evaluate_probe_test(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    config_sha256: str,
    test_records: Sequence[CubProbeRecord],
    test_targets: dict[str, Any],
    classification_rows: Sequence[dict[str, Any]],
    selections: list[dict[str, Any]],
    validation_hash: str,
    batch_size: int,
    encoder_seed: int,
    prior_profile_count: int,
    device: torch.device,
) -> None:
    probe_config = config["frozen_probe"]["probe"]
    by_variant = {row["variant"]: row for row in classification_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[row["variant"]].append(row)
    profile_dir = _profile_dir(args, batch_size, encoder_seed)
    for variant in EXPECTED_VARIANTS:
        rows = grouped[variant]
        if all(int(row["official_test_evaluations"]) == 1 for row in rows):
            continue
        classification = by_variant[variant]
        model, encoder_audit = _load_encoder(
            classification,
            config_sha256=config_sha256,
            validation_hash=validation_hash,
            device=device,
        )
        cache_path = (
            args.cache_dir
            / f"batch{batch_size}"
            / f"encoder_seed{encoder_seed}"
            / "features"
            / variant
            / "official_test.pt"
        )
        test_features, _ = _feature_cache(
            model,
            test_records,
            split="official_test",
            variant=f"b{batch_size}_s{encoder_seed}_{variant}",
            encoder_seed=encoder_seed,
            checkpoint_sha256=encoder_audit["checkpoint_sha256"],
            state_sha256=encoder_audit["student_state_sha256"],
            config=config,
            config_sha256=config_sha256,
            cache_path=cache_path,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        del model
        for row in sorted(rows, key=lambda item: int(item["probe_seed"])):
            if int(row["official_test_evaluations"]) == 1:
                continue
            saved = torch.load(
                row["probe_checkpoint_path"], map_location="cpu", weights_only=True
            )
            metadata = saved.get("metadata", {})
            expected = {
                "purpose": _probe_checkpoint_purpose(
                    args.seed_extension_full,
                    loader_followup=args.loader_followup_full,
                ),
                "config_sha256": config_sha256,
                "batch_size": batch_size,
                "batch_profile_role": _batch_role(
                    batch_size,
                    seed_extension=args.seed_extension_full,
                    loader_followup=args.loader_followup_full,
                ),
                "cub_loader_profile": (
                    L2_CONSERVATIVE_SPATIAL
                    if args.loader_followup_full
                    else "l0_current_strong"
                ),
                "variant": row["variant"],
                "encoder_seed": encoder_seed,
                "probe_seed": row["probe_seed"],
                "official_test_evaluations_at_checkpoint_write": 0,
            }
            failures = [key for key, value in expected.items() if metadata.get(key) != value]
            if failures:
                raise RuntimeError("selected probe metadata mismatch: " + ",".join(failures))
            probe = probe_from_state(
                probe_config, int(row["probe_seed"]), saved["model"], device
            )
            metrics, _ = evaluate_probe_both_resolutions(
                probe,
                test_features["features"],
                test_targets["grid_targets"],
                test_targets["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=int(config["frozen_probe"]["image_input"]["size"]),
                ignore_index=int(probe_config["loss"]["ignore_index"]),
            )
            if not all(_finite_metrics(value) for value in metrics.values()):
                raise RuntimeError("non-finite official-test probe metrics")
            row["official_test"] = metrics
            row["official_test_evaluations"] = 1
            _save_selections(
                profile_dir,
                config_sha256=config_sha256,
                batch_size=batch_size,
                encoder_seed=encoder_seed,
                selections=selections,
                official_test_masks_accessed=True,
            )
            completed = sum(
                int(value["official_test_evaluations"]) for value in selections
            )
            _write_status(
                args.output_dir / "sequence_status.json",
                status="running",
                phase="probe_official_test",
                batch_size=batch_size,
                classification_complete=prior_profile_count * 4 + 4,
                probe_candidates_complete=prior_profile_count * 60 + 60,
                probe_selections_complete=prior_profile_count * 20 + 20,
                test_evaluations_complete=prior_profile_count * 20 + completed,
                profile_total=len(_execution_profiles(args)),
                active=(
                    f"batch{batch_size}/encoder_seed{encoder_seed}/"
                    f"{variant}/probe_seed{row['probe_seed']}"
                ),
            )
            log(
                f"[CUB_R50_FULL_PROBE_TEST] batch={batch_size} "
                f"encoder_seed={encoder_seed} variant={variant} "
                f"probe_seed={row['probe_seed']} selected_lr="
                f"{row['selected_learning_rate']:g} selected_epoch={row['selected_epoch']} "
                f"test_input_miou={metrics['input_224']['mean_iou']:.6f}"
            )
            del probe
        del test_features
        _cleanup_feature_files([cache_path])
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _probe_flat_rows(selections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in selections:
        validation = item["validation"]
        test = item["official_test"]
        rows.append(
            {
                "batch_size": item["batch_size"],
                "batch_profile_role": item["batch_profile_role"],
                "cub_loader_profile": item["cub_loader_profile"],
                "confirmatory_main_result": item["confirmatory_main_result"],
                "variant": item["variant"],
                "method": item["method"],
                "fusion_ratio_lambda": item["fusion_ratio_lambda"],
                "controller_warmup_epochs": item["controller_warmup_epochs"],
                "encoder_seed": item["encoder_seed"],
                "probe_seed": item["probe_seed"],
                "selected_learning_rate": item["selected_learning_rate"],
                "selected_epoch": item["selected_epoch"],
                "validation_grid_mean_iou": validation["grid_14x14"]["mean_iou"],
                "validation_input_224_mean_iou": validation["input_224"]["mean_iou"],
                "test_grid_mean_iou": test["grid_14x14"]["mean_iou"],
                "test_input_224_mean_iou": test["input_224"]["mean_iou"],
                "test_input_224_foreground_iou": test["input_224"]["foreground_iou"],
                "test_input_224_background_iou": test["input_224"]["background_iou"],
                "test_input_224_foreground_dice": test["input_224"]["foreground_dice"],
                "test_input_224_pixel_accuracy": test["input_224"]["pixel_accuracy"],
                "encoder_checkpoint_path": item["encoder_checkpoint_path"],
                "encoder_checkpoint_sha256": item["encoder_checkpoint_sha256"],
                "probe_checkpoint_path": item["probe_checkpoint_path"],
                "probe_checkpoint_sha256": item["probe_checkpoint_sha256"],
                "official_test_evaluations": item["official_test_evaluations"],
                "scientific_result": True,
            }
        )
    return rows


def aggregate_probe_seed_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    encoder_seeds = {int(row.get("encoder_seed", 1)) for row in rows}
    if len(encoder_seeds) != 1:
        raise RuntimeError("probe aggregation requires exactly one encoder seed")
    encoder_seed = next(iter(encoder_seeds))
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(float(row["test_input_224_mean_iou"]))
    if set(grouped) != set(EXPECTED_VARIANTS):
        raise RuntimeError("probe result variants changed")
    aggregates = []
    for variant in EXPECTED_VARIANTS:
        values = grouped[variant]
        if len(values) != 5:
            raise RuntimeError("expected five probe seeds per batch/variant")
        aggregates.append(
            {
                "variant": variant,
                "encoder_seed": encoder_seed,
                "probe_seed_values": values,
                "mean_over_probe_seeds": statistics.mean(values),
                "sample_standard_deviation_over_probe_seeds": statistics.stdev(values),
                "independent_encoder_n": 1,
                "probe_seeds_per_encoder": 5,
                "encoder_seed_standard_deviation_not_estimable": True,
            }
        )
    return aggregates


def _batch_checkpoint_manifest(
    teacher: dict[str, Any],
    classification_rows: Sequence[dict[str, Any]],
    probe_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    entries = [
        {
            "kind": "external_shared_teacher",
            "source_h200_issue": 722,
            "path": teacher["checkpoint"],
            "sha256": teacher["checkpoint_sha256"],
        }
    ]
    entries.extend(
        {
            "kind": "classification_encoder",
            "batch_size": row["batch_size"],
            "variant": row["variant"],
            "encoder_seed": row["encoder_seed"],
            "path": row["checkpoint_path"],
            "sha256": row["summary"]["checkpoint_sha256"],
        }
        for row in classification_rows
    )
    entries.extend(
        {
            "kind": "selected_probe",
            "batch_size": row["batch_size"],
            "variant": row["variant"],
            "encoder_seed": row["encoder_seed"],
            "probe_seed": row["probe_seed"],
            "path": row["probe_checkpoint_path"],
            "sha256": row["probe_checkpoint_sha256"],
        }
        for row in probe_rows
    )
    return {"count": len(entries), "new_checkpoint_count": len(entries) - 1, "entries": entries}


def _load_complete_batch_summary(
    path: Path, *, config_sha256: str, batch_size: int, encoder_seed: int
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _load_json(path)
    if (
        payload.get("status") != "complete"
        or payload.get("config_sha256") != config_sha256
        or payload.get("batch_size") != batch_size
        or payload.get("encoder_seed") != encoder_seed
        or payload.get("counts")
        != {
            "classification_students": 4,
            "probe_lr_candidates": 60,
            "selected_probes": 20,
            "probe_official_test_evaluations": 20,
            "new_checkpoints": 24,
        }
    ):
        raise RuntimeError(
            f"existing batch{batch_size}/seed{encoder_seed} summary is incompatible"
        )
    manifest_path = path.parent / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"batch{batch_size}/seed{encoder_seed} checkpoint manifest is missing"
        )
    manifest = _load_json(manifest_path)
    if (
        manifest.get("count") != 25
        or manifest.get("new_checkpoint_count") != 24
        or len(manifest.get("entries", [])) != 25
    ):
        raise RuntimeError(
            f"batch{batch_size}/seed{encoder_seed} checkpoint manifest count changed"
        )
    for entry in manifest.get("entries", [])[1:]:
        checkpoint = Path(entry["path"])
        if not checkpoint.is_file() or file_sha256(checkpoint) != entry.get("sha256"):
            raise RuntimeError(
                f"batch{batch_size}/seed{encoder_seed} retained checkpoint changed"
            )
    log(
        f"[CUB_R50_FULL_RESUME] batch={batch_size} encoder_seed={encoder_seed} "
        "status=complete"
    )
    return payload


def _run_profile(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    config_sha256: str,
    records: dict[str, list[CubProbeRecord]],
    split_manifest: dict[str, Any],
    teacher: dict[str, Any],
    batch_size: int,
    encoder_seed: int,
    prior_profile_count: int,
    device: torch.device,
) -> dict[str, Any]:
    batch_started = time.monotonic()
    profile_dir = _profile_dir(args, batch_size, encoder_seed)
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_path = profile_dir / (
        "profile_full_summary.json"
        if args.seed_extension_full or args.loader_followup_full
        else "batch_full_summary.json"
    )
    resumed = _load_complete_batch_summary(
        summary_path,
        config_sha256=config_sha256,
        batch_size=batch_size,
        encoder_seed=encoder_seed,
    )
    if resumed is not None:
        return resumed

    validation_hash = split_manifest["validation_image_ids_sha256"]
    classification_rows = _classification_rows(
        args,
        config_sha256=config_sha256,
        validation_hash=validation_hash,
        batch_size=batch_size,
        encoder_seed=encoder_seed,
        prior_profile_count=prior_profile_count,
    )
    initial_hashes = {
        row["summary"]["initial_student_state_sha256"] for row in classification_rows
    }
    if len(initial_hashes) != 1:
        raise RuntimeError(f"paired student initialization failed for batch {batch_size}")
    if {
        row["summary"]["teacher_model_state_sha256"] for row in classification_rows
    } != {EXPECTED_TEACHER_STATE_SHA256}:
        raise RuntimeError("guided students did not share the audited teacher")
    _evaluate_deferred_classification_tests(
        args,
        classification_rows,
        config_sha256=config_sha256,
        validation_hash=validation_hash,
        batch_size=batch_size,
        encoder_seed=encoder_seed,
        device=device,
    )
    _classification_csv(
        classification_rows, profile_dir / "classification_results.csv"
    )

    targets: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation"):
        targets[split_name], _ = _target_cache(
            records[split_name],
            split=split_name,
            config=config,
            config_sha256=config_sha256,
            cache_path=args.cache_dir / "targets" / f"{split_name}.pt",
        )
    selections, selection_contracts = _select_probes(
        args,
        config=config,
        config_sha256=config_sha256,
        records=records,
        targets=targets,
        classification_rows=classification_rows,
        validation_hash=validation_hash,
        batch_size=batch_size,
        encoder_seed=encoder_seed,
        prior_profile_count=prior_profile_count,
        device=device,
    )
    selection_marker_path = (
        profile_dir / "probe" / "selection_complete_before_test.json"
    )
    test_was_started = any(
        int(row.get("official_test_evaluations", 0)) == 1 for row in selections
    )
    if test_was_started:
        if not selection_marker_path.is_file():
            raise RuntimeError("official-test resume is missing the pre-test selection marker")
        marker = _load_json(selection_marker_path)
        if (
            marker.get("status") != "complete"
            or marker.get("completed_probe_selections") != 20
            or marker.get("encoder_seed", 1) != encoder_seed
            or marker.get("official_test_masks_accessed") is not False
        ):
            raise RuntimeError("pre-test selection marker changed before resume")
    else:
        _atomic_json_save(
            {
                "status": "complete",
                "config_sha256": config_sha256,
                "batch_size": batch_size,
                "encoder_seed": encoder_seed,
                "completed_probe_selections": 20,
                "expected_probe_selections": 20,
                "official_test_masks_accessed": False,
                "selection_contracts": selection_contracts,
            },
            selection_marker_path,
        )
    log(
        f"[CUB_R50_FULL_SELECTION_COMPLETE] batch={batch_size} "
        f"encoder_seed={encoder_seed} selections=20/20 "
        "official_test_masks_accessed=false opening_test=true"
    )

    test_records, test_source = load_official_test_records(args.data_dir, download=True)
    if len(test_records) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("CUB official-test record count changed")
    test_targets, _ = _target_cache(
        test_records,
        split="official_test",
        config=config,
        config_sha256=config_sha256,
        cache_path=args.cache_dir / "targets" / "official_test.pt",
    )
    _evaluate_probe_test(
        args,
        config=config,
        config_sha256=config_sha256,
        test_records=test_records,
        test_targets=test_targets,
        classification_rows=classification_rows,
        selections=selections,
        validation_hash=validation_hash,
        batch_size=batch_size,
        encoder_seed=encoder_seed,
        prior_profile_count=prior_profile_count,
        device=device,
    )
    if len(selections) != 20 or any(
        int(row["official_test_evaluations"]) != 1 for row in selections
    ):
        raise RuntimeError("probe official-test once contract failed")
    probe_rows = _probe_flat_rows(selections)
    _write_csv(probe_rows, profile_dir / "probe" / "raw_results.csv")
    aggregates = aggregate_probe_seed_results(probe_rows)
    _atomic_json_save(
        {
            "status": "complete",
            "scientific_result": True,
            "confirmatory_main_result": (
                batch_size == 128 and not args.loader_followup_full
            ),
            "batch_size": batch_size,
            "batch_profile_role": _batch_role(
                batch_size,
                seed_extension=args.seed_extension_full,
                loader_followup=args.loader_followup_full,
            ),
            "cub_loader_profile": (
                L2_CONSERVATIVE_SPATIAL
                if args.loader_followup_full
                else "l0_current_strong"
            ),
            "encoder_seed": encoder_seed,
            "rows": probe_rows,
            "aggregates": aggregates,
        },
        profile_dir / "probe" / "results.json",
    )
    manifest = _batch_checkpoint_manifest(teacher, classification_rows, probe_rows)
    if manifest["count"] != 25 or manifest["new_checkpoint_count"] != 24:
        raise RuntimeError("batch checkpoint manifest count changed")
    _atomic_json_save(manifest, profile_dir / "checkpoint_manifest.json")
    elapsed = time.monotonic() - batch_started
    summary = {
        "status": "complete",
        "scientific_result": True,
        "confirmatory_main_result": (
            batch_size == 128 and not args.loader_followup_full
        ),
        "exploratory_selected_loader_followup": args.loader_followup_full,
        "final_confirmatory_matrix_complete": False,
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "batch_size": batch_size,
        "batch_profile_role": _batch_role(
            batch_size,
            seed_extension=args.seed_extension_full,
            loader_followup=args.loader_followup_full,
        ),
        "cub_loader_profile": (
            L2_CONSERVATIVE_SPATIAL
            if args.loader_followup_full
            else "l0_current_strong"
        ),
        "encoder_seed": encoder_seed,
        "variants": list(EXPECTED_VARIANTS),
        "counts": {
            "classification_students": 4,
            "probe_lr_candidates": 60,
            "selected_probes": 20,
            "probe_official_test_evaluations": 20,
            "new_checkpoints": 24,
        },
        "initial_student_state_sha256": next(iter(initial_hashes)),
        "teacher": teacher,
        "classification_rows": [
            {key: value for key, value in row.items() if key != "summary"}
            for row in classification_rows
        ],
        "classification_results": _classification_flat_rows(classification_rows),
        "classification_results_csv": str(
            (profile_dir / "classification_results.csv").resolve()
        ),
        "probe_aggregates": aggregates,
        "probe_results_json": str(
            (profile_dir / "probe" / "results.json").resolve()
        ),
        "selection_contracts": selection_contracts,
        "official_test_source": test_source,
        "official_test_used_for_selection": False,
        "elapsed_seconds": elapsed,
        "checkpoint_manifest": str(
            (profile_dir / "checkpoint_manifest.json").resolve()
        ),
    }
    _atomic_json_save(summary, summary_path)
    log(
        f"[CUB_R50_FULL_BATCH{batch_size}_SEED{encoder_seed}_"
        "CLASSIFICATION_RESULTS]"
    )
    for row in classification_rows:
        result = row["summary"]
        official_test = row.get("official_test", result.get("official_test"))
        if not isinstance(official_test, dict):
            raise RuntimeError("classification official-test metrics are missing")
        log(
            f"[CUB_R50_FULL_CLASSIFICATION_RESULT] batch={batch_size} "
            f"variant={row['variant']} encoder_seed={encoder_seed} selected_epoch="
            f"{result['selected_epoch']} test_macro_top1="
            f"{official_test['macro_top1']:.4f}"
        )
    log(f"[CUB_R50_FULL_BATCH{batch_size}_SEED{encoder_seed}_PROBE_RESULTS]")
    for aggregate in aggregates:
        log(
            f"[CUB_R50_FULL_PROBE_AGGREGATE] batch={batch_size} "
            f"encoder_seed={encoder_seed} "
            f"variant={aggregate['variant']} test_input_miou_probe_seed_mean="
            f"{aggregate['mean_over_probe_seeds']:.6f} test_input_miou_probe_seed_sd="
            f"{aggregate['sample_standard_deviation_over_probe_seeds']:.6f} "
            "independent_encoder_n=1 probe_seeds_per_encoder=5"
        )
    log(
        f"[CUB_R50_FULL_PROFILE_DONE] batch={batch_size} "
        f"encoder_seed={encoder_seed} classification=4/4 "
        f"probe_candidates=60/60 selections=20/20 test_once=20/20 "
        f"new_checkpoints=24 elapsed={format_duration(elapsed)}"
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUB R50 guided full run requires CUDA")
    if (
        args.num_workers != 4
        or args.feature_batch_size != 32
        or args.eval_batch_size != 200
    ):
        raise ValueError(
            "locked CUB R50 loaders require workers=4, feature-batch=32, "
            "eval-batch=200"
        )
    profiles = _execution_profiles(args)
    config_path = args.config.expanduser().resolve()
    args.config = config_path
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    config = _load_json(config_path)
    if args.loader_followup_full:
        _validate_loader_followup_config(config, config_path)
        parent_path = _resolve_repository_path(
            config["protocol_provenance"]["pre_smoke_locked_full"]["path"]
        )
        runtime_config = _load_json(parent_path)
        runtime_config["protocol_id"] = config["protocol_id"]
    elif args.seed_extension_full:
        _validate_seed_extension_config(
            config, config_path, partition=str(args.partition)
        )
        reference_config = _load_json(DEFAULT_CONFIG)
        runtime_config = dict(config)
        runtime_config["frozen_probe"] = reference_config["frozen_probe"]
    else:
        _validate_config(config, config_path)
        runtime_config = config
    config_sha256 = file_sha256(config_path)
    if not args.teacher_checkpoint.is_file():
        raise RuntimeError(f"audited teacher checkpoint is missing: {args.teacher_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _write_status(
        args.output_dir / "sequence_status.json",
        status="running",
        phase="teacher_and_dataset_audit",
        batch_size=None,
        classification_complete=0,
        probe_candidates_complete=0,
        probe_selections_complete=0,
        test_evaluations_complete=0,
        profile_total=len(profiles),
    )
    log("=" * 104)
    if args.loader_followup_full:
        log(
            "CUB PHASE 1 — RESNET-50/224 SELECTED L2 GUIDED "
            "SEED-1 PRELIMINARY FULL -> PROBE"
        )
    elif args.seed_extension_full:
        log(
            "CUB PHASE 1 — RESNET-50/224 GUIDED SEED EXTENSION FULL -> PROBE"
        )
    else:
        log(
            "CUB PHASE 1 — RESNET-50/224 GUIDED SEED-1 "
            "BATCH 128 + 64 FULL -> PROBE"
        )
    log("=" * 104)
    profile_text = ",".join(
        f"b{batch_size}s{encoder_seed}"
        for batch_size, encoder_seed in profiles
    )
    log(
        f"[CUB_R50_FULL_POLICY] config_sha256={config_sha256} "
        f"partition={args.partition} profiles={profile_text} "
        "variants=lg,alg_warmup20,ibkd_lambda_0.25,ibkd_lambda_0.5 "
        "probe_seeds=1,2,3,4,5 final_confirmatory_matrix_complete=false"
    )

    teacher_model, teacher_metadata, teacher_hash, teacher_state_hash = (
        load_scientific_teacher(
            args.teacher_checkpoint,
            device=device,
            validation_hash=EXPECTED_VALIDATION_HASH,
        )
    )
    del teacher_model
    torch.cuda.empty_cache()
    if (
        teacher_hash != EXPECTED_TEACHER_SHA256
        or teacher_state_hash != EXPECTED_TEACHER_STATE_SHA256
    ):
        raise RuntimeError("issue-722 teacher checkpoint identity changed")
    teacher = {
        "source_h200_issue": 722,
        "reused": True,
        "checkpoint": str(args.teacher_checkpoint),
        "checkpoint_sha256": teacher_hash,
        "model_state_sha256": teacher_state_hash,
        "metadata": teacher_metadata,
    }
    log(
        "[CUB_R50_FULL_TEACHER_REUSED] issue=722 strict_load=true frozen=true "
        f"checkpoint_sha256={teacher_hash} model_state_sha256={teacher_state_hash}"
    )

    records, split_manifest, source = load_train_validation_records(
        args.data_dir, download=True
    )
    if (
        len(records["train"]) != DERIVED_TRAIN_COUNT
        or len(records["validation"]) != DERIVED_VALIDATION_COUNT
        or split_manifest.get("validation_image_ids_sha256")
        != EXPECTED_VALIDATION_HASH
    ):
        raise RuntimeError("CUB train/validation split contract failed")
    archives = _archive_audit(args.data_dir)
    _atomic_json_save(
        {
            "status": "pass",
            "source": source,
            "archives": archives,
            "split_manifest": split_manifest,
            "counts": {"train": 5394, "validation": 600, "test": 5794},
            "official_test_masks_opened_only_after_each_profile_selection_marker": True,
        },
        args.output_dir / "dataset_audit.json",
    )

    profile_summaries = []
    for index, (batch_size, encoder_seed) in enumerate(profiles):
        profile_summaries.append(
            _run_profile(
                args,
                config=runtime_config,
                config_sha256=config_sha256,
                records=records,
                split_manifest=split_manifest,
                teacher=teacher,
                batch_size=batch_size,
                encoder_seed=encoder_seed,
                prior_profile_count=index,
                device=device,
            )
        )
    initial_hashes_by_seed: dict[int, set[str]] = defaultdict(set)
    for profile_summary in profile_summaries:
        initial_hashes_by_seed[int(profile_summary["encoder_seed"])].add(
            profile_summary["initial_student_state_sha256"]
        )
    if any(len(values) != 1 for values in initial_hashes_by_seed.values()):
        raise RuntimeError("student initialization changed within an encoder seed")
    distinct_initial_hashes = {
        next(iter(values)) for values in initial_hashes_by_seed.values()
    }
    if args.seed_extension_full and len(distinct_initial_hashes) != len(
        initial_hashes_by_seed
    ):
        raise RuntimeError("different encoder seeds reused the same initialization")
    if args.batch_profile_full and len(distinct_initial_hashes) != 1:
        raise RuntimeError("student seed-1 initialization changed between batches")

    manifest_entries = [
        {
            "kind": "external_shared_teacher",
            "source_h200_issue": 722,
            "path": teacher["checkpoint"],
            "sha256": teacher_hash,
        }
    ]
    for batch_size, encoder_seed in profiles:
        profile_dir = _profile_dir(args, batch_size, encoder_seed)
        batch_manifest = _load_json(
            profile_dir / "checkpoint_manifest.json"
        )
        manifest_entries.extend(batch_manifest["entries"][1:])
    manifest = {
        "count": len(manifest_entries),
        "external_shared_teacher_count": 1,
        "new_checkpoint_count": len(manifest_entries) - 1,
        "entries": manifest_entries,
    }
    expected_new_checkpoints = len(profiles) * 24
    if (
        manifest["count"] != expected_new_checkpoints + 1
        or manifest["new_checkpoint_count"] != expected_new_checkpoints
    ):
        raise RuntimeError("combined checkpoint manifest count changed")
    _atomic_json_save(manifest, args.output_dir / "checkpoint_manifest.json")
    elapsed = time.monotonic() - started
    summary = {
        "status": "complete",
        "scientific_result": True,
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "mode": (
            "selected_l2_guided_preliminary_v1"
            if args.loader_followup_full
            else "seed_extension_v5"
            if args.seed_extension_full
            else "batch_profile_v4"
        ),
        "partition": args.partition,
        "profile_order": [
            {"batch_size": batch_size, "encoder_seed": encoder_seed}
            for batch_size, encoder_seed in profiles
        ],
        "variants": list(EXPECTED_VARIANTS),
        "encoder_seeds": sorted(initial_hashes_by_seed),
        "final_confirmatory_matrix_complete": False,
        "batch128_eligible_for_locked_v3_matrix": not args.loader_followup_full,
        "exploratory_selected_loader_followup": args.loader_followup_full,
        "cub_loader_profile": (
            L2_CONSERVATIVE_SPATIAL
            if args.loader_followup_full
            else "l0_current_strong"
        ),
        "batch64_is_sensitivity_profile": any(
            batch_size == 64 for batch_size, _ in profiles
        ),
        "counts": {
            "teacher_reused": 1,
            "classification_students": len(profiles) * 4,
            "probe_lr_candidates": len(profiles) * 60,
            "selected_probes": len(profiles) * 20,
            "classification_official_test_evaluations": len(profiles) * 4,
            "probe_official_test_evaluations": len(profiles) * 20,
            "new_checkpoints": expected_new_checkpoints,
        },
        "teacher": teacher,
        "same_initial_student_state_across_variants_within_encoder_seed": True,
        "different_initial_student_state_across_encoder_seeds": (
            len(distinct_initial_hashes) == len(initial_hashes_by_seed)
        ),
        "initial_student_state_sha256_by_seed": {
            str(seed): next(iter(values))
            for seed, values in sorted(initial_hashes_by_seed.items())
        },
        "profiles": profile_summaries,
        "checkpoint_manifest": str(
            (args.output_dir / "checkpoint_manifest.json").resolve()
        ),
        "elapsed_seconds": elapsed,
        "runtime": _runtime(device),
    }
    _atomic_json_save(summary, args.output_dir / "combined_full_summary.json")
    _write_status(
        args.output_dir / "sequence_status.json",
        status="complete",
        phase="complete",
        batch_size=None,
        classification_complete=len(profiles) * 4,
        probe_candidates_complete=len(profiles) * 60,
        probe_selections_complete=len(profiles) * 20,
        test_evaluations_complete=len(profiles) * 20,
        profile_total=len(profiles),
    )
    log(
        "[CUB_R50_L2_GUIDED_PRELIMINARY_FINAL_RESULTS]"
        if args.loader_followup_full
        else "[CUB_R50_SEED_EXTENSION_FINAL_RESULTS]"
        if args.seed_extension_full
        else "[CUB_R50_BATCH_PROFILE_FINAL_RESULTS]"
    )
    for profile_summary in profile_summaries:
        for result in profile_summary["classification_results"]:
            log(
                f"[CUB_R50_BATCH_PROFILE_CLASSIFICATION_RESULT] "
                f"batch={profile_summary['batch_size']} "
                f"encoder_seed={profile_summary['encoder_seed']} "
                f"variant={result['variant']} "
                f"selected_epoch={result['selected_epoch']} validation_macro_top1="
                f"{result['validation_macro_top1']:.4f} test_macro_top1="
                f"{result['test_macro_top1']:.4f}"
            )
        for aggregate in profile_summary["probe_aggregates"]:
            log(
                f"[CUB_R50_BATCH_PROFILE_RESULT] batch={profile_summary['batch_size']} "
                f"encoder_seed={profile_summary['encoder_seed']} "
                f"role={profile_summary['batch_profile_role']} variant="
                f"{aggregate['variant']} test_input_miou_probe_seed_mean="
                f"{aggregate['mean_over_probe_seeds']:.6f} probe_seed_sd="
                f"{aggregate['sample_standard_deviation_over_probe_seeds']:.6f}"
            )
    if args.loader_followup_full:
        log(
            "[CUB_R50_L2_GUIDED_PRELIMINARY_FULL_DONE] status=complete "
            "profiles=1/1 classification=4/4 probe_candidates=60/60 "
            "selections=20/20 test_once=20/20 "
            f"new_checkpoints={expected_new_checkpoints} "
            f"elapsed={format_duration(elapsed)} "
            f"summary={(args.output_dir / 'combined_full_summary.json').resolve()}"
        )
    elif args.seed_extension_full:
        log(
            "[CUB_R50_SEED_EXTENSION_FULL_DONE] status=complete "
            f"partition={args.partition} profiles={len(profiles)}/{len(profiles)} "
            f"classification={len(profiles) * 4}/{len(profiles) * 4} "
            f"probe_candidates={len(profiles) * 60}/{len(profiles) * 60} "
            f"selections={len(profiles) * 20}/{len(profiles) * 20} "
            f"test_once={len(profiles) * 20}/{len(profiles) * 20} "
            f"new_checkpoints={expected_new_checkpoints} "
            f"elapsed={format_duration(elapsed)} "
            f"summary={(args.output_dir / 'combined_full_summary.json').resolve()}"
        )
    else:
        log(
            "[CUB_R50_BATCH_PROFILE_FULL_DONE] status=complete batches=2/2 "
            "classification=8/8 probe_candidates=120/120 selections=40/40 "
            f"test_once=40/40 new_checkpoints=48 elapsed={format_duration(elapsed)} "
            f"summary={(args.output_dir / 'combined_full_summary.json').resolve()}"
        )
    return summary


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_save(
            {
                "status": "failed",
                "scientific_result": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            args.output_dir / "failure.json",
        )
        marker = (
            "CUB_R50_L2_GUIDED_PRELIMINARY_FULL_FAILED"
            if args.loader_followup_full
            else "CUB_R50_SEED_EXTENSION_FULL_FAILED"
            if args.seed_extension_full
            else "CUB_R50_BATCH_PROFILE_FULL_FAILED"
        )
        log(f"[{marker}] {type(error).__name__}:{error}")
        raise


if __name__ == "__main__":
    main()
