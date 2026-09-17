#!/usr/bin/env python3
"""Run one locked CUB main-L0 iBKD connection-ablation seed shard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .cub_data import DATASET_NAME, NUM_CLASSES, build_official_test_loader
from .cub_loader_profiles import L0_CURRENT_STRONG
from .cub_probe_data import (
    CubProbeRecord,
    load_official_test_records,
    load_train_validation_records,
)
from .models import IBKD_AGGREGATION_MODES, create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .cub_experiment_support import (
    _atomic_json_save,
    _atomic_torch_save,
    _feature_cache,
    _finite_metrics,
    _load_json,
    _runtime,
    _target_cache,
)
from .run_cub_r50_batch_profile_full import _archive_audit
from .run_cub_r50_teacher_full import load_scientific_teacher
from .train_timing import evaluate, file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/mechanism_analysis/configs/"
    "cub200_r50_224_b128_main_l0_ibkd_connection_full_v1.json"
)
EXECUTION_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/mechanism_analysis/configs/"
    "cub200_r50_224_b128_main_l0_ibkd_connection_full_execution_v1.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "1650e76da40c235292fca166b8058c1a9ee279165a275b59122f505f3b7235d8"
)
EXPECTED_EXECUTION_SHA256 = (
    "145eb88ebba5134f8fcd4b5bf11861465407baff15aaaa41c69809762f0e469c"
)
EXPECTED_MATCHED_PROTOCOL_SHA256 = (
    "a2fadff0b93e1878b27cbd78e9249141b67202bcd6911ce27f4b19810ea3ca5b"
)
EXPECTED_MATCHED_EXECUTION_SHA256 = (
    "30ef6886128a23bce5ee991ce55060cff93f2e74761397203dd8310660e52572"
)
EXPECTED_BASE_SHA256 = (
    "e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc"
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
AGGREGATION_VARIANTS = (
    "learned_all",
    "fixed_uniform_all",
    "fixed_stage_match",
    "fixed_last",
)
PROBE_SEEDS = (1, 2, 3, 4, 5)
ROLE = "posthoc_main_l0_mechanism_ablation"
CLASSIFICATION_PURPOSE = (
    "phase1_cub_r50_224_main_l0_ibkd_connection_full_student_v1"
)
PROBE_PURPOSE = "phase1_cub_r50_224_main_l0_ibkd_connection_full_probe_v1"
MATCHED_CLASSIFICATION_PURPOSE = (
    "phase1_cub_r50_224_main_l0_ibkd_connection_matched_full_student_v2"
)
MATCHED_PROBE_PURPOSE = (
    "phase1_cub_r50_224_main_l0_ibkd_connection_matched_full_probe_v2"
)


def log(message: str = "") -> None:
    print(message, flush=True)


def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_matched_configs(
    protocol: dict[str, Any],
    protocol_path: Path,
    execution: dict[str, Any],
    execution_path: Path,
    encoder_seed: int,
) -> tuple[dict[str, Any], Path]:
    if file_sha256(execution_path) != EXPECTED_MATCHED_EXECUTION_SHA256:
        raise RuntimeError("matched mechanism execution config SHA-256 changed")
    base_path = (
        REPOSITORY_ROOT
        / "phase1/phase1_cub/configs/cub200_r50_224_b128_full_v3.json"
    )
    base = _load_json(base_path)
    dataset = protocol.get("dataset", {})
    student = protocol.get("student", {})
    guidance = protocol.get("guidance", {})
    probe = protocol.get("frozen_probe", {})
    shared = execution.get("shared", {})
    checks = {
        "protocol_id": protocol.get("protocol_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_connection_matched_duration_full_v2",
        "scientific": protocol.get("scientific_result") is True
        and protocol.get("posthoc_mechanism_analysis") is True,
        "lineage": protocol.get("lineage", {}).get("input_lineage_id")
        == "main_l0_v3"
        and protocol.get("lineage", {}).get("source_teacher_issue") == 722
        and protocol.get("lineage", {}).get("source_canonical_student_issue")
        == 727,
        "base": base_path.is_file()
        and file_sha256(base_path) == EXPECTED_BASE_SHA256,
        "dataset": dataset.get("name") == DATASET_NAME
        and dataset.get("train") == 5394
        and dataset.get("validation") == 600
        and dataset.get("official_test") == 5794
        and dataset.get("split_seed") == 2027
        and dataset.get("validation_image_ids_sha256")
        == EXPECTED_VALIDATION_HASH
        and dataset.get("loader_profile") == L0_CURRENT_STRONG
        and dataset.get("input_size") == 224,
        "teacher": protocol.get("teacher", {}).get("checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and protocol.get("teacher", {}).get("model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "classification": student.get("batch_size") == 128
        and student.get("encoder_seeds") == [1, 2, 3]
        and student.get("epochs") == 300
        and student.get("fusion_ratio_lambda") == 0.25
        and student.get("precision") == "float32",
        "fixed_guidance": guidance.get("beta_on") == 2.5
        and guidance.get("policy") == "fixed_common_horizon"
        and guidance.get("active_epochs_inclusive") == [1, 123]
        and guidance.get("beta_zero_from_epoch") == 124
        and guidance.get("adaptive_controller_selects_stop") is False
        and guidance.get("same_duration_for_every_variant_and_seed") is True,
        "variants": protocol.get("aggregation_variants")
        == list(AGGREGATION_VARIANTS)
        and set(AGGREGATION_VARIANTS).issubset(set(IBKD_AGGREGATION_MODES)),
        "probe": probe.get("run_after_classification") is True
        and probe.get("encoder_frozen") is True
        and probe.get("probe_seeds") == list(PROBE_SEEDS)
        and probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 100,
        "test_policy": protocol.get("official_test_policy", {}).get(
            "smoke_access"
        )
        is False
        and protocol.get("official_test_policy", {}).get("selection_uses_test")
        is False,
        "execution_id": execution.get("execution_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_connection_matched_duration_full_execution_v2",
        "execution_ready": str(execution.get("status", "")).startswith(
            "ready_for_unconditional_three_seed_full"
        )
        and execution.get("release_decision", {}).get("source_h200_issue") == 776
        and execution.get("release_decision", {}).get("smoke_variants_complete")
        == 4
        and execution.get("release_decision", {}).get(
            "smoke_cross_gates_complete"
        )
        == 6
        and execution.get("release_decision", {}).get(
            "all_encoder_seeds_run_regardless_of_earlier_results"
        )
        is True
        and execution.get("release_decision", {}).get("retain_smoke_outputs")
        is False,
        "execution_protocol": execution.get("scientific_protocol")
        == {
            "path": str(protocol_path.relative_to(REPOSITORY_ROOT)),
            "sha256": EXPECTED_MATCHED_PROTOCOL_SHA256,
        },
        "execution_base": execution.get("base_protocol")
        == {
            "path": str(base_path.relative_to(REPOSITORY_ROOT)),
            "sha256": EXPECTED_BASE_SHA256,
        },
        "execution_shared": shared.get("teacher_source_h200_issue") == 722
        and shared.get("teacher_checkpoint_sha256") == EXPECTED_TEACHER_SHA256
        and shared.get("teacher_model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256
        and shared.get("batch_size") == 128
        and shared.get("classification_epochs") == 300
        and shared.get("fixed_guidance_epochs_inclusive") == 123
        and shared.get("beta_zero_from_epoch") == 124
        and shared.get("aggregation_variants") == list(AGGREGATION_VARIANTS)
        and shared.get("probe_seeds") == list(PROBE_SEEDS)
        and shared.get("probe_learning_rates") == [0.01, 0.03, 0.1]
        and shared.get("probe_epochs") == 100
        and shared.get("official_test_after_all_validation_selections_within_shard")
        is True,
        "seed_shard": encoder_seed in {1, 2, 3}
        and encoder_seed
        in {
            int(row.get("encoder_seed", -1))
            for row in execution.get("h200_shards", [])
        },
        "one_seed_per_issue": [
            (row.get("encoder_seed"), row.get("planned_issue_group"))
            for row in execution.get("h200_shards", [])
        ]
        == [(1, 1), (2, 2), (3, 3)],
        "runtime": execution.get("runtime", {}).get("requested_mig_slices") == 7
        and execution.get("runtime", {}).get("num_workers") == 0
        and execution.get("runtime", {}).get("ten_hour_limit_seconds") == 36000,
        "per_shard_gate": execution.get("per_shard_completion_gate")
        == {
            "teacher_download_and_audit": 1,
            "classification_students": 4,
            "classification_validation_selections": 4,
            "classification_official_test_evaluations": 4,
            "segmentation_probe_lr_candidates": 60,
            "segmentation_probe_validation_selections": 20,
            "segmentation_probe_official_test_evaluations": 20,
            "retained_new_checkpoints": 24,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid matched CUB iBKD connection full contract: "
            + ", ".join(failures)
        )
    return base, base_path


def _validate_configs(
    protocol: dict[str, Any],
    protocol_path: Path,
    execution: dict[str, Any],
    execution_path: Path,
    encoder_seed: int,
) -> tuple[dict[str, Any], Path]:
    protocol_sha256 = file_sha256(protocol_path)
    if protocol_sha256 == EXPECTED_MATCHED_PROTOCOL_SHA256:
        return _validate_matched_configs(
            protocol,
            protocol_path,
            execution,
            execution_path,
            encoder_seed,
        )
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("mechanism scientific protocol SHA-256 changed")
    if file_sha256(execution_path) != EXPECTED_EXECUTION_SHA256:
        raise RuntimeError("mechanism execution config SHA-256 changed")
    base_info = protocol.get("lineage", {}).get("base_protocol", {})
    base_path = _resolve_repository_path(str(base_info.get("path")))
    base = _load_json(base_path)
    variants = [
        row.get("id")
        for row in protocol.get("classification", {}).get("variants", [])
    ]
    frozen = protocol.get("frozen_segmentation_probe", {})
    checks = {
        "protocol_id": protocol.get("protocol_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_layer_connection_ablation_full_v1",
        "scientific": protocol.get("scientific_result") is True
        and protocol.get("posthoc_mechanism_analysis") is True,
        "lineage": protocol.get("lineage", {}).get("input_lineage_id")
        == "main_l0_v3"
        and protocol.get("lineage", {}).get("loader_profile")
        == L0_CURRENT_STRONG,
        "base": base_path.is_file()
        and base_info.get("sha256") == EXPECTED_BASE_SHA256
        and file_sha256(base_path) == EXPECTED_BASE_SHA256,
        "dataset": protocol.get("dataset", {}).get("name") == DATASET_NAME
        and protocol.get("dataset", {}).get("num_classes") == NUM_CLASSES
        and protocol.get("dataset", {}).get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_HASH,
        },
        "teacher": protocol.get("teacher", {}).get("source_h200_issue") == 722
        and protocol.get("teacher", {}).get("checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and protocol.get("teacher", {}).get("model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "classification": protocol.get("classification", {}).get("batch_size") == 128
        and protocol.get("classification", {}).get("epochs") == 300
        and protocol.get("classification", {}).get("encoder_seeds") == [1, 2, 3]
        and protocol.get("classification", {}).get("fusion_ratio_lambda") == 0.25
        and variants == list(AGGREGATION_VARIANTS),
        "known_modes": set(AGGREGATION_VARIANTS).issubset(
            set(IBKD_AGGREGATION_MODES)
        ),
        "probe": frozen.get("probe_seeds") == list(PROBE_SEEDS)
        and frozen.get("learning_rates") == [0.01, 0.03, 0.1]
        and frozen.get("epochs") == 100
        and frozen.get("batch_size") == 64,
        "test_policy": protocol.get("official_test_policy", {}).get(
            "test_used_for_selection"
        )
        is False
        and protocol.get("official_test_policy", {}).get(
            "settings_changed_from_test"
        )
        is False,
        "execution_id": execution.get("execution_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_connection_full_execution_v1",
        "execution_ready": str(execution.get("status", "")).startswith(
            "ready_for_unconditional_three_seed_full"
        )
        and execution.get("release_decision", {}).get(
            "all_encoder_seeds_run_regardless_of_earlier_results"
        )
        is True
        and execution.get("release_decision", {}).get("retain_smoke_outputs")
        is False,
        "execution_protocol": execution.get("scientific_protocol")
        == {
            "path": str(protocol_path.relative_to(REPOSITORY_ROOT)),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "execution_base": execution.get("base_protocol")
        == {
            "path": str(base_path.relative_to(REPOSITORY_ROOT)),
            "sha256": EXPECTED_BASE_SHA256,
        },
        "execution_shared": execution.get("shared")
        == {
            "teacher_source_h200_issue": 722,
            "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
            "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
            "batch_size": 128,
            "classification_epochs": 300,
            "aggregation_variants": list(AGGREGATION_VARIANTS),
            "probe_seeds": list(PROBE_SEEDS),
            "probe_learning_rates": [0.01, 0.03, 0.1],
            "probe_epochs": 100,
            "official_test_after_all_validation_selections_within_shard": True,
        },
        "seed_shard": encoder_seed in {1, 2, 3}
        and encoder_seed
        in {
            int(row.get("encoder_seed", -1))
            for row in execution.get("h200_shards", [])
        },
        "one_seed_per_issue": [
            (row.get("encoder_seed"), row.get("planned_issue_group"))
            for row in execution.get("h200_shards", [])
        ]
        == [(1, 1), (2, 2), (3, 3)],
        "per_shard_gate": execution.get("per_shard_completion_gate")
        == {
            "teacher_download_and_audit": 1,
            "classification_students": 4,
            "classification_validation_selections": 4,
            "classification_official_test_evaluations": 4,
            "segmentation_probe_lr_candidates": 60,
            "segmentation_probe_validation_selections": 20,
            "segmentation_probe_official_test_evaluations": 20,
            "retained_new_checkpoints": 24,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB iBKD connection full contract: " + ", ".join(failures)
        )
    return base, base_path


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    classification_complete: int,
    probe_candidates_complete: int,
    probe_selections_complete: int,
    classification_tests_complete: int,
    probe_tests_complete: int,
    active: str | None = None,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "classification_complete": classification_complete,
            "classification_expected": 4,
            "probe_candidates_complete": probe_candidates_complete,
            "probe_candidates_expected": 60,
            "probe_selections_complete": probe_selections_complete,
            "probe_selections_expected": 20,
            "classification_tests_complete": classification_tests_complete,
            "classification_tests_expected": 4,
            "probe_tests_complete": probe_tests_complete,
            "probe_tests_expected": 20,
            "official_test_used_for_selection": False,
            "active": active,
            "failure": failure,
        },
        output_dir / "sequence_status.json",
    )


def _run_or_resume(
    command: list[str],
    summary_path: Path,
    checkpoint_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if summary_path.is_file():
        summary = _load_json(summary_path)
        if (
            summary.get("status") == "complete"
            and checkpoint_path.is_file()
            and summary.get("checkpoint_sha256") == file_sha256(checkpoint_path)
        ):
            log(f"[MECHANISM_FULL_TASK_RESUME] label={label}")
            return summary
    log(f"[MECHANISM_FULL_TASK_START] label={label} command={' '.join(command)}")
    subprocess.run(command, check=True)
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"full task did not produce required artifacts: {label}")
    summary = _load_json(summary_path)
    log(f"[MECHANISM_FULL_TASK_DONE] label={label}")
    return summary


def _load_encoder(
    row: dict[str, Any],
    *,
    protocol_sha256: str,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(row["checkpoint_path"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    mode = str(row["aggregation_mode"])
    expected = {
        "purpose": (
            MATCHED_CLASSIFICATION_PURPOSE
            if protocol_sha256 == EXPECTED_MATCHED_PROTOCOL_SHA256
            else CLASSIFICATION_PURPOSE
        ),
        "scientific_result": True,
        "confirmatory_main_result": False,
        "canonical_phase1_result_replaced": False,
        "mechanism_ablation_full": True,
        "ibkd_aggregation_mode": mode,
        "cub_loader_profile": L0_CURRENT_STRONG,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": "ibkd",
        "fusion_ratio_lambda": 0.25,
        "batch_size": 128,
        "epochs": 300,
        "seed": row["encoder_seed"],
        "validation_image_ids_sha256": validation_hash,
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
        "teacher_architecture": "resnet50_224_scratch",
        "protocol_config_sha256": protocol_sha256,
        "batch_profile_role": ROLE,
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    if protocol_sha256 == EXPECTED_MATCHED_PROTOCOL_SHA256:
        expected["ibkd_fixed_guidance_epochs"] = 123
    failures = [key for key, value in expected.items() if metadata.get(key) != value]
    aggregation = metadata.get("ibkd_aggregation")
    if not isinstance(aggregation, dict) or aggregation.get("mode") != mode:
        failures.append("ibkd_aggregation")
    if failures:
        raise RuntimeError(
            f"mechanism encoder metadata mismatch {mode}: " + ",".join(failures)
        )
    checkpoint_hash = file_sha256(path)
    if row["summary"].get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"mechanism encoder checkpoint hash changed: {mode}")
    model = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"mechanism encoder strict load failed: {mode}")
    state_hash = state_dict_sha256(model)
    if state_hash != metadata.get("student_state_sha256"):
        raise RuntimeError(f"mechanism encoder state hash changed: {mode}")
    model.to(device).eval().requires_grad_(False)
    audit = {
        "strict_load": True,
        "eval_mode": not model.training,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "checkpoint_sha256": checkpoint_hash,
        "student_state_sha256": state_hash,
        "aggregation": aggregation,
    }
    if not audit["eval_mode"] or audit["trainable_parameter_count"] != 0:
        raise RuntimeError(f"mechanism encoder freeze failed: {mode}")
    return model, audit


def _train_classifiers(
    args: argparse.Namespace,
    *,
    protocol_sha256: str,
    validation_hash: str,
    teacher_hash: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = args.output_dir / "classification" / "students"
    matched_duration = protocol_sha256 == EXPECTED_MATCHED_PROTOCOL_SHA256
    for mode in AGGREGATION_VARIANTS:
        _write_status(
            args.output_dir,
            status="running",
            phase="classification_validation_selection",
            classification_complete=len(rows),
            probe_candidates_complete=0,
            probe_selections_complete=0,
            classification_tests_complete=0,
            probe_tests_complete=0,
            active=mode,
        )
        run_name = (
            f"cub_r50_224_ibkd_lambda_0.25_{mode}_b128_full_300ep_"
            f"seed{args.encoder_seed}"
        )
        run_dir = root / run_name
        checkpoint_path = run_dir / "student_best_validation.pt"
        summary_path = run_dir / "summary.json"
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
            "ibkd",
            "--fusion-ratio",
            "0.25",
            "--teacher-architecture",
            "resnet50_224_scratch",
            "--scientific-cub-r50-teacher",
            "--mechanism-ablation-full",
            "--ibkd-aggregation-mode",
            mode,
            "--defer-official-test",
            "--cub-loader-profile",
            L0_CURRENT_STRONG,
            "--protocol-config",
            str(args.protocol_config),
            "--batch-profile-role",
            ROLE,
            "--batch-size",
            "128",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(root),
            "--run-name",
            run_name,
            "--teacher-checkpoint",
            str(args.teacher_checkpoint),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.encoder_seed),
        ]
        if matched_duration:
            command.extend(["--ibkd-fixed-guidance-epochs", "123"])
        summary = _run_or_resume(
            command,
            summary_path,
            checkpoint_path,
            label=f"classification_seed{args.encoder_seed}_{mode}",
        )
        expected = {
            "status": "complete",
            "scientific_result": True,
            "confirmatory_main_result": False,
            "mechanism_ablation_full": True,
            "ibkd_aggregation_mode": mode,
            "official_test": None,
            "official_test_evaluations": 0,
            "official_test_accessed": False,
            "dataset": DATASET_NAME,
            "method": "ibkd",
            "fusion_ratio_lambda": 0.25,
            "batch_size": 128,
            "epochs": 300,
            "seed": args.encoder_seed,
            "protocol_config_sha256": protocol_sha256,
            "batch_profile_role": ROLE,
            "teacher_checkpoint_sha256": teacher_hash,
        }
        if matched_duration:
            expected["ibkd_fixed_guidance_epochs"] = 123
        failures = [key for key, value in expected.items() if summary.get(key) != value]
        controller = summary.get("controller_final") or {}
        if matched_duration and not (
            controller.get("stop_policy") == "fixed_epoch"
            and controller.get("fixed_stop_epoch") == 123
            and controller.get("stop_epoch") == 123
            and controller.get("active") is False
            and controller.get("beta_history")
            == [2.5] * 123 + [0.0] * 177
        ):
            failures.append("fixed_guidance_schedule")
        if failures:
            raise RuntimeError(
                f"mechanism classification summary mismatch {mode}: "
                + ",".join(failures)
            )
        if summary.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise RuntimeError(f"mechanism checkpoint missing or changed: {mode}")
        row = {
            "aggregation_mode": mode,
            "encoder_seed": args.encoder_seed,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "summary_path": str(summary_path.resolve()),
            "summary": summary,
        }
        model, _ = _load_encoder(
            row,
            protocol_sha256=protocol_sha256,
            validation_hash=validation_hash,
            device=device,
        )
        del model
        torch.cuda.empty_cache()
        rows.append(row)
    initial_hashes = {
        row["summary"]["initial_student_state_sha256"] for row in rows
    }
    if len(initial_hashes) != 1:
        raise RuntimeError("paired student initialization changed across connections")
    if matched_duration:
        input_streams = {
            tuple(
                epoch.get("input_stream_sha256")
                for epoch in row["summary"].get("history", [])
            )
            for row in rows
        }
        if len(input_streams) != 1 or len(next(iter(input_streams), ())) != 300:
            raise RuntimeError(
                "paired augmented input stream changed across matched connections"
            )
    _atomic_json_save(
        {
            "status": "complete",
            "encoder_seed": args.encoder_seed,
            "protocol_sha256": protocol_sha256,
            "completed_validation_selections": 4,
            "expected_validation_selections": 4,
            "paired_initial_student_state_sha256": next(iter(initial_hashes)),
            "paired_epoch_input_stream_sha256": (
                list(next(iter(input_streams))) if matched_duration else None
            ),
            "fixed_guidance_epochs": 123 if matched_duration else None,
            "official_test_accessed": False,
        },
        args.output_dir / "classification_selection_complete_before_test.json",
    )
    return rows


def _train_probes(
    args: argparse.Namespace,
    *,
    base: dict[str, Any],
    protocol_sha256: str,
    execution_sha256: str,
    records: dict[str, list[CubProbeRecord]],
    split_manifest: dict[str, Any],
    classification_rows: Sequence[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    validation_hash = str(split_manifest["validation_image_ids_sha256"])
    runtime_config = {"frozen_probe": base["frozen_probe"]}
    probe_config = runtime_config["frozen_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    if learning_rates != [0.01, 0.03, 0.1]:
        raise RuntimeError("locked probe learning rates changed")
    targets: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        targets[split], _ = _target_cache(
            records[split],
            split=split,
            config=runtime_config,
            config_sha256=protocol_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
    selections: list[dict[str, Any]] = []
    for classification in classification_rows:
        mode = str(classification["aggregation_mode"])
        model, encoder_audit = _load_encoder(
            classification,
            protocol_sha256=protocol_sha256,
            validation_hash=validation_hash,
            device=device,
        )
        features: dict[str, dict[str, Any]] = {}
        feature_paths: list[Path] = []
        for split in ("train", "validation"):
            feature_path = (
                args.cache_dir
                / "features"
                / mode
                / f"seed{args.encoder_seed}_{split}.pt"
            )
            feature_paths.append(feature_path)
            features[split], _ = _feature_cache(
                model,
                records[split],
                split=split,
                variant=mode,
                encoder_seed=args.encoder_seed,
                checkpoint_sha256=encoder_audit["checkpoint_sha256"],
                state_sha256=encoder_audit["student_state_sha256"],
                config=runtime_config,
                config_sha256=protocol_sha256,
                cache_path=feature_path,
                device=device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
        del model
        torch.cuda.empty_cache()
        for probe_seed in PROBE_SEEDS:
            _write_status(
                args.output_dir,
                status="running",
                phase="probe_validation_selection",
                classification_complete=4,
                probe_candidates_complete=len(selections) * 3,
                probe_selections_complete=len(selections),
                classification_tests_complete=0,
                probe_tests_complete=0,
                active=f"{mode}/probe_seed{probe_seed}",
            )
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
                    f"[MECHANISM_FULL_PROBE_CANDIDATE] encoder_seed={args.encoder_seed} "
                    f"mode={mode} probe_seed={probe_seed} lr={learning_rate:g} "
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
            probe = probe_from_state(
                probe_config, probe_seed, selected_state, device
            )
            validation_metrics, _ = evaluate_probe_both_resolutions(
                probe,
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                targets["validation"]["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=int(runtime_config["frozen_probe"]["image_input"]["size"]),
                ignore_index=int(probe_config["loss"]["ignore_index"]),
            )
            if not all(
                _finite_metrics(validation_metrics[key])
                for key in ("grid_14x14", "input_224")
            ):
                raise RuntimeError(f"non-finite validation probe metric: {mode}")
            probe_path = (
                args.output_dir
                / "probe"
                / "checkpoints"
                / mode
                / f"encoder_seed{args.encoder_seed}_probe_seed{probe_seed}_best_validation.pt"
            )
            _atomic_torch_save(
                {
                    "metadata": {
                        "purpose": (
                            MATCHED_PROBE_PURPOSE
                            if protocol_sha256 == EXPECTED_MATCHED_PROTOCOL_SHA256
                            else PROBE_PURPOSE
                        ),
                        "scientific_result": True,
                        "posthoc_mechanism_analysis": True,
                        "protocol_config_sha256": protocol_sha256,
                        "execution_config_sha256": execution_sha256,
                        "input_lineage_id": "main_l0_v3",
                        "cub_loader_profile": L0_CURRENT_STRONG,
                        "aggregation_mode": mode,
                        "encoder_seed": args.encoder_seed,
                        "encoder_checkpoint_sha256": encoder_audit[
                            "checkpoint_sha256"
                        ],
                        "probe_seed": probe_seed,
                        "selection_split": "validation",
                        "selection_metric": "grid_14x14_two_class_mean_iou",
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
            strict_probe = probe_from_state(
                probe_config, probe_seed, saved["model"], device
            )
            del strict_probe, probe
            selections.append(
                {
                    "aggregation_mode": mode,
                    "encoder_seed": args.encoder_seed,
                    "encoder_checkpoint_path": classification["checkpoint_path"],
                    "encoder_checkpoint_sha256": encoder_audit[
                        "checkpoint_sha256"
                    ],
                    "encoder_state_sha256": encoder_audit["student_state_sha256"],
                    "probe_seed": probe_seed,
                    "candidates": [candidate for _, candidate in candidates],
                    "selected_learning_rate": selected["learning_rate"],
                    "selected_epoch": selected["best_epoch"],
                    "validation": validation_metrics,
                    "probe_checkpoint_path": str(probe_path.resolve()),
                    "probe_checkpoint_sha256": file_sha256(probe_path),
                    "selected_probe_strict_reloaded": True,
                    "official_test": None,
                    "official_test_evaluations": 0,
                }
            )
            _atomic_json_save(
                {
                    "status": "in_progress",
                    "protocol_config_sha256": protocol_sha256,
                    "execution_config_sha256": execution_sha256,
                    "encoder_seed": args.encoder_seed,
                    "official_test_accessed": False,
                    "selections": selections,
                },
                args.output_dir / "probe" / "validation_selections.json",
            )
            del candidates
        del features
        for path in feature_paths:
            path.unlink(missing_ok=True)
        torch.cuda.empty_cache()
    if (
        len(selections) != 20
        or {row["aggregation_mode"] for row in selections}
        != set(AGGREGATION_VARIANTS)
        or any(len(row["candidates"]) != 3 for row in selections)
    ):
        raise RuntimeError("probe validation-selection completion gate failed")
    _atomic_json_save(
        {
            "status": "complete",
            "protocol_config_sha256": protocol_sha256,
            "execution_config_sha256": execution_sha256,
            "encoder_seed": args.encoder_seed,
            "classification_validation_selections": 4,
            "probe_validation_selections": 20,
            "probe_lr_candidates": 60,
            "official_test_accessed": False,
        },
        args.output_dir / "all_validation_selections_complete_before_test.json",
    )
    return selections


def _evaluate_official_test(
    args: argparse.Namespace,
    *,
    base: dict[str, Any],
    protocol_sha256: str,
    execution_sha256: str,
    validation_hash: str,
    classification_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = _load_json(
        args.output_dir / "all_validation_selections_complete_before_test.json"
    )
    if marker != {
        "status": "complete",
        "protocol_config_sha256": protocol_sha256,
        "execution_config_sha256": execution_sha256,
        "encoder_seed": args.encoder_seed,
        "classification_validation_selections": 4,
        "probe_validation_selections": 20,
        "probe_lr_candidates": 60,
        "official_test_accessed": False,
    }:
        raise RuntimeError("pre-test selection marker changed")

    classification_loader = build_official_test_loader(
        args.data_dir,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    classification_journal: list[dict[str, Any]] = []
    for row in classification_rows:
        model, audit = _load_encoder(
            row,
            protocol_sha256=protocol_sha256,
            validation_hash=validation_hash,
            device=device,
        )
        metrics = evaluate(
            model,
            classification_loader,
            device,
            teacher=False,
            num_classes=NUM_CLASSES,
        )
        del model
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError("non-finite classification official-test metric")
        row["official_test"] = metrics
        classification_journal.append(
            {
                "aggregation_mode": row["aggregation_mode"],
                "encoder_checkpoint_sha256": audit["checkpoint_sha256"],
                "metrics": metrics,
                "official_test_evaluations": 1,
            }
        )
        _atomic_json_save(
            {
                "status": (
                    "complete" if len(classification_journal) == 4 else "in_progress"
                ),
                "encoder_seed": args.encoder_seed,
                "settings_locked_before_first_test": True,
                "entries": classification_journal,
            },
            args.output_dir / "classification" / "official_test_results.json",
        )
        log(
            f"[MECHANISM_FULL_CLASSIFICATION_TEST] encoder_seed={args.encoder_seed} "
            f"mode={row['aggregation_mode']} test_macro_top1="
            f"{metrics['macro_top1']:.4f}"
        )
        torch.cuda.empty_cache()
    if len(classification_journal) != 4:
        raise RuntimeError("classification official-test completion gate failed")

    test_records, test_source = load_official_test_records(
        args.data_dir, download=True
    )
    if len(test_records) != 5794:
        raise RuntimeError("official-test probe record count changed")
    runtime_config = {"frozen_probe": base["frozen_probe"]}
    probe_config = runtime_config["frozen_probe"]["probe"]
    test_targets, _ = _target_cache(
        test_records,
        split="official_test",
        config=runtime_config,
        config_sha256=protocol_sha256,
        cache_path=args.cache_dir / "targets" / "official_test.pt",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        grouped[str(row["aggregation_mode"])].append(row)
    completed = 0
    for classification in classification_rows:
        mode = str(classification["aggregation_mode"])
        rows = sorted(grouped[mode], key=lambda item: int(item["probe_seed"]))
        if len(rows) != 5:
            raise RuntimeError(f"expected five selected probes for {mode}")
        model, encoder_audit = _load_encoder(
            classification,
            protocol_sha256=protocol_sha256,
            validation_hash=validation_hash,
            device=device,
        )
        feature_path = (
            args.cache_dir
            / "features"
            / mode
            / f"seed{args.encoder_seed}_official_test.pt"
        )
        test_features, _ = _feature_cache(
            model,
            test_records,
            split="official_test",
            variant=mode,
            encoder_seed=args.encoder_seed,
            checkpoint_sha256=encoder_audit["checkpoint_sha256"],
            state_sha256=encoder_audit["student_state_sha256"],
            config=runtime_config,
            config_sha256=protocol_sha256,
            cache_path=feature_path,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        del model
        for row in rows:
            saved = torch.load(
                row["probe_checkpoint_path"],
                map_location="cpu",
                weights_only=True,
            )
            metadata = saved.get("metadata", {})
            expected = {
                "purpose": (
                    MATCHED_PROBE_PURPOSE
                    if protocol_sha256 == EXPECTED_MATCHED_PROTOCOL_SHA256
                    else PROBE_PURPOSE
                ),
                "scientific_result": True,
                "protocol_config_sha256": protocol_sha256,
                "execution_config_sha256": execution_sha256,
                "aggregation_mode": mode,
                "encoder_seed": args.encoder_seed,
                "probe_seed": row["probe_seed"],
                "official_test_evaluations_at_checkpoint_write": 0,
            }
            failures = [
                key for key, value in expected.items() if metadata.get(key) != value
            ]
            if failures:
                raise RuntimeError(
                    f"selected probe metadata mismatch {mode}: "
                    + ",".join(failures)
                )
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
                input_size=int(runtime_config["frozen_probe"]["image_input"]["size"]),
                ignore_index=int(probe_config["loss"]["ignore_index"]),
            )
            del probe
            if not all(
                _finite_metrics(metrics[key])
                for key in ("grid_14x14", "input_224")
            ):
                raise RuntimeError("non-finite probe official-test metric")
            row["official_test"] = metrics
            row["official_test_evaluations"] = 1
            completed += 1
            _atomic_json_save(
                {
                    "status": "complete" if completed == 20 else "in_progress",
                    "encoder_seed": args.encoder_seed,
                    "completed": completed,
                    "expected": 20,
                    "settings_locked_before_first_test": True,
                    "selections": probe_rows,
                },
                args.output_dir / "probe" / "official_test_results.json",
            )
            _write_status(
                args.output_dir,
                status="running",
                phase="official_test_evaluation",
                classification_complete=4,
                probe_candidates_complete=60,
                probe_selections_complete=20,
                classification_tests_complete=4,
                probe_tests_complete=completed,
                active=f"{mode}/probe_seed{row['probe_seed']}",
            )
            log(
                f"[MECHANISM_FULL_PROBE_TEST] encoder_seed={args.encoder_seed} "
                f"mode={mode} probe_seed={row['probe_seed']} selected_lr="
                f"{row['selected_learning_rate']:g} selected_epoch="
                f"{row['selected_epoch']} test_input_miou="
                f"{metrics['input_224']['mean_iou']:.6f}"
            )
        del test_features
        feature_path.unlink(missing_ok=True)
        torch.cuda.empty_cache()
    if completed != 20:
        raise RuntimeError("probe official-test completion gate failed")
    return {"classification": classification_journal}, test_source


def _classification_flat_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        summary = row["summary"]
        test = row.get("official_test")
        if not isinstance(test, dict):
            raise RuntimeError("classification test result missing")
        audit = summary.get("ibkd_aggregation") or {}
        output.append(
            {
                "aggregation_mode": row["aggregation_mode"],
                "encoder_seed": row["encoder_seed"],
                "selected_epoch": summary["selected_epoch"],
                "validation_macro_top1": summary["selected_validation"]["macro_top1"],
                "test_macro_top1": test["macro_top1"],
                "test_overall_top1": test["overall_top1"],
                "test_top5": test["top5"],
                "controller_stop_epoch": summary["controller_final"]["stop_epoch"],
                "aggregation_normalized_entropy": audit.get(
                    "normalized_entropy_by_teacher_stage"
                ),
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "scientific_result": True,
            }
        )
    reference = next(
        row for row in output if row["aggregation_mode"] == "learned_all"
    )
    for row in output:
        row["validation_macro_top1_delta_from_learned_all"] = float(
            row["validation_macro_top1"]
        ) - float(reference["validation_macro_top1"])
        row["test_macro_top1_delta_from_learned_all"] = float(
            row["test_macro_top1"]
        ) - float(reference["test_macro_top1"])
    return output


def _probe_flat_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        test = row.get("official_test")
        if not isinstance(test, dict):
            raise RuntimeError("probe test result missing")
        validation = row["validation"]
        output.append(
            {
                "aggregation_mode": row["aggregation_mode"],
                "encoder_seed": row["encoder_seed"],
                "probe_seed": row["probe_seed"],
                "selected_learning_rate": row["selected_learning_rate"],
                "selected_epoch": row["selected_epoch"],
                "validation_grid_mean_iou": validation["grid_14x14"]["mean_iou"],
                "validation_input_224_mean_iou": validation["input_224"]["mean_iou"],
                "test_grid_mean_iou": test["grid_14x14"]["mean_iou"],
                "test_input_224_mean_iou": test["input_224"]["mean_iou"],
                "test_input_224_foreground_iou": test["input_224"][
                    "foreground_iou"
                ],
                "test_input_224_background_iou": test["input_224"][
                    "background_iou"
                ],
                "test_input_224_foreground_dice": test["input_224"][
                    "foreground_dice"
                ],
                "test_input_224_pixel_accuracy": test["input_224"][
                    "pixel_accuracy"
                ],
                "encoder_checkpoint_path": row["encoder_checkpoint_path"],
                "encoder_checkpoint_sha256": row["encoder_checkpoint_sha256"],
                "probe_checkpoint_path": row["probe_checkpoint_path"],
                "probe_checkpoint_sha256": row["probe_checkpoint_sha256"],
                "official_test_evaluations": row["official_test_evaluations"],
                "scientific_result": True,
            }
        )
    return output


def _probe_aggregates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        mode = str(row["aggregation_mode"])
        probe_seed = int(row["probe_seed"])
        if probe_seed in grouped[mode]:
            raise RuntimeError(f"duplicate probe seed for {mode}: {probe_seed}")
        grouped[mode][probe_seed] = float(row["test_input_224_mean_iou"])
    if set(grouped) != set(AGGREGATION_VARIANTS):
        raise RuntimeError("probe aggregate variants changed")
    expected_probe_seeds = set(PROBE_SEEDS)
    if any(set(values) != expected_probe_seeds for values in grouped.values()):
        raise RuntimeError("probe aggregate seed coverage changed")
    learned = grouped["learned_all"]
    output = []
    for mode in AGGREGATION_VARIANTS:
        values = [grouped[mode][seed] for seed in PROBE_SEEDS]
        paired_differences = [
            grouped[mode][seed] - learned[seed] for seed in PROBE_SEEDS
        ]
        output.append(
            {
                "aggregation_mode": mode,
                "probe_seed_values": values,
                "mean_over_probe_seeds": statistics.mean(values),
                "sample_standard_deviation_over_probe_seeds": statistics.stdev(
                    values
                ),
                "paired_differences_from_learned_all_by_probe_seed": (
                    paired_differences
                ),
                "mean_paired_difference_from_learned_all": statistics.mean(
                    paired_differences
                ),
                "independent_encoder_n": 1,
                "probe_seeds_per_encoder": 5,
            }
        )
    return output


def _checkpoint_manifest(
    args: argparse.Namespace,
    *,
    teacher_hash: str,
    classification_rows: Sequence[dict[str, Any]],
    probe_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = [
        {
            "kind": "external_shared_teacher",
            "source_h200_issue": 722,
            "path": str(args.teacher_checkpoint.resolve()),
            "sha256": teacher_hash,
        }
    ]
    entries.extend(
        {
            "kind": "classification_encoder",
            "aggregation_mode": row["aggregation_mode"],
            "encoder_seed": row["encoder_seed"],
            "path": row["checkpoint_path"],
            "sha256": row["summary"]["checkpoint_sha256"],
        }
        for row in classification_rows
    )
    entries.extend(
        {
            "kind": "selected_probe",
            "aggregation_mode": row["aggregation_mode"],
            "encoder_seed": row["encoder_seed"],
            "probe_seed": row["probe_seed"],
            "path": row["probe_checkpoint_path"],
            "sha256": row["probe_checkpoint_sha256"],
        }
        for row in probe_rows
    )
    return {
        "count_including_external_teacher": len(entries),
        "retained_new_checkpoint_count": len(entries) - 1,
        "entries": entries,
    }


def _completion_counts(
    classification_rows: Sequence[dict[str, Any]],
    probe_rows: Sequence[dict[str, Any]],
) -> dict[str, int]:
    return {
        "teacher_download_and_audit": 1,
        "classification_students": len(classification_rows),
        "classification_validation_selections": len(classification_rows),
        "classification_official_test_evaluations": sum(
            isinstance(row.get("official_test"), dict) for row in classification_rows
        ),
        "segmentation_probe_lr_candidates": sum(
            len(row.get("candidates", [])) for row in probe_rows
        ),
        "segmentation_probe_validation_selections": len(probe_rows),
        "segmentation_probe_official_test_evaluations": sum(
            int(row.get("official_test_evaluations", 0)) for row in probe_rows
        ),
        "retained_new_checkpoints": len(classification_rows) + len(probe_rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    args.protocol_config = args.protocol_config.resolve()
    args.execution_config = args.execution_config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUB mechanism full requires CUDA")
    if args.feature_batch_size <= 0 or args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader settings")
    protocol = _load_json(args.protocol_config)
    execution = _load_json(args.execution_config)
    base, base_path = _validate_configs(
        protocol,
        args.protocol_config,
        execution,
        args.execution_config,
        args.encoder_seed,
    )
    protocol_sha256 = file_sha256(args.protocol_config)
    execution_sha256 = file_sha256(args.execution_config)
    matched_duration = protocol_sha256 == EXPECTED_MATCHED_PROTOCOL_SHA256
    if matched_duration and args.num_workers != 0:
        raise RuntimeError("matched-duration full requires num_workers=0")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        args.output_dir,
        status="running",
        phase="input_audit",
        classification_complete=0,
        probe_candidates_complete=0,
        probe_selections_complete=0,
        classification_tests_complete=0,
        probe_tests_complete=0,
    )
    log("=" * 88)
    log("CUB PHASE 1 — MAIN-L0 iBKD CONNECTION MECHANISM FULL")
    log("=" * 88)
    log(
        f"[MECHANISM_FULL_POLICY] encoder_seed={args.encoder_seed} "
        "variants=4 classification_epochs=300 probe_seeds=5 probe_lrs=3 "
        "probe_epochs=100 official_test_after_all_validation_selections=true "
        f"matched_duration={str(matched_duration).lower()} "
        f"fixed_guidance_epochs={123 if matched_duration else 'adaptive'}"
    )

    partitions, split_manifest, dataset_source = load_train_validation_records(
        args.data_dir, download=True
    )
    records = {
        "train": partitions["train"],
        "validation": partitions["validation"],
    }
    counts = {key: len(value) for key, value in records.items()}
    validation_hash = str(split_manifest["validation_image_ids_sha256"])
    if counts != {"train": 5394, "validation": 600}:
        raise RuntimeError(f"unexpected train/validation counts: {counts}")
    if validation_hash != EXPECTED_VALIDATION_HASH:
        raise RuntimeError("validation split hash changed")
    teacher, teacher_metadata, teacher_hash, teacher_state_hash = (
        load_scientific_teacher(
            args.teacher_checkpoint,
            validation_hash=validation_hash,
            device=device,
        )
    )
    del teacher
    torch.cuda.empty_cache()
    if (
        teacher_hash != EXPECTED_TEACHER_SHA256
        or teacher_state_hash != EXPECTED_TEACHER_STATE_SHA256
    ):
        raise RuntimeError("shared teacher identity changed")
    input_audit = {
        "status": "pass",
        "encoder_seed": args.encoder_seed,
        "protocol_config_sha256": protocol_sha256,
        "execution_config_sha256": execution_sha256,
        "base_protocol_path": str(base_path),
        "base_protocol_sha256": file_sha256(base_path),
        "dataset_source": dataset_source,
        "dataset_archives": _archive_audit(args.data_dir),
        "counts": counts,
        "split_manifest": split_manifest,
        "teacher_checkpoint_sha256": teacher_hash,
        "teacher_model_state_sha256": teacher_state_hash,
        "teacher_metadata": teacher_metadata,
        "official_test_instantiated": False,
    }
    _atomic_json_save(input_audit, args.output_dir / "input_audit.json")

    classification_rows = _train_classifiers(
        args,
        protocol_sha256=protocol_sha256,
        validation_hash=validation_hash,
        teacher_hash=teacher_hash,
        device=device,
    )
    probe_rows = _train_probes(
        args,
        base=base,
        protocol_sha256=protocol_sha256,
        execution_sha256=execution_sha256,
        records=records,
        split_manifest=split_manifest,
        classification_rows=classification_rows,
        device=device,
    )
    official_journal, official_probe_source = _evaluate_official_test(
        args,
        base=base,
        protocol_sha256=protocol_sha256,
        execution_sha256=execution_sha256,
        validation_hash=validation_hash,
        classification_rows=classification_rows,
        probe_rows=probe_rows,
        device=device,
    )

    classification_flat = _classification_flat_rows(classification_rows)
    probe_flat = _probe_flat_rows(probe_rows)
    probe_aggregates = _probe_aggregates(probe_flat)
    manifest = _checkpoint_manifest(
        args,
        teacher_hash=teacher_hash,
        classification_rows=classification_rows,
        probe_rows=probe_rows,
    )
    completion = _completion_counts(classification_rows, probe_rows)
    if completion != execution["per_shard_completion_gate"]:
        raise RuntimeError(
            f"mechanism full completion gate failed: {completion!r}"
        )
    if manifest["retained_new_checkpoint_count"] != 24:
        raise RuntimeError("mechanism checkpoint retention gate failed")
    _write_csv(
        classification_flat,
        args.output_dir / "classification_results.csv",
    )
    _write_csv(probe_flat, args.output_dir / "probe_results.csv")
    _write_csv(probe_aggregates, args.output_dir / "probe_aggregates.csv")
    _atomic_json_save(manifest, args.output_dir / "checkpoint_manifest.json")
    elapsed = time.monotonic() - started
    summary = {
        "status": "complete",
        "scientific_result": True,
        "posthoc_mechanism_analysis": True,
        "completed_main_l0_v3_result_replaced": False,
        "encoder_seed": args.encoder_seed,
        "protocol_config_sha256": protocol_sha256,
        "execution_config_sha256": execution_sha256,
        "input_lineage_id": "main_l0_v3",
        "loader_profile": L0_CURRENT_STRONG,
        "matched_guidance_duration": matched_duration,
        "fixed_guidance_epochs": 123 if matched_duration else None,
        "classification": classification_flat,
        "frozen_probe": probe_flat,
        "frozen_probe_aggregates": probe_aggregates,
        "completion": completion,
        "checkpoint_manifest": manifest,
        "official_test_journal": official_journal,
        "official_probe_source": official_probe_source,
        "official_test_used_for_training_or_selection": False,
        "elapsed_seconds": elapsed,
        "runtime": _runtime(device),
    }
    _atomic_json_save(summary, args.output_dir / "full_summary.json")
    _write_status(
        args.output_dir,
        status="complete",
        phase="complete",
        classification_complete=4,
        probe_candidates_complete=60,
        probe_selections_complete=20,
        classification_tests_complete=4,
        probe_tests_complete=20,
    )
    log("")
    log("[MECHANISM_FULL_RESULTS]")
    by_classification = {
        row["aggregation_mode"]: row for row in classification_flat
    }
    by_probe = {row["aggregation_mode"]: row for row in probe_aggregates}
    for mode in AGGREGATION_VARIANTS:
        log(
            f"  {mode}: test_macro_top1="
            f"{float(by_classification[mode]['test_macro_top1']):.4f}% "
            f"test_probe_input_miou_mean="
            f"{float(by_probe[mode]['mean_over_probe_seeds']):.6f} "
            f"probe_seed_sd="
            f"{float(by_probe[mode]['sample_standard_deviation_over_probe_seeds']):.6f}"
        )
    log(
        f"[MECHANISM_FULL_TIME] encoder_seed={args.encoder_seed} "
        f"elapsed={format_duration(elapsed)}"
    )
    log(
        f"[MECHANISM_FULL_COMPLETE] status=pass encoder_seed={args.encoder_seed} "
        "teacher_audit=1/1 classification=4/4 classification_test=4/4 "
        "probe_candidates=60/60 probe_selections=20/20 probe_test=20/20 "
        "new_checkpoints=24/24"
    )
    terminal_results = []
    for row in classification_rows:
        mode = str(row["aggregation_mode"])
        terminal_results.append(
            {
                "aggregation_mode": mode,
                "final_train_loss": row["summary"]["history"][-1]["train_loss"],
                "selected_epoch": row["summary"]["selected_epoch"],
                "validation_macro_top1": by_classification[mode][
                    "validation_macro_top1"
                ],
                "test_macro_top1": by_classification[mode]["test_macro_top1"],
                "test_overall_top1": by_classification[mode][
                    "test_overall_top1"
                ],
                "test_top5": by_classification[mode]["test_top5"],
                "controller_stop_epoch": by_classification[mode][
                    "controller_stop_epoch"
                ],
                "test_probe_input_miou_mean": by_probe[mode][
                    "mean_over_probe_seeds"
                ],
                "test_probe_input_miou_sample_sd": by_probe[mode][
                    "sample_standard_deviation_over_probe_seeds"
                ],
            }
        )
    log(
        "[MECHANISM_FULL_FINAL] "
        + json.dumps(
            {
                "status": "pass",
                "encoder_seed": args.encoder_seed,
                "matched_guidance_duration": matched_duration,
                "fixed_guidance_epochs": 123 if matched_duration else None,
                "classification": terminal_results,
                "frozen_probe": probe_flat,
                "frozen_probe_aggregates": probe_aggregates,
                "completion": completion,
                "official_test_used_for_selection": False,
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true", required=True)
    parser.add_argument("--encoder-seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, default=PROTOCOL_CONFIG)
    parser.add_argument("--execution-config", type=Path, default=EXECUTION_CONFIG)
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
        _write_status(
            args.output_dir,
            status="failed",
            phase="failed",
            classification_complete=0,
            probe_candidates_complete=0,
            probe_selections_complete=0,
            classification_tests_complete=0,
            probe_tests_complete=0,
            failure=f"{type(error).__name__}: {error}",
        )
        log(f"[MECHANISM_FULL_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
