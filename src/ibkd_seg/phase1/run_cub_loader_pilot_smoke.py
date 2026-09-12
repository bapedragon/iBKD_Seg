#!/usr/bin/env python3
"""Run a validation-only CUB loader-pilot smoke.

By default the smoke trains all twelve L0/L1/L2 two-epoch guided students.  A
canonical subset may be requested only to validate an operational job
partition after the complete smoke has passed.  Every encoder is frozen before
the binary-segmentation, part-localization, spatial-CKA, and attention--GT
paths.  The runner never constructs an official-test loader and none of its
metrics may select a loader or change the locked full-pilot matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .cub_data import DATASET_NAME, NUM_CLASSES, resolve_dataset_root
from .cub_direct_spatial import (
    load_part_supervision,
    load_spatial_annotations,
    select_lowest_image_id_per_class,
)
from .cub_loader_profiles import LOADER_PROFILE_ORDER, loader_profile_contract
from .cub_probe_data import CubProbeRecord, load_train_validation_records
from .models import create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .run_cub_combined_smoke import (
    _atomic_json_save,
    _atomic_torch_save,
    _device,
    _feature_cache,
    _finite_metrics,
    _target_cache,
)
from .run_cub_direct_spatial_smoke import (
    _attention_smoke,
    _cka_smoke,
    _part_smoke,
)
from .run_cub_r50_teacher_full import load_scientific_teacher
from .train_timing import file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_loader_pilot_smoke_v2.json"
)
EXPECTED_CONFIG_SHA256 = "8ea14d480d64dcc6ad1ab2fafd2754bc22c29867d0b8a20126bcd4ebab537a5f"
EXPECTED_VALIDATION_SHA256 = (
    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
)
EXPECTED_TEACHER_SHA256 = (
    "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
)
EXPECTED_TEACHER_STATE_SHA256 = (
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _normalize_profiles(values: Sequence[str] | None) -> tuple[str, ...]:
    """Return a non-empty, duplicate-free subset in the locked profile order."""

    if values is None:
        return LOADER_PROFILE_ORDER
    requested = tuple(values)
    if not requested:
        raise ValueError("at least one loader profile is required")
    if len(set(requested)) != len(requested):
        raise ValueError("loader profiles must not be repeated")
    unknown = [value for value in requested if value not in LOADER_PROFILE_ORDER]
    if unknown:
        raise ValueError(f"unknown loader profiles: {unknown}")
    canonical = tuple(
        profile for profile in LOADER_PROFILE_ORDER if profile in requested
    )
    if requested != canonical:
        raise ValueError(
            "loader profile subset must follow the locked L0 -> L1 -> L2 order"
        )
    return requested


def _completion_gate_for_profiles(profiles: Sequence[str]) -> dict[str, int]:
    profile_count = len(tuple(profiles))
    classification_count = profile_count * len(EXPECTED_VARIANTS)
    return {
        "loader_profiles": profile_count,
        "classification_students": classification_count,
        "classification_checkpoints": classification_count,
        "segmentation_probe_lr_candidates": classification_count * 3,
        "segmentation_probe_validation_selections": classification_count,
        "part_probe_lr_candidates": classification_count * 3,
        "part_probe_validation_selections": classification_count,
        "spatial_cka_values": classification_count * 12,
        "attention_metric_rows": classification_count,
        "attention_qualitative_pngs": classification_count * 4,
        "official_test_evaluations": 0,
    }


def _write_csv(
    rows: Sequence[dict[str, Any]], path: Path, fields: Sequence[str]
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_config(config: dict[str, Any], path: Path) -> None:
    classification = config.get("classification", {})
    frozen = config.get("frozen_probe", {})
    direct = config.get("direct_spatial", {})
    gate = config.get("completion_gate", {})
    checks = {
        "config_hash": file_sha256(path) == EXPECTED_CONFIG_SHA256,
        "smoke_id": config.get("smoke_id")
        == "cub200_phase1_r50_224_b128_seed1_loader_pilot_smoke_v2",
        "locked_before_results": config.get("status")
        == "locked_after_issue746_preprobe_schema_fix_before_retry_results_2026-09-12",
        "non_scientific": config.get("scientific_result") is False,
        "selection_forbidden": config.get(
            "selection_from_smoke_metrics_forbidden"
        )
        is True,
        "test_forbidden": config.get("official_test_accessed") is False,
        "dataset": config.get("dataset", {}).get("name") == DATASET_NAME
        and config.get("dataset", {}).get("num_classes") == NUM_CLASSES
        and config.get("dataset", {}).get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test_loaded": 0,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_SHA256,
        },
        "teacher": config.get("teacher", {}).get("architecture")
        == "torchvision_resnet50"
        and config.get("teacher", {}).get("initialization") == "scratch"
        and config.get("teacher", {}).get("input_size") == 224
        and config.get("teacher", {}).get("checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and config.get("teacher", {}).get("model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "profiles": tuple(config.get("loader_profiles", ()))
        == LOADER_PROFILE_ORDER,
        "classification": classification.get("student_architecture")
        == "deit_tiny_patch16_224"
        and classification.get("student_batch_size") == 128
        and classification.get("encoder_seed") == 1
        and classification.get("actual_epochs") == 2
        and classification.get("planned_epochs") == 300
        and tuple(classification.get("variants", ())) == EXPECTED_VARIANTS
        and classification.get("same_initial_state_across_all_profiles_and_variants")
        is True
        and classification.get("same_shuffle_order_across_all_profiles_and_variants")
        is True
        and classification.get("teacher_and_student_receive_same_augmented_tensor")
        is True,
        "segmentation_probe": frozen.get("probe", {}).get("probe_seeds") == [1]
        and frozen.get("probe", {}).get("learning_rates") == [0.01, 0.03, 0.1]
        and frozen.get("probe", {}).get("epochs") == 2
        and frozen.get("probe", {}).get("planned_probe_seeds")
        == [1, 2, 3, 4, 5]
        and frozen.get("probe", {}).get("planned_epochs") == 100
        and frozen.get("probe", {}).get("selection_tie_breakers")
        == ["lower_learning_rate", "earlier_epoch"]
        and frozen.get("probe", {}).get("parameter_count") == 386
        and frozen.get("probe", {}).get("initialization")
        == {"weight": "normal", "weight_std": 0.01, "bias": 0.0}
        and frozen.get("probe", {}).get("optimizer")
        == {
            "name": "sgd",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "nesterov": False,
        }
        and frozen.get("probe", {}).get("scheduler")
        == {"name": "cosine", "minimum_learning_rate": 0.0}
        and frozen.get("probe", {}).get("loss")
        == {
            "name": "cross_entropy",
            "ignore_index": 255,
            "class_weighting": "none",
        }
        and frozen.get("encoder", {}).get("frozen") is True,
        "direct_spatial": direct.get("subset")
        == {
            "train": "lowest_image_id_per_class_200",
            "validation": "lowest_image_id_per_class_200",
        }
        and direct.get("part_probe", {}).get("probe_seeds") == [1]
        and direct.get("part_probe", {}).get("learning_rates")
        == [0.01, 0.03, 0.1]
        and direct.get("part_probe", {}).get("epochs") == 2
        and direct.get("part_probe", {}).get("selection_tie_breakers")
        == ["lower_learning_rate", "earlier_epoch"]
        and direct.get("attention_gt", {}).get(
            "qualitative_examples_per_encoder"
        )
        == 4,
        "full_selection_locked": config.get("planned_full_pilot", {}).get(
            "loader_selection_primary"
        )
        == "mean_validation_part_PCK_at_0.1_across_four_guided_variants"
        and config.get("planned_full_pilot", {}).get(
            "classification_CKA_and_attention_not_used_for_loader_selection"
        )
        is True
        and config.get("planned_full_pilot", {}).get("official_test_accessed")
        is False,
        "gate": gate
        == {
            "loader_profiles": 3,
            "classification_students": 12,
            "classification_checkpoints": 12,
            "segmentation_probe_lr_candidates": 36,
            "segmentation_probe_validation_selections": 12,
            "part_probe_lr_candidates": 36,
            "part_probe_validation_selections": 12,
            "spatial_cka_values": 144,
            "attention_metric_rows": 12,
            "attention_qualitative_pngs": 48,
            "official_test_evaluations": 0,
        },
        "failed_v1_preserved": config.get("failed_predecessor")
        == {
            "source_h200_issue": 746,
            "config_path": (
                "phase1/phase1_cub/configs/"
                "cub200_r50_224_b128_loader_pilot_smoke_v1.json"
            ),
            "config_sha256": (
                "db95bf6eb04410e1da2cfffcc97887086f36c0cc7f766f3f5ba407af417785dc"
            ),
            "completed_before_failure": {
                "classification_students": 12,
                "classification_checkpoints": 12,
                "segmentation_probe_candidates": 0,
                "official_test_evaluations": 0,
            },
            "failure": "segmentation_probe_runtime_schema_missing",
            "scientific_result": False,
            "metric_or_selection_semantics_changed_in_v2": False,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid CUB loader-pilot smoke config: " + ", ".join(failures))

    provenance = config["protocol_provenance"]
    sources = (
        ("base_v3", "path", "sha256"),
        ("loader_damage_audit_v1", "config_path", "config_sha256"),
        ("loader_damage_audit_v1", "result_path", "result_sha256"),
    )
    for source_name, path_key, hash_key in sources:
        source = provenance[source_name]
        source_path = _repository_path(source[path_key])
        if not source_path.is_file() or file_sha256(source_path) != source[hash_key]:
            raise RuntimeError(
                f"loader-pilot provenance changed: {source_name}/{path_key}"
            )
    failed_v1 = config["failed_predecessor"]
    failed_v1_path = _repository_path(failed_v1["config_path"])
    if (
        not failed_v1_path.is_file()
        or file_sha256(failed_v1_path) != failed_v1["config_sha256"]
    ):
        raise RuntimeError("loader-pilot failed-v1 provenance changed")
    teacher_manifest = _repository_path(config["teacher"]["release_manifest"])
    if (
        not teacher_manifest.is_file()
        or file_sha256(teacher_manifest)
        != config["teacher"]["release_manifest_sha256"]
    ):
        raise RuntimeError("loader-pilot teacher release manifest changed")


def _validate_cli(args: argparse.Namespace, config: dict[str, Any]) -> None:
    execution = config["execution"]
    actual = {
        "feature_batch_size": args.feature_batch_size,
        "cka_batch_size": args.cka_batch_size,
        "attention_batch_size": args.attention_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
    }
    expected = {key: execution[key] for key in actual}
    if actual != expected:
        raise RuntimeError(f"loader-pilot runtime values changed: {actual} != {expected}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUB loader-pilot smoke requires CUDA")


def _variant_metadata(variant: str) -> tuple[str, float | None, int]:
    try:
        return VARIANT_ARGUMENTS[variant]
    except KeyError as error:
        raise RuntimeError(f"unexpected loader-pilot variant: {variant}") from error


def _complete_student_summary(
    path: Path,
    *,
    profile: str,
    variant: str,
    validation_hash: str,
) -> dict[str, Any]:
    summary = _load_json(path)
    method, fusion_ratio, warmup = _variant_metadata(variant)
    expected = {
        "status": "complete",
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "dataset": DATASET_NAME,
        "kind": "student",
        "method": method,
        "batch_size": 128,
        "seed": 1,
        "loader_pilot_smoke": True,
        "cub_loader_profile": profile,
        "fusion_ratio_lambda": fusion_ratio,
        "actual_epochs": 2,
        "planned_epochs": 300,
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
        "teacher_checkpoint_kind": "cub_r50_v3_scientific",
        "alg_controller_warmup_epochs": 20 if method == "alg" else 0,
        "guidance_controller_warmup_epochs": warmup,
    }
    failures = [key for key, value in expected.items() if summary.get(key) != value]
    split = summary.get("split_manifest", {})
    if split.get("validation_image_ids_sha256") != validation_hash:
        failures.append("validation_image_ids_sha256")
    if split.get("student_train_loader") != loader_profile_contract(profile):
        failures.append("student_train_loader")
    if failures:
        raise RuntimeError(
            f"loader-pilot summary mismatch {profile}/{variant}: "
            + ",".join(failures)
        )
    return summary


def _load_student(
    checkpoint: Path,
    summary: dict[str, Any],
    *,
    profile: str,
    variant: str,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not checkpoint.is_file() or file_sha256(checkpoint) != summary.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError(f"loader-pilot checkpoint bytes changed: {profile}/{variant}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    method, fusion_ratio, warmup = _variant_metadata(variant)
    expected = {
        "purpose": "phase1_cub_r50_224_loader_pilot_smoke_student_v1",
        "scientific_result": False,
        "official_test_accessed": False,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": method,
        "batch_size": 128,
        "seed": 1,
        "loader_pilot_smoke": True,
        "cub_loader_profile": profile,
        "actual_epochs": 2,
        "planned_epochs": 300,
        "teacher_architecture": "resnet50_224_scratch",
        "teacher_checkpoint_kind": "cub_r50_v3_scientific",
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
        "controller_warmup_epochs": 20 if method == "alg" else 0,
        "guidance_controller_warmup_epochs": warmup,
        "validation_image_ids_sha256": validation_hash,
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    failures = [key for key, value in expected.items() if metadata.get(key) != value]
    if metadata.get("fusion_ratio_lambda") != fusion_ratio:
        # Old smoke checkpoints do not store lambda in metadata; this pilot does
        # not rely on that omission because the summary is checked above.
        if "fusion_ratio_lambda" in metadata:
            failures.append("fusion_ratio_lambda")
    if failures:
        raise RuntimeError(
            f"loader-pilot checkpoint mismatch {profile}/{variant}: "
            + ",".join(failures)
        )
    student = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = student.load_state_dict(payload["student"], strict=True)
    state_hash = state_dict_sha256(student)
    floating_finite = all(
        bool(torch.isfinite(value).all())
        for value in payload["student"].values()
        if value.is_floating_point()
    )
    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
        or state_hash != metadata.get("student_state_sha256")
        or state_hash != summary.get("student_state_sha256")
        or not floating_finite
    ):
        raise RuntimeError(f"loader-pilot encoder state audit failed: {profile}/{variant}")
    student.to(device).eval().requires_grad_(False)
    trainable = sum(
        parameter.numel()
        for parameter in student.parameters()
        if parameter.requires_grad
    )
    if student.training or trainable != 0:
        raise RuntimeError(f"loader-pilot encoder was not frozen: {profile}/{variant}")
    return student, {
        "profile": profile,
        "variant": variant,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "student_state_sha256": state_hash,
        "strict_load": True,
        "all_floating_tensors_finite": True,
        "eval_mode": True,
        "trainable_parameters": 0,
    }


def _classification_command(
    *,
    data_dir: Path,
    output_dir: Path,
    teacher_checkpoint: Path,
    profile: str,
    variant: str,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[list[str], str, Path]:
    method, fusion_ratio, warmup = _variant_metadata(variant)
    run_name = f"{profile}__{variant}__b128_seed1_smoke_2ep"
    run_dir = output_dir / run_name
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
        "--scientific-cub-r50-teacher",
        "--loader-pilot-smoke",
        "--cub-loader-profile",
        profile,
        "--batch-size",
        "128",
        "--seed",
        "1",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--teacher-checkpoint",
        str(teacher_checkpoint),
        "--eval-batch-size",
        str(eval_batch_size),
        "--num-workers",
        str(num_workers),
        "--save-student-checkpoint",
    ]
    if fusion_ratio is not None:
        command.extend(["--fusion-ratio", str(fusion_ratio)])
    if method == "alg":
        command.extend(["--alg-controller-warmup-epochs", str(warmup)])
    return command, run_name, run_dir


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    classification_complete: int,
    segmentation_selections: int,
    part_selections: int,
    direct_rows: int,
    profile_count: int = len(LOADER_PROFILE_ORDER),
    active_profile: str | None = None,
    active_variant: str | None = None,
    failure: str | None = None,
) -> None:
    classification_expected = profile_count * len(EXPECTED_VARIANTS)
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "classification_complete": classification_complete,
            "classification_expected": classification_expected,
            "segmentation_probe_selections": segmentation_selections,
            "segmentation_probe_selections_expected": classification_expected,
            "part_probe_selections": part_selections,
            "part_probe_selections_expected": classification_expected,
            "direct_metric_rows": direct_rows,
            "direct_metric_rows_expected": classification_expected * 13,
            "active_profile": active_profile,
            "active_variant": active_variant,
            "scientific_result": False,
            "official_test_accessed": False,
            "failure": failure,
        },
        output_dir / "sequence_status.json",
    )


def _segmentation_probe_smoke(
    *,
    profile: str,
    variant: str,
    student: torch.nn.Module,
    encoder_audit: dict[str, Any],
    records: dict[str, list[CubProbeRecord]],
    targets: dict[str, dict[str, Any]],
    config: dict[str, Any],
    config_sha256: str,
    output_dir: Path,
    cache_dir: Path,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    feature_started = time.perf_counter()
    features: dict[str, dict[str, Any]] = {}
    feature_reload_checks: list[bool] = []
    for split in ("train", "validation"):
        features[split], reloaded = _feature_cache(
            student,
            records[split],
            split=split,
            variant=f"{profile}__{variant}",
            encoder_seed=1,
            checkpoint_sha256=encoder_audit["checkpoint_sha256"],
            state_sha256=encoder_audit["student_state_sha256"],
            config=config,
            config_sha256=config_sha256,
            cache_path=cache_dir / "features" / profile / variant / f"{split}.pt",
            device=device,
            batch_size=feature_batch_size,
            num_workers=num_workers,
        )
        feature_reload_checks.append(reloaded)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    feature_seconds = time.perf_counter() - feature_started

    probe_config = config["frozen_probe"]["probe"]
    probe_seed = int(probe_config["probe_seeds"][0])
    probe_epochs = int(probe_config["epochs"])
    probe_started = time.perf_counter()
    candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
    for learning_rate in probe_config["learning_rates"]:
        state, candidate = train_candidate(
            features["train"]["features"],
            targets["train"]["grid_targets"],
            features["validation"]["features"],
            targets["validation"]["grid_targets"],
            probe_config=probe_config,
            learning_rate=float(learning_rate),
            seed=probe_seed,
            device=device,
            epochs=probe_epochs,
        )
        candidates.append((state, candidate))
        log(
            "[LOADER_PILOT_SEG_CANDIDATE] "
            f"profile={profile} variant={variant} lr={learning_rate:g} "
            f"best_epoch={candidate['best_epoch']} "
            "validation_grid_miou="
            f"{candidate['best_validation_grid_mean_iou']:.6f}"
        )
    selected_state, selected = max(
        candidates,
        key=lambda value: (
            value[1]["best_validation_grid_mean_iou"],
            -value[1]["learning_rate"],
            -value[1]["best_epoch"],
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
        raise RuntimeError(f"non-finite loader-pilot segmentation metric: {profile}/{variant}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    probe_seconds = time.perf_counter() - probe_started
    checkpoint = output_dir / "segmentation_probe/checkpoints" / f"{profile}__{variant}.pt"
    _atomic_torch_save(
        {
            "model": selected_state,
            "metadata": {
                "purpose": "non_scientific_cub_loader_pilot_smoke_segmentation_probe_v2",
                "scientific_result": False,
                "official_test_accessed": False,
                "config_sha256": config_sha256,
                "loader_profile": profile,
                "variant": variant,
                "encoder_seed": 1,
                "encoder_checkpoint_sha256": encoder_audit["checkpoint_sha256"],
                "probe_seed": probe_seed,
                "selected_learning_rate": selected["learning_rate"],
                "selected_epoch": selected["best_epoch"],
                "selection_split": "validation",
            },
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    strict_probe = probe_from_state(probe_config, probe_seed, saved["model"], device)
    del strict_probe, probe
    return {
        "profile": profile,
        "variant": variant,
        "encoder_seed": 1,
        "probe_seed": probe_seed,
        "candidates": [candidate for _state, candidate in candidates],
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
        "initial_probe_state_sha256": selected["initial_probe_state_sha256"],
        "batch_order_sha256_by_epoch": selected["batch_order_sha256_by_epoch"],
        "feature_cache_safe_reload": all(feature_reload_checks),
        "selected_probe_strict_reloaded": True,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "timing": {
            "feature_cache_seconds": feature_seconds,
            "probe_training_seconds": probe_seconds,
        },
        "scientific_result": False,
        "official_test_evaluations": 0,
    }


def _direct_config(config: dict[str, Any]) -> dict[str, Any]:
    part = config["direct_spatial"]["part_probe"]
    return {
        "protocol_id": config["smoke_id"],
        "smoke": {
            "part_probe": {
                "probe_seeds": part["probe_seeds"],
                "learning_rates": part["learning_rates"],
                "epochs": part["epochs"],
            }
        },
        "part_localization_probe": {"batch_size": part["batch_size"]},
    }


def _classification_csv_rows(
    classifications: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in classifications:
        final_epoch = item["summary"]["epochs"][-1]
        rows.append(
            {
                "profile": item["profile"],
                "variant": item["variant"],
                "validation_macro_top1_epoch2": final_epoch["validation"][
                    "macro_top1"
                ],
                "validation_overall_top1_epoch2": final_epoch["validation"][
                    "overall_top1"
                ],
                "avg_epoch_seconds": item["summary"]["avg_epoch_seconds"],
                "checkpoint_sha256": item["summary"]["checkpoint_sha256"],
                "scientific_result": False,
            }
        )
    return rows


def _segmentation_csv_rows(
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "profile": item["profile"],
            "variant": item["variant"],
            "selected_learning_rate": item["selection"]["learning_rate"],
            "selected_epoch": item["selection"]["epoch"],
            "validation_grid_mean_iou": item["validation"]["grid_14x14"][
                "mean_iou"
            ],
            "validation_input_224_mean_iou": item["validation"]["input_224"][
                "mean_iou"
            ],
            "validation_input_224_foreground_iou": item["validation"][
                "input_224"
            ]["foreground_iou"],
            "feature_cache_seconds": item["timing"]["feature_cache_seconds"],
            "probe_training_seconds": item["timing"]["probe_training_seconds"],
            "scientific_result": False,
        }
        for item in results
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    profiles = _normalize_profiles(getattr(args, "profiles", None))
    profile_count = len(profiles)
    classification_expected = profile_count * len(EXPECTED_VARIANTS)
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
    _validate_config(config, config_path)
    config_sha256 = file_sha256(config_path)
    _validate_cli(args, config)
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUB loader-pilot smoke requires H200 CUDA")

    import timm

    if timm.__version__ != "1.0.27":
        raise RuntimeError(f"expected timm==1.0.27, found {timm.__version__}")
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.cuda.reset_peak_memory_stats(device)

    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    _write_status(
        output_dir,
        status="running",
        phase="data_and_teacher_audit",
        classification_complete=0,
        segmentation_selections=0,
        part_selections=0,
        direct_rows=0,
        profile_count=profile_count,
    )
    log("=" * 96)
    log("CUB PHASE 1 — LOADER PILOT SMOKE")
    log("=" * 96)
    log(
        "[LOADER_PILOT_POLICY] scientific_result=false "
        "smoke_selection=forbidden official_test_accessed=false "
        f"profiles={','.join(profiles)}"
    )

    records, split_manifest, source = load_train_validation_records(
        args.data_dir.expanduser().resolve(), download=True
    )
    counts = {name: len(values) for name, values in records.items()}
    validation_hash = split_manifest.get("validation_image_ids_sha256")
    if counts != {"train": 5394, "validation": 600} or (
        validation_hash != EXPECTED_VALIDATION_SHA256
    ):
        raise RuntimeError("loader-pilot CUB train/validation split changed")

    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    teacher, _metadata, teacher_hash, teacher_state_hash = load_scientific_teacher(
        teacher_checkpoint,
        device=device,
        validation_hash=validation_hash,
    )
    if (
        teacher_hash != EXPECTED_TEACHER_SHA256
        or teacher_state_hash != EXPECTED_TEACHER_STATE_SHA256
    ):
        raise RuntimeError("loader-pilot teacher identity changed")

    targets: dict[str, dict[str, Any]] = {}
    target_reload_checks: list[bool] = []
    for split in ("train", "validation"):
        targets[split], reloaded = _target_cache(
            records[split],
            split=split,
            config=config,
            config_sha256=config_sha256,
            cache_path=cache_dir / "targets" / f"{split}.pt",
        )
        target_reload_checks.append(reloaded)
    target_values = {
        int(value)
        for payload in targets.values()
        for key in ("input_targets", "grid_targets")
        for value in torch.unique(payload[key])
    }
    if target_values != {0, 1}:
        raise RuntimeError(f"loader-pilot mask values changed: {target_values}")

    direct_train_records = select_lowest_image_id_per_class(records["train"])
    direct_validation_records = select_lowest_image_id_per_class(
        records["validation"]
    )
    dataset_root = resolve_dataset_root(args.data_dir.expanduser().resolve())
    part_names, annotations = load_spatial_annotations(dataset_root)
    direct_train_supervision = load_part_supervision(
        direct_train_records, annotations
    )
    direct_validation_supervision = load_part_supervision(
        direct_validation_records, annotations
    )
    if len(part_names) != 15 or len(annotations) != 11788:
        raise RuntimeError("loader-pilot CUB spatial annotation inventory changed")
    _atomic_json_save(
        {
            "status": "pass",
            "dataset": DATASET_NAME,
            "source": source,
            "split_manifest": split_manifest,
            "counts": {**counts, "official_test_loaded": 0},
            "binary_mask_values": sorted(target_values),
            "direct_subset_counts": {"train": 200, "validation": 200},
            "part_names": list(part_names),
            "global_annotation_index_parsed_for_integrity": True,
            "spatial_annotations_used_for_student_training": False,
            "official_test_images_masks_or_metrics_accessed": False,
        },
        output_dir / "dataset_audit.json",
    )

    classification_root = output_dir / "classification"
    classifications: list[dict[str, Any]] = []
    for profile in profiles:
        for variant in EXPECTED_VARIANTS:
            _write_status(
                output_dir,
                status="running",
                phase="classification",
                classification_complete=len(classifications),
                segmentation_selections=0,
                part_selections=0,
                direct_rows=0,
                profile_count=profile_count,
                active_profile=profile,
                active_variant=variant,
            )
            command, run_name, run_dir = _classification_command(
                data_dir=args.data_dir.expanduser().resolve(),
                output_dir=classification_root,
                teacher_checkpoint=teacher_checkpoint,
                profile=profile,
                variant=variant,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
            )
            log(
                "[LOADER_PILOT_TASK_START] "
                f"profile={profile} variant={variant} command={' '.join(command)}"
            )
            subprocess.run(command, check=True)
            summary_path = run_dir / "summary.json"
            summary = _complete_student_summary(
                summary_path,
                profile=profile,
                variant=variant,
                validation_hash=validation_hash,
            )
            checkpoint = run_dir / "timing_student_latest.pt"
            if not checkpoint.is_file() or summary.get("checkpoint_sha256") != file_sha256(
                checkpoint
            ):
                raise RuntimeError(
                    f"loader-pilot student artifact failed: {profile}/{variant}"
                )
            classifications.append(
                {
                    "profile": profile,
                    "variant": variant,
                    "run_name": run_name,
                    "summary_path": str(summary_path.resolve()),
                    "checkpoint_path": str(checkpoint.resolve()),
                    "summary": summary,
                }
            )
            log(
                f"[LOADER_PILOT_TASK_DONE] profile={profile} variant={variant}"
            )

    initial_student_hashes = {
        item["summary"]["initial_student_state_sha256"]
        for item in classifications
    }
    teacher_hashes = {
        item["summary"]["teacher_checkpoint_sha256"] for item in classifications
    }
    split_hashes = {
        item["summary"]["split_manifest"]["validation_image_ids_sha256"]
        for item in classifications
    }
    if (
        len(initial_student_hashes) != 1
        or teacher_hashes != {EXPECTED_TEACHER_SHA256}
        or split_hashes != {EXPECTED_VALIDATION_SHA256}
    ):
        raise RuntimeError("loader-pilot paired student/teacher/split contract failed")

    classification_csv = _classification_csv_rows(classifications)
    _write_csv(
        classification_csv,
        output_dir / "classification/results.csv",
        list(classification_csv[0]),
    )

    encoder_audits: list[dict[str, Any]] = []
    segmentation_results: list[dict[str, Any]] = []
    part_candidates: list[dict[str, Any]] = []
    part_results: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    direct_adapter = _direct_config(config)
    shared_part_initial_hashes: set[str] = set()
    fixed_diagnostic_seconds = 0.0

    for item in classifications:
        profile = item["profile"]
        variant = item["variant"]
        _write_status(
            output_dir,
            status="running",
            phase="frozen_spatial_probes",
            classification_complete=classification_expected,
            segmentation_selections=len(segmentation_results),
            part_selections=len(part_results),
            direct_rows=len(cka_rows) + len(attention_rows),
            profile_count=profile_count,
            active_profile=profile,
            active_variant=variant,
        )
        student, encoder_audit = _load_student(
            Path(item["checkpoint_path"]),
            item["summary"],
            profile=profile,
            variant=variant,
            validation_hash=validation_hash,
            device=device,
        )
        encoder_audits.append(encoder_audit)
        segmentation_results.append(
            _segmentation_probe_smoke(
                profile=profile,
                variant=variant,
                student=student,
                encoder_audit=encoder_audit,
                records=records,
                targets=targets,
                config=config,
                config_sha256=config_sha256,
                output_dir=output_dir,
                cache_dir=cache_dir,
                device=device,
                feature_batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
        )
        direct_variant = f"{profile}__{variant}"
        candidates, part_result = _part_smoke(
            variant=direct_variant,
            encoder_seed=1,
            model=student,
            train_records=direct_train_records,
            validation_records=direct_validation_records,
            train_supervision=direct_train_supervision,
            validation_supervision=direct_validation_supervision,
            config=direct_adapter,
            config_sha256=config_sha256,
            output_dir=output_dir,
            device=device,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        for row in candidates:
            row["profile"] = profile
            row["base_variant"] = variant
        part_result["profile"] = profile
        part_result["base_variant"] = variant
        part_candidates.extend(candidates)
        part_results.append(part_result)
        shared_part_initial_hashes.add(part_result["initial_probe_state_sha256"])

        fixed_started = time.perf_counter()
        profile_cka = _cka_smoke(
            variant=direct_variant,
            encoder_seed=1,
            student=student,
            teacher=teacher,
            records=direct_validation_records,
            batch_size=args.cka_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        for row in profile_cka:
            row["profile"] = profile
            row["base_variant"] = variant
        cka_rows.extend(profile_cka)
        attention = _attention_smoke(
            variant=direct_variant,
            encoder_seed=1,
            student=student,
            records=direct_validation_records,
            batch_size=args.attention_batch_size,
            num_workers=args.num_workers,
            output_dir=output_dir,
            device=device,
        )
        attention["profile"] = profile
        attention["base_variant"] = variant
        attention_rows.append(attention)
        fixed_diagnostic_seconds += time.perf_counter() - fixed_started
        del student
        torch.cuda.empty_cache()

    segmentation_initial_hashes = {
        row["initial_probe_state_sha256"] for row in segmentation_results
    }
    segmentation_batch_orders = {
        epoch: {
            row["batch_order_sha256_by_epoch"][epoch - 1]
            for row in segmentation_results
        }
        for epoch in range(1, int(config["frozen_probe"]["probe"]["epochs"]) + 1)
    }
    if len(segmentation_initial_hashes) != 1 or not all(
        len(values) == 1 for values in segmentation_batch_orders.values()
    ):
        raise RuntimeError("loader-pilot segmentation probe pairing changed")
    if len(shared_part_initial_hashes) != 1:
        raise RuntimeError("loader-pilot part probe pairing changed")

    qualitative_pngs = sorted((output_dir / "attention_gt/qualitative").glob("*.png"))
    gate = {
        "loader_profiles": profile_count,
        "classification_students": len(classifications),
        "classification_checkpoints": sum(
            Path(item["checkpoint_path"]).is_file() for item in classifications
        ),
        "segmentation_probe_lr_candidates": sum(
            len(item["candidates"]) for item in segmentation_results
        ),
        "segmentation_probe_validation_selections": len(segmentation_results),
        "part_probe_lr_candidates": len(part_candidates),
        "part_probe_validation_selections": len(part_results),
        "spatial_cka_values": len(cka_rows),
        "attention_metric_rows": len(attention_rows),
        "attention_qualitative_pngs": len(qualitative_pngs),
        "official_test_evaluations": 0,
    }
    expected_gate = _completion_gate_for_profiles(profiles)
    if (
        profiles == LOADER_PROFILE_ORDER
        and expected_gate != config["completion_gate"]
    ):
        raise RuntimeError("locked full-smoke completion gate changed")
    if gate != expected_gate:
        raise RuntimeError(
            f"loader-pilot completion gate failed: {gate} != {expected_gate}"
        )

    segmentation_csv = _segmentation_csv_rows(segmentation_results)
    _write_csv(
        segmentation_csv,
        output_dir / "segmentation_probe/results.csv",
        list(segmentation_csv[0]),
    )
    _atomic_json_save(
        {
            "status": "complete",
            "same_initial_state_across_all_candidates": True,
            "same_batch_order_across_all_candidates": True,
            "results": segmentation_results,
            "scientific_result": False,
            "official_test_evaluations": 0,
        },
        output_dir / "segmentation_probe/results.json",
    )
    _write_csv(
        part_candidates,
        output_dir / "part_probe/candidates.csv",
        list(part_candidates[0]),
    )
    _atomic_json_save(
        {
            "status": "complete",
            "same_initial_state_across_all_encoders": True,
            "results": part_results,
            "scientific_result": False,
            "official_test_evaluations": 0,
        },
        output_dir / "part_probe/results.json",
    )
    _write_csv(
        cka_rows,
        output_dir / "spatial_cka/results.csv",
        list(cka_rows[0]),
    )
    _atomic_json_save(
        {
            "status": "complete",
            "rows": cka_rows,
            "scientific_result": False,
            "official_test_used": False,
        },
        output_dir / "spatial_cka/results.json",
    )
    _write_csv(
        attention_rows,
        output_dir / "attention_gt/results.csv",
        list(attention_rows[0]),
    )
    _atomic_json_save(
        {
            "status": "complete",
            "rows": attention_rows,
            "scientific_result": False,
            "official_test_evaluations": 0,
        },
        output_dir / "attention_gt/results.json",
    )
    _atomic_json_save(
        {
            "status": "pass",
            "teacher": {
                "checkpoint_sha256": teacher_hash,
                "model_state_sha256": teacher_state_hash,
                "strict_load": True,
                "eval_mode": True,
                "trainable_parameters": 0,
            },
            "students": encoder_audits,
        },
        output_dir / "checkpoint_audit.json",
    )

    classification_full_seconds = sum(
        float(item["summary"]["avg_epoch_seconds"]) * 300
        for item in classifications
    )
    segmentation_full_seconds = sum(
        float(item["timing"]["probe_training_seconds"])
        * (100 / int(config["frozen_probe"]["probe"]["epochs"]))
        * 5
        for item in segmentation_results
    )
    # This deliberately over-scales feature extraction as well as optimization,
    # so it is a conservative partitioning estimate rather than a runtime claim.
    part_full_seconds_upper = sum(
        float(item["elapsed_seconds"])
        * (100 / int(config["direct_spatial"]["part_probe"]["epochs"]))
        * 5
        for item in part_results
    )
    estimated_full_seconds_upper = (
        classification_full_seconds
        + segmentation_full_seconds
        + part_full_seconds_upper
        + fixed_diagnostic_seconds
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "status": "pass",
        "smoke_id": config["smoke_id"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "profiles": list(profiles),
        "locked_profile_order": list(LOADER_PROFILE_ORDER),
        "operational_subset_smoke": profiles != LOADER_PROFILE_ORDER,
        "variants": list(EXPECTED_VARIANTS),
        "classification": classification_csv,
        "segmentation_probe": segmentation_csv,
        "part_probe": part_results,
        "spatial_cka": cka_rows,
        "attention_gt": attention_rows,
        "contracts": {
            "same_initial_student_state_across_profiles_and_variants": True,
            "one_teacher_shared_by_all_students": True,
            "same_train_validation_split": True,
            "target_cache_safe_reload": all(target_reload_checks),
            "all_encoders_strict_loaded_frozen_eval": True,
            "segmentation_probe_matched_initialization_and_batch_order": True,
            "part_probe_matched_initialization": True,
            "smoke_metrics_not_used_for_loader_selection": True,
        },
        "completion_gate": gate,
        "timing": {
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "smoke_suite_seconds": elapsed,
            "rough_full_pilot_upper_estimate": {
                "selected_profile_count": profile_count,
                "classification_student_count": classification_expected,
                "classification_x300ep_seconds": classification_full_seconds,
                "segmentation_probe_x5seeds_x3lr_x100ep_seconds": segmentation_full_seconds,
                "part_probe_conservative_upper_seconds": part_full_seconds_upper,
                "cka_attention_fixed_seconds": fixed_diagnostic_seconds,
                "total_seconds": estimated_full_seconds_upper,
                "warning": "linear_conservative_extrapolation_for_job_partitioning_only",
            },
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _atomic_json_save(summary, output_dir / "summary.json")
    _write_status(
        output_dir,
        status="pass",
        phase="complete",
        classification_complete=classification_expected,
        segmentation_selections=classification_expected,
        part_selections=classification_expected,
        direct_rows=classification_expected * 13,
        profile_count=profile_count,
    )

    segmentation_lookup = {
        (row["profile"], row["variant"]): row for row in segmentation_results
    }
    part_lookup = {
        (row["profile"], row["base_variant"]): row for row in part_results
    }
    cka_lookup = {
        (row["profile"], row["base_variant"], row["student_block"]): row
        for row in cka_rows
    }
    attention_lookup = {
        (row["profile"], row["base_variant"]): row for row in attention_rows
    }
    log("")
    log("[LOADER_PILOT_SMOKE_RESULTS]")
    for item in classifications:
        profile, variant = item["profile"], item["variant"]
        final_epoch = item["summary"]["epochs"][-1]
        seg = segmentation_lookup[(profile, variant)]
        part = part_lookup[(profile, variant)]
        cka = cka_lookup[(profile, variant, 11)]
        attention = attention_lookup[(profile, variant)]
        log(
            f"[LOADER_PILOT_SMOKE_RESULT] profile={profile} variant={variant} "
            f"class_val_macro={final_epoch['validation']['macro_top1']:.4f} "
            f"seg_val_miou={seg['validation']['input_224']['mean_iou']:.6f} "
            f"part_val_pck={part['validation']['micro_pck_at_0.1']:.6f} "
            f"cka_block11={cka['centered_linear_cka']:.6f} "
            "attention_patch_ap="
            f"{attention['global_micro_patch_average_precision']:.6f} "
            "attention_pointing="
            f"{attention['pointing_game_peak_inside_mask']:.6f} "
            "foreground_attention_mass="
            f"{attention['foreground_attention_mass_mean']:.6f}"
        )
    log(
        "[LOADER_PILOT_ROUGH_FULL_UPPER_ESTIMATE] "
        f"total={format_duration(estimated_full_seconds_upper)} "
        "linear_conservative_extrapolation_only=true"
    )
    log(
        "[LOADER_PILOT_SMOKE_DONE] status=pass "
        f"profiles={profile_count}/{profile_count} "
        f"classification={classification_expected}/{classification_expected} "
        f"segmentation_candidates={classification_expected * 3}/"
        f"{classification_expected * 3} "
        f"segmentation_selections={classification_expected}/"
        f"{classification_expected} "
        f"part_candidates={classification_expected * 3}/"
        f"{classification_expected * 3} "
        f"part_selections={classification_expected}/{classification_expected} "
        f"cka_values={classification_expected * 12}/"
        f"{classification_expected * 12} "
        f"attention_rows={classification_expected}/{classification_expected} "
        f"qualitative_pngs={classification_expected * 4}/"
        f"{classification_expected * 4} official_test=0 "
        f"elapsed_seconds={elapsed:.2f} summary={output_dir / 'summary.json'}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=LOADER_PROFILE_ORDER,
        help=(
            "Canonical loader-profile subset for operational smoke only; "
            "defaults to all L0/L1/L2 profiles."
        ),
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--cka-batch-size", type=int, default=8)
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            previous = _load_json(output_dir / "sequence_status.json")
        except Exception:
            previous = {}
        _write_status(
            output_dir,
            status="failed",
            phase="failed",
            classification_complete=int(
                previous.get("classification_complete", 0)
            ),
            segmentation_selections=int(
                previous.get("segmentation_probe_selections", 0)
            ),
            part_selections=int(previous.get("part_probe_selections", 0)),
            direct_rows=int(previous.get("direct_metric_rows", 0)),
            profile_count=len(
                getattr(args, "profiles", None) or LOADER_PROFILE_ORDER
            ),
            active_profile=previous.get("active_profile"),
            active_variant=previous.get("active_variant"),
            failure=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
