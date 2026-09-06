#!/usr/bin/env python3
"""Run the post-hoc ALG-warm-up-20 classification-to-probe full diagnostic."""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw

from .data import OFFICIAL_TEST_COUNT, prepare_and_audit_dataset, save_json
from .models import create_student
from .probe import (
    evaluate_probe_both_resolutions,
    module_state_sha256,
    probe_from_state,
    train_candidate,
)
from .probe_data import load_official_test_records, load_train_validation_records
from .run_alg_warmup20_smoke import (
    DEFAULT_DIAGNOSTIC_CONFIG,
    DIAGNOSTIC_ID,
    _validate_config as _validate_smoke_diagnostic_config,
)
from .run_full import (
    aggregate_students,
    build_final_result_lines,
    completed_row,
    load_complete_summary,
    write_csv,
    write_text,
)
from .run_probe_full import (
    DEFAULT_PROTOCOL,
    METRIC_PATHS,
    _artifact_path,
    _baseline_templates,
    _evaluate_baselines,
    _mask_image,
    _metric,
    _remove_feature_caches,
    _select_candidate,
    _summary_stat,
    _target_values_are_valid,
    _verify_dataset_identity,
    _write_raw_csv,
)
from .run_probe_smoke import (
    _atomic_json_save,
    _atomic_torch_save,
    _device,
    _feature_cache,
    _finite_metrics,
    _load_json,
    _runtime,
    _synchronize,
    _target_cache,
    _validate_classification_input,
    log,
)
from .train_full import ALG_WARMUP20_DIAGNOSTIC_ID, load_full_teacher
from .train_timing import file_sha256, format_duration


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FULL_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/configs/oxford_iiit_pet_alg_warmup20_full_v1.json"
)
DEFAULT_RELEASE_MANIFEST = (
    REPOSITORY_ROOT
    / "phase1/reports/classification/batch128/checkpoint_release.json"
)
EXPERIMENT_ID = "oxford_iiit_pet_alg_controller_warmup20_posthoc_full_v1"
VARIANT = "alg_controller_warmup20"
ENCODER_SEEDS = (1, 2, 3)
PROBE_SEEDS = (1, 2, 3, 4, 5)
EXPECTED_CLASSIFIERS = 3
EXPECTED_LR_CANDIDATES = 45
EXPECTED_SELECTIONS = 15


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    classification_complete: int,
    probe_selections_complete: int,
    probe_test_evaluations_complete: int,
    active: dict[str, Any] | None,
    probe_official_test_accessed: bool = False,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "experiment_id": EXPERIMENT_ID,
            "posthoc_diagnostic": True,
            "confirmatory_main_result": False,
            "canonical_phase1_result_replaced": False,
            "classification_complete": classification_complete,
            "classification_expected": EXPECTED_CLASSIFIERS,
            "classification_official_test_evaluations": classification_complete,
            "probe_selections_complete": probe_selections_complete,
            "probe_selections_expected": EXPECTED_SELECTIONS,
            "probe_test_evaluations_complete": probe_test_evaluations_complete,
            "probe_test_evaluations_expected": EXPECTED_SELECTIONS,
            "probe_official_test_accessed": probe_official_test_accessed,
            "active": active,
            "failure": failure,
            "updated_at_utc": _utc_now(),
        },
        output_dir / "sequence_status.json",
    )


def _validate_full_config(
    full: dict[str, Any],
    diagnostic: dict[str, Any],
    protocol: dict[str, Any],
    *,
    diagnostic_path: Path,
    protocol_path: Path,
    release_manifest_path: Path,
) -> None:
    _validate_smoke_diagnostic_config(diagnostic, protocol, protocol_path)
    classification = full.get("classification", {})
    controller = classification.get("controller", {})
    probe = full.get("frozen_probe", {})
    expected_output = full.get("expected_output", {})
    reference = full.get("reference_classification", {})
    checks = {
        "experiment_id": full.get("experiment_id") == EXPERIMENT_ID,
        "status_locked": full.get("status") == "locked_after_smoke_pass",
        "posthoc": full.get("posthoc_diagnostic") is True,
        "scientific": full.get("scientific_result") is True,
        "not_confirmatory": full.get("confirmatory_main_result") is False,
        "canonical_retained": full.get("canonical_phase1_result_replaced") is False,
        "source_diagnostic_id": full.get("source_diagnostic_id") == DIAGNOSTIC_ID,
        "source_diagnostic_hash": full.get("source_diagnostic_config_sha256")
        == file_sha256(diagnostic_path),
        "source_protocol_id": full.get("source_protocol")
        == protocol.get("protocol_id"),
        "source_protocol_hash": full.get("source_protocol_config_sha256")
        == file_sha256(protocol_path),
        "single_changed_field": full.get("changed_field")
        == {
            "name": "alg_controller_warmup_epochs",
            "canonical_value": 0,
            "diagnostic_value": 20,
        },
        "reference_batch": reference.get("batch_size") == 128,
        "reference_manifest_path": reference.get("release_manifest")
        == "phase1/reports/classification/batch128/checkpoint_release.json",
        "reference_release_hash": reference.get("release_manifest_sha256")
        == file_sha256(release_manifest_path),
        "classification_method": classification.get("method") == "alg",
        "classification_batch": classification.get("batch_size") == 128,
        "classification_seeds": classification.get("encoder_seeds")
        == list(ENCODER_SEEDS),
        "classification_epochs": classification.get("epochs") == 300,
        "optimizer_lr_warmup": classification.get("optimizer_lr_warmup_epochs")
        == 20,
        "controller_kind": controller.get("kind") == "alg",
        "controller_beta": controller.get("beta_on") == 2.5,
        "controller_threshold": controller.get("threshold") == -0.02,
        "controller_window": controller.get("smoothing_window") == 50,
        "controller_warmup": controller.get("warmup_epochs") == 20,
        "controller_boundary": controller.get("stop_comparison")
        == "greater_or_equal",
        "classification_selection": classification.get("selection")
        == {
            "split": "validation",
            "metric": "macro_top1",
            "tie_break": "earlier_epoch",
        },
        "classification_test_once": classification.get(
            "official_test_evaluations"
        )
        == EXPECTED_CLASSIFIERS,
        "probe_encoder_seeds": probe.get("encoder_seeds") == list(ENCODER_SEEDS),
        "probe_seeds": probe.get("probe_seeds") == list(PROBE_SEEDS),
        "probe_lrs": probe.get("learning_rates") == [0.01, 0.03, 0.1],
        "probe_epochs": probe.get("epochs_per_candidate") == 100,
        "probe_parameters": probe.get("probe_parameter_count") == 386,
        "probe_candidates": probe.get("lr_candidate_count")
        == EXPECTED_LR_CANDIDATES,
        "probe_selections": probe.get("selected_probe_count")
        == EXPECTED_SELECTIONS,
        "probe_selection": probe.get("selection")
        == {
            "split": "validation",
            "metric": "grid_14x14_two_class_mean_iou",
            "tie_break": ["lower_learning_rate", "earlier_epoch"],
        },
        "probe_test_policy": probe.get("official_test_policy")
        == "once_after_all_15_probe_selections",
        "probe_test_once": probe.get("official_test_evaluations")
        == EXPECTED_SELECTIONS,
        "probe_qualitative": probe.get("qualitative")
        == "reuse_locked_8_test_ids_encoder_seed1_probe_seed1",
        "expected_output": expected_output
        == {
            "classification_students": EXPECTED_CLASSIFIERS,
            "probe_lr_candidates": EXPECTED_LR_CANDIDATES,
            "selected_probes": EXPECTED_SELECTIONS,
            "classification_test_evaluations": EXPECTED_CLASSIFIERS,
            "probe_test_evaluations": EXPECTED_SELECTIONS,
        },
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "ALG warm-up-20 full config failed: " + ", ".join(failures)
        )


def _validate_reference(
    reference_root: Path,
    full_config: dict[str, Any],
) -> tuple[Path, dict[str, Any], str, dict[str, Any]]:
    entries, suite, validation_hash = _validate_classification_input(
        reference_root,
        expected_batch_size=128,
        encoder_seeds=ENCODER_SEEDS,
    )
    reference = full_config["reference_classification"]
    alg_entries = [entry for entry in entries if entry["variant"] == "alg"]
    observed_initial_hashes = {
        str(entry["encoder_seed"]): entry["summary"][
            "initial_student_state_sha256"
        ]
        for entry in alg_entries
    }
    checks = {
        "all_reference_entries": len(entries) == 18,
        "three_canonical_alg_entries": len(alg_entries) == 3,
        "validation_hash": validation_hash
        == reference["validation_image_ids_sha256"],
        "canonical_alg_warmup_zero": all(
            entry["summary"]["controller_final"]["warmup_epochs"] == 0
            for entry in alg_entries
        ),
        "matched_initial_states": observed_initial_hashes
        == reference["alg_initial_student_state_sha256_by_seed"],
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "batch-128 reference classification failed: " + ", ".join(failures)
        )

    teacher_path = reference_root / reference["teacher_checkpoint_relative_path"]
    if file_sha256(teacher_path) != reference["teacher_checkpoint_sha256"]:
        raise RuntimeError("reference teacher checkpoint SHA-256 mismatch")
    teacher, teacher_metadata, checkpoint_hash, state_hash = load_full_teacher(
        teacher_path,
        validation_hash=validation_hash,
        device=torch.device("cpu"),
    )
    del teacher
    if checkpoint_hash != reference["teacher_checkpoint_sha256"]:
        raise RuntimeError("strict-loaded reference teacher file hash mismatch")
    if state_hash != reference["teacher_model_state_sha256"]:
        raise RuntimeError("strict-loaded reference teacher state hash mismatch")
    return teacher_path, suite, validation_hash, {
        "status": "pass",
        "classification_suite_complete": True,
        "classification_suite_contracts": suite["contracts"],
        "canonical_alg_controller_warmup_epochs": 0,
        "canonical_alg_initial_student_state_sha256_by_seed": (
            observed_initial_hashes
        ),
        "teacher_checkpoint_sha256": checkpoint_hash,
        "teacher_model_state_sha256": state_hash,
        "teacher_metadata": teacher_metadata,
    }


def _classification_command(
    args: argparse.Namespace,
    *,
    seed: int,
    teacher_checkpoint: Path,
    student_root: Path,
    run_name: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_full",
        "--full-run",
        "--kind",
        "student",
        "--method",
        "alg",
        "--batch-size",
        "128",
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
        str(seed),
        "--alg-controller-warmup-epochs",
        "20",
        "--posthoc-diagnostic-id",
        DIAGNOSTIC_ID,
    ]


def _validate_classification_result(
    summary: dict[str, Any],
    checkpoint_path: Path,
    *,
    seed: int,
    validation_hash: str,
    teacher_checkpoint_sha256: str,
    teacher_state_sha256: str,
    expected_initial_hash: str,
) -> dict[str, Any]:
    controller = summary.get("controller_final", {})
    stop_epoch = controller.get("stop_epoch")
    beta_history = controller.get("beta_history", [])
    checks = {
        "complete": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "not_confirmatory": summary.get("confirmatory_main_result") is False,
        "posthoc": summary.get("posthoc_diagnostic") is True,
        "diagnostic_id": summary.get("posthoc_diagnostic_id") == DIAGNOSTIC_ID,
        "canonical_retained": summary.get("canonical_phase1_result_replaced")
        is False,
        "method": summary.get("method") == "alg",
        "batch": summary.get("batch_size") == 128,
        "epochs": summary.get("epochs") == 300,
        "seed": summary.get("seed") == seed,
        "initial_state": summary.get("initial_student_state_sha256")
        == expected_initial_hash,
        "teacher_checkpoint": summary.get("teacher_checkpoint_sha256")
        == teacher_checkpoint_sha256,
        "teacher_state": summary.get("teacher_model_state_sha256")
        == teacher_state_sha256,
        "validation_split": summary.get("split_manifest", {}).get(
            "validation_image_ids_sha256"
        )
        == validation_hash,
        "strict_reload": summary.get("selected_checkpoint_strict_reloaded") is True,
        "classification_test_once": summary.get("official_test_evaluations") == 1,
        "classification_test_not_selected": summary.get(
            "official_test_used_for_training_or_selection"
        )
        is False,
        "history_300": len(summary.get("history", [])) == 300,
        "controller_kind": controller.get("kind") == "alg",
        "controller_beta": controller.get("beta_on") == 2.5,
        "controller_threshold": controller.get("threshold") == -0.02,
        "controller_window": controller.get("smoothing_window") == 50,
        "controller_warmup": controller.get("warmup_epochs") == 20,
        "controller_boundary": controller.get("stop_comparison")
        == "greater_or_equal",
        "no_premature_stop": stop_epoch is None or int(stop_epoch) >= 20,
        "beta_history_300": len(beta_history) == 300,
        "guidance_active_first_20": beta_history[:20] == [2.5] * 20,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"ALG warm-up-20 classification seed {seed} failed: "
            + ", ".join(failures)
        )

    if file_sha256(checkpoint_path) != summary["checkpoint_sha256"]:
        raise RuntimeError("classification checkpoint SHA-256 differs from summary")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    checkpoint_controller = payload.get("controller", {})
    checkpoint_stop = checkpoint_controller.get("stop_epoch")
    metadata_checks = {
        "purpose": metadata.get("purpose")
        == "phase1_posthoc_alg_warmup20_full_student",
        "posthoc": metadata.get("posthoc_diagnostic") is True,
        "diagnostic_id": metadata.get("posthoc_diagnostic_id") == DIAGNOSTIC_ID,
        "canonical_retained": metadata.get("canonical_phase1_result_replaced")
        is False,
        "method": metadata.get("method") == "alg",
        "batch": metadata.get("batch_size") == 128,
        "epochs": metadata.get("epochs") == 300,
        "seed": metadata.get("seed") == seed,
        "controller_warmup": metadata.get("controller_warmup_epochs") == 20,
        "initial_state": metadata.get("initial_student_state_sha256")
        == expected_initial_hash,
        "student_state": metadata.get("student_state_sha256")
        == summary["student_state_sha256"],
        "teacher_state": metadata.get("teacher_model_state_sha256")
        == teacher_state_sha256,
        "validation_split": metadata.get("validation_image_ids_sha256")
        == validation_hash,
        "test_zero_at_write": metadata.get(
            "official_test_evaluations_at_checkpoint_write"
        )
        == 0,
        "checkpoint_controller_warmup": checkpoint_controller.get("warmup_epochs")
        == 20,
        "checkpoint_no_premature_stop": checkpoint_stop is None
        or int(checkpoint_stop) >= 20,
    }
    if not all(metadata_checks.values()):
        failures = [name for name, passed in metadata_checks.items() if not passed]
        raise RuntimeError(
            f"ALG warm-up-20 checkpoint seed {seed} failed: "
            + ", ".join(failures)
        )
    model = create_student(num_classes=37, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("classification checkpoint strict load returned keys")
    if module_state_sha256(model) != summary["student_state_sha256"]:
        raise RuntimeError("classification checkpoint student-state hash mismatch")
    return {
        "all_passed": True,
        "controller_stop_epoch": stop_epoch,
        "controller_active_after_epoch_300": controller.get("active"),
        "guidance_active_first_20_epochs": True,
        "checkpoint_strict_loaded": True,
    }


def _load_diagnostic_encoder(
    entry: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path: Path = entry["checkpoint_path"]
    summary = entry["summary"]
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != summary["checkpoint_sha256"]:
        raise RuntimeError("diagnostic encoder checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    checks = {
        "purpose": metadata.get("purpose")
        == "phase1_posthoc_alg_warmup20_full_student",
        "diagnostic_id": metadata.get("posthoc_diagnostic_id") == DIAGNOSTIC_ID,
        "method": metadata.get("method") == "alg",
        "batch": metadata.get("batch_size") == 128,
        "epochs": metadata.get("epochs") == 300,
        "seed": metadata.get("seed") == entry["encoder_seed"],
        "warmup": metadata.get("controller_warmup_epochs") == 20,
        "student_state": metadata.get("student_state_sha256")
        == summary["student_state_sha256"],
        "test_zero_at_write": metadata.get(
            "official_test_evaluations_at_checkpoint_write"
        )
        == 0,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("diagnostic encoder metadata failed: " + ", ".join(failures))
    model = create_student(num_classes=37, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("diagnostic encoder strict load returned incompatible keys")
    state_hash = module_state_sha256(model)
    if state_hash != summary["student_state_sha256"]:
        raise RuntimeError("diagnostic encoder model-state SHA-256 mismatch")
    model.requires_grad_(False)
    model.eval()
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError("diagnostic encoder freeze/eval contract failed")
    return model.to(device), {
        "checkpoint_sha256": checkpoint_hash,
        "student_state_sha256": state_hash,
        "strict_load": True,
        "eval_mode": True,
        "trainable_parameter_count": 0,
    }


def _aggregate_probe(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != EXPECTED_SELECTIONS:
        raise RuntimeError(
            f"diagnostic probe has {len(results)} rows, expected {EXPECTED_SELECTIONS}"
        )
    report: dict[str, Any] = {"by_encoder_seed": {}}
    for encoder_seed in ENCODER_SEEDS:
        rows = sorted(
            (
                row
                for row in results
                if int(row["encoder_seed"]) == encoder_seed
            ),
            key=lambda row: int(row["probe_seed"]),
        )
        if [int(row["probe_seed"]) for row in rows] != list(PROBE_SEEDS):
            raise RuntimeError(
                f"encoder seed {encoder_seed} diagnostic probe matrix is incomplete"
            )
        seed_report: dict[str, Any] = {}
        for split in ("validation", "test"):
            seed_report[split] = {
                metric_name: _summary_stat(
                    [_metric(row, split, metric_name) for row in rows]
                )
                for metric_name in METRIC_PATHS
            }
        report["by_encoder_seed"][str(encoder_seed)] = seed_report

    report["across_encoder_seed_means"] = {
        split: {
            metric_name: _summary_stat(
                [
                    report["by_encoder_seed"][str(seed)][split][metric_name][
                        "mean"
                    ]
                    for seed in ENCODER_SEEDS
                ]
            )
            for metric_name in METRIC_PATHS
        }
        for split in ("validation", "test")
    }
    return report


def _save_qualitative_panels(
    records: Sequence[Any],
    test_targets: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    qualitative = protocol["frozen_spatial_probe"]["qualitative"]
    image_ids = list(qualitative["test_image_ids"])
    if set(predictions) != set(image_ids):
        raise RuntimeError("diagnostic qualitative prediction set is incomplete")
    record_by_id = {record.image_id: record for record in records}
    index_by_id = {record.image_id: index for index, record in enumerate(records)}
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    panel_paths: list[str] = []
    mask_paths: list[str] = []
    labels = ["input", "ground truth", "ALG warm-up20"]
    for image_id in image_ids:
        with Image.open(record_by_id[image_id].image_path) as handle:
            rgb_input = handle.convert("RGB").resize(
                (224, 224),
                Image.Resampling.BILINEAR,
            )
        ground_truth = _mask_image(
            test_targets["input_targets"][index_by_id[image_id]],
            ground_truth=True,
        )
        prediction = _mask_image(predictions[image_id], ground_truth=False)
        mask_path = mask_dir / f"{image_id}_{VARIANT}.png"
        prediction.save(mask_path)
        mask_paths.append(str(mask_path.relative_to(output_dir)))
        images = [rgb_input, ground_truth, prediction]
        header_height = 28
        panel = Image.new(
            "RGB",
            (224 * len(images), 224 + header_height),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            left = column * 224
            draw.text((left + 6, 7), label, fill=(0, 0, 0))
            panel.paste(image, (left, header_height))
        panel_path = output_dir / f"{image_id}.png"
        panel.save(panel_path)
        panel_paths.append(str(panel_path.relative_to(output_dir)))
    manifest = {
        "status": "complete",
        "posthoc_diagnostic": True,
        "variant": VARIANT,
        "encoder_seed": int(qualitative["encoder_seed"]),
        "probe_seed": int(qualitative["probe_seed"]),
        "test_image_ids": image_ids,
        "selection_rule": qualitative["selection_rule"],
        "panel_order": labels,
        "panels": panel_paths,
        "prediction_masks": mask_paths,
        "test_inference_reused_from_metric_pass": True,
        "posthoc_example_selection": False,
    }
    _atomic_json_save(manifest, output_dir / "manifest.json")
    return manifest


def _run_probe(
    args: argparse.Namespace,
    *,
    entries: list[dict[str, Any]],
    classification_suite: dict[str, Any],
    classification_split_hash: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
    full_config_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    probe_output = args.output_dir / "probe"
    probe_output.mkdir(parents=True, exist_ok=True)
    records, split_manifest = load_train_validation_records(
        args.data_dir,
        download=False,
    )
    if split_manifest["validation_image_ids_sha256"] != classification_split_hash:
        raise RuntimeError("classification and probe validation split hashes differ")
    if {split: len(rows) for split, rows in records.items()} != {
        "train": 2940,
        "validation": 740,
    }:
        raise RuntimeError("diagnostic probe train/validation counts changed")
    dataset_identity = _verify_dataset_identity(args.data_dir, classification_suite)

    targets: dict[str, dict[str, Any]] = {}
    target_cache_audits: list[bool] = []
    target_started = time.monotonic()
    for split in ("train", "validation"):
        targets[split], reloaded = _target_cache(
            records[split],
            split=split,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
        target_cache_audits.append(reloaded)
    target_seconds = time.monotonic() - target_started

    probe_config = protocol["frozen_spatial_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    epochs = int(probe_config["epochs"])
    input_size = int(protocol["frozen_spatial_probe"]["image_input"]["size"])
    ignore_index = int(protocol["frozen_spatial_probe"]["mask"]["ignore_index"])
    if not _target_values_are_valid(
        [targets["train"], targets["validation"]],
        ignore_index=ignore_index,
    ):
        raise RuntimeError("diagnostic train/validation targets are invalid")
    baseline_templates = _baseline_templates(
        targets["train"],
        ignore_index=ignore_index,
        input_size=input_size,
    )
    validation_baselines = _evaluate_baselines(
        baseline_templates,
        targets["validation"],
        ignore_index=ignore_index,
    )

    results: list[dict[str, Any]] = []
    encoder_audits: list[dict[str, Any]] = []
    feature_cache_audits: list[bool] = []
    initial_hashes: dict[int, set[str]] = {seed: set() for seed in PROBE_SEEDS}
    batch_order_hashes: dict[int, dict[int, set[str]]] = {
        seed: {epoch: set() for epoch in range(1, epochs + 1)}
        for seed in PROBE_SEEDS
    }
    gradient_contracts: list[dict[str, int]] = []
    candidate_count = 0
    peak_cuda_memory_bytes = 0
    selection_started = time.monotonic()
    log(
        "[ALG_W20_PROBE_FULL_TASK_COUNT] encoders=3 probe_selections=15 "
        "lr_candidates=45 epochs_per_candidate=100"
    )

    for encoder_index, entry in enumerate(entries, start=1):
        encoder_seed = int(entry["encoder_seed"])
        active = {
            "stage": "probe_validation_selection",
            "encoder_seed": encoder_seed,
            "encoder_index": encoder_index,
        }
        _write_status(
            args.output_dir,
            status="running",
            phase="probe_validation_selection",
            classification_complete=EXPECTED_CLASSIFIERS,
            probe_selections_complete=len(results),
            probe_test_evaluations_complete=0,
            active=active,
        )
        torch.cuda.reset_peak_memory_stats(device)
        model, encoder_audit = _load_diagnostic_encoder(entry, device)
        encoder_audits.append({"encoder_seed": encoder_seed, **encoder_audit})
        features: dict[str, dict[str, Any]] = {}
        feature_paths: list[Path] = []
        feature_started = time.monotonic()
        for split in ("train", "validation"):
            cache_path = (
                args.cache_dir
                / "features"
                / VARIANT
                / f"encoder_seed_{encoder_seed}"
                / f"{split}.pt"
            )
            feature_paths.append(cache_path)
            features[split], reloaded = _feature_cache(
                model,
                entry,
                records[split],
                split=split,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                cache_path=cache_path,
                device=device,
                feature_batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
            feature_cache_audits.append(reloaded)
        _synchronize(device)
        feature_seconds = time.monotonic() - feature_started
        del model
        torch.cuda.empty_cache()

        for probe_seed in PROBE_SEEDS:
            candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
            probe_started = time.monotonic()
            for learning_rate in learning_rates:
                candidate_started = time.monotonic()
                state, candidate = train_candidate(
                    features["train"]["features"],
                    targets["train"]["grid_targets"],
                    features["validation"]["features"],
                    targets["validation"]["grid_targets"],
                    probe_config=probe_config,
                    learning_rate=learning_rate,
                    seed=probe_seed,
                    device=device,
                    epochs=epochs,
                )
                _synchronize(device)
                candidate["elapsed_seconds"] = (
                    time.monotonic() - candidate_started
                )
                candidates.append((state, candidate))
                candidate_count += 1
                initial_hashes[probe_seed].add(
                    candidate["initial_probe_state_sha256"]
                )
                gradient_contracts.append(candidate["gradient_contract"])
                for epoch, digest in enumerate(
                    candidate["batch_order_sha256_by_epoch"],
                    start=1,
                ):
                    batch_order_hashes[probe_seed][epoch].add(digest)
                log(
                    "[ALG_W20_PROBE_CANDIDATE_DONE] "
                    f"completed={candidate_count}/{EXPECTED_LR_CANDIDATES} "
                    f"encoder_seed={encoder_seed} probe_seed={probe_seed} "
                    f"lr={learning_rate:g} best_epoch={candidate['best_epoch']} "
                    "best_val_grid_miou="
                    f"{candidate['best_validation_grid_mean_iou']:.6f}"
                )

            selected_state, selected = _select_candidate(candidates)
            probe = probe_from_state(
                probe_config,
                probe_seed,
                selected_state,
                device,
            )
            validation_metrics, _ = evaluate_probe_both_resolutions(
                probe,
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                targets["validation"]["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=input_size,
                ignore_index=ignore_index,
            )
            if not all(
                _finite_metrics(metrics) for metrics in validation_metrics.values()
            ):
                raise RuntimeError("diagnostic validation metrics are non-finite")
            selection = {
                "split": "validation",
                "metric": "grid_14x14_two_class_mean_iou",
                "learning_rate": float(selected["learning_rate"]),
                "epoch": int(selected["best_epoch"]),
                "validation_grid_mean_iou": float(
                    selected["best_validation_grid_mean_iou"]
                ),
                "tie_break": ["lower_learning_rate", "earlier_epoch"],
            }
            if not math.isclose(
                validation_metrics["grid_14x14"]["mean_iou"],
                selection["validation_grid_mean_iou"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("selected validation metric changed after reload")
            artifact_path = _artifact_path(
                probe_output,
                VARIANT,
                encoder_seed,
                probe_seed,
            )
            candidate_reports = [candidate for _, candidate in candidates]
            _atomic_torch_save(
                {
                    "purpose": "phase1_posthoc_alg_warmup20_full_frozen_probe",
                    "scientific_result": True,
                    "confirmatory_main_result": False,
                    "posthoc_diagnostic": True,
                    "canonical_phase1_result_replaced": False,
                    "experiment_id": EXPERIMENT_ID,
                    "diagnostic_id": DIAGNOSTIC_ID,
                    "full_config_sha256": full_config_sha256,
                    "protocol_sha256": protocol_sha256,
                    "classification_batch_size": 128,
                    "variant": VARIANT,
                    "method": "alg",
                    "fusion_ratio_lambda": None,
                    "encoder_seed": encoder_seed,
                    "encoder_checkpoint_sha256": encoder_audit[
                        "checkpoint_sha256"
                    ],
                    "probe_seed": probe_seed,
                    "selection": selection,
                    "candidate_audits": candidate_reports,
                    "official_test_evaluations_at_checkpoint_write": 0,
                    "model": selected_state,
                },
                artifact_path,
            )
            artifact = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=True,
            )
            artifact_checks = {
                "purpose": artifact.get("purpose")
                == "phase1_posthoc_alg_warmup20_full_frozen_probe",
                "experiment": artifact.get("experiment_id") == EXPERIMENT_ID,
                "diagnostic": artifact.get("diagnostic_id") == DIAGNOSTIC_ID,
                "config": artifact.get("full_config_sha256")
                == full_config_sha256,
                "protocol": artifact.get("protocol_sha256") == protocol_sha256,
                "encoder_seed": artifact.get("encoder_seed") == encoder_seed,
                "probe_seed": artifact.get("probe_seed") == probe_seed,
                "test_zero": artifact.get(
                    "official_test_evaluations_at_checkpoint_write"
                )
                == 0,
            }
            if not all(artifact_checks.values()):
                raise RuntimeError("diagnostic probe artifact metadata failed")
            strict_probe = probe_from_state(
                probe_config,
                probe_seed,
                artifact["model"],
                device,
            )
            del strict_probe, artifact, probe
            results.append(
                {
                    "variant": VARIANT,
                    "method": "alg",
                    "fusion_ratio_lambda": None,
                    "encoder_seed": encoder_seed,
                    "probe_seed": probe_seed,
                    "classification_checkpoint": {
                        **encoder_audit,
                        "relative_path": str(
                            entry["checkpoint_path"].relative_to(
                                args.output_dir / "classification"
                            )
                        ),
                    },
                    "probe_artifact": {
                        "relative_path": str(
                            artifact_path.relative_to(probe_output)
                        ),
                        "sha256": file_sha256(artifact_path),
                        "strict_reloaded": True,
                        "official_test_evaluations_at_write": 0,
                    },
                    "selection": selection,
                    "candidate_audits": candidate_reports,
                    "validation": validation_metrics,
                    "test": None,
                    "official_test_evaluations": 0,
                    "timing": {
                        "feature_cache_seconds_for_encoder": feature_seconds,
                        "probe_seed_total_seconds": time.monotonic()
                        - probe_started,
                    },
                    "scientific_result": True,
                    "posthoc_diagnostic": True,
                }
            )
            log(
                "[ALG_W20_PROBE_SELECTION_DONE] "
                f"completed={len(results)}/{EXPECTED_SELECTIONS} "
                f"encoder_seed={encoder_seed} probe_seed={probe_seed} "
                f"lr={selection['learning_rate']:g} "
                f"epoch={selection['epoch']} val_grid_miou="
                f"{selection['validation_grid_mean_iou']:.6f} "
                "probe_test_accessed=false"
            )
            _write_status(
                args.output_dir,
                status="running",
                phase="probe_validation_selection",
                classification_complete=EXPECTED_CLASSIFIERS,
                probe_selections_complete=len(results),
                probe_test_evaluations_complete=0,
                active=active,
            )
            del candidates, selected_state

        peak_cuda_memory_bytes = max(
            peak_cuda_memory_bytes,
            int(torch.cuda.max_memory_allocated(device)),
        )
        del features
        _remove_feature_caches(feature_paths)
        gc.collect()
        torch.cuda.empty_cache()

    selection_seconds = time.monotonic() - selection_started
    selection_gates = {
        "three_classification_encoders_strict_loaded": len(encoder_audits) == 3
        and all(
            audit["strict_load"]
            and audit["eval_mode"]
            and audit["trainable_parameter_count"] == 0
            for audit in encoder_audits
        ),
        "forty_five_lr_candidates_completed": candidate_count
        == EXPECTED_LR_CANDIDATES,
        "fifteen_probes_selected": len(results) == EXPECTED_SELECTIONS,
        "probe_test_not_accessed_before_all_selections": True,
        "same_probe_initialization_per_probe_seed": all(
            len(values) == 1 for values in initial_hashes.values()
        ),
        "same_batch_order_per_probe_seed_and_epoch": all(
            len(values) == 1
            for by_epoch in batch_order_hashes.values()
            for values in by_epoch.values()
        ),
        "only_probe_parameters_receive_gradients": all(
            contract
            == {
                "cached_feature_gradient_tensor_count": 0,
                "probe_gradient_tensor_count": 2,
            }
            for contract in gradient_contracts
        ),
        "target_cache_safe_reload": all(target_cache_audits),
        "feature_cache_safe_reload": all(feature_cache_audits),
        "dataset_identity_passed": dataset_identity["status"] == "pass",
        "train_validation_target_values_valid": _target_values_are_valid(
            [targets["train"], targets["validation"]],
            ignore_index=ignore_index,
        ),
        "classification_validation_split_reused": split_manifest[
            "validation_image_ids_sha256"
        ]
        == classification_split_hash,
    }
    if not all(selection_gates.values()):
        failures = [name for name, passed in selection_gates.items() if not passed]
        raise RuntimeError("diagnostic probe selection failed: " + ", ".join(failures))
    _atomic_json_save(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "experiment_id": EXPERIMENT_ID,
            "posthoc_diagnostic": True,
            "confirmatory_main_result": False,
            "classification_official_test_evaluations": 3,
            "probe_selection_split": "validation",
            "completed_probe_selections": len(results),
            "probe_official_test_accessed": False,
            "probe_official_test_evaluations": 0,
            "selection_gates": selection_gates,
            "selected_probes": results,
        },
        probe_output / "selection_complete_before_test.json",
    )
    log(
        "[ALG_W20_PROBE_SELECTION_ALL_DONE] completed=15/15 gates=pass "
        "probe_official_test_evaluations=0 opening_probe_test_now=true"
    )

    _write_status(
        args.output_dir,
        status="running",
        phase="probe_official_test",
        classification_complete=EXPECTED_CLASSIFIERS,
        probe_selections_complete=EXPECTED_SELECTIONS,
        probe_test_evaluations_complete=0,
        active={"stage": "probe_official_test_preparation"},
        probe_official_test_accessed=True,
    )
    test_started = time.monotonic()
    test_records = load_official_test_records(args.data_dir)
    if len(test_records) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("official test count changed")
    targets["test"], test_target_reloaded = _target_cache(
        test_records,
        split="test",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        cache_path=args.cache_dir / "targets" / "test.pt",
    )
    if not _target_values_are_valid(
        [targets["test"]],
        ignore_index=ignore_index,
    ):
        raise RuntimeError("diagnostic test target values are invalid")
    test_baselines = _evaluate_baselines(
        baseline_templates,
        targets["test"],
        ignore_index=ignore_index,
    )
    qualitative = protocol["frozen_spatial_probe"]["qualitative"]
    qualitative_ids = list(qualitative["test_image_ids"])
    test_index_by_id = {
        record.image_id: index for index, record in enumerate(test_records)
    }
    if not set(qualitative_ids).issubset(test_index_by_id):
        raise RuntimeError("locked qualitative ids are missing from official test")
    capture_indices = {test_index_by_id[image_id] for image_id in qualitative_ids}
    qualitative_predictions: dict[str, torch.Tensor] = {}
    result_by_key = {
        (result["encoder_seed"], result["probe_seed"]): result
        for result in results
    }
    test_evaluations = 0

    for encoder_index, entry in enumerate(entries, start=1):
        encoder_seed = int(entry["encoder_seed"])
        active = {
            "stage": "probe_official_test",
            "encoder_seed": encoder_seed,
            "encoder_index": encoder_index,
        }
        model, encoder_audit = _load_diagnostic_encoder(entry, device)
        test_cache_path = (
            args.cache_dir
            / "features"
            / VARIANT
            / f"encoder_seed_{encoder_seed}"
            / "test.pt"
        )
        test_features, reloaded = _feature_cache(
            model,
            entry,
            test_records,
            split="test",
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            cache_path=test_cache_path,
            device=device,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        feature_cache_audits.append(reloaded)
        del model
        torch.cuda.empty_cache()

        for probe_seed in PROBE_SEEDS:
            result = result_by_key[(encoder_seed, probe_seed)]
            artifact_path = probe_output / result["probe_artifact"]["relative_path"]
            if file_sha256(artifact_path) != result["probe_artifact"]["sha256"]:
                raise RuntimeError("diagnostic probe artifact changed before test")
            artifact = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=True,
            )
            checks = {
                "purpose": artifact.get("purpose")
                == "phase1_posthoc_alg_warmup20_full_frozen_probe",
                "experiment": artifact.get("experiment_id") == EXPERIMENT_ID,
                "encoder_seed": artifact.get("encoder_seed") == encoder_seed,
                "probe_seed": artifact.get("probe_seed") == probe_seed,
                "encoder_checkpoint": artifact.get(
                    "encoder_checkpoint_sha256"
                )
                == encoder_audit["checkpoint_sha256"],
                "test_zero_at_write": artifact.get(
                    "official_test_evaluations_at_checkpoint_write"
                )
                == 0,
            }
            if not all(checks.values()):
                raise RuntimeError("diagnostic probe pre-test audit failed")
            probe = probe_from_state(
                probe_config,
                probe_seed,
                artifact["model"],
                device,
            )
            should_capture = (
                encoder_seed == int(qualitative["encoder_seed"])
                and probe_seed == int(qualitative["probe_seed"])
            )
            test_metrics, captured = evaluate_probe_both_resolutions(
                probe,
                test_features["features"],
                targets["test"]["grid_targets"],
                targets["test"]["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=input_size,
                ignore_index=ignore_index,
                capture_indices=capture_indices if should_capture else None,
            )
            if not all(
                _finite_metrics(metrics) for metrics in test_metrics.values()
            ):
                raise RuntimeError("diagnostic probe test metrics are non-finite")
            test_evaluations += 1
            result["test"] = test_metrics
            result["official_test_evaluations"] = 1
            if should_capture:
                qualitative_predictions = {
                    image_id: captured[test_index_by_id[image_id]]
                    for image_id in qualitative_ids
                }
            log(
                "[ALG_W20_PROBE_TEST_ONCE] "
                f"completed={test_evaluations}/{EXPECTED_SELECTIONS} "
                f"encoder_seed={encoder_seed} probe_seed={probe_seed} "
                f"input_miou={test_metrics['input_224']['mean_iou']:.6f}"
            )
            _write_status(
                args.output_dir,
                status="running",
                phase="probe_official_test",
                classification_complete=EXPECTED_CLASSIFIERS,
                probe_selections_complete=EXPECTED_SELECTIONS,
                probe_test_evaluations_complete=test_evaluations,
                active=active,
                probe_official_test_accessed=True,
            )
            del probe, artifact, captured

        del test_features
        _remove_feature_caches([test_cache_path])
        gc.collect()
        torch.cuda.empty_cache()

    test_seconds = time.monotonic() - test_started
    qualitative_manifest = _save_qualitative_panels(
        test_records,
        targets["test"],
        qualitative_predictions,
        protocol,
        probe_output / "qualitative",
    )
    final_gates = {
        **selection_gates,
        "probe_test_opened_only_after_all_fifteen_selections": (
            probe_output / "selection_complete_before_test.json"
        ).is_file()
        and len(results) == EXPECTED_SELECTIONS,
        "probe_test_evaluated_once_per_selected_probe": test_evaluations
        == EXPECTED_SELECTIONS
        and all(result["official_test_evaluations"] == 1 for result in results),
        "all_probe_test_metrics_finite": all(
            all(_finite_metrics(metrics) for metrics in result["test"].values())
            for result in results
        ),
        "test_target_cache_safe_reload": test_target_reloaded,
        "all_feature_caches_safe_reload": all(feature_cache_audits),
        "both_nonlearned_baselines_reported": set(validation_baselines)
        == {"all_background", "train_mean_mask"}
        and set(test_baselines) == {"all_background", "train_mean_mask"},
        "eight_locked_qualitative_ids_reported": len(
            qualitative_manifest["test_image_ids"]
        )
        == 8,
        "qualitative_reused_from_metric_pass": qualitative_manifest[
            "test_inference_reused_from_metric_pass"
        ],
        "no_posthoc_qualitative_selection": not qualitative_manifest[
            "posthoc_example_selection"
        ],
    }
    if not all(final_gates.values()):
        failures = [name for name, passed in final_gates.items() if not passed]
        raise RuntimeError(
            "diagnostic probe final gates failed: " + ", ".join(failures)
        )

    aggregate = _aggregate_probe(results)
    summary = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_id": DIAGNOSTIC_ID,
        "posthoc_diagnostic": True,
        "scientific_result": True,
        "confirmatory_main_result": False,
        "canonical_phase1_result_replaced": False,
        "full_config_sha256": full_config_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "classification_batch_size": 128,
        "runtime": _runtime(device),
        "matrix": {
            "variant": VARIANT,
            "encoder_seeds": list(ENCODER_SEEDS),
            "probe_seeds": list(PROBE_SEEDS),
            "learning_rates": learning_rates,
            "epochs_per_lr_candidate": epochs,
            "lr_candidate_count": candidate_count,
            "selected_probe_count": len(results),
        },
        "data": {
            "counts": {"train": 2940, "validation": 740, "test": 3669},
            "split_manifest": split_manifest,
            "dataset_identity": dataset_identity,
        },
        "probe_contract": protocol["frozen_spatial_probe"],
        "probe_test_policy": {
            "selection_uses": "validation_grid_14x14_two_class_mean_iou",
            "official_test_used_for_training_or_selection": False,
            "official_test_opened_after_all_selections": True,
            "official_test_evaluations": test_evaluations,
            "expected_official_test_evaluations": EXPECTED_SELECTIONS,
        },
        "baselines": {
            "validation": validation_baselines,
            "test": test_baselines,
        },
        "raw_results": results,
        "aggregate": aggregate,
        "qualitative": qualitative_manifest,
        "contracts": {"all_passed": True, **final_gates},
        "timing": {
            "target_cache_seconds_before_selection": target_seconds,
            "validation_selection_seconds": selection_seconds,
            "official_test_and_qualitative_seconds": test_seconds,
        },
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }
    _atomic_json_save(summary, probe_output / "probe_summary.json")
    _write_raw_csv(results, probe_output / "probe_raw_results.csv")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.num_workers < 0 or args.feature_batch_size <= 0:
        raise ValueError("invalid data-loader settings")
    if args.eval_batch_size <= 0:
        raise ValueError("invalid classification evaluation batch size")
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("ALG warm-up-20 full diagnostic requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        args.output_dir,
        status="running",
        phase="preflight",
        classification_complete=0,
        probe_selections_complete=0,
        probe_test_evaluations_complete=0,
        active=None,
    )

    full_config = _load_json(args.full_config)
    diagnostic_config = _load_json(args.diagnostic_config)
    protocol = _load_json(args.protocol_config)
    _validate_full_config(
        full_config,
        diagnostic_config,
        protocol,
        diagnostic_path=args.diagnostic_config,
        protocol_path=args.protocol_config,
        release_manifest_path=args.release_manifest,
    )
    full_config_sha256 = file_sha256(args.full_config)
    protocol_sha256 = file_sha256(args.protocol_config)
    log("=" * 96)
    log("OXFORD-IIIT PET — ALG CONTROLLER WARM-UP 20 FULL DIAGNOSTIC")
    log("=" * 96)
    log(
        "[ALG_W20_FULL_POLICY] posthoc_diagnostic=true "
        "confirmatory_main_result=false canonical_phase1_result_replaced=false "
        "changed_field=alg_controller_warmup_epochs:0_to_20"
    )

    (
        reference_teacher,
        reference_suite,
        validation_hash,
        reference_audit,
    ) = _validate_reference(args.reference_classification_root, full_config)
    teacher_output_dir = args.output_dir / "classification" / "reference_teacher"
    teacher_output_dir.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint = teacher_output_dir / "teacher_best_validation.pt"
    shutil.copy2(reference_teacher, teacher_checkpoint)
    if file_sha256(teacher_checkpoint) != reference_audit[
        "teacher_checkpoint_sha256"
    ]:
        raise RuntimeError("copied reference teacher SHA-256 mismatch")
    _atomic_json_save(reference_audit, teacher_output_dir / "reference_audit.json")
    log(
        "[ALG_W20_REFERENCE] batch128_release=pass canonical_alg_warmup=0 "
        f"teacher_sha256={reference_audit['teacher_checkpoint_sha256']} "
        "teacher_reused_for_diagnostic=true"
    )

    dataset_audit = prepare_and_audit_dataset(args.data_dir)
    if dataset_audit != reference_suite["dataset_audit"]:
        raise RuntimeError("current dataset audit differs from batch-128 reference")
    save_json(dataset_audit, args.output_dir / "dataset_audit.json")

    student_root = args.output_dir / "classification" / "students"
    student_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    classification_audits: list[dict[str, Any]] = []
    classification_started = time.monotonic()
    initial_hashes = full_config["reference_classification"][
        "alg_initial_student_state_sha256_by_seed"
    ]
    for seed in ENCODER_SEEDS:
        run_name = f"pet_alg_controller_warmup20_b128_full_300ep_seed{seed}"
        run_dir = student_root / run_name
        active = {"stage": "classification", "seed": seed}
        _write_status(
            args.output_dir,
            status="running",
            phase="classification",
            classification_complete=len(summaries),
            probe_selections_complete=0,
            probe_test_evaluations_complete=0,
            active=active,
        )
        command = _classification_command(
            args,
            seed=seed,
            teacher_checkpoint=teacher_checkpoint,
            student_root=student_root,
            run_name=run_name,
        )
        log(
            f"[ALG_W20_CLASSIFICATION_START] seed={seed}/3 "
            f"command={' '.join(command)}"
        )
        attempt_started = time.monotonic()
        completed = subprocess.run(command, check=False)
        attempt_seconds = time.monotonic() - attempt_started
        if completed.returncode != 0:
            raise RuntimeError(
                f"ALG warm-up-20 classification seed {seed} exited with "
                f"code {completed.returncode}"
            )
        summary_path = run_dir / "summary.json"
        summary = load_complete_summary(summary_path)
        checkpoint_path = run_dir / "student_best_validation.pt"
        audit = _validate_classification_result(
            summary,
            checkpoint_path,
            seed=seed,
            validation_hash=validation_hash,
            teacher_checkpoint_sha256=reference_audit[
                "teacher_checkpoint_sha256"
            ],
            teacher_state_sha256=reference_audit["teacher_model_state_sha256"],
            expected_initial_hash=initial_hashes[str(seed)],
        )
        row = completed_row(
            index=seed,
            payload=summary,
            attempt_elapsed_seconds=attempt_seconds,
        )
        rows.append(row)
        summaries.append(summary)
        classification_audits.append({"seed": seed, **audit})
        log(
            "[ALG_W20_CLASSIFICATION_DONE] "
            f"completed={len(summaries)}/3 seed={seed} "
            f"selected_epoch={summary['selected_epoch']} "
            f"test_macro={summary['official_test']['macro_top1']:.3f} "
            f"controller_stop_epoch={audit['controller_stop_epoch']} "
            f"elapsed={format_duration(attempt_seconds)}"
        )

    classification_aggregates = aggregate_students(rows)
    if len(classification_aggregates) != 1:
        raise RuntimeError("classification aggregation did not produce one ALG row")
    classification_contracts = {
        "three_alg_warmup20_seeds_complete": len(summaries) == 3,
        "all_classification_audits_passed": all(
            audit["all_passed"] for audit in classification_audits
        ),
        "same_reference_teacher": len(
            {summary["teacher_model_state_sha256"] for summary in summaries}
        )
        == 1,
        "canonical_initial_state_reused_per_seed": {
            str(summary["seed"]): summary["initial_student_state_sha256"]
            for summary in summaries
        }
        == initial_hashes,
        "classification_validation_split_fixed": len(
            {
                summary["split_manifest"]["validation_image_ids_sha256"]
                for summary in summaries
            }
        )
        == 1,
        "classification_test_once_per_seed": all(
            summary["official_test_evaluations"] == 1 for summary in summaries
        ),
        "classification_test_not_used_for_selection": all(
            not summary["official_test_used_for_training_or_selection"]
            for summary in summaries
        ),
        "canonical_phase1_result_retained": True,
    }
    if not all(classification_contracts.values()):
        failures = [
            name
            for name, passed in classification_contracts.items()
            if not passed
        ]
        raise RuntimeError(
            "classification suite contracts failed: " + ", ".join(failures)
        )
    classification_dir = args.output_dir / "classification"
    classification_results_path = classification_dir / "classification_results.txt"
    classification_lines = build_final_result_lines(
        rows,
        classification_aggregates,
        profile_batch_size=128,
    )
    write_csv(rows, classification_dir / "classification_summary.csv")
    write_text(classification_lines, classification_results_path)
    classification_suite = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "posthoc_diagnostic": True,
        "scientific_result": True,
        "confirmatory_main_result": False,
        "canonical_phase1_result_replaced": False,
        "batch_size": 128,
        "epochs": 300,
        "completed_tasks": 3,
        "failed_tasks": 0,
        "dataset_audit": dataset_audit,
        "reference_audit": reference_audit,
        "contracts": {"all_passed": True, **classification_contracts},
        "rows": rows,
        "aggregates": classification_aggregates,
        "classification_audits": classification_audits,
        "final_results_text": str(classification_results_path),
        "elapsed_seconds": time.monotonic() - classification_started,
    }
    classification_summary_path = classification_dir / "classification_summary.json"
    _atomic_json_save(classification_suite, classification_summary_path)
    for line in classification_lines:
        log(line)

    entries = [
        {
            "variant": VARIANT,
            "method": "alg",
            "fusion_ratio_lambda": None,
            "encoder_seed": int(summary["seed"]),
            "summary_path": (
                student_root
                / f"pet_alg_controller_warmup20_b128_full_300ep_seed{summary['seed']}"
                / "summary.json"
            ),
            "checkpoint_path": (
                student_root
                / f"pet_alg_controller_warmup20_b128_full_300ep_seed{summary['seed']}"
                / "student_best_validation.pt"
            ),
            "summary": summary,
        }
        for summary in sorted(summaries, key=lambda item: int(item["seed"]))
    ]
    probe_summary = _run_probe(
        args,
        entries=entries,
        classification_suite=classification_suite,
        classification_split_hash=validation_hash,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        full_config_sha256=full_config_sha256,
        device=device,
    )

    classification_macro = classification_aggregates[0]["test_macro_top1"]
    probe_primary = probe_summary["aggregate"]["across_encoder_seed_means"][
        "test"
    ]["input_224_mean_iou"]
    elapsed_seconds = time.monotonic() - started
    final_summary = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_id": DIAGNOSTIC_ID,
        "posthoc_diagnostic": True,
        "scientific_result": True,
        "confirmatory_main_result": False,
        "canonical_phase1_result_replaced": False,
        "changed_field": "alg_controller_warmup_epochs:0_to_20",
        "full_config": {
            "path": str(args.full_config),
            "sha256": full_config_sha256,
        },
        "protocol": {
            "path": str(args.protocol_config),
            "sha256": protocol_sha256,
        },
        "classification": {
            "summary_path": str(classification_summary_path),
            "summary_sha256": file_sha256(classification_summary_path),
            "test_macro_top1": classification_macro,
            "official_test_evaluations": 3,
        },
        "probe": {
            "summary_path": str(args.output_dir / "probe" / "probe_summary.json"),
            "summary_sha256": file_sha256(
                args.output_dir / "probe" / "probe_summary.json"
            ),
            "test_input_224_mean_iou": probe_primary,
            "selected_probes": 15,
            "official_test_evaluations": 15,
        },
        "contracts": {
            "all_passed": True,
            "classification": classification_contracts,
            "probe": probe_summary["contracts"],
        },
        "elapsed_seconds": elapsed_seconds,
        "runtime": _runtime(device),
    }
    final_summary_path = args.output_dir / "alg_warmup20_full_summary.json"
    _atomic_json_save(final_summary, final_summary_path)
    _write_status(
        args.output_dir,
        status="complete",
        phase="complete",
        classification_complete=EXPECTED_CLASSIFIERS,
        probe_selections_complete=EXPECTED_SELECTIONS,
        probe_test_evaluations_complete=EXPECTED_SELECTIONS,
        active=None,
        probe_official_test_accessed=True,
    )

    for row in rows:
        log(
            "[ALG_W20_FULL_CLASSIFICATION_RAW] "
            f"seed={row['seed']} selected_epoch={row['selected_epoch']} "
            f"validation_macro_top1={row['validation_macro_top1']:.3f} "
            f"test_macro_top1={row['test_macro_top1']:.3f}"
        )
    log(
        "[ALG_W20_FULL_CLASSIFICATION_RESULT] "
        f"test_macro_top1_mean={classification_macro['mean']:.3f} "
        "test_macro_top1_sample_sd="
        f"{classification_macro['sample_standard_deviation']:.3f}"
    )
    for result in probe_summary["raw_results"]:
        log(
            "[ALG_W20_FULL_PROBE_RAW] "
            f"encoder_seed={result['encoder_seed']} "
            f"probe_seed={result['probe_seed']} "
            f"lr={result['selection']['learning_rate']:g} "
            f"epoch={result['selection']['epoch']} "
            "test_input_miou="
            f"{result['test']['input_224']['mean_iou']:.6f}"
        )
    log(
        "[ALG_W20_FULL_PROBE_RESULT] "
        f"test_input_miou_mean={probe_primary['mean']:.6f} "
        f"sample_sd={probe_primary['sample_standard_deviation']:.6f} "
        "encoder_seed_means="
        + ",".join(f"{value:.6f}" for value in probe_primary["values"])
    )
    log(
        "[ALG_W20_FULL_DONE] status=pass classification=3/3 "
        "classification_test_once=3/3 probe_candidates=45/45 "
        "probe_selections=15/15 probe_test_once=15/15 "
        "posthoc_diagnostic=true confirmatory_main_result=false "
        "canonical_phase1_result_replaced=false "
        f"seconds={elapsed_seconds:.2f} summary={final_summary_path.resolve()}"
    )
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-diagnostic", action="store_true")
    parser.add_argument("--reference-classification-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, default=DEFAULT_FULL_CONFIG)
    parser.add_argument(
        "--diagnostic-config",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_CONFIG,
    )
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=DEFAULT_RELEASE_MANIFEST,
    )
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.full_diagnostic:
        parser.error("--full-diagnostic is required")
    return args


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
            probe_selections_complete=int(
                previous.get("probe_selections_complete", 0)
            ),
            probe_test_evaluations_complete=int(
                previous.get("probe_test_evaluations_complete", 0)
            ),
            active=previous.get("active"),
            probe_official_test_accessed=bool(
                previous.get("probe_official_test_accessed", False)
            ),
            failure=f"{type(error).__name__}: {error}",
        )
        log(f"[ALG_W20_FULL_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
