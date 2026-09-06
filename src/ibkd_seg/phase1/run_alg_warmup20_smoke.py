#!/usr/bin/env python3
"""Smoke the post-hoc ALG controller-warm-up-20 classification-to-probe path.

This entry point is deliberately non-scientific.  It trains a two-epoch timing
teacher and one two-epoch ALG student, freezes that student, and runs the three
locked probe learning-rate candidates for two epochs on official trainval only.
The Oxford-IIIT Pet official test split is never instantiated.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .models import create_student
from .probe import (
    evaluate_probe_both_resolutions,
    module_state_sha256,
    probe_from_state,
    train_candidate,
)
from .probe_data import load_train_validation_records
from .run_probe_smoke import (
    DEFAULT_PROTOCOL,
    _atomic_json_save,
    _atomic_torch_save,
    _device,
    _feature_cache,
    _finite_metrics,
    _load_json,
    _runtime,
    _synchronize,
    _target_cache,
    log,
)
from .train_timing import file_sha256, format_duration


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIAGNOSTIC_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/configs/oxford_iiit_pet_alg_warmup20_diagnostic_v1.json"
)
LOCKED_PROTOCOL_SHA256 = (
    "38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143"
)
DIAGNOSTIC_ID = "oxford_iiit_pet_alg_controller_warmup20_posthoc_v1"
VARIANT = "alg_controller_warmup20"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    classification_complete: int,
    probe_candidates_complete: int,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "diagnostic_id": DIAGNOSTIC_ID,
            "classification_complete": classification_complete,
            "classification_expected": 1,
            "probe_candidates_complete": probe_candidates_complete,
            "probe_candidates_expected": 3,
            "scientific_result": False,
            "official_test_accessed": False,
            "failure": failure,
            "updated_at_utc": _utc_now(),
        },
        output_dir / "sequence_status.json",
    )


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


def _run_command(command: Sequence[str], *, label: str) -> None:
    log(f"[ALG_W20_SMOKE_TASK_START] {label} command={' '.join(command)}")
    completed = subprocess.run(list(command), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} exited with code {completed.returncode}")
    log(f"[ALG_W20_SMOKE_TASK_DONE] {label}")


def _complete_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing task summary: {path}")
    payload = _load_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"task summary is not complete: {path}")
    return payload


def _load_smoke_encoder(
    checkpoint_path: Path,
    summary: dict[str, Any],
    *,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != summary.get("checkpoint_sha256"):
        raise RuntimeError("timing student checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("student"), dict):
        raise RuntimeError("invalid timing student checkpoint payload")
    metadata = payload.get("metadata", {})
    checks = {
        "purpose": metadata.get("purpose")
        == "phase1_alg_warmup20_combined_smoke_student",
        "non_scientific": metadata.get("scientific_result") is False,
        "test_forbidden": metadata.get("official_test_accessed") is False,
        "dataset": metadata.get("dataset") == "Oxford-IIIT Pet",
        "architecture": metadata.get("architecture") == "deit_tiny_patch16_224",
        "method": metadata.get("method") == "alg",
        "batch": metadata.get("batch_size") == 128,
        "seed": metadata.get("seed") == 1,
        "actual_epochs": metadata.get("actual_epochs") == 2,
        "planned_epochs": metadata.get("planned_epochs") == 300,
        "controller_warmup": metadata.get("controller_warmup_epochs") == 20,
        "validation_split": metadata.get("validation_image_ids_sha256")
        == validation_hash,
        "test_zero_at_write": metadata.get(
            "official_test_evaluations_at_checkpoint_write"
        )
        == 0,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("timing student metadata failed: " + ", ".join(failures))

    model = create_student(num_classes=37, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict timing student load returned incompatible keys")
    state_hash = module_state_sha256(model)
    if state_hash != summary.get("student_state_sha256"):
        raise RuntimeError("timing student state hash differs from summary")
    if state_hash != metadata.get("student_state_sha256"):
        raise RuntimeError("timing student state hash differs from metadata")
    model.requires_grad_(False)
    model.eval()
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError("frozen encoder contract failed")
    return model.to(device), {
        "checkpoint_sha256": checkpoint_hash,
        "student_state_sha256": state_hash,
        "strict_load": True,
        "eval_mode": True,
        "trainable_parameter_count": 0,
    }


def _select_candidate(
    candidates: Sequence[tuple[dict[str, torch.Tensor], dict[str, Any]]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not candidates:
        raise RuntimeError("no probe candidates completed")
    index = max(
        range(len(candidates)),
        key=lambda candidate_index: float(
            candidates[candidate_index][1]["best_validation_grid_mean_iou"]
        ),
    )
    return candidates[index]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if (
        args.num_workers < 0
        or args.feature_batch_size <= 0
        or args.eval_batch_size <= 0
    ):
        raise ValueError("invalid loader settings")
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("ALG warm-up-20 H200 smoke requires CUDA")

    diagnostic = _load_json(args.diagnostic_config)
    protocol = _load_json(args.protocol_config)
    _validate_config(diagnostic, protocol, args.protocol_config)
    diagnostic_sha256 = file_sha256(args.diagnostic_config)
    protocol_sha256 = file_sha256(args.protocol_config)
    smoke = diagnostic["smoke"]
    probe_config = protocol["frozen_spatial_probe"]["probe"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        args.output_dir,
        status="running",
        phase="timing_teacher",
        classification_complete=0,
        probe_candidates_complete=0,
    )
    log("=" * 96)
    log("OXFORD-IIIT PET — ALG CONTROLLER WARM-UP 20 COMBINED SMOKE")
    log("=" * 96)
    log(
        "[ALG_W20_SMOKE_POLICY] posthoc_diagnostic=true canonical_alg_retained=true "
        "changed_field=controller_warmup_epochs:0_to_20 scientific_result=false "
        "official_test_accessed=false"
    )

    classification_root = args.output_dir / "classification"
    teacher_name = "pet_teacher_resnet56_32_b128_timing_2ep_seed1"
    teacher_root = classification_root / "teacher"
    teacher_dir = teacher_root / teacher_name
    teacher_checkpoint = teacher_dir / "timing_teacher_latest.pt"
    teacher_command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_timing",
        "--timing-run",
        "--kind",
        "teacher",
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
    _run_command(teacher_command, label="timing_teacher")
    teacher_summary = _complete_summary(teacher_dir / "summary.json")
    if teacher_summary.get("official_test_accessed") is not False:
        raise RuntimeError("timing teacher accessed official test")

    _write_status(
        args.output_dir,
        status="running",
        phase="alg_classification",
        classification_complete=0,
        probe_candidates_complete=0,
    )
    student_name = "pet_alg_controller_warmup20_b128_timing_2ep_seed1"
    student_root = classification_root / "students"
    student_dir = student_root / student_name
    student_command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_timing",
        "--timing-run",
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
        student_name,
        "--teacher-checkpoint",
        str(teacher_checkpoint),
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--seed",
        "1",
        "--alg-controller-warmup-epochs",
        "20",
        "--save-student-checkpoint",
    ]
    _run_command(student_command, label="alg_warmup20_classification")
    classification_summary_path = student_dir / "summary.json"
    classification_summary = _complete_summary(classification_summary_path)
    controller = classification_summary.get("controller", {})
    classification_checks = {
        "non_scientific": classification_summary.get("scientific_result") is False,
        "test_forbidden": classification_summary.get("official_test_accessed") is False,
        "method": classification_summary.get("method") == "alg",
        "batch": classification_summary.get("batch_size") == 128,
        "seed": classification_summary.get("seed") == 1,
        "actual_epochs": classification_summary.get("actual_epochs") == 2,
        "controller_kind": controller.get("kind") == "alg",
        "controller_warmup": controller.get("warmup_epochs") == 20,
        "controller_boundary": controller.get("stop_comparison")
        == "greater_or_equal",
        "controller_active": controller.get("active") is True,
        "controller_not_stopped": controller.get("stop_epoch") is None,
        "controller_beta_both_epochs": controller.get("beta_history") == [2.5, 2.5],
        "controller_decision_deferred": controller.get(
            "smoothed_derivative_history"
        )
        == [None, None],
    }
    if not all(classification_checks.values()):
        failures = [
            name for name, passed in classification_checks.items() if not passed
        ]
        raise RuntimeError("classification smoke checks failed: " + ", ".join(failures))
    validation_hash = classification_summary["split_manifest"][
        "validation_image_ids_sha256"
    ]
    student_checkpoint = student_dir / "timing_student_latest.pt"
    _write_status(
        args.output_dir,
        status="running",
        phase="frozen_feature_cache",
        classification_complete=1,
        probe_candidates_complete=0,
    )

    records, split_manifest = load_train_validation_records(
        args.data_dir,
        download=True,
    )
    if split_manifest["validation_image_ids_sha256"] != validation_hash:
        raise RuntimeError("classification and probe validation split hashes differ")
    if {split: len(rows) for split, rows in records.items()} != {
        "train": 2940,
        "validation": 740,
    }:
        raise RuntimeError("smoke train/validation counts changed")

    targets: dict[str, dict[str, Any]] = {}
    target_cache_checks: list[bool] = []
    for split in ("train", "validation"):
        targets[split], reloaded = _target_cache(
            records[split],
            split=split,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
        target_cache_checks.append(reloaded)

    model, encoder_audit = _load_smoke_encoder(
        student_checkpoint,
        classification_summary,
        validation_hash=validation_hash,
        device=device,
    )
    entry = {
        "variant": VARIANT,
        "method": "alg",
        "fusion_ratio_lambda": None,
        "encoder_seed": 1,
        "checkpoint_path": student_checkpoint,
        "summary": classification_summary,
    }
    features: dict[str, dict[str, Any]] = {}
    feature_cache_checks: list[bool] = []
    feature_started = time.monotonic()
    for split in ("train", "validation"):
        features[split], reloaded = _feature_cache(
            model,
            entry,
            records[split],
            split=split,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            cache_path=args.cache_dir / "features" / VARIANT / f"{split}.pt",
            device=device,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        feature_cache_checks.append(reloaded)
    _synchronize(device)
    feature_seconds = time.monotonic() - feature_started
    del model
    torch.cuda.empty_cache()

    learning_rates = [float(value) for value in smoke["probe"]["learning_rates"]]
    probe_seed = int(smoke["probe"]["probe_seeds"][0])
    probe_epochs = int(smoke["probe"]["epochs_per_candidate"])
    candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
    probe_started = time.monotonic()
    for candidate_index, learning_rate in enumerate(learning_rates, start=1):
        _write_status(
            args.output_dir,
            status="running",
            phase="probe_validation_smoke",
            classification_complete=1,
            probe_candidates_complete=len(candidates),
        )
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
        log(
            "[ALG_W20_PROBE_CANDIDATE] "
            f"completed={candidate_index}/3 lr={learning_rate:g} "
            f"best_epoch={candidate['best_epoch']} "
            "validation_grid_miou="
            f"{candidate['best_validation_grid_mean_iou']:.6f} "
            "scientific_result=false"
        )

    selected_state, selected = _select_candidate(candidates)
    probe = probe_from_state(probe_config, probe_seed, selected_state, device)
    validation_metrics, _ = evaluate_probe_both_resolutions(
        probe,
        features["validation"]["features"],
        targets["validation"]["grid_targets"],
        targets["validation"]["input_targets"],
        batch_size=int(probe_config["batch_size"]),
        device=device,
        input_size=int(protocol["frozen_spatial_probe"]["image_input"]["size"]),
        ignore_index=int(protocol["frozen_spatial_probe"]["mask"]["ignore_index"]),
    )
    _synchronize(device)
    probe_seconds = time.monotonic() - probe_started
    probe_path = args.output_dir / "probe" / "alg_warmup20_seed1_smoke.pt"
    _atomic_torch_save(
        {
            "purpose": "phase1_alg_warmup20_combined_probe_smoke",
            "scientific_result": False,
            "official_test_accessed": False,
            "diagnostic_id": DIAGNOSTIC_ID,
            "diagnostic_config_sha256": diagnostic_sha256,
            "source_protocol_sha256": protocol_sha256,
            "variant": VARIANT,
            "encoder_seed": 1,
            "encoder_checkpoint_sha256": encoder_audit["checkpoint_sha256"],
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
            "model": selected_state,
        },
        probe_path,
    )
    probe_payload = torch.load(probe_path, map_location="cpu", weights_only=True)
    strict_probe = probe_from_state(
        probe_config,
        probe_seed,
        probe_payload["model"],
        device,
    )
    del strict_probe

    initial_hashes = {
        candidate["initial_probe_state_sha256"] for _, candidate in candidates
    }
    batch_order_hashes = {
        epoch: {
            candidate["batch_order_sha256_by_epoch"][epoch - 1]
            for _, candidate in candidates
        }
        for epoch in range(1, probe_epochs + 1)
    }
    ignore_index = int(protocol["frozen_spatial_probe"]["mask"]["ignore_index"])
    target_values = {
        int(value)
        for split in targets.values()
        for name in ("input_targets", "grid_targets")
        for value in torch.unique(split[name])
    }
    contracts = {
        "diagnostic_config_passed": True,
        "canonical_alg_result_retained": True,
        "only_controller_warmup_changed_to_20": True,
        "classification_checks_passed": all(classification_checks.values()),
        "classification_and_probe_split_match": split_manifest[
            "validation_image_ids_sha256"
        ]
        == validation_hash,
        "official_trainval_counts_2940_740": True,
        "official_test_not_accessed": True,
        "smoke_metrics_not_scientific": True,
        "encoder_strict_loaded_frozen_eval": (
            encoder_audit["strict_load"]
            and encoder_audit["eval_mode"]
            and encoder_audit["trainable_parameter_count"] == 0
        ),
        "features_float32_192x14x14_without_grad": all(
            payload["features"].dtype == torch.float32
            and tuple(payload["features"].shape[1:]) == (192, 14, 14)
            and not payload["features"].requires_grad
            for payload in features.values()
        ),
        "targets_only_background_foreground_ignore": target_values.issubset(
            {0, 1, ignore_index}
        ),
        "target_cache_safe_reload": all(target_cache_checks),
        "feature_cache_safe_reload": all(feature_cache_checks),
        "three_probe_lr_candidates_completed": len(candidates) == 3,
        "same_probe_initial_state_across_lrs": len(initial_hashes) == 1,
        "same_probe_batch_order_across_lrs": all(
            len(values) == 1 for values in batch_order_hashes.values()
        ),
        "only_probe_parameters_receive_gradients": all(
            candidate["gradient_contract"]
            == {
                "cached_feature_gradient_tensor_count": 0,
                "probe_gradient_tensor_count": 2,
            }
            for _, candidate in candidates
        ),
        "selected_probe_strict_reloaded": True,
        "finite_validation_metrics": all(
            _finite_metrics(metrics) for metrics in validation_metrics.values()
        ),
    }
    status = "pass" if all(contracts.values()) else "fail"
    elapsed_seconds = time.monotonic() - started
    summary = {
        "status": status,
        "completed_at_utc": _utc_now(),
        "diagnostic_id": DIAGNOSTIC_ID,
        "posthoc_diagnostic": True,
        "canonical_alg_result_replaced": False,
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "diagnostic_config": {
            "path": str(args.diagnostic_config),
            "sha256": diagnostic_sha256,
        },
        "source_protocol": {
            "path": str(args.protocol_config),
            "sha256": protocol_sha256,
        },
        "classification": {
            "summary_path": str(classification_summary_path),
            "summary_sha256": file_sha256(classification_summary_path),
            "checkpoint_path": str(student_checkpoint),
            "checkpoint_sha256": encoder_audit["checkpoint_sha256"],
            "controller": controller,
            "validation_only_metrics": classification_summary["epochs"],
            "avg_epoch_seconds": classification_summary["avg_epoch_seconds"],
            "estimated_three_seed_300_epoch_seconds": (
                float(classification_summary["avg_epoch_seconds"]) * 300 * 3
            ),
        },
        "probe": {
            "encoder_seed": 1,
            "probe_seed": probe_seed,
            "learning_rates": learning_rates,
            "epochs_per_candidate": probe_epochs,
            "candidates": [candidate for _, candidate in candidates],
            "selected": {
                "learning_rate": selected["learning_rate"],
                "epoch": selected["best_epoch"],
                "validation_grid_mean_iou": selected[
                    "best_validation_grid_mean_iou"
                ],
            },
            "validation": validation_metrics,
            "artifact_path": str(probe_path),
            "artifact_sha256": file_sha256(probe_path),
        },
        "data": {
            "source": "official_trainval_only",
            "counts": {"train": 2940, "validation": 740, "test": 0},
            "validation_image_ids_sha256": validation_hash,
        },
        "contracts": {"all_passed": status == "pass", **contracts},
        "timing": {
            "frozen_feature_cache_seconds": feature_seconds,
            "probe_candidate_and_validation_seconds": probe_seconds,
            "suite_seconds": elapsed_seconds,
        },
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime": _runtime(device),
    }
    summary_path = args.output_dir / "alg_warmup20_smoke_summary.json"
    _atomic_json_save(summary, summary_path)
    _write_status(
        args.output_dir,
        status=status,
        phase="complete",
        classification_complete=1,
        probe_candidates_complete=3,
    )
    log(
        "[ALG_W20_CLASSIFICATION_SMOKE_RESULT] "
        f"controller_warmup={controller['warmup_epochs']} "
        f"stop_epoch={controller['stop_epoch']} active={controller['active']} "
        f"avg_epoch_seconds={classification_summary['avg_epoch_seconds']:.3f} "
        "official_test_accessed=false"
    )
    log(
        "[ALG_W20_PROBE_SMOKE_RESULT] "
        f"selected_lr={selected['learning_rate']:g} "
        f"selected_epoch={selected['best_epoch']} "
        "validation_grid_miou="
        f"{validation_metrics['grid_14x14']['mean_iou']:.6f} "
        "validation_input_miou="
        f"{validation_metrics['input_224']['mean_iou']:.6f} "
        "official_test_accessed=false"
    )
    log(
        "[ALG_W20_FULL_ESTIMATE] classification_3seed_300epoch="
        + format_duration(
            float(classification_summary["avg_epoch_seconds"]) * 300 * 3
        )
        + " probe_full_not_estimated_from_two_epoch_smoke=true"
    )
    log(
        f"[ALG_W20_SMOKE_DONE] status={status} classification=1/1 "
        f"probe_candidates=3/3 selected_probes=1/1 "
        f"scientific_result=false official_test_accessed=false "
        f"seconds={elapsed_seconds:.2f} summary={summary_path.resolve()}"
    )
    if status != "pass":
        raise RuntimeError("ALG warm-up-20 combined smoke failed one or more contracts")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--diagnostic-config",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_CONFIG,
    )
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
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
            classification_complete=int(previous.get("classification_complete", 0)),
            probe_candidates_complete=int(
                previous.get("probe_candidates_complete", 0)
            ),
            failure=f"{type(error).__name__}: {error}",
        )
        log(f"[ALG_W20_SMOKE_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
