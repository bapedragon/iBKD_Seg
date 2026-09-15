#!/usr/bin/env python3
"""Smoke the post-hoc CUB main-L0 iBKD layer-connection ablation.

The official test split is never instantiated.  Two-epoch classification and
linear frozen-probe values are execution diagnostics only; they cannot select
an aggregation variant or change the pre-locked full protocol.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .cub_data import DATASET_NAME, NUM_CLASSES
from .cub_loader_profiles import L0_CURRENT_STRONG
from .cub_probe_data import CubProbeRecord, load_train_validation_records
from .models import IBKD_AGGREGATION_MODES, create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .run_cub_combined_smoke import (
    _atomic_json_save,
    _atomic_torch_save,
    _feature_cache,
    _finite_metrics,
    _load_json,
    _target_cache,
)
from .train_timing import file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/mechanism_analysis/configs/"
    "cub200_r50_224_b128_main_l0_ibkd_connection_smoke_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "1fb71d3e076e592df3209c176bcea05a296f4745b243664b09afccef59de2400"
)
EXPECTED_FULL_CONFIG_SHA256 = (
    "1650e76da40c235292fca166b8058c1a9ee279165a275b59122f505f3b7235d8"
)
EXPECTED_TEACHER_CHECKPOINT_SHA256 = (
    "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
)
EXPECTED_TEACHER_STATE_SHA256 = (
    "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
)
EXPECTED_VALIDATION_IDS_SHA256 = (
    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
)
AGGREGATION_VARIANTS = (
    "learned_all",
    "fixed_uniform_all",
    "fixed_stage_match",
    "fixed_last",
)


def log(message: str = "") -> None:
    print(message, flush=True)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_repository_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _validate_config(config: dict[str, Any], config_path: Path) -> Path:
    full_path = _resolve_repository_path(str(config["full_protocol_config"]))
    teacher_manifest = _resolve_repository_path(
        str(config["teacher"]["release_manifest"])
    )
    checks = {
        "smoke_config_hash": file_sha256(config_path) == EXPECTED_CONFIG_SHA256,
        "smoke_id": config.get("smoke_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_layer_connection_smoke_v1",
        "non_scientific": config.get("scientific_result") is False,
        "selection_forbidden": config.get("selection_from_smoke_metrics_forbidden")
        is True
        and config.get("method_or_ablation_selection_from_smoke_forbidden") is True,
        "official_test_closed": config.get("official_test_accessed") is False,
        "main_l0_only": config.get("lineage")
        == {
            "input_lineage_id": "main_l0_v3",
            "loader_profile": L0_CURRENT_STRONG,
            "loader_pilot_l0_checkpoints_allowed": False,
            "selected_l2_checkpoints_allowed": False,
        },
        "dataset": config.get("dataset", {}).get("name") == DATASET_NAME
        and config.get("dataset", {}).get("num_classes") == NUM_CLASSES,
        "split": config.get("dataset", {}).get("split", {}).get("counts")
        == {"train": 5394, "validation": 600, "test": 5794}
        and config.get("dataset", {}).get("split", {}).get(
            "validation_image_ids_sha256"
        )
        == EXPECTED_VALIDATION_IDS_SHA256,
        "teacher": config.get("teacher", {}).get("reuse_h200_issue") == 722
        and config.get("teacher", {}).get("train_in_smoke") is False
        and config.get("teacher", {}).get("checkpoint_sha256")
        == EXPECTED_TEACHER_CHECKPOINT_SHA256
        and config.get("teacher", {}).get("model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "teacher_manifest": teacher_manifest.is_file()
        and file_sha256(teacher_manifest)
        == config.get("teacher", {}).get("release_manifest_sha256"),
        "classification": config.get("classification", {}).get(
            "student_batch_size"
        )
        == 128
        and config.get("classification", {}).get("encoder_seed") == 1
        and config.get("classification", {}).get("actual_epochs") == 2
        and config.get("classification", {}).get("planned_epochs") == 300
        and config.get("classification", {}).get("fusion_ratio_lambda") == 0.25
        and config.get("classification", {}).get("loader_profile")
        == L0_CURRENT_STRONG
        and tuple(config.get("classification", {}).get("variants", ()))
        == AGGREGATION_VARIANTS,
        "known_modes": set(AGGREGATION_VARIANTS).issubset(
            set(IBKD_AGGREGATION_MODES)
        ),
        "probe": config.get("frozen_probe", {}).get("probe", {}).get(
            "learning_rates"
        )
        == [0.01, 0.03, 0.1]
        and config.get("frozen_probe", {}).get("probe", {}).get("epochs") == 2
        and config.get("frozen_probe", {}).get("probe", {}).get("probe_seeds")
        == [1],
        "completion_gate": config.get("completion_gate")
        == {
            "teacher_download_and_audit": 1,
            "classification_students": 4,
            "classification_checkpoints": 4,
            "aggregation_audits": 4,
            "probe_lr_candidates": 12,
            "probe_validation_selections": 4,
            "selected_probe_checkpoints": 4,
            "official_test_evaluations": 0,
        },
        "full_config": full_path.is_file()
        and file_sha256(full_path) == EXPECTED_FULL_CONFIG_SHA256
        and config.get("full_protocol_config_sha256")
        == EXPECTED_FULL_CONFIG_SHA256,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB iBKD connection smoke contract: " + ", ".join(failures)
        )
    full = _load_json(full_path)
    if (
        full.get("protocol_id")
        != "cub200_phase1_r50_224_b128_main_l0_ibkd_layer_connection_ablation_full_v1"
        or full.get("execution_gate", {}).get("current_state")
        != "full_blocked_until_smoke_passes"
        or [row.get("id") for row in full.get("classification", {}).get("variants", [])]
        != list(AGGREGATION_VARIANTS)
    ):
        raise RuntimeError("locked full mechanism protocol contract changed")
    return full_path


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    classification_complete: int,
    probe_candidates_complete: int,
    probe_selections_complete: int,
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
            "probe_selections_complete": probe_selections_complete,
            "probe_selections_expected": 4,
            "active_variant": active_variant,
            "scientific_result": False,
            "official_test_accessed": False,
            "failure": failure,
        },
        output_dir / "sequence_status.json",
    )


def _smoke_completion_counts(
    classification_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    learning_rates: list[float],
    *,
    teacher_download_and_audit: int,
) -> dict[str, int]:
    """Build the exact completion contract checked at the end of smoke."""

    return {
        "teacher_download_and_audit": teacher_download_and_audit,
        "classification_students": len(classification_rows),
        "classification_checkpoints": len(classification_rows),
        "aggregation_audits": sum(
            row["summary"].get("ibkd_aggregation") is not None
            for row in classification_rows
        ),
        "probe_lr_candidates": len(probe_rows) * len(learning_rates),
        "probe_validation_selections": len(probe_rows),
        "selected_probe_checkpoints": len(probe_rows),
        "official_test_evaluations": 0,
    }


def _run_command(command: list[str], *, label: str) -> None:
    log(f"[MECHANISM_SMOKE_TASK_START] label={label} command={' '.join(command)}")
    subprocess.run(command, check=True)
    log(f"[MECHANISM_SMOKE_TASK_DONE] label={label}")


def _load_encoder(
    checkpoint_path: Path,
    summary: dict[str, Any],
    *,
    mode: str,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": "phase1_cub_r50_224_main_l0_ibkd_connection_smoke_student_v1",
        "scientific_result": False,
        "official_test_accessed": False,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": "ibkd",
        "fusion_ratio_lambda": 0.25,
        "batch_size": 128,
        "seed": 1,
        "actual_epochs": 2,
        "planned_epochs": 300,
        "teacher_architecture": "resnet50_224_scratch",
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_CHECKPOINT_SHA256,
        "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
        "mechanism_ablation_smoke": True,
        "cub_loader_profile": L0_CURRENT_STRONG,
        "validation_image_ids_sha256": validation_hash,
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"mechanism smoke checkpoint mismatch {mode}:{key}: "
                f"expected={value!r} got={metadata.get(key)!r}"
            )
    aggregation = metadata.get("ibkd_aggregation")
    if not isinstance(aggregation, dict) or aggregation.get("mode") != mode:
        raise RuntimeError(f"missing aggregation audit for {mode}")
    if summary.get("ibkd_aggregation") != aggregation:
        raise RuntimeError(f"checkpoint/summary aggregation audit mismatch for {mode}")
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != summary.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint SHA-256 mismatch for {mode}")
    student = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    student.load_state_dict(payload["student"], strict=True)
    student_hash = state_dict_sha256(student)
    if student_hash != metadata.get("student_state_sha256"):
        raise RuntimeError(f"student-state SHA-256 mismatch for {mode}")
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
        "student_state_sha256": student_hash,
        "aggregation": aggregation,
    }
    if not audit["eval_mode"] or audit["trainable_parameter_count"] != 0:
        raise RuntimeError(f"encoder freeze contract failed for {mode}")
    return student, audit


def _classification_csv_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        summary = row["summary"]
        aggregation = summary["ibkd_aggregation"]
        final_epoch = summary["epochs"][-1]
        output.append(
            {
                "aggregation_mode": row["mode"],
                "validation_macro_top1_epoch2": final_epoch["validation"][
                    "macro_top1"
                ],
                "avg_epoch_seconds": summary["avg_epoch_seconds"],
                "peak_cuda_memory_bytes": max(
                    int(epoch.get("peak_cuda_memory_bytes") or 0)
                    for epoch in summary["epochs"]
                ),
                "controller_stop_epoch": summary["controller"]["stop_epoch"],
                "teacher_stage0_top_block": aggregation[
                    "top_block_by_teacher_stage"
                ][0],
                "teacher_stage1_top_block": aggregation[
                    "top_block_by_teacher_stage"
                ][1],
                "teacher_stage2_top_block": aggregation[
                    "top_block_by_teacher_stage"
                ][2],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "scientific_result": False,
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUB iBKD connection smoke requires CUDA")
    if args.feature_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid feature loader settings")
    config = _load_json(args.config)
    full_config_path = _validate_config(config, args.config)
    config_hash = file_sha256(args.config)
    full_config_hash = file_sha256(full_config_path)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        args.output_dir,
        status="running",
        phase="dataset_and_teacher_audit",
        classification_complete=0,
        probe_candidates_complete=0,
        probe_selections_complete=0,
    )

    log("=" * 88)
    log("CUB PHASE 1 — MAIN-L0 iBKD LAYER-CONNECTION MECHANISM SMOKE")
    log("=" * 88)
    log(
        "[MECHANISM_SMOKE_POLICY] scientific_result=false official_test=false "
        "input_lineage=main_l0_v3 loader=l0_current_strong lambda=0.25 "
        "variant_selection_from_smoke=false"
    )

    partitions, split_manifest, source = load_train_validation_records(
        args.data_dir, download=True
    )
    records: dict[str, list[CubProbeRecord]] = {
        "train": partitions["train"],
        "validation": partitions["validation"],
    }
    counts = {name: len(values) for name, values in records.items()}
    validation_hash = split_manifest["validation_image_ids_sha256"]
    if counts != {"train": 5394, "validation": 600}:
        raise RuntimeError(f"unexpected CUB train/validation counts: {counts}")
    if validation_hash != EXPECTED_VALIDATION_IDS_SHA256:
        raise RuntimeError("CUB validation split hash changed")

    from .run_cub_r50_teacher_full import load_scientific_teacher

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
        teacher_hash != EXPECTED_TEACHER_CHECKPOINT_SHA256
        or teacher_state_hash != EXPECTED_TEACHER_STATE_SHA256
    ):
        raise RuntimeError("audited CUB teacher contract changed")
    _atomic_json_save(
        {
            "status": "pass",
            "dataset_source": source,
            "counts": counts,
            "split_manifest": split_manifest,
            "teacher_checkpoint_sha256": teacher_hash,
            "teacher_model_state_sha256": teacher_state_hash,
            "teacher_metadata": teacher_metadata,
            "official_test_instantiated": False,
        },
        args.output_dir / "input_audit.json",
    )

    classification_root = args.output_dir / "classification" / "students"
    classification_rows: list[dict[str, Any]] = []
    for mode in AGGREGATION_VARIANTS:
        _write_status(
            args.output_dir,
            status="running",
            phase="classification",
            classification_complete=len(classification_rows),
            probe_candidates_complete=0,
            probe_selections_complete=0,
            active_variant=mode,
        )
        run_name = f"cub_r50_224_ibkd_lambda_0.25_{mode}_b128_smoke_2ep_seed1"
        run_dir = classification_root / run_name
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
            "ibkd",
            "--fusion-ratio",
            "0.25",
            "--teacher-architecture",
            "resnet50_224_scratch",
            "--scientific-cub-r50-teacher",
            "--mechanism-ablation-smoke",
            "--ibkd-aggregation-mode",
            mode,
            "--batch-size",
            "128",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(classification_root),
            "--run-name",
            run_name,
            "--teacher-checkpoint",
            str(args.teacher_checkpoint),
            "--num-workers",
            str(args.num_workers),
            "--eval-batch-size",
            "200",
            "--seed",
            "1",
            "--save-student-checkpoint",
        ]
        _run_command(command, label=f"classification_{mode}")
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "timing_student_latest.pt"
        summary = _load_json(summary_path)
        expected_summary = {
            "status": "complete",
            "scientific_result": False,
            "official_test_accessed": False,
            "dataset": DATASET_NAME,
            "method": "ibkd",
            "batch_size": 128,
            "seed": 1,
            "fusion_ratio_lambda": 0.25,
            "actual_epochs": 2,
            "planned_epochs": 300,
            "mechanism_ablation_smoke": True,
            "cub_loader_profile": L0_CURRENT_STRONG,
        }
        for key, value in expected_summary.items():
            if summary.get(key) != value:
                raise RuntimeError(
                    f"classification summary mismatch {mode}:{key}"
                )
        if (
            not checkpoint_path.is_file()
            or summary.get("ibkd_aggregation", {}).get("mode") != mode
            or summary.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        ):
            raise RuntimeError(f"classification artifact contract failed for {mode}")
        classification_rows.append(
            {
                "mode": mode,
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
    if len(initial_hashes) != 1 or teacher_hashes != {teacher_hash}:
        raise RuntimeError("paired initialization or shared-teacher contract failed")
    _write_csv(
        _classification_csv_rows(classification_rows),
        args.output_dir / "classification_smoke_results.csv",
    )

    target_started = time.monotonic()
    targets: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        targets[split], _ = _target_cache(
            records[split],
            split=split,
            config=config,
            config_sha256=config_hash,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
    target_seconds = time.monotonic() - target_started
    target_values = {
        int(value)
        for payload in targets.values()
        for key in ("input_targets", "grid_targets")
        for value in torch.unique(payload[key])
    }
    if target_values != {0, 1}:
        raise RuntimeError(f"unexpected binary target values: {target_values}")

    probe_config = config["frozen_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    probe_seed = int(probe_config["probe_seeds"][0])
    probe_epochs = int(probe_config["epochs"])
    probe_rows: list[dict[str, Any]] = []
    feature_seconds_total = 0.0
    probe_seconds_total = 0.0
    for classification in classification_rows:
        mode = classification["mode"]
        _write_status(
            args.output_dir,
            status="running",
            phase="frozen_segmentation_probe",
            classification_complete=4,
            probe_candidates_complete=len(probe_rows) * len(learning_rates),
            probe_selections_complete=len(probe_rows),
            active_variant=mode,
        )
        model, encoder_audit = _load_encoder(
            Path(classification["checkpoint_path"]),
            classification["summary"],
            mode=mode,
            validation_hash=validation_hash,
            device=device,
        )
        feature_started = time.monotonic()
        features: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation"):
            features[split], _ = _feature_cache(
                model,
                records[split],
                split=split,
                variant=mode,
                encoder_seed=1,
                checkpoint_sha256=encoder_audit["checkpoint_sha256"],
                state_sha256=encoder_audit["student_state_sha256"],
                config=config,
                config_sha256=config_hash,
                cache_path=args.cache_dir / "features" / mode / f"{split}.pt",
                device=device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
        torch.cuda.synchronize(device)
        feature_seconds = time.monotonic() - feature_started
        feature_seconds_total += feature_seconds
        del model
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
            log(
                f"[MECHANISM_PROBE_CANDIDATE] mode={mode} lr={learning_rate:g} "
                f"best_epoch={candidate['best_epoch']} validation_grid_miou="
                f"{candidate['best_validation_grid_mean_iou']:.6f} "
                "scientific_result=false"
            )
        selected_state, selected = max(
            candidates,
            key=lambda item: item[1]["best_validation_grid_mean_iou"],
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
        if not all(
            _finite_metrics(validation_metrics[key])
            for key in ("grid_14x14", "input_224")
        ):
            raise RuntimeError(f"non-finite probe metrics for {mode}")
        probe_seconds = time.monotonic() - probe_started
        probe_seconds_total += probe_seconds
        probe_path = args.output_dir / "probes" / f"{mode}_seed1_smoke.pt"
        _atomic_torch_save(
            {
                "purpose": "phase1_cub_main_l0_ibkd_connection_probe_smoke_v1",
                "scientific_result": False,
                "official_test_accessed": False,
                "config_sha256": config_hash,
                "full_protocol_config_sha256": full_config_hash,
                "aggregation_mode": mode,
                "encoder_seed": 1,
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
                "model": selected_state,
            },
            probe_path,
        )
        saved = torch.load(probe_path, map_location="cpu", weights_only=True)
        strict_probe = probe_from_state(
            probe_config, probe_seed, saved["model"], device
        )
        del strict_probe, probe
        probe_rows.append(
            {
                "aggregation_mode": mode,
                "selected_learning_rate": selected["learning_rate"],
                "selected_epoch": selected["best_epoch"],
                "validation_grid_mean_iou": validation_metrics["grid_14x14"][
                    "mean_iou"
                ],
                "validation_input_224_mean_iou": validation_metrics[
                    "input_224"
                ]["mean_iou"],
                "feature_cache_seconds": feature_seconds,
                "probe_seconds": probe_seconds,
                "probe_checkpoint_sha256": file_sha256(probe_path),
                "scientific_result": False,
            }
        )
        del features
        torch.cuda.empty_cache()

    _write_csv(probe_rows, args.output_dir / "probe_smoke_results.csv")
    classification_full_estimate = sum(
        float(row["summary"]["avg_epoch_seconds"]) * 300.0
        for row in classification_rows
    )
    probe_full_train_estimate = probe_seconds_total * (5.0 * 100.0 / 2.0)
    elapsed = time.monotonic() - started
    summary = {
        "status": "pass",
        "smoke_id": config["smoke_id"],
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "official_test_evaluations": 0,
        "input_lineage_id": "main_l0_v3",
        "loader_profile": L0_CURRENT_STRONG,
        "config_sha256": config_hash,
        "full_protocol_config_sha256": full_config_hash,
        "dataset_counts": counts,
        "validation_image_ids_sha256": validation_hash,
        "teacher_checkpoint_sha256": teacher_hash,
        "teacher_model_state_sha256": teacher_state_hash,
        "paired_initial_student_state_sha256": next(iter(initial_hashes)),
        "classification": classification_rows,
        "frozen_probe": probe_rows,
        "completion": _smoke_completion_counts(
            classification_rows,
            probe_rows,
            learning_rates,
            teacher_download_and_audit=1,
        ),
        "timing": {
            "target_cache_seconds": target_seconds,
            "feature_cache_seconds": feature_seconds_total,
            "probe_smoke_seconds": probe_seconds_total,
            "smoke_suite_seconds": elapsed,
            "rough_full_classification_seconds_four_variants_three_seeds": (
                classification_full_estimate * 3.0
            ),
            "rough_full_probe_training_seconds_four_variants_three_encoder_seeds": (
                probe_full_train_estimate * 3.0
            ),
            "rough_estimate_excludes_test_feature_cache_and_io": True,
        },
    }
    expected_gate = config["completion_gate"]
    for key, expected in expected_gate.items():
        if summary["completion"].get(key) != expected:
            raise RuntimeError(
                f"mechanism smoke completion gate failed {key}: "
                f"{summary['completion'].get(key)} != {expected}"
            )
    _atomic_json_save(summary, args.output_dir / "mechanism_smoke_summary.json")
    _write_status(
        args.output_dir,
        status="pass",
        phase="complete",
        classification_complete=4,
        probe_candidates_complete=12,
        probe_selections_complete=4,
    )

    log("")
    log("[MECHANISM_SMOKE_RESULTS] non_scientific=true")
    by_mode = {row["aggregation_mode"]: row for row in probe_rows}
    classification_by_mode = {
        row["aggregation_mode"]: row
        for row in _classification_csv_rows(classification_rows)
    }
    for mode in AGGREGATION_VARIANTS:
        classification = classification_by_mode[mode]
        probe_row = by_mode[mode]
        log(
            f"  {mode}: val_cls={float(classification['validation_macro_top1_epoch2']):.4f}% "
            f"val_probe_mIoU={100.0 * float(probe_row['validation_input_224_mean_iou']):.4f}% "
            f"lr={float(probe_row['selected_learning_rate']):g}"
        )
    rough_seconds = (
        summary["timing"]["rough_full_classification_seconds_four_variants_three_seeds"]
        + summary["timing"][
            "rough_full_probe_training_seconds_four_variants_three_encoder_seeds"
        ]
    )
    log(
        "[MECHANISM_SMOKE_TIME] elapsed="
        f"{format_duration(elapsed)} rough_full_compute_without_test_or_io="
        f"{format_duration(rough_seconds)}"
    )
    log(
        "[MECHANISM_SMOKE_COMPLETE] status=pass teacher_audit=1/1 classification=4/4 "
        "aggregation_audits=4/4 probe_candidates=12/12 "
        "probe_selections=4/4 official_test=0"
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
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=32)
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
            failure=f"{type(error).__name__}: {error}",
        )
        log(f"[MECHANISM_SMOKE_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
