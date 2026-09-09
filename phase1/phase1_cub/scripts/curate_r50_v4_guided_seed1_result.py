#!/usr/bin/env python3
"""Audit and report CUB ResNet-50/224 v4 guided seed-1 batch profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ibkd_seg.phase1.models import create_student
from ibkd_seg.phase1.train_timing import state_dict_sha256


PROTOCOL_ID = "cub200_phase1_r50_224_guided_b128_b64_seed1_full_v4"
CONFIG_SHA256 = (
    "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633"
)
VALIDATION_SHA256 = (
    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
)
TEACHER_CHECKPOINT_SHA256 = (
    "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
)
TEACHER_STATE_SHA256 = (
    "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
)
BATCHES = (128, 64)
VARIANTS = ("lg", "alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5")
PROBE_SEEDS = (1, 2, 3, 4, 5)
BATCH_ROLES = {128: "locked_v3_partial_cell", 64: "batch64_sensitivity"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if source.suffix in {".csv", ".json"}:
        temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _finite_state(state: dict[str, torch.Tensor]) -> bool:
    return all(
        bool(torch.isfinite(tensor).all())
        for tensor in state.values()
        if tensor.is_floating_point()
    )


def _variant_from_summary(summary: dict[str, Any]) -> str:
    method = summary.get("method")
    if method == "lg":
        return "lg"
    if method == "alg":
        return "alg_warmup20"
    if method == "ibkd":
        value = float(summary["fusion_ratio_lambda"])
        return f"ibkd_lambda_{value:g}"
    raise RuntimeError(f"unsupported guided method: {method}")


def _checkpoint_relative_path(absolute_path: str) -> Path:
    marker = "/phase1_cub_r50_224_b128_b64_guided_probe_seed1_full_v4/"
    if marker not in absolute_path:
        raise RuntimeError(f"unexpected v4 checkpoint path: {absolute_path}")
    relative = Path(absolute_path.split(marker, 1)[1])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe v4 checkpoint path: {absolute_path}")
    return relative


def _verify_import(raw_dir: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("status") != "pass"
        or manifest.get("experiment_kind") != "resnet50-v4-guided-seed1"
        or manifest.get("h200_issue_id") != "727"
        or manifest.get("checkpoint_count") != 48
        or manifest.get("imported_file_count_excluding_this_manifest") != 77
    ):
        raise RuntimeError("wrong or incomplete CUB v4 guided seed-1 import manifest")
    files = manifest.get("files", [])
    if len(files) != 77:
        raise RuntimeError("unexpected CUB v4 imported file count")
    for item in files:
        path = raw_dir / item["path"]
        if not path.is_file():
            raise RuntimeError(f"imported v4 artifact is missing: {path}")
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"imported v4 artifact digest mismatch: {path}")


def _validate_combined(raw_dir: Path, combined: dict[str, Any]) -> None:
    expected_counts = {
        "classification_official_test_evaluations": 8,
        "classification_students": 8,
        "new_checkpoints": 48,
        "probe_lr_candidates": 120,
        "probe_official_test_evaluations": 40,
        "selected_probes": 40,
        "teacher_reused": 1,
    }
    teacher = combined.get("teacher", {})
    if (
        combined.get("status") != "complete"
        or combined.get("scientific_result") is not True
        or combined.get("protocol_id") != PROTOCOL_ID
        or combined.get("config_sha256") != CONFIG_SHA256
        or combined.get("batch_order") != [128, 64]
        or combined.get("encoder_seeds") != [1]
        or combined.get("variants") != list(VARIANTS)
        or combined.get("counts") != expected_counts
        or combined.get("same_initial_student_state_across_variants_and_batches")
        is not True
        or combined.get("final_confirmatory_matrix_complete") is not False
        or teacher.get("checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256
        or teacher.get("model_state_sha256") != TEACHER_STATE_SHA256
        or teacher.get("source_h200_issue") != 722
        or teacher.get("reused") is not True
    ):
        raise RuntimeError("CUB v4 combined completion contract failed")

    sequence = _load_json(raw_dir / "sequence_status.json")
    expected_sequence = {
        "classification_complete": 8,
        "classification_expected": 8,
        "official_test_evaluations_complete": 40,
        "official_test_evaluations_expected": 40,
        "probe_candidates_complete": 120,
        "probe_candidates_expected": 120,
        "probe_selections_complete": 40,
        "probe_selections_expected": 40,
    }
    if (
        sequence.get("status") != "complete"
        or sequence.get("phase") != "complete"
        or sequence.get("failure") is not None
        or any(sequence.get(key) != value for key, value in expected_sequence.items())
    ):
        raise RuntimeError("CUB v4 sequence did not complete every expected task")

    dataset = _load_json(raw_dir / "dataset_audit.json")
    if (
        dataset.get("status") != "pass"
        or dataset.get("counts") != {"train": 5394, "validation": 600, "test": 5794}
        or dataset.get("split_manifest", {}).get("validation_image_ids_sha256")
        != VALIDATION_SHA256
        or dataset.get("official_test_masks_opened_only_after_each_batch_selection_marker")
        is not True
    ):
        raise RuntimeError("CUB v4 dataset/split audit failed")


def _classification_audit(
    raw_dir: Path, combined: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    expected = {(batch, variant) for batch in BATCHES for variant in VARIANTS}
    csv_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for batch in BATCHES:
        with (raw_dir / f"batch{batch}/classification_results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != 4:
            raise RuntimeError(f"batch {batch} classification must contain four rows")
        for source in source_rows:
            row = {
                "batch_size": int(source["batch_size"]),
                "batch_profile_role": source["batch_profile_role"],
                "confirmatory_main_result": source["confirmatory_main_result"] == "True",
                "variant": source["variant"],
                "method": source["method"],
                "fusion_ratio_lambda": (
                    None
                    if source["fusion_ratio_lambda"] == ""
                    else float(source["fusion_ratio_lambda"])
                ),
                "controller_warmup_epochs": int(source["controller_warmup_epochs"]),
                "encoder_seed": int(source["encoder_seed"]),
                "selected_epoch": int(source["selected_epoch"]),
                "validation_macro_top1": float(source["validation_macro_top1"]),
                "test_macro_top1": float(source["test_macro_top1"]),
                "test_overall_top1": float(source["test_overall_top1"]),
                "test_top5": float(source["test_top5"]),
                "checkpoint_sha256": source["checkpoint_sha256"],
            }
            key = (row["batch_size"], row["variant"])
            if (
                key not in expected
                or key in csv_rows
                or row["batch_size"] != batch
                or row["batch_profile_role"] != BATCH_ROLES[batch]
                or row["confirmatory_main_result"] is not (batch == 128)
                or row["encoder_seed"] != 1
                or any(
                    not math.isfinite(row[name])
                    for name in (
                        "validation_macro_top1",
                        "test_macro_top1",
                        "test_overall_top1",
                        "test_top5",
                    )
                )
            ):
                raise RuntimeError(f"invalid classification CSV row: {key}")
            csv_rows[key] = row
    if set(csv_rows) != expected:
        raise RuntimeError("CUB v4 classification batch/variant matrix is incomplete")

    summary_rows: dict[tuple[int, str], dict[str, Any]] = {}
    checkpoint_entries: list[dict[str, Any]] = []
    initial_hashes: set[str] = set()
    for summary_path in sorted(raw_dir.glob("batch*/classification/students/*/summary.json")):
        summary = _load_json(summary_path)
        batch = int(summary["batch_size"])
        variant = _variant_from_summary(summary)
        key = (batch, variant)
        split = summary.get("split_manifest", {})
        warmup_expected = 0 if variant == "lg" else 20
        if (
            key not in expected
            or key in summary_rows
            or summary.get("status") != "complete"
            or summary.get("scientific_result") is not True
            or summary.get("protocol_config_sha256") != CONFIG_SHA256
            or summary.get("epochs") != 300
            or summary.get("seed") != 1
            or summary.get("batch_profile_role") != BATCH_ROLES[batch]
            or summary.get("confirmatory_main_result") is not (batch == 128)
            or summary.get("final_confirmatory_matrix_complete") is not False
            or summary.get("official_test_evaluations") != 1
            or summary.get("official_test_used_for_training_or_selection") is not False
            or summary.get("selected_checkpoint_strict_reloaded") is not True
            or summary.get("teacher_checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256
            or summary.get("teacher_model_state_sha256") != TEACHER_STATE_SHA256
            or split.get("validation_image_ids_sha256") != VALIDATION_SHA256
            or summary.get("controller_final", {}).get("warmup_epochs")
            != warmup_expected
        ):
            raise RuntimeError(f"student completion contract failed: {key}")

        initial_hashes.add(summary["initial_student_state_sha256"])
        checkpoint_path = summary_path.parent / "student_best_validation.pt"
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != summary["checkpoint_sha256"]:
            raise RuntimeError(f"student checkpoint hash mismatch: {key}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = create_student(num_classes=200, drop_path_rate=0.1)
        incompatible = model.load_state_dict(payload["student"], strict=True)
        state_hash = state_dict_sha256(model)
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
            or state_hash != summary["student_state_sha256"]
            or not _finite_state(payload["student"])
        ):
            raise RuntimeError(f"student state audit failed: {key}")
        checkpoint_entries.append(
            {
                "kind": "classification_encoder",
                "batch_size": batch,
                "variant": variant,
                "encoder_seed": 1,
                "path_under_ignored_raw_root": checkpoint_path.relative_to(raw_dir).as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": checkpoint_hash,
                "model_state_sha256": state_hash,
                "strict_load": True,
                "all_floating_tensors_finite": True,
            }
        )
        summary_rows[key] = summary
        del model, payload

    if set(summary_rows) != expected or len(initial_hashes) != 1:
        raise RuntimeError("matched student initialization or summary matrix failed")
    if initial_hashes != {combined["initial_student_state_sha256"]}:
        raise RuntimeError("student initialization hash differs from combined summary")

    rows = [csv_rows[(batch, variant)] for batch in BATCHES for variant in VARIANTS]
    for row in rows:
        summary = summary_rows[(row["batch_size"], row["variant"])]
        if (
            row["checkpoint_sha256"] != summary["checkpoint_sha256"]
            or row["selected_epoch"] != summary["selected_epoch"]
            or not math.isclose(
                row["test_macro_top1"],
                summary["official_test"]["macro_top1"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("classification CSV and student summary disagree")

    summary_payload = {
        "schema_version": 1,
        "status": "complete_audited_partial_v4",
        "protocol_id": PROTOCOL_ID,
        "independent_unit": "encoder_seed",
        "encoder_seed": 1,
        "independent_encoder_n_per_batch": 1,
        "encoder_seed_standard_deviation_estimable": False,
        "batch128_role": BATCH_ROLES[128],
        "batch64_role": BATCH_ROLES[64],
        "matched_initial_student_state_across_variants_and_batches": True,
        "initial_student_state_sha256": next(iter(initial_hashes)),
        "cells": rows,
        "descriptive_batch64_minus_batch128": {
            variant: {
                "test_macro_top1_percentage_points": (
                    csv_rows[(64, variant)]["test_macro_top1"]
                    - csv_rows[(128, variant)]["test_macro_top1"]
                )
            }
            for variant in VARIANTS
        },
        "official_test_evaluations": 8,
        "final_six_method_three_seed_matrix_complete": False,
    }
    return rows, summary_payload, checkpoint_entries


def _probe_audit(
    raw_dir: Path, combined: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    expected = {
        (batch, variant, probe_seed)
        for batch in BATCHES
        for variant in VARIANTS
        for probe_seed in PROBE_SEEDS
    }
    rows: list[dict[str, Any]] = []
    checkpoint_entries: list[dict[str, Any]] = []
    for batch in BATCHES:
        with (raw_dir / f"batch{batch}/probe/raw_results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != 20:
            raise RuntimeError(f"batch {batch} probe must contain twenty selected rows")
        for source in source_rows:
            row = {
                "batch_size": int(source["batch_size"]),
                "batch_profile_role": source["batch_profile_role"],
                "confirmatory_main_result": source["confirmatory_main_result"] == "True",
                "variant": source["variant"],
                "encoder_seed": int(source["encoder_seed"]),
                "probe_seed": int(source["probe_seed"]),
                "selected_learning_rate": float(source["selected_learning_rate"]),
                "selected_epoch": int(source["selected_epoch"]),
                "validation_grid_mean_iou": float(source["validation_grid_mean_iou"]),
                "validation_input_224_mean_iou": float(
                    source["validation_input_224_mean_iou"]
                ),
                "test_grid_mean_iou": float(source["test_grid_mean_iou"]),
                "test_input_224_mean_iou": float(source["test_input_224_mean_iou"]),
                "test_input_224_foreground_iou": float(
                    source["test_input_224_foreground_iou"]
                ),
                "test_input_224_background_iou": float(
                    source["test_input_224_background_iou"]
                ),
                "test_input_224_foreground_dice": float(
                    source["test_input_224_foreground_dice"]
                ),
                "test_input_224_pixel_accuracy": float(
                    source["test_input_224_pixel_accuracy"]
                ),
                "encoder_checkpoint_sha256": source["encoder_checkpoint_sha256"],
                "probe_checkpoint_sha256": source["probe_checkpoint_sha256"],
                "official_test_evaluations": int(source["official_test_evaluations"]),
                "scientific_result": source["scientific_result"] == "True",
            }
            key = (row["batch_size"], row["variant"], row["probe_seed"])
            if (
                key not in expected
                or row["batch_size"] != batch
                or row["batch_profile_role"] != BATCH_ROLES[batch]
                or row["confirmatory_main_result"] is not (batch == 128)
                or row["encoder_seed"] != 1
                or row["official_test_evaluations"] != 1
                or row["scientific_result"] is not True
                or any(
                    not math.isfinite(value)
                    for name, value in row.items()
                    if isinstance(value, float) and name != "selected_learning_rate"
                )
            ):
                raise RuntimeError(f"invalid selected probe row: {key}")

            checkpoint_path = raw_dir / _checkpoint_relative_path(
                source["probe_checkpoint_path"]
            )
            checkpoint_hash = file_sha256(checkpoint_path)
            if checkpoint_hash != row["probe_checkpoint_sha256"]:
                raise RuntimeError(f"selected probe checkpoint hash mismatch: {key}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            probe = nn.Conv2d(192, 2, kernel_size=1, bias=True)
            incompatible = probe.load_state_dict(payload["model"], strict=True)
            if (
                incompatible.missing_keys
                or incompatible.unexpected_keys
                or not _finite_state(payload["model"])
            ):
                raise RuntimeError(f"selected probe state audit failed: {key}")
            checkpoint_entries.append(
                {
                    "kind": "selected_probe",
                    "batch_size": batch,
                    "variant": row["variant"],
                    "encoder_seed": 1,
                    "probe_seed": row["probe_seed"],
                    "path_under_ignored_raw_root": checkpoint_path.relative_to(raw_dir).as_posix(),
                    "bytes": checkpoint_path.stat().st_size,
                    "checkpoint_sha256": checkpoint_hash,
                    "model_state_sha256": state_dict_sha256(probe),
                    "strict_load": True,
                    "all_floating_tensors_finite": True,
                }
            )
            rows.append(row)
            del probe, payload

        selection = _load_json(
            raw_dir / f"batch{batch}/probe/selection_complete_before_test.json"
        )
        if (
            selection.get("status") != "complete"
            or selection.get("config_sha256") != CONFIG_SHA256
            or selection.get("batch_size") != batch
            or selection.get("completed_probe_selections") != 20
            or selection.get("expected_probe_selections") != 20
            or selection.get("official_test_masks_accessed") is not False
            or not all(selection.get("selection_contracts", {}).values())
        ):
            raise RuntimeError(f"batch {batch} selection-before-test contract failed")

    if {(r["batch_size"], r["variant"], r["probe_seed"]) for r in rows} != expected:
        raise RuntimeError("CUB v4 selected probe matrix is incomplete")

    batch_summaries = {
        batch: _load_json(raw_dir / f"batch{batch}/batch_full_summary.json")
        for batch in BATCHES
    }
    per_encoder_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for batch in BATCHES:
        reported = {
            item["variant"]: item
            for item in batch_summaries[batch]["probe_aggregates"]
        }
        for variant in VARIANTS:
            values = [
                row["test_input_224_mean_iou"]
                for row in rows
                if row["batch_size"] == batch and row["variant"] == variant
            ]
            if len(values) != 5:
                raise RuntimeError("expected five probe seeds per encoder")
            cell = {
                "batch_size": batch,
                "batch_profile_role": BATCH_ROLES[batch],
                "confirmatory_main_result": batch == 128,
                "variant": variant,
                "encoder_seed": 1,
                "independent_encoder_n": 1,
                "probe_seeds_per_encoder": 5,
                "test_input_224_mean_iou_probe_seed_values": values,
                "test_input_224_mean_iou_probe_seed_mean": statistics.mean(values),
                "test_input_224_mean_iou_probe_seed_sample_sd": statistics.stdev(values),
                "encoder_seed_standard_deviation_estimable": False,
            }
            source = reported[variant]
            if (
                not math.isclose(
                    cell["test_input_224_mean_iou_probe_seed_mean"],
                    source["mean_over_probe_seeds"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    cell["test_input_224_mean_iou_probe_seed_sample_sd"],
                    source["sample_standard_deviation_over_probe_seeds"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise RuntimeError("recomputed probe aggregate differs from H200 summary")
            lookup[(batch, variant)] = cell
            per_encoder_rows.append(cell)

    summary_payload = {
        "schema_version": 1,
        "status": "complete_audited_partial_v4",
        "protocol_id": PROTOCOL_ID,
        "primary_metric": "official_test_input_224_two_class_mean_iou",
        "independent_unit": "encoder_seed",
        "encoder_seed": 1,
        "independent_encoder_n_per_batch": 1,
        "probe_seed_is_not_independent_replication": True,
        "probe_seeds_per_encoder": 5,
        "encoder_seed_standard_deviation_estimable": False,
        "cells": per_encoder_rows,
        "descriptive_within_batch_contrasts": {
            str(batch): {
                variant: {
                    "minus_lg_mean_iou": (
                        lookup[(batch, variant)][
                            "test_input_224_mean_iou_probe_seed_mean"
                        ]
                        - lookup[(batch, "lg")][
                            "test_input_224_mean_iou_probe_seed_mean"
                        ]
                    ),
                    "minus_alg_warmup20_mean_iou": (
                        lookup[(batch, variant)][
                            "test_input_224_mean_iou_probe_seed_mean"
                        ]
                        - lookup[(batch, "alg_warmup20")][
                            "test_input_224_mean_iou_probe_seed_mean"
                        ]
                    ),
                }
                for variant in ("ibkd_lambda_0.25", "ibkd_lambda_0.5")
            }
            for batch in BATCHES
        },
        "descriptive_batch64_minus_batch128": {
            variant: {
                "test_input_224_mean_iou": (
                    lookup[(64, variant)]["test_input_224_mean_iou_probe_seed_mean"]
                    - lookup[(128, variant)]["test_input_224_mean_iou_probe_seed_mean"]
                )
            }
            for variant in VARIANTS
        },
        "all_40_selections_completed_before_each_batch_official_test": True,
        "official_test_evaluations": 40,
        "final_six_method_three_seed_matrix_complete": False,
    }
    return per_encoder_rows, summary_payload, checkpoint_entries


def _copy_source_evidence(raw_dir: Path, report_dir: Path) -> None:
    root_files = {
        "combined_full_summary.json": "h200_combined_full_summary.json",
        "dataset_audit.json": "h200_dataset_audit.json",
        "sequence_status.json": "h200_sequence_status.json",
        "checkpoint_manifest.json": "h200_checkpoint_manifest.json",
    }
    batch_files = {
        "batch_full_summary.json": "batch_full_summary.json",
        "checkpoint_manifest.json": "checkpoint_manifest.json",
        "classification_results.csv": "classification_results.csv",
        "probe/raw_results.csv": "probe_raw_results.csv",
        "probe/results.json": "probe_results.json",
        "probe/selection_complete_before_test.json": "probe_selection_complete_before_test.json",
    }
    for source_name, destination_name in root_files.items():
        _copy(raw_dir / source_name, report_dir / destination_name)
    for batch in BATCHES:
        for source_name, destination_name in batch_files.items():
            _copy(
                raw_dir / f"batch{batch}" / source_name,
                report_dir / f"h200_batch{batch}_{destination_name}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    repository_root = Path(__file__).resolve().parents[3]

    manifest = _load_json(raw_dir / "artifact_manifest.json")
    _verify_import(raw_dir, manifest)
    combined = _load_json(raw_dir / "combined_full_summary.json")
    _validate_combined(raw_dir, combined)
    config_path = (
        repository_root
        / "phase1/phase1_cub/configs/"
        "cub200_r50_224_b128_b64_guided_seed1_full_v4.json"
    )
    if file_sha256(config_path) != CONFIG_SHA256:
        raise RuntimeError("repository v4 config no longer matches the H200 result")

    classification_rows, classification_summary, encoder_entries = (
        _classification_audit(raw_dir, combined)
    )
    probe_rows, probe_summary, probe_entries = _probe_audit(raw_dir, combined)

    h200_manifest = _load_json(raw_dir / "checkpoint_manifest.json")
    h200_entries = h200_manifest.get("entries", [])
    external = [entry for entry in h200_entries if entry.get("kind") == "external_shared_teacher"]
    if (
        h200_manifest.get("count") != 49
        or len(h200_entries) != 49
        or len(external) != 1
        or external[0].get("sha256") != TEACHER_CHECKPOINT_SHA256
        or external[0].get("source_h200_issue") != 722
    ):
        raise RuntimeError("H200 v4 checkpoint manifest is incomplete")
    audit_entries = [*encoder_entries, *probe_entries]
    if len(audit_entries) != 48:
        raise RuntimeError("independent checkpoint audit did not cover 48 new files")
    h200_new_hashes = {
        entry["sha256"]
        for entry in h200_entries
        if entry.get("kind") != "external_shared_teacher"
    }
    audited_hashes = {entry["checkpoint_sha256"] for entry in audit_entries}
    if h200_new_hashes != audited_hashes or len(h200_new_hashes) != 48:
        raise RuntimeError("independent checkpoint hashes differ from H200 manifest")

    checkpoint_audit = {
        "schema_version": 1,
        "status": "pass",
        "safe_torch_load": "weights_only_true",
        "external_teacher": {
            "source_h200_issue": 722,
            "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "model_state_sha256": TEACHER_STATE_SHA256,
            "not_duplicated_in_issue727_archive": True,
        },
        "new_checkpoint_count": 48,
        "classification_encoder_checkpoint_count": 8,
        "selected_probe_checkpoint_count": 40,
        "all_file_hashes_match": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "entries": sorted(
            audit_entries,
            key=lambda item: (
                item["kind"],
                int(item["batch_size"]),
                VARIANTS.index(item["variant"]),
                int(item.get("probe_seed", 0)),
            ),
        ),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    _copy_source_evidence(raw_dir, report_dir)
    _save_json(manifest, report_dir / "source_manifest.json")
    _save_json(checkpoint_audit, report_dir / "checkpoint_audit.json")
    _save_json(classification_summary, report_dir / "classification_summary.json")
    _save_json(probe_summary, report_dir / "probe_summary.json")
    _write_csv(classification_rows, report_dir / "classification_seed1.csv")

    probe_csv_rows = [
        {
            key: value
            for key, value in row.items()
            if key != "test_input_224_mean_iou_probe_seed_values"
        }
        for row in probe_rows
    ]
    _write_csv(probe_csv_rows, report_dir / "probe_per_encoder_seed1.csv")

    class_lookup = {
        (row["batch_size"], row["variant"]): row for row in classification_rows
    }
    probe_lookup = {
        (row["batch_size"], row["variant"]): row for row in probe_rows
    }
    labels = {
        "lg": "LG",
        "alg_warmup20": "ALG-w20",
        "ibkd_lambda_0.25": "iBKD λ=0.25",
        "ibkd_lambda_0.5": "iBKD λ=0.5",
    }

    classification_lines = []
    probe_lines = []
    for variant in VARIANTS:
        c128 = class_lookup[(128, variant)]
        c64 = class_lookup[(64, variant)]
        p128 = probe_lookup[(128, variant)]
        p64 = probe_lookup[(64, variant)]
        classification_lines.append(
            f"| {labels[variant]} | {c128['selected_epoch']} | "
            f"{c128['test_macro_top1']:.3f}% | {c64['selected_epoch']} | "
            f"{c64['test_macro_top1']:.3f}% |"
        )
        probe_lines.append(
            f"| {labels[variant]} | "
            f"{100 * p128['test_input_224_mean_iou_probe_seed_mean']:.3f} ± "
            f"{100 * p128['test_input_224_mean_iou_probe_seed_sample_sd']:.3f}% | "
            f"{100 * p64['test_input_224_mean_iou_probe_seed_mean']:.3f} ± "
            f"{100 * p64['test_input_224_mean_iou_probe_seed_sample_sd']:.3f}% |"
        )

    p128_i025_alg = 100 * (
        probe_lookup[(128, "ibkd_lambda_0.25")][
            "test_input_224_mean_iou_probe_seed_mean"
        ]
        - probe_lookup[(128, "alg_warmup20")][
            "test_input_224_mean_iou_probe_seed_mean"
        ]
    )
    p128_i025_lg = 100 * (
        probe_lookup[(128, "ibkd_lambda_0.25")][
            "test_input_224_mean_iou_probe_seed_mean"
        ]
        - probe_lookup[(128, "lg")]["test_input_224_mean_iou_probe_seed_mean"]
    )
    p64_i025_alg = 100 * (
        probe_lookup[(64, "ibkd_lambda_0.25")][
            "test_input_224_mean_iou_probe_seed_mean"
        ]
        - probe_lookup[(64, "alg_warmup20")][
            "test_input_224_mean_iou_probe_seed_mean"
        ]
    )
    p64_i025_lg = 100 * (
        probe_lookup[(64, "ibkd_lambda_0.25")][
            "test_input_224_mean_iou_probe_seed_mean"
        ]
        - probe_lookup[(64, "lg")]["test_input_224_mean_iou_probe_seed_mean"]
    )

    result_markdown = f"""# CUB ResNet-50/224 v4 guided seed-1 결과

상태: **H200 issue 727 완료 · 독립 감사 통과 · 부분 매트릭스**

이 실행은 issue 722의 한 ResNet-50/224 scratch Teacher를 공유해 LG, ALG-w20,
iBKD λ=0.25/0.5를 encoder seed 1에서 학습하고 frozen segmentation probe까지
완료했습니다. Batch 128은 잠긴 v3 주 비교의 네 셀이며, batch 64는 사전 표기한
sensitivity profile입니다. 아직 Vanilla/KD와 batch-128 encoder seed 2·3이 없으므로
최종 6방법×3seed 결과가 아닙니다.

## 완결성 및 감사

- 실행시간: {combined['elapsed_seconds'] / 3600:.2f}시간
- 분류: 8/8, probe LR 후보: 120/120, validation 선택: 40/40
- 분류 official test: 8/8, probe official test: 40/40
- 새 checkpoint: encoder 8개 + 선택 probe 40개 = 48개
- 48개 모두 파일 SHA-256, `weights_only=True`, strict load, 유한값 감사 통과
- 공용 Teacher hash와 train/validation/test `5,394 / 600 / 5,794` split hash 일치
- 원시 checkpoint는 [검증된 GitHub Release](artifact_release.json)에 보존하고
  Git history에는 hash·요약만 기록

## 200종 분류

| 방법 | batch128 선택 epoch | batch128 test macro top-1 | batch64 선택 epoch | batch64 test macro top-1 |
|---|---:|---:|---:|---:|
{chr(10).join(classification_lines)}

두 batch 모두 iBKD λ=0.25의 seed-1 분류 정확도가 네 방법 중 가장 높았습니다.
반면 λ=0.5는 두 batch 모두 크게 낮아, 현재 설정에서는 λ 민감도가 큽니다.

## Frozen segmentation probe

아래 `±`는 한 encoder에서 반복한 **5개 probe seed의 sample SD**입니다. 독립
encoder는 batch별 하나뿐이므로 encoder-seed SD나 방법 간 통계적 유의성으로
해석하면 안 됩니다.

| 방법 | batch128 test input-224 mIoU | batch64 test input-224 mIoU |
|---|---:|---:|
{chr(10).join(probe_lines)}

## 현재 해석

- Batch 128에서 iBKD λ=0.25는 ALG-w20보다 `{p128_i025_alg:+.3f}%p` 높지만,
  LG보다는 `{p128_i025_lg:+.3f}%p` 낮습니다.
- Batch 64에서는 iBKD λ=0.25가 ALG-w20보다 `{p64_i025_alg:+.3f}%p`, LG보다
  `{p64_i025_lg:+.3f}%p` 낮습니다.
- 따라서 seed 1만 놓고 보면 “iBKD가 ALG보다 항상 높다”거나 “guided 방법 중
  iBKD가 공간정보를 가장 잘 보존한다”는 결론은 성립하지 않습니다. Batch 128의
  ALG 대비 우위는 관찰됐지만 LG가 더 높고, batch 64에서는 방향도 바뀝니다.
- 분류에서는 iBKD λ=0.25가 가장 높지만 probe에서는 LG가 가장 높습니다. 즉 이번
  seed-1 결과는 분류 성능과 frozen spatial probe 성능이 같은 순서가 아님을
  보여줍니다.
- Batch 크기 효과도 방법마다 방향과 크기가 다릅니다. Batch 64는 탐색적
  sensitivity이므로 batch 선택 근거로 사후 사용하지 않습니다.

최종 guided 판단은 이미 고정해 실행 중인 batch-128 seed 2·3을 합쳐 encoder
seed `[1,2,3]` 기준으로 내려야 합니다. 이후 Vanilla/KD까지 같은 계약으로 채워야
전체 6방법 비교가 됩니다. Part localization, spatial CKA, attention–GT 분석에는
이번에 보존한 8개 encoder checkpoint를 후속 입력으로 사용할 수 있습니다.
"""
    (report_dir / "RESULTS.md").write_text(result_markdown, encoding="utf-8")
    print(
        "[CUB_R50_V4_GUIDED_SEED1_CURATE_DONE] status=pass checkpoints=48 "
        f"b128_lg_miou={probe_lookup[(128, 'lg')]['test_input_224_mean_iou_probe_seed_mean']:.6f} "
        f"b128_ibkd025_miou={probe_lookup[(128, 'ibkd_lambda_0.25')]['test_input_224_mean_iou_probe_seed_mean']:.6f} "
        f"report={report_dir}",
        flush=True,
    )
    return {
        "classification": classification_summary,
        "probe": probe_summary,
        "checkpoint_audit": checkpoint_audit,
    }


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
