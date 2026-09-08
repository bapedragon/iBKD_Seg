#!/usr/bin/env python3
"""Audit and report the completed, superseded CUB ResNet-56 v2 guided shard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ibkd_seg.phase1.models import ResNet56, create_student
from ibkd_seg.phase1.train_timing import state_dict_sha256


V2_CONFIG_SHA256 = (
    "0cf751c28168872a4108274644f80dadc7466d5c1210995e7da3abfc0737e575"
)
VALIDATION_HASH = (
    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
)
VARIANTS = ("alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5")
ENCODER_SEEDS = (1, 2, 3)
PROBE_SEEDS = (1, 2, 3, 4, 5)
TRACKED_SOURCE_ARTIFACTS = {
    "combined_full_summary.json": "h200_combined_full_summary.json",
    "classification_results.csv": "h200_classification_results.csv",
    "dataset_audit.json": "h200_dataset_audit.json",
    "sequence_status.json": "h200_sequence_status.json",
    "checkpoint_manifest.json": "h200_checkpoint_manifest.json",
    "probe/raw_results.csv": "h200_probe_raw_results.csv",
    "probe/results.json": "h200_probe_results.json",
    "probe/selection_complete_before_test.json": (
        "h200_probe_selection_complete_before_test.json"
    ),
    "probe/qualitative_manifest.json": "h200_qualitative_manifest.json",
}


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
    if source.suffix == ".csv":
        temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _variant_from_summary(summary: dict[str, Any]) -> str:
    if summary["method"] == "alg":
        return "alg_warmup20"
    value = float(summary["fusion_ratio_lambda"])
    return f"ibkd_lambda_{value:g}"


def _checkpoint_relative_path(absolute_path: str) -> Path:
    marker = "/phase1_cub_b128_full_v2_guided/"
    if marker not in absolute_path:
        raise RuntimeError(f"unexpected v2 checkpoint path: {absolute_path}")
    return Path(absolute_path.split(marker, 1)[1])


def _finite_state(state: dict[str, torch.Tensor]) -> bool:
    return all(
        bool(torch.isfinite(tensor).all())
        for tensor in state.values()
        if tensor.is_floating_point()
    )


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "raw": values,
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def _paired(
    by_variant_seed: dict[str, dict[int, float]], left: str, right: str
) -> dict[str, Any]:
    raw = {
        str(seed): by_variant_seed[left][seed] - by_variant_seed[right][seed]
        for seed in ENCODER_SEEDS
    }
    values = list(raw.values())
    return {
        "contrast": f"{left}_minus_{right}",
        "raw_by_encoder_seed": raw,
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def _verify_import(raw_dir: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("status") != "pass"
        or manifest.get("experiment_kind") != "resnet56-v2-guided"
        or manifest.get("h200_issue_id") != "716"
        or manifest.get("checkpoint_count") != 55
    ):
        raise RuntimeError("wrong or incomplete CUB v2 guided import manifest")
    files = manifest.get("files", [])
    if len(files) != 78:
        raise RuntimeError("unexpected CUB v2 imported file count")
    for item in files:
        path = raw_dir / item["path"]
        if not path.is_file():
            raise RuntimeError(f"imported v2 artifact is missing: {path}")
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"imported v2 artifact digest mismatch: {path}")


def _classification_audit(
    raw_dir: Path, combined: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    with (raw_dir / "classification_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 9:
        raise RuntimeError("v2 guided classification must contain nine rows")

    rows: list[dict[str, Any]] = []
    for row in source_rows:
        parsed = {
            "variant": row["variant"],
            "method": row["method"],
            "fusion_ratio_lambda": (
                None if row["fusion_ratio_lambda"] == "" else float(row["fusion_ratio_lambda"])
            ),
            "controller_warmup_epochs": int(row["controller_warmup_epochs"]),
            "encoder_seed": int(row["encoder_seed"]),
            "selected_epoch": int(row["selected_epoch"]),
            "validation_macro_top1": float(row["validation_macro_top1"]),
            "test_macro_top1": float(row["test_macro_top1"]),
            "test_overall_top1": float(row["test_overall_top1"]),
            "test_top5": float(row["test_top5"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        if any(
            not math.isfinite(parsed[key])
            for key in (
                "validation_macro_top1",
                "test_macro_top1",
                "test_overall_top1",
                "test_top5",
            )
        ):
            raise RuntimeError("classification CSV contains non-finite metrics")
        rows.append(parsed)
    keys = {(row["variant"], row["encoder_seed"]) for row in rows}
    expected = {(variant, seed) for variant in VARIANTS for seed in ENCODER_SEEDS}
    if keys != expected:
        raise RuntimeError("v2 classification variant/seed matrix is incomplete")

    individual: dict[tuple[str, int], dict[str, Any]] = {}
    controller_rows: list[dict[str, Any]] = []
    initial_hashes: dict[int, set[str]] = defaultdict(set)
    checkpoint_entries: list[dict[str, Any]] = []
    teacher_checkpoint_hash = combined["teacher"]["checkpoint_sha256"]
    for summary_path in sorted(raw_dir.glob("classification/students/*/summary.json")):
        summary = _load_json(summary_path)
        variant = _variant_from_summary(summary)
        seed = int(summary["seed"])
        key = (variant, seed)
        if key in individual or key not in expected:
            raise RuntimeError(f"unexpected or duplicate student summary: {key}")
        if (
            summary.get("status") != "complete"
            or summary.get("scientific_result") is not True
            or summary.get("official_test_evaluations") != 1
            or summary.get("official_test_used_for_training_or_selection") is not False
            or summary.get("selected_checkpoint_strict_reloaded") is not True
            or summary.get("teacher_checkpoint_sha256") != teacher_checkpoint_hash
            or summary.get("split_manifest", {}).get("validation_image_ids_sha256")
            != VALIDATION_HASH
        ):
            raise RuntimeError(f"student completion contract failed: {key}")
        individual[key] = summary
        initial_hashes[seed].add(summary["initial_student_state_sha256"])
        controller = summary.get("controller_final")
        if controller is None or controller.get("warmup_epochs") != 20:
            raise RuntimeError(f"guided controller contract failed: {key}")
        controller_rows.append(
            {
                "variant": variant,
                "encoder_seed": seed,
                "controller_stop_epoch": controller.get("stop_epoch"),
                "controller_active_at_end": controller.get("active"),
            }
        )

        checkpoint_path = summary_path.parent / "student_best_validation.pt"
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != summary["checkpoint_sha256"]:
            raise RuntimeError(f"student checkpoint hash mismatch: {key}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = create_student(num_classes=200, drop_path_rate=0.1)
        incompatible = model.load_state_dict(payload["student"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"student strict load failed: {key}")
        state_hash = state_dict_sha256(model)
        if state_hash != summary["student_state_sha256"] or not _finite_state(
            payload["student"]
        ):
            raise RuntimeError(f"student state audit failed: {key}")
        checkpoint_entries.append(
            {
                "kind": "classification_encoder",
                "variant": variant,
                "encoder_seed": seed,
                "path_under_ignored_raw_root": checkpoint_path.relative_to(raw_dir).as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": checkpoint_hash,
                "model_state_sha256": state_hash,
                "strict_load": True,
                "all_floating_tensors_finite": True,
            }
        )
        del model, payload

    if set(individual) != expected or any(len(values) != 1 for values in initial_hashes.values()):
        raise RuntimeError("matched student initialization contract failed")
    for row in rows:
        summary = individual[(row["variant"], row["encoder_seed"])]
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
            raise RuntimeError("classification CSV and individual summary disagree")

    by_variant_seed = {
        variant: {
            seed: next(
                row["test_macro_top1"]
                for row in rows
                if row["variant"] == variant and row["encoder_seed"] == seed
            )
            for seed in ENCODER_SEEDS
        }
        for variant in VARIANTS
    }
    aggregates = [
        {
            "variant": variant,
            "metric": "official_test_macro_top1_percent",
            **_summary([by_variant_seed[variant][seed] for seed in ENCODER_SEEDS]),
        }
        for variant in VARIANTS
    ]
    summary_payload = {
        "schema_version": 1,
        "status": "complete_audited_superseded_v2",
        "independent_unit": "encoder_seed",
        "independent_n": 3,
        "aggregates": aggregates,
        "paired_contrasts": [
            {
                **_paired(by_variant_seed, variant, "alg_warmup20"),
                "metric": "official_test_macro_top1_percentage_points",
            }
            for variant in ("ibkd_lambda_0.25", "ibkd_lambda_0.5")
        ],
        "matched_initial_student_state_per_seed": True,
        "initial_student_state_sha256_by_seed": {
            str(seed): next(iter(initial_hashes[seed])) for seed in ENCODER_SEEDS
        },
        "controller": sorted(
            controller_rows, key=lambda row: (VARIANTS.index(row["variant"]), row["encoder_seed"])
        ),
    }
    return rows, summary_payload, checkpoint_entries


def _teacher_audit(raw_dir: Path, combined: dict[str, Any]) -> dict[str, Any]:
    summary_path = next(raw_dir.glob("classification/teacher/*/summary.json"))
    summary = _load_json(summary_path)
    checkpoint_path = summary_path.parent / "teacher_best_validation.pt"
    if (
        summary.get("status") != "complete"
        or summary.get("scientific_result") is not True
        or summary.get("epochs") != 300
        or summary.get("batch_size") != 128
        or summary.get("official_test_evaluations") != 1
        or summary.get("official_test_used_for_training_or_selection") is not False
        or summary.get("selected_checkpoint_strict_reloaded") is not True
        or summary.get("split_manifest", {}).get("validation_image_ids_sha256")
        != VALIDATION_HASH
    ):
        raise RuntimeError("v2 ResNet-56 teacher completion contract failed")
    checkpoint_hash = file_sha256(checkpoint_path)
    if (
        checkpoint_hash != summary["checkpoint_sha256"]
        or checkpoint_hash != combined["teacher"]["checkpoint_sha256"]
    ):
        raise RuntimeError("v2 teacher checkpoint file hash mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ResNet56(num_classes=200)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("v2 teacher strict load failed")
    state_hash = state_dict_sha256(model)
    if (
        state_hash != summary["model_state_sha256"]
        or state_hash != combined["teacher"]["model_state_sha256"]
        or not _finite_state(payload["model"])
    ):
        raise RuntimeError("v2 teacher state audit failed")
    del model, payload
    return {
        "kind": "teacher",
        "architecture": "cifar_style_resnet56_6n_plus_2_n9",
        "input_size": 32,
        "seed": 1,
        "selected_epoch": summary["selected_epoch"],
        "selected_validation": summary["selected_validation"],
        "official_test": summary["official_test"],
        "path_under_ignored_raw_root": checkpoint_path.relative_to(raw_dir).as_posix(),
        "bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": state_hash,
        "strict_load": True,
        "all_floating_tensors_finite": True,
    }


def _probe_audit(
    raw_dir: Path, combined: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    with (raw_dir / "probe/raw_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 45:
        raise RuntimeError("v2 guided probe must contain 45 selected results")
    rows: list[dict[str, Any]] = []
    checkpoint_entries: list[dict[str, Any]] = []
    expected = {
        (variant, encoder_seed, probe_seed)
        for variant in VARIANTS
        for encoder_seed in ENCODER_SEEDS
        for probe_seed in PROBE_SEEDS
    }
    for source in source_rows:
        row = {
            "variant": source["variant"],
            "encoder_seed": int(source["encoder_seed"]),
            "probe_seed": int(source["probe_seed"]),
            "selected_learning_rate": float(source["selected_learning_rate"]),
            "selected_epoch": int(source["selected_epoch"]),
            "validation_grid_mean_iou": float(source["validation_grid_mean_iou"]),
            "validation_input_224_mean_iou": float(source["validation_input_224_mean_iou"]),
            "test_grid_mean_iou": float(source["test_grid_mean_iou"]),
            "test_input_224_mean_iou": float(source["test_input_224_mean_iou"]),
            "test_input_224_foreground_iou": float(source["test_input_224_foreground_iou"]),
            "test_input_224_background_iou": float(source["test_input_224_background_iou"]),
            "test_input_224_foreground_dice": float(source["test_input_224_foreground_dice"]),
            "test_input_224_pixel_accuracy": float(source["test_input_224_pixel_accuracy"]),
            "encoder_checkpoint_sha256": source["encoder_checkpoint_sha256"],
            "probe_checkpoint_sha256": source["probe_checkpoint_sha256"],
            "official_test_evaluations": int(source["official_test_evaluations"]),
            "scientific_result": source["scientific_result"] == "True",
        }
        if any(
            not math.isfinite(value)
            for key, value in row.items()
            if isinstance(value, float) and key != "selected_learning_rate"
        ):
            raise RuntimeError("probe CSV contains non-finite metrics")
        if row["official_test_evaluations"] != 1 or not row["scientific_result"]:
            raise RuntimeError("probe test-once/scientific flag failed")
        rows.append(row)

        checkpoint_path = raw_dir / _checkpoint_relative_path(source["probe_checkpoint_path"])
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != row["probe_checkpoint_sha256"]:
            raise RuntimeError("selected probe checkpoint file hash mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        probe = nn.Conv2d(192, 2, kernel_size=1, bias=True)
        incompatible = probe.load_state_dict(payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("selected probe strict load failed")
        if not _finite_state(payload["model"]):
            raise RuntimeError("selected probe has non-finite tensors")
        checkpoint_entries.append(
            {
                "kind": "selected_probe",
                "variant": row["variant"],
                "encoder_seed": row["encoder_seed"],
                "probe_seed": row["probe_seed"],
                "path_under_ignored_raw_root": checkpoint_path.relative_to(raw_dir).as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": checkpoint_hash,
                "model_state_sha256": state_dict_sha256(probe),
                "strict_load": True,
                "all_floating_tensors_finite": True,
            }
        )
        del probe, payload
    if {(row["variant"], row["encoder_seed"], row["probe_seed"]) for row in rows} != expected:
        raise RuntimeError("v2 selected probe matrix is incomplete")

    selection = _load_json(raw_dir / "probe/selection_complete_before_test.json")
    if (
        selection.get("status") != "complete"
        or selection.get("completed_probe_selections") != 45
        or selection.get("expected_probe_selections") != 45
        or selection.get("official_test_masks_accessed") is not False
        or not all(selection.get("selection_contracts", {}).values())
    ):
        raise RuntimeError("probe selections were not all finalized before official test")

    by_variant_encoder: dict[str, dict[int, float]] = defaultdict(dict)
    per_encoder_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for encoder_seed in ENCODER_SEEDS:
            values = [
                row["test_input_224_mean_iou"]
                for row in rows
                if row["variant"] == variant and row["encoder_seed"] == encoder_seed
            ]
            if len(values) != 5:
                raise RuntimeError("expected five probe seeds per encoder")
            mean = statistics.mean(values)
            by_variant_encoder[variant][encoder_seed] = mean
            per_encoder_rows.append(
                {
                    "variant": variant,
                    "encoder_seed": encoder_seed,
                    "probe_seeds": 5,
                    "test_input_224_mean_iou_mean": mean,
                    "test_input_224_mean_iou_probe_seed_sample_sd": statistics.stdev(values),
                }
            )

    reported = {item["variant"]: item for item in combined["probe_aggregates"]}
    aggregates: list[dict[str, Any]] = []
    for variant in VARIANTS:
        encoder_values = [by_variant_encoder[variant][seed] for seed in ENCODER_SEEDS]
        aggregate = {
            "variant": variant,
            "metric": "official_test_input_224_two_class_mean_iou",
            "encoder_seed_means": {
                str(seed): by_variant_encoder[variant][seed] for seed in ENCODER_SEEDS
            },
            "mean": statistics.mean(encoder_values),
            "sample_standard_deviation": statistics.stdev(encoder_values),
            "independent_n": 3,
            "probe_seeds_per_encoder": 5,
        }
        source = reported[variant]
        if not math.isclose(
            aggregate["mean"], source["mean_over_encoder_seed_means"], abs_tol=1e-15
        ) or not math.isclose(
            aggregate["sample_standard_deviation"],
            source["sample_standard_deviation_over_encoder_seed_means"],
            abs_tol=1e-15,
        ):
            raise RuntimeError("recomputed probe aggregate differs from H200 summary")
        aggregates.append(aggregate)

    summary_payload = {
        "schema_version": 1,
        "status": "complete_audited_superseded_v2",
        "primary_metric": "official_test_input_224_two_class_mean_iou",
        "independent_unit": "encoder_seed",
        "probe_seed_is_not_independent_replication": True,
        "independent_n": 3,
        "probe_seeds_per_encoder": 5,
        "aggregates": aggregates,
        "paired_contrasts": [
            {
                **_paired(by_variant_encoder, variant, "alg_warmup20"),
                "metric": "official_test_input_224_mean_iou",
            }
            for variant in ("ibkd_lambda_0.25", "ibkd_lambda_0.5")
        ],
        "all_45_selections_completed_before_official_test": True,
        "official_test_evaluations": 45,
    }
    return per_encoder_rows, summary_payload, checkpoint_entries


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
    if (
        combined.get("status") != "complete"
        or combined.get("scientific_result") is not True
        or combined.get("protocol_id")
        != "cub200_phase1_b128_frozen_spatial_probe_full_v2"
        or combined.get("config_sha256") != V2_CONFIG_SHA256
        or not all(combined.get("contracts", {}).values())
    ):
        raise RuntimeError("v2 guided combined result contract failed")
    config_path = repository_root / "phase1/phase1_cub/configs/cub200_b128_full_v2.json"
    if file_sha256(config_path) != V2_CONFIG_SHA256:
        raise RuntimeError("repository v2 config no longer matches the H200 result")

    sequence = _load_json(raw_dir / "sequence_status.json")
    if (
        sequence.get("status") != "complete"
        or sequence.get("classification_complete") != 9
        or sequence.get("probe_candidates_complete") != 135
        or sequence.get("probe_selections_complete") != 45
        or sequence.get("official_test_evaluations_complete") != 45
        or sequence.get("failure") is not None
    ):
        raise RuntimeError("v2 sequence did not complete every expected task")

    teacher_entry = _teacher_audit(raw_dir, combined)
    classification_rows, classification_summary, student_entries = (
        _classification_audit(raw_dir, combined)
    )
    per_encoder_rows, probe_summary, probe_entries = _probe_audit(raw_dir, combined)

    h200_manifest = _load_json(raw_dir / "checkpoint_manifest.json")
    if h200_manifest.get("count") != 55 or len(h200_manifest.get("entries", [])) != 55:
        raise RuntimeError("H200 checkpoint manifest is incomplete")
    audit_entries = [teacher_entry, *student_entries, *probe_entries]
    if len(audit_entries) != 55:
        raise RuntimeError("independent checkpoint audit did not cover 55 files")
    h200_hashes = {entry["sha256"] for entry in h200_manifest["entries"]}
    audited_hashes = {entry["checkpoint_sha256"] for entry in audit_entries}
    if h200_hashes != audited_hashes:
        raise RuntimeError("independent checkpoint hashes differ from H200 manifest")
    checkpoint_audit = {
        "schema_version": 1,
        "status": "pass",
        "safe_torch_load": "weights_only_true",
        "checkpoint_count": 55,
        "teacher_checkpoint_count": 1,
        "classification_encoder_checkpoint_count": 9,
        "selected_probe_checkpoint_count": 45,
        "all_file_hashes_match": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "entries": sorted(
            audit_entries,
            key=lambda item: (
                item["kind"],
                str(item.get("variant", "")),
                int(item.get("encoder_seed", 0)),
                int(item.get("probe_seed", 0)),
            ),
        ),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in TRACKED_SOURCE_ARTIFACTS.items():
        _copy(raw_dir / source_name, report_dir / destination_name)
    _save_json(manifest, report_dir / "source_manifest.json")
    _save_json(checkpoint_audit, report_dir / "checkpoint_audit.json")
    _save_json(classification_summary, report_dir / "classification_summary.json")
    _save_json(probe_summary, report_dir / "probe_summary.json")
    _write_csv(classification_rows, report_dir / "classification_per_seed.csv")
    _write_csv(per_encoder_rows, report_dir / "probe_per_encoder_seed.csv")

    class_aggregate = {
        item["variant"]: item for item in classification_summary["aggregates"]
    }
    probe_aggregate = {item["variant"]: item for item in probe_summary["aggregates"]}
    class_contrast = {
        item["contrast"]: item for item in classification_summary["paired_contrasts"]
    }
    probe_contrast = {
        item["contrast"]: item for item in probe_summary["paired_contrasts"]
    }
    teacher = teacher_entry
    result_markdown = f"""# CUB ResNet-56/32 v2 guided shard 결과

상태: **H200 issue 716 완료 · 독립 감사 통과 · 최종 v3 비교에서는 제외**

이 실행은 당시 잠근 v2 계약 아래에서는 완전한 scientific guided shard입니다.
다만 이후 최종 CUB 프로토콜을 ResNet-50/224 scratch Teacher v3로 교체했으므로,
이 결과를 v3의 Vanilla/KD/LG 또는 향후 ALG/iBKD 결과와 합치면 안 됩니다.

## 실행 완결성

- Teacher 1개, guided encoder 3방법 × 3 seed = 9개
- Probe LR 후보 135개, validation 선택 45개, official-test 평가 45/45
- Teacher/encoder/probe checkpoint 55개 모두 파일 hash, strict load, 유한값 감사 통과
- train/validation/test 5,394 / 600 / 5,794 및 validation split hash 일치
- 전체 실행시간: {combined['elapsed_seconds'] / 3600:.2f}시간

## Teacher

| 구조 | 선택 epoch | validation macro top-1 | test macro top-1 |
|---|---:|---:|---:|
| ResNet-56/32 scratch | {teacher['selected_epoch']} | {teacher['selected_validation']['macro_top1']:.3f}% | {teacher['official_test']['macro_top1']:.3f}% |

## Guided 분류 결과

| 방법 | test macro top-1 (3 encoder seeds, mean ± sample SD) | ALG-w20 대비 paired 차이 |
|---|---:|---:|
| ALG-w20 | {class_aggregate['alg_warmup20']['mean']:.3f} ± {class_aggregate['alg_warmup20']['sample_standard_deviation']:.3f}% | 기준 |
| iBKD λ=0.25 | {class_aggregate['ibkd_lambda_0.25']['mean']:.3f} ± {class_aggregate['ibkd_lambda_0.25']['sample_standard_deviation']:.3f}% | {class_contrast['ibkd_lambda_0.25_minus_alg_warmup20']['mean']:+.3f}%p |
| iBKD λ=0.5 | {class_aggregate['ibkd_lambda_0.5']['mean']:.3f} ± {class_aggregate['ibkd_lambda_0.5']['sample_standard_deviation']:.3f}% | {class_contrast['ibkd_lambda_0.5_minus_alg_warmup20']['mean']:+.3f}%p |

## Frozen probe 결과

독립 반복 단위는 encoder seed입니다. 각 encoder의 5개 probe seed 평균을 먼저 낸
뒤 3개 encoder seed의 mean ± sample SD를 계산했습니다.

| 방법 | test input-224 mIoU | ALG-w20 대비 matched-seed 차이 |
|---|---:|---:|
| ALG-w20 | {100 * probe_aggregate['alg_warmup20']['mean']:.3f} ± {100 * probe_aggregate['alg_warmup20']['sample_standard_deviation']:.3f}% | 기준 |
| iBKD λ=0.25 | {100 * probe_aggregate['ibkd_lambda_0.25']['mean']:.3f} ± {100 * probe_aggregate['ibkd_lambda_0.25']['sample_standard_deviation']:.3f}% | {100 * probe_contrast['ibkd_lambda_0.25_minus_alg_warmup20']['mean']:+.3f}%p |
| iBKD λ=0.5 | {100 * probe_aggregate['ibkd_lambda_0.5']['mean']:.3f} ± {100 * probe_aggregate['ibkd_lambda_0.5']['sample_standard_deviation']:.3f}% | {100 * probe_contrast['ibkd_lambda_0.5_minus_alg_warmup20']['mean']:+.3f}%p |

## 해석

구버전 v2 조건에서는 ALG-w20이 두 iBKD보다 probe mIoU가 약 0.34%p 높았습니다.
분류 정확도는 ALG-w20과 iBKD λ=0.25가 거의 같았지만, 공간 probe에서는 iBKD의
우위가 관찰되지 않았습니다. 다만 차이는 seed별로 완전히 일관되지는 않았고
(seed 2에서는 두 iBKD가 ALG-w20보다 근소하게 높음), 세 방법만 포함한 shard라
Vanilla/KD/LG까지 포함한 전체 순위도 알 수 없습니다.

가장 중요한 제한은 이것이 **ResNet-56/32 v2 결과**라는 점입니다. 현재 최종
프로토콜은 ResNet-50/224 v3이므로 이 수치는 참고·민감도 분석으로만 보존하고,
iBKD 핵심 주장에 대한 최종 판정은 v3의 6방법 × 3 seed 결과로 내려야 합니다.
"""
    (report_dir / "RESULTS.md").write_text(result_markdown, encoding="utf-8")
    print(
        "[CUB_R56_V2_GUIDED_CURATE_DONE] status=pass checkpoints=55 "
        f"alg_probe_miou={probe_aggregate['alg_warmup20']['mean']:.6f} "
        f"ibkd025_probe_miou={probe_aggregate['ibkd_lambda_0.25']['mean']:.6f} "
        f"ibkd05_probe_miou={probe_aggregate['ibkd_lambda_0.5']['mean']:.6f} "
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
