#!/usr/bin/env python3
"""Validate and summarize the two CUB ResNet-50/224 batch-profile smokes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_cub_combined_smoke import _atomic_json_save
from .run_cub_r50_guided_smoke import (
    BATCH_PROFILE_SMOKE_ID,
    EXPECTED_SCIENTIFIC_TEACHER_SHA256,
    EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
    EXPECTED_VARIANTS,
)
from .train_timing import format_duration


EXPECTED_BATCHES = (128, 64)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _variant_map(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    mapped = {str(row.get("variant")): row for row in rows}
    if tuple(mapped) != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"{label} variants are incomplete or out of order: {tuple(mapped)}"
        )
    return mapped


def _validate_batch_summary(
    summary: dict[str, Any], *, expected_batch: int
) -> dict[str, Any]:
    classification = _variant_map(
        summary.get("classification", []), label=f"batch{expected_batch} classification"
    )
    probes = _variant_map(
        summary.get("frozen_probe", []), label=f"batch{expected_batch} probe"
    )
    teacher = summary.get("teacher", {})
    checks = {
        "status": summary.get("status") == "pass",
        "contracts": summary.get("contracts", {}).get("all_passed") is True,
        "smoke_id": summary.get("smoke_id") == BATCH_PROFILE_SMOKE_ID,
        "non_scientific": summary.get("scientific_result") is False,
        "batch_profile_mode": summary.get("batch_profile_mode") is True,
        "student_batch": summary.get("student_batch_size") == expected_batch,
        "capacity_batch": summary.get("capacity", {}).get("requested_batch_size")
        == expected_batch,
        "teacher_checkpoint": teacher.get("checkpoint_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_SHA256,
        "teacher_state": teacher.get("model_state_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
        "teacher_reused": teacher.get("reused") is True
        and teacher.get("trained_in_smoke") is False,
        "four_classification": len(classification) == 4,
        "four_probes": len(probes) == 4,
        "classification_batches": all(
            row.get("summary", {}).get("batch_size") == expected_batch
            for row in classification.values()
        ),
        "test_once_classification": all(
            row.get("summary", {}).get("official_test_accessed") is True
            for row in classification.values()
        ),
        "test_once_probe": all(
            row.get("official_test_evaluations") == 1 for row in probes.values()
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            f"batch{expected_batch} smoke summary contract failed: "
            + ", ".join(failures)
        )
    return {
        "summary": summary,
        "classification": classification,
        "probes": probes,
    }


def run(root: Path) -> dict[str, Any]:
    batches: dict[int, dict[str, Any]] = {}
    for batch_size in EXPECTED_BATCHES:
        path = root / f"batch{batch_size}" / "combined_smoke_summary.json"
        if not path.is_file():
            raise RuntimeError(f"missing batch-profile smoke summary: {path}")
        batches[batch_size] = _validate_batch_summary(
            _load_json(path), expected_batch=batch_size
        )

    initial_hashes = {
        row["summary"]["initial_student_state_sha256"]
        for batch in batches.values()
        for row in batch["classification"].values()
    }
    teacher_hashes = {
        row["summary"]["teacher_checkpoint_sha256"]
        for batch in batches.values()
        for row in batch["classification"].values()
    }
    if len(initial_hashes) != 1:
        raise RuntimeError("student initialization differs across variants or batches")
    if teacher_hashes != {EXPECTED_SCIENTIFIC_TEACHER_SHA256}:
        raise RuntimeError("students did not share the audited issue-722 teacher")

    batch_rows: list[dict[str, Any]] = []
    for batch_size in EXPECTED_BATCHES:
        summary = batches[batch_size]["summary"]
        estimate = summary["timing"][
            "rough_one_encoder_seed_four_variant_estimate"
        ]
        classification_rows: list[dict[str, Any]] = []
        probe_rows: list[dict[str, Any]] = []
        for variant in EXPECTED_VARIANTS:
            classification = batches[batch_size]["classification"][variant]
            classification_summary = classification["summary"]
            final_epoch = classification_summary["epochs"][-1]
            probe = batches[batch_size]["probes"][variant]
            classification_rows.append(
                {
                    "variant": variant,
                    "avg_epoch_seconds": classification_summary[
                        "avg_epoch_seconds"
                    ],
                    "epoch2_validation_macro_top1": final_epoch["validation"][
                        "macro_top1"
                    ],
                    "epoch2_official_test_macro_top1": classification_summary[
                        "official_test"
                    ]["macro_top1"],
                    "peak_cuda_memory_bytes": summary["capacity"][
                        "peak_cuda_memory_bytes_by_student"
                    ][variant],
                    "peak_cuda_memory_reserved_bytes": summary["capacity"][
                        "peak_cuda_memory_reserved_bytes_by_student"
                    ][variant],
                }
            )
            probe_rows.append(
                {
                    "variant": variant,
                    "selected_learning_rate": probe["selection"]["learning_rate"],
                    "selected_epoch": probe["selection"]["epoch"],
                    "validation_input_224_mean_iou": probe["validation"][
                        "input_224"
                    ]["mean_iou"],
                    "official_test_input_224_mean_iou": probe["official_test"][
                        "input_224"
                    ]["mean_iou"],
                    "feature_cache_seconds": probe["timing"][
                        "feature_cache_seconds"
                    ],
                    "probe_training_and_validation_seconds": probe["timing"][
                        "probe_training_and_validation_seconds"
                    ],
                    "official_test_evaluation_seconds": probe["timing"][
                        "official_test_evaluation_seconds"
                    ],
                }
            )
        batch_rows.append(
            {
                "student_batch_size": batch_size,
                "smoke_suite_seconds": summary["timing"]["smoke_suite_seconds"],
                "classification": classification_rows,
                "frozen_probe": probe_rows,
                "rough_one_encoder_seed_four_variant_estimate": estimate,
            }
        )

    combined_estimate_seconds = sum(
        float(row["rough_one_encoder_seed_four_variant_estimate"]["total_seconds"])
        for row in batch_rows
    )
    output = {
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_id": BATCH_PROFILE_SMOKE_ID,
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "teacher": {
            "source_h200_issue": 722,
            "checkpoint_sha256": EXPECTED_SCIENTIFIC_TEACHER_SHA256,
            "model_state_sha256": EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
            "reused_by_both_batches": True,
        },
        "paired_initialization": {
            "same_across_all_eight_students": True,
            "student_state_sha256": next(iter(initial_hashes)),
        },
        "batches": batch_rows,
        "rough_combined_one_encoder_seed_four_variant_estimate_seconds": (
            combined_estimate_seconds
        ),
        "warning": "all_smoke_metrics_and_linear_time_estimates_are_non_scientific",
    }
    output_path = root / "batch_profile_smoke_summary.json"
    _atomic_json_save(output, output_path)

    print("[CUB_R50_BATCH_PROFILE_FINAL_CLASSIFICATION_RESULTS]", flush=True)
    for batch in batch_rows:
        batch_size = batch["student_batch_size"]
        for row in batch["classification"]:
            print(
                "[CUB_R50_BATCH_PROFILE_CLASSIFICATION_RESULT] "
                f"batch={batch_size} variant={row['variant']} "
                f"avg_epoch_seconds={row['avg_epoch_seconds']:.3f} "
                f"peak_cuda_bytes={row['peak_cuda_memory_bytes']} "
                f"peak_cuda_reserved_bytes={row['peak_cuda_memory_reserved_bytes']} "
                "scientific_result=false",
                flush=True,
            )
    print("[CUB_R50_BATCH_PROFILE_FINAL_PROBE_RESULTS]", flush=True)
    for batch in batch_rows:
        batch_size = batch["student_batch_size"]
        for row in batch["frozen_probe"]:
            print(
                "[CUB_R50_BATCH_PROFILE_PROBE_RESULT] "
                f"encoder_batch={batch_size} variant={row['variant']} "
                f"selected_lr={row['selected_learning_rate']:g} "
                f"selected_epoch={row['selected_epoch']} "
                f"validation_input_miou={row['validation_input_224_mean_iou']:.6f} "
                f"official_test_input_miou={row['official_test_input_224_mean_iou']:.6f} "
                "scientific_result=false",
                flush=True,
            )
    for batch in batch_rows:
        estimate = batch["rough_one_encoder_seed_four_variant_estimate"]
        print(
            "[CUB_R50_BATCH_PROFILE_ESTIMATE] "
            f"batch={batch['student_batch_size']} "
            f"four_variants_one_encoder_seed={format_duration(estimate['total_seconds'])} "
            "teacher_reused=true linear_extrapolation_only=true",
            flush=True,
        )
    print(
        "[CUB_R50_BATCH_PROFILE_COMBINED_ESTIMATE] "
        f"both_batches={format_duration(combined_estimate_seconds)} "
        "teacher_reused=true linear_extrapolation_only=true",
        flush=True,
    )
    print(
        "[CUB_R50_BATCH_PROFILE_SMOKE_DONE] status=pass batch_profiles=2/2 "
        "classification=8/8 probe_candidates=24/24 selected_probes=8/8 "
        "gpu_tasks=32/32 teacher_reused=true scientific_result=false "
        f"summary={output_path.resolve()}",
        flush=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args().root)


if __name__ == "__main__":
    main()
