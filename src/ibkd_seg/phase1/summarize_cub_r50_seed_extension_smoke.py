#!/usr/bin/env python3
"""Validate and summarize the three CUB guided seed-extension smokes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_cub_combined_smoke import _atomic_json_save
from .run_cub_r50_guided_smoke import (
    EXPECTED_SCIENTIFIC_TEACHER_SHA256,
    EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
    EXPECTED_VARIANTS,
    SEED_EXTENSION_SMOKE_ID,
)
from .train_timing import format_duration


EXPECTED_PROFILES = ((128, 2), (128, 3), (64, 2))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _variant_map(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    mapped = {str(row.get("variant")): row for row in rows}
    if tuple(mapped) != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"{label} variants are incomplete or out of order: {tuple(mapped)}"
        )
    return mapped


def _validate_profile_summary(
    summary: dict[str, Any], *, batch_size: int, encoder_seed: int
) -> dict[str, Any]:
    label = f"batch{batch_size}/encoder_seed{encoder_seed}"
    classification = _variant_map(
        summary.get("classification", []), label=f"{label} classification"
    )
    probes = _variant_map(summary.get("frozen_probe", []), label=f"{label} probe")
    teacher = summary.get("teacher", {})
    checks = {
        "status": summary.get("status") == "pass",
        "contracts": summary.get("contracts", {}).get("all_passed") is True,
        "smoke_id": summary.get("smoke_id") == SEED_EXTENSION_SMOKE_ID,
        "non_scientific": summary.get("scientific_result") is False,
        "seed_extension_mode": summary.get("seed_extension_mode") is True,
        "batch_profile_mode": summary.get("batch_profile_mode") is False,
        "batch": summary.get("student_batch_size") == batch_size,
        "encoder_seed": summary.get("encoder_seed") == encoder_seed,
        "capacity_batch": summary.get("capacity", {}).get("requested_batch_size")
        == batch_size,
        "teacher_checkpoint": teacher.get("checkpoint_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_SHA256,
        "teacher_state": teacher.get("model_state_sha256")
        == EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
        "teacher_reused": teacher.get("reused") is True
        and teacher.get("trained_in_smoke") is False,
        "four_classification": len(classification) == 4,
        "four_probes": len(probes) == 4,
        "classification_profile": all(
            row.get("summary", {}).get("batch_size") == batch_size
            and row.get("summary", {}).get("seed") == encoder_seed
            and row.get("summary", {}).get("official_test_accessed") is True
            for row in classification.values()
        ),
        "probe_profile": all(
            row.get("encoder_seed") == encoder_seed
            and row.get("official_test_evaluations") == 1
            for row in probes.values()
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"{label} smoke contract failed: " + ", ".join(failures))
    return {
        "summary": summary,
        "classification": classification,
        "probes": probes,
    }


def run(root: Path) -> dict[str, Any]:
    profiles: dict[tuple[int, int], dict[str, Any]] = {}
    for batch_size, encoder_seed in EXPECTED_PROFILES:
        profile_dir = root / f"batch{batch_size}_seed{encoder_seed}"
        path = profile_dir / "combined_smoke_summary.json"
        if not path.is_file():
            raise RuntimeError(f"missing seed-extension smoke summary: {path}")
        profiles[(batch_size, encoder_seed)] = _validate_profile_summary(
            _load_json(path),
            batch_size=batch_size,
            encoder_seed=encoder_seed,
        )

    teacher_hashes = {
        row["summary"]["teacher_checkpoint_sha256"]
        for profile in profiles.values()
        for row in profile["classification"].values()
    }
    if teacher_hashes != {EXPECTED_SCIENTIFIC_TEACHER_SHA256}:
        raise RuntimeError("students did not share the audited issue-722 teacher")

    initial_hashes_by_seed: dict[int, set[str]] = {2: set(), 3: set()}
    for (_, encoder_seed), profile in profiles.items():
        initial_hashes_by_seed[encoder_seed].update(
            row["summary"]["initial_student_state_sha256"]
            for row in profile["classification"].values()
        )
    if any(len(values) != 1 for values in initial_hashes_by_seed.values()):
        raise RuntimeError("student initialization differs within an encoder seed")
    if initial_hashes_by_seed[2] == initial_hashes_by_seed[3]:
        raise RuntimeError("encoder seeds 2 and 3 produced the same initial state")

    profile_rows: list[dict[str, Any]] = []
    for batch_size, encoder_seed in EXPECTED_PROFILES:
        profile = profiles[(batch_size, encoder_seed)]
        summary = profile["summary"]
        classification_rows: list[dict[str, Any]] = []
        probe_rows: list[dict[str, Any]] = []
        for variant in EXPECTED_VARIANTS:
            classification = profile["classification"][variant]
            classification_summary = classification["summary"]
            final_epoch = classification_summary["epochs"][-1]
            probe = profile["probes"][variant]
            classification_rows.append(
                {
                    "variant": variant,
                    "avg_epoch_seconds": classification_summary["avg_epoch_seconds"],
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
                    "validation_input_224_mean_iou": probe["validation"]["input_224"][
                        "mean_iou"
                    ],
                    "official_test_input_224_mean_iou": probe["official_test"][
                        "input_224"
                    ]["mean_iou"],
                }
            )
        profile_rows.append(
            {
                "student_batch_size": batch_size,
                "encoder_seed": encoder_seed,
                "role": (
                    "locked_v3_confirmatory_continuation"
                    if batch_size == 128
                    else "posthoc_exploratory_batch_sensitivity"
                ),
                "smoke_suite_seconds": summary["timing"]["smoke_suite_seconds"],
                "classification": classification_rows,
                "frozen_probe": probe_rows,
                "rough_one_encoder_seed_four_variant_estimate": summary["timing"][
                    "rough_one_encoder_seed_four_variant_estimate"
                ],
            }
        )

    batch128_estimate = sum(
        float(row["rough_one_encoder_seed_four_variant_estimate"]["total_seconds"])
        for row in profile_rows
        if row["student_batch_size"] == 128
    )
    batch64_estimate = sum(
        float(row["rough_one_encoder_seed_four_variant_estimate"]["total_seconds"])
        for row in profile_rows
        if row["student_batch_size"] == 64
    )
    output = {
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_id": SEED_EXTENSION_SMOKE_ID,
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "teacher": {
            "source_h200_issue": 722,
            "checkpoint_sha256": EXPECTED_SCIENTIFIC_TEACHER_SHA256,
            "model_state_sha256": EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
            "reused_by_all_profiles": True,
        },
        "paired_initialization": {
            "same_within_each_encoder_seed": True,
            "same_seed2_across_batches": True,
            "different_between_seed2_and_seed3": True,
            "state_sha256_by_encoder_seed": {
                str(seed): next(iter(values))
                for seed, values in initial_hashes_by_seed.items()
            },
        },
        "profiles": profile_rows,
        "rough_recommended_full_job_estimates": {
            "batch128_encoder_seeds_2_3_seconds": batch128_estimate,
            "batch64_encoder_seed_2_seconds": batch64_estimate,
            "combined_seconds": batch128_estimate + batch64_estimate,
            "warning": "linear_extrapolation_for_job_partitioning_only",
        },
        "warning": "all_smoke_metrics_are_non_scientific_and_must_not_select_a_method_lambda_batch_or_seed",
    }
    output_path = root / "seed_extension_smoke_summary.json"
    _atomic_json_save(output, output_path)

    print("[CUB_R50_SEED_EXTENSION_FINAL_CLASSIFICATION_RESULTS]", flush=True)
    for profile in profile_rows:
        for row in profile["classification"]:
            print(
                "[CUB_R50_SEED_EXTENSION_CLASSIFICATION_RESULT] "
                f"batch={profile['student_batch_size']} "
                f"encoder_seed={profile['encoder_seed']} variant={row['variant']} "
                f"avg_epoch_seconds={row['avg_epoch_seconds']:.3f} "
                f"peak_cuda_bytes={row['peak_cuda_memory_bytes']} "
                f"peak_cuda_reserved_bytes={row['peak_cuda_memory_reserved_bytes']} "
                "scientific_result=false",
                flush=True,
            )
    print("[CUB_R50_SEED_EXTENSION_FINAL_PROBE_RESULTS]", flush=True)
    for profile in profile_rows:
        for row in profile["frozen_probe"]:
            print(
                "[CUB_R50_SEED_EXTENSION_PROBE_RESULT] "
                f"encoder_batch={profile['student_batch_size']} "
                f"encoder_seed={profile['encoder_seed']} variant={row['variant']} "
                f"selected_lr={row['selected_learning_rate']:g} "
                f"selected_epoch={row['selected_epoch']} "
                f"validation_input_miou={row['validation_input_224_mean_iou']:.6f} "
                f"official_test_input_miou={row['official_test_input_224_mean_iou']:.6f} "
                "scientific_result=false",
                flush=True,
            )
    print(
        "[CUB_R50_SEED_EXTENSION_ESTIMATE] "
        f"batch128_seeds2_3={format_duration(batch128_estimate)} "
        f"batch64_seed2={format_duration(batch64_estimate)} "
        f"combined={format_duration(batch128_estimate + batch64_estimate)} "
        "teacher_reused=true linear_extrapolation_only=true",
        flush=True,
    )
    print(
        "[CUB_R50_SEED_EXTENSION_SMOKE_DONE] status=pass profiles=3/3 "
        "classification=12/12 probe_candidates=36/36 selected_probes=12/12 "
        "gpu_tasks=48/48 teacher_reused=true scientific_result=false "
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
