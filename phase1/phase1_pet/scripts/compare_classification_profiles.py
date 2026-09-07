#!/usr/bin/env python3
"""Audit and compare the pre-registered Phase 1 batch 64/128 profiles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


VARIANT_ORDER = (
    "vanilla",
    "kd",
    "lg",
    "alg",
    "ibkd_lambda0.25",
    "ibkd_lambda0.5",
)


def load_summary(path: Path, expected_batch_size: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("batch_size") != expected_batch_size:
        raise RuntimeError(f"unexpected batch profile in {path}")
    if not payload.get("protocol_contracts", {}).get("all_passed"):
        raise RuntimeError(f"classification contracts did not pass in {path}")
    if payload.get("checkpoint_audit", {}).get("status") != "pass":
        raise RuntimeError(f"checkpoint audit did not pass in {path}")
    methods = {row["variant"]: row for row in payload.get("methods", [])}
    if set(methods) != set(VARIANT_ORDER):
        raise RuntimeError(f"classification method matrix is incomplete in {path}")
    return payload


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch64-summary", type=Path, required=True)
    parser.add_argument("--batch128-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    batch64 = load_summary(args.batch64_summary, 64)
    batch128 = load_summary(args.batch128_summary, 128)
    equality_checks = {
        "same_dataset": batch64["dataset"] == batch128["dataset"],
        "same_dataset_audit": batch64["dataset_audit"] == batch128["dataset_audit"],
        "same_teacher_model_state": batch64["teacher"]["model_state_sha256"]
        == batch128["teacher"]["model_state_sha256"],
        "same_teacher_test_metrics": {
            key: batch64["teacher"][key]
            for key in ("test_macro_top1", "test_overall_top1", "test_top5")
        }
        == {
            key: batch128["teacher"][key]
            for key in ("test_macro_top1", "test_overall_top1", "test_top5")
        },
        "same_validation_split": batch64["protocol_contracts"][
            "validation_split_hashes"
        ]
        == batch128["protocol_contracts"]["validation_split_hashes"],
        "same_student_initial_states": batch64["protocol_contracts"][
            "student_initial_state_hashes_by_seed"
        ]
        == batch128["protocol_contracts"]["student_initial_state_hashes_by_seed"],
        "test_not_used_for_selection_in_either_profile": not batch64[
            "protocol_contracts"
        ]["official_test_used_for_training_or_selection"]
        and not batch128["protocol_contracts"][
            "official_test_used_for_training_or_selection"
        ],
    }
    if not all(equality_checks.values()):
        failures = [name for name, passed in equality_checks.items() if not passed]
        raise RuntimeError("cross-profile contract mismatch: " + ", ".join(failures))

    methods64 = {row["variant"]: row for row in batch64["methods"]}
    methods128 = {row["variant"]: row for row in batch128["methods"]}
    rows: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        row64 = methods64[variant]
        row128 = methods128[variant]
        raw64 = [float(value) for value in row64["test_macro_top1"]["raw"]]
        raw128 = [float(value) for value in row128["test_macro_top1"]["raw"]]
        differences = [right - left for left, right in zip(raw64, raw128)]
        method = {
            "variant": variant,
            "batch64_test_macro_top1": row64["test_macro_top1"],
            "batch128_test_macro_top1": row128["test_macro_top1"],
            "batch128_minus_batch64_test_macro_top1": {
                "raw_by_seed": {
                    str(seed): value
                    for seed, value in zip((1, 2, 3), differences)
                },
                "mean": statistics.mean(differences),
                "sample_standard_deviation": statistics.stdev(differences),
            },
            "batch64_selected_epoch_by_seed": row64["selected_epoch_by_seed"],
            "batch128_selected_epoch_by_seed": row128["selected_epoch_by_seed"],
        }
        if "guidance_controller" in row64 or "guidance_controller" in row128:
            method["guidance_controller"] = {
                "batch64": row64.get("guidance_controller"),
                "batch128": row128.get("guidance_controller"),
            }
        methods.append(method)
        rows.append(
            {
                "variant": variant,
                "batch64_test_macro_top1_mean": row64["test_macro_top1"]["mean"],
                "batch64_test_macro_top1_sample_sd": row64["test_macro_top1"][
                    "sample_standard_deviation"
                ],
                "batch128_test_macro_top1_mean": row128["test_macro_top1"]["mean"],
                "batch128_test_macro_top1_sample_sd": row128["test_macro_top1"][
                    "sample_standard_deviation"
                ],
                "batch128_minus_batch64_mean": statistics.mean(differences),
                "batch128_minus_batch64_sample_sd": statistics.stdev(differences),
                "batch64_controller_stop_epoch_s1_s2_s3": "/".join(
                    str(value)
                    for value in row64.get("guidance_controller", {})
                    .get("stop_epoch_by_seed", {})
                    .values()
                ),
                "batch128_controller_stop_epoch_s1_s2_s3": "/".join(
                    str(value)
                    for value in row128.get("guidance_controller", {})
                    .get("stop_epoch_by_seed", {})
                    .values()
                ),
            }
        )

    report = {
        "schema_version": 1,
        "status": "pass",
        "dataset": batch64["dataset"],
        "profiles": [64, 128],
        "encoder_seeds": [1, 2, 3],
        "metric": "37_class_macro_top1_percentage",
        "cross_profile_contracts": equality_checks,
        "teacher_model_state_sha256": batch64["teacher"]["model_state_sha256"],
        "validation_split_sha256": batch64["protocol_contracts"][
            "validation_split_hashes"
        ][0],
        "ranking": {
            "batch64": batch64["classification_ranking_by_test_macro_top1"],
            "batch128": batch128["classification_ranking_by_test_macro_top1"],
        },
        "methods": methods,
        "interpretation_scope": {
            "both_pre_registered_profiles_must_be_reported": True,
            "posthoc_batch_selection_forbidden": True,
            "method_by_batch_interaction_observed": True,
            "classification_only_not_spatial_information_evidence": True,
            "batch64_probe_available": True,
            "batch128_probe_pending": True,
        },
    }
    save_json(report, args.output_json)
    write_csv(rows, args.output_csv)
    print(
        f"[PROFILE_COMPARISON_DONE] methods={len(methods)} "
        f"output={args.output_json}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
