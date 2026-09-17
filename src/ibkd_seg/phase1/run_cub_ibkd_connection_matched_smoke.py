#!/usr/bin/env python3
"""Controlled four-mode smoke for the matched-duration CUB connection ablation."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .data import save_json
from .train_timing import file_sha256, format_duration


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/mechanism_analysis/configs/"
    "cub200_r50_224_b128_main_l0_ibkd_connection_matched_smoke_v2.json"
)
EXPECTED_CONFIG_SHA256 = (
    "b0ec4af0f1eb350725fdc14d33be9ea09446dd9143cbe477accb3ae568fc897d"
)
EXPECTED_FULL_CONFIG_SHA256 = (
    "a2fadff0b93e1878b27cbd78e9249141b67202bcd6911ce27f4b19810ea3ca5b"
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
CONTROLLED_ENVIRONMENT = {
    "PYTHONHASHSEED": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NVIDIA_TF32_OVERRIDE": "0",
}


def log(message: str = "") -> None:
    print(message, flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_config(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("matched connection smoke config SHA-256 changed")
    config = _load_json(path)
    full_path = REPOSITORY_ROOT / str(config.get("full_protocol", ""))
    scope = config.get("scope", {})
    dataset = config.get("dataset", {})
    teacher = config.get("teacher", {})
    student = config.get("student", {})
    control = config.get("execution_control", {})
    checks = {
        "protocol": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_connection_matched_duration_smoke_v2",
        "non_scientific": config.get("scientific_result") is False,
        "scope": scope.get("classification_only") is True
        and scope.get("actual_epochs") == 2
        and scope.get("planned_full_epochs") == 300
        and scope.get("encoder_seed") == 1
        and tuple(scope.get("aggregation_variants", ())) == AGGREGATION_VARIANTS
        and scope.get("official_test_accessed") is False
        and scope.get("frozen_probe_runs") == 0,
        "dataset": dataset.get("name") == "CUB-200-2011"
        and dataset.get("train") == 5394
        and dataset.get("validation") == 600
        and dataset.get("official_test") == 5794
        and dataset.get("split_seed") == 2027
        and dataset.get("validation_image_ids_sha256") == EXPECTED_VALIDATION_HASH
        and dataset.get("loader_profile") == "l0_current_strong",
        "teacher": teacher.get("source_h200_issue") == 722
        and teacher.get("checkpoint_sha256") == EXPECTED_TEACHER_SHA256
        and teacher.get("model_state_sha256") == EXPECTED_TEACHER_STATE_SHA256,
        "student": student.get("architecture") == "deit_tiny_patch16_224"
        and student.get("batch_size") == 128
        and student.get("fusion_ratio_lambda") == 0.25
        and student.get("precision") == "float32"
        and student.get("fixed_guidance_epochs") == 123
        and student.get("smoke_beta_history_expected") == [2.5, 2.5],
        "control": control.get("independent_process_per_variant") is True
        and control.get("torch_use_deterministic_algorithms") is True
        and control.get("warn_only") is True
        and control.get("formal_bitwise_determinism") is False
        and control.get("scientific_model_path_changed") is False
        and control.get("num_workers") == 0,
        "full_protocol": full_path.is_file()
        and file_sha256(full_path) == EXPECTED_FULL_CONFIG_SHA256,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid matched connection smoke contract: " + ", ".join(failures)
        )
    return config


def _probability_contract(mode: str, aggregation: dict[str, Any]) -> bool:
    probabilities = aggregation.get("probabilities")
    if aggregation.get("mode") != mode or not isinstance(probabilities, list):
        return False
    if len(probabilities) != 3 or any(len(row) != 12 for row in probabilities):
        return False
    if any(not math.isclose(sum(row), 1.0, abs_tol=1e-6) for row in probabilities):
        return False
    if mode == "learned_all":
        logits = aggregation.get("raw_logits")
        return isinstance(logits, list) and len(logits) == 3
    if aggregation.get("raw_logits") is not None:
        return False
    if mode == "fixed_uniform_all":
        return all(
            math.isclose(value, 1.0 / 12.0, abs_tol=1e-7)
            for row in probabilities
            for value in row
        )
    expected = (
        (0, 6, 11) if mode == "fixed_stage_match" else (11, 11, 11)
    )
    return all(
        all(value == (1.0 if block == expected[stage] else 0.0)
            for block, value in enumerate(row))
        for stage, row in enumerate(probabilities)
    )


def _summary_checks(mode: str, summary: dict[str, Any], checkpoint: Path) -> dict[str, bool]:
    epochs = summary.get("epochs", [])
    controller = summary.get("controller", {})
    controlled = summary.get("controlled_reproducibility", {})
    return {
        "complete": summary.get("status") == "complete",
        "scope": summary.get("method") == "ibkd"
        and summary.get("fusion_ratio_lambda") == 0.25
        and summary.get("batch_size") == 128
        and summary.get("seed") == 1
        and summary.get("mechanism_ablation_smoke") is True
        and summary.get("ibkd_aggregation_mode") == mode
        and summary.get("ibkd_fixed_guidance_epochs") == 123,
        "two_epochs": summary.get("actual_epochs") == 2
        and summary.get("planned_epochs") == 300
        and len(epochs) == 2,
        "test_sealed": summary.get("official_test_accessed") is False
        and summary.get("official_test") is None,
        "teacher": summary.get("teacher_checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and summary.get("teacher_model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "split": summary.get("split_manifest", {}).get(
            "validation_image_ids_sha256"
        )
        == EXPECTED_VALIDATION_HASH,
        "fixed_schedule": controller.get("stop_policy") == "fixed_epoch"
        and controller.get("fixed_stop_epoch") == 123
        and controller.get("stop_epoch") is None
        and controller.get("active") is True
        and controller.get("beta_history") == [2.5, 2.5]
        and all(row.get("beta") == 2.5 for row in epochs),
        "trace": len(epochs) == 2
        and all(
            row.get("input_stream_sha256")
            and row.get("rng_state_sha256_after_epoch")
            and row.get("student_state_sha256_after_epoch")
            and row.get("guidance_state_sha256_after_epoch")
            for row in epochs
        ),
        "controlled": controlled.get("enabled") is True
        and controlled.get("formal_bitwise_determinism") is False
        and controlled.get("scientific_path_preserved") is True,
        "aggregation": _probability_contract(
            mode, summary.get("ibkd_aggregation", {})
        ),
        "checkpoint": checkpoint.is_file()
        and summary.get("checkpoint_sha256") == file_sha256(checkpoint),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = validate_config(args.config.resolve())
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classification_root = args.output_dir / "classification"
    if classification_root.exists():
        raise RuntimeError(
            f"refusing to reuse matched connection smoke output: {classification_root}"
        )
    environment = dict(os.environ)
    environment.update(CONTROLLED_ENVIRONMENT)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for mode in AGGREGATION_VARIANTS:
        run_dir = classification_root / mode
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
            "--ibkd-fixed-guidance-epochs",
            "123",
            "--batch-size",
            "128",
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output-dir",
            str(classification_root.resolve()),
            "--run-name",
            mode,
            "--teacher-checkpoint",
            str(args.teacher_checkpoint.resolve()),
            "--eval-batch-size",
            "200",
            "--num-workers",
            "0",
            "--seed",
            "1",
            "--save-student-checkpoint",
            "--cub-loader-profile",
            "l0_current_strong",
        ]
        log(f"[MATCHED_CONNECTION_SMOKE_START] mode={mode} epochs=2")
        subprocess.run(
            command,
            check=True,
            env=environment,
            cwd=REPOSITORY_ROOT,
        )
        summary = _load_json(run_dir / "summary.json")
        checkpoint = run_dir / "timing_student_latest.pt"
        checks = _summary_checks(mode, summary, checkpoint)
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise RuntimeError(
                f"matched connection smoke failed for {mode}: "
                + ", ".join(failures)
            )
        rows.append(
            {
                "mode": mode,
                "summary": summary,
                "checks": checks,
                "checkpoint": str(checkpoint.resolve()),
            }
        )
        log(f"[MATCHED_CONNECTION_SMOKE_DONE] mode={mode} gates={len(checks)}/{len(checks)}")

    student_hashes = {
        row["summary"]["initial_student_state_sha256"] for row in rows
    }
    teacher_hashes = {
        row["summary"]["teacher_checkpoint_sha256"] for row in rows
    }
    split_hashes = {
        row["summary"]["split_manifest"]["validation_image_ids_sha256"]
        for row in rows
    }
    input_streams = {
        tuple(epoch["input_stream_sha256"] for epoch in row["summary"]["epochs"])
        for row in rows
    }
    cross_checks = {
        "four_variants_complete": len(rows) == 4
        and tuple(row["mode"] for row in rows) == AGGREGATION_VARIANTS,
        "same_initial_student_state": len(student_hashes) == 1,
        "same_teacher": teacher_hashes == {EXPECTED_TEACHER_SHA256},
        "same_validation_split": split_hashes == {EXPECTED_VALIDATION_HASH},
        "same_augmented_input_stream": len(input_streams) == 1,
        "official_test_zero": all(
            row["summary"].get("official_test") is None for row in rows
        ),
    }
    failures = [name for name, passed in cross_checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "matched connection cross-variant gate failed: " + ", ".join(failures)
        )
    elapsed = time.perf_counter() - started
    result = {
        "status": "pass",
        "schema_version": 2,
        "protocol_id": config["protocol_id"],
        "config_sha256": file_sha256(args.config.resolve()),
        "full_protocol_sha256": EXPECTED_FULL_CONFIG_SHA256,
        "scientific_result": False,
        "aggregation_variants": list(AGGREGATION_VARIANTS),
        "fixed_guidance_epochs": 123,
        "classification_runs": rows,
        "cross_variant_checks": cross_checks,
        "paired_initial_student_state_sha256": next(iter(student_hashes)),
        "paired_epoch_input_stream_sha256": list(next(iter(input_streams))),
        "official_test_evaluations": 0,
        "frozen_probe_runs": 0,
        "elapsed_seconds": elapsed,
        "all_guidance_on_300_epoch_upper_bound_seconds_by_mode": {
            row["mode"]: row["summary"]["avg_epoch_seconds"] * 300.0
            for row in rows
        },
    }
    save_json(result, args.output_dir / "smoke_summary.json")
    save_json(
        {
            "status": "pass",
            "phase": "matched_connection_smoke_complete",
            "classification_runs_complete": 4,
            "classification_runs_expected": 4,
            "official_test_evaluations": 0,
            "frozen_probe_runs": 0,
            "scientific_result": False,
        },
        args.output_dir / "completion_status.json",
    )
    log(
        "[MATCHED_CONNECTION_SMOKE_COMPLETE] status=PASS variants=4/4 "
        f"cross_gates={len(cross_checks)}/{len(cross_checks)} "
        "fixed_guidance_epochs=123 official_test=0 frozen_probe=0 "
        f"elapsed={format_duration(elapsed)} scientific_result=false"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "status": "failed",
                "phase": "matched_connection_smoke_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "official_test_evaluations": 0,
                "frozen_probe_runs": 0,
                "scientific_result": False,
            },
            args.output_dir / "completion_status.json",
        )
        log(f"[MATCHED_CONNECTION_SMOKE_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
