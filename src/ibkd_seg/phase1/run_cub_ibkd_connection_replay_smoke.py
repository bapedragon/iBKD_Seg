#!/usr/bin/env python3
"""Smoke the single learned-all replay path derived from issue 760."""

from __future__ import annotations

import argparse
import json
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
    "cub200_r50_224_b128_main_l0_ibkd_learned_all_replay_smoke_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "7253617d873093b20f28394be0fe9e85e953448f79151e15e12053c8d0538b15"
)
EXPECTED_MECHANISM_PROTOCOL_SHA256 = (
    "1650e76da40c235292fca166b8058c1a9ee279165a275b59122f505f3b7235d8"
)
EXPECTED_BASE_SHA256 = (
    "e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc"
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


def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def validate_config(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Mechanism replay smoke config SHA-256 changed")
    config = _load_json(path)
    lineage = config.get("lineage", {})
    mechanism_info = lineage.get("issue760_mechanism_protocol", {})
    mechanism_path = _resolve_repository_path(str(mechanism_info.get("path", "")))
    base_info = lineage.get("base_main_l0_protocol", {})
    base_path = _resolve_repository_path(str(base_info.get("path", "")))
    scope = config.get("scope", {})
    dataset = config.get("dataset", {})
    teacher = config.get("teacher", {})
    student = config.get("student", {})
    control = config.get("execution_control", {})
    checks = {
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_learned_all_replay_smoke_v1",
        "diagnostic_scope": config.get("scientific_result") is False
        and config.get("posthoc_mechanism_path_diagnostic") is True,
        "single_path": scope.get("source_h200_issue") == 760
        and scope.get("aggregation_variants") == ["learned_all"]
        and scope.get("encoder_seeds") == [1]
        and scope.get("classification_only") is True
        and scope.get("frozen_probe") is False
        and scope.get("official_test_accessed") is False
        and scope.get("actual_epochs") == 2
        and scope.get("planned_full_epochs") == 300,
        "mechanism_lineage": mechanism_path.is_file()
        and mechanism_info.get("sha256") == EXPECTED_MECHANISM_PROTOCOL_SHA256
        and file_sha256(mechanism_path) == EXPECTED_MECHANISM_PROTOCOL_SHA256,
        "base_lineage": base_path.is_file()
        and base_info.get("sha256") == EXPECTED_BASE_SHA256
        and file_sha256(base_path) == EXPECTED_BASE_SHA256,
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
        and student.get("method") == "ibkd"
        and student.get("aggregation_mode") == "learned_all"
        and student.get("fusion_ratio_lambda") == 0.25
        and student.get("batch_size") == 128
        and student.get("eval_batch_size") == 200
        and student.get("seed") == 1
        and student.get("precision") == "float32"
        and student.get("guidance_controller_warmup_epochs") == 20,
        "execution_control": control.get("independent_processes") == 1
        and control.get("torch_use_deterministic_algorithms") is True
        and control.get("warn_only") is True
        and control.get("formal_bitwise_determinism") is False
        and control.get("scientific_model_path_changed") is False
        and control.get("cudnn_benchmark") is False
        and control.get("cudnn_deterministic") is True
        and control.get("cuda_matmul_allow_tf32") is False
        and control.get("cudnn_allow_tf32") is False
        and control.get("float32_matmul_precision") == "highest"
        and control.get("pythonhashseed") == 1
        and control.get("cublas_workspace_config") == ":4096:8"
        and control.get("omp_num_threads") == 1
        and control.get("mkl_num_threads") == 1
        and control.get("num_workers") == 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Invalid mechanism replay smoke contract: " + ", ".join(failures)
        )
    return config


def _validate_summary(summary: dict[str, Any]) -> dict[str, bool]:
    epochs = summary.get("epochs", [])
    aggregation = summary.get("ibkd_aggregation", {})
    split = summary.get("split_manifest", {})
    controlled = summary.get("controlled_reproducibility", {})
    return {
        "run_complete": summary.get("status") == "complete",
        "single_learned_all_path": summary.get("method") == "ibkd"
        and summary.get("fusion_ratio_lambda") == 0.25
        and summary.get("batch_size") == 128
        and summary.get("seed") == 1
        and summary.get("cub_loader_profile") == "l0_current_strong"
        and summary.get("mechanism_replay_smoke") is True
        and summary.get("controlled_aa_smoke") is False
        and aggregation.get("mode") == "learned_all",
        "two_full_data_epochs": summary.get("actual_epochs") == 2
        and summary.get("planned_epochs") == 300
        and len(epochs) == 2,
        "test_and_probe_sealed": summary.get("official_test_accessed") is False
        and summary.get("official_test") is None,
        "split_identity": split.get("validation_image_ids_sha256")
        == EXPECTED_VALIDATION_HASH,
        "teacher_identity": summary.get("teacher_checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and summary.get("teacher_model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "initial_state_hashes": bool(summary.get("initial_student_state_sha256"))
        and bool(summary.get("initial_guidance_state_sha256")),
        "epoch_trace_hashes": len(epochs) == 2
        and all(
            row.get("input_stream_sha256")
            and row.get("rng_state_sha256_after_epoch")
            and row.get("student_state_sha256_after_epoch")
            and row.get("guidance_state_sha256_after_epoch")
            for row in epochs
        ),
        "memory_recorded": len(epochs) == 2
        and all(
            isinstance(row.get("peak_cuda_memory_bytes"), int)
            and row.get("peak_cuda_memory_bytes", 0) > 0
            and isinstance(row.get("peak_cuda_memory_reserved_bytes"), int)
            and row.get("peak_cuda_memory_reserved_bytes", 0) > 0
            for row in epochs
        ),
        "checkpoint_written": bool(summary.get("checkpoint_sha256"))
        and bool(summary.get("student_state_sha256"))
        and bool(summary.get("guidance_state_sha256")),
        "controlled_trace_disclosed": controlled.get("enabled") is True
        and controlled.get("formal_bitwise_determinism") is False
        and controlled.get("scientific_path_preserved") is True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = validate_config(args.config.resolve())
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "student"
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse mechanism replay smoke: {run_dir}")
    environment = dict(os.environ)
    environment.update(CONTROLLED_ENVIRONMENT)
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
        "--mechanism-replay-smoke",
        "--batch-size",
        "128",
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--run-name",
        "student",
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
    started = time.perf_counter()
    log("[MECHANISM_REPLAY_SMOKE_START] variant=learned_all seed=1 epochs=2")
    subprocess.run(command, check=True, env=environment, cwd=REPOSITORY_ROOT)
    summary = _load_json(run_dir / "summary.json")
    checks = _validate_summary(summary)
    failures = [name for name, passed in checks.items() if not passed]
    peak_allocated = max(
        int(row["peak_cuda_memory_bytes"]) for row in summary["epochs"]
    )
    peak_reserved = max(
        int(row["peak_cuda_memory_reserved_bytes"]) for row in summary["epochs"]
    )
    result = {
        "status": "pass" if not failures else "fail",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "config_sha256": file_sha256(args.config.resolve()),
        "scientific_result": False,
        "source_h200_issue": 760,
        "aggregation_mode": "learned_all",
        "encoder_seed": 1,
        "checks": checks,
        "failures": failures,
        "official_test_evaluations": 0,
        "frozen_probe_runs": 0,
        "avg_epoch_seconds": summary["avg_epoch_seconds"],
        "estimated_300_epoch_seconds": summary["estimated_planned_seconds"],
        "peak_cuda_memory_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "initial_student_state_sha256": summary[
            "initial_student_state_sha256"
        ],
        "initial_guidance_state_sha256": summary[
            "initial_guidance_state_sha256"
        ],
        "teacher_checkpoint_sha256": summary["teacher_checkpoint_sha256"],
        "teacher_model_state_sha256": summary["teacher_model_state_sha256"],
        "validation_image_ids_sha256": summary["split_manifest"][
            "validation_image_ids_sha256"
        ],
        "epoch_input_stream_sha256": [
            row["input_stream_sha256"] for row in summary["epochs"]
        ],
        "epoch_rng_state_sha256": [
            row["rng_state_sha256_after_epoch"] for row in summary["epochs"]
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "decision_note": (
            "A smoke pass validates only execution, identity, tracing, memory, and "
            "runtime. It does not establish that the prior 10.41% result is or is "
            "not reproducible."
        ),
    }
    save_json(result, args.output_dir / "smoke_summary.json")
    save_json(
        {
            "status": result["status"],
            "phase": "issue760_learned_all_single_path_smoke",
            "classification_runs_complete": int(not failures),
            "classification_runs_expected": 1,
            "actual_epochs": 2,
            "official_test_evaluations": 0,
            "frozen_probe_runs": 0,
            "scientific_result": False,
        },
        args.output_dir / "completion_status.json",
    )
    log(
        "[MECHANISM_REPLAY_SMOKE_RESULT] "
        f"status={result['status'].upper()} gates="
        f"{sum(checks.values())}/{len(checks)} variant=learned_all seed=1 "
        f"peak_cuda_bytes={peak_allocated} peak_cuda_reserved_bytes={peak_reserved} "
        f"estimated_300ep={format_duration(result['estimated_300_epoch_seconds'])} "
        "official_test_evaluations=0 frozen_probe_runs=0 scientific_result=false"
    )
    if failures:
        raise RuntimeError(
            "Mechanism replay smoke gate failed: " + ", ".join(failures)
        )
    return result


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "status": "failed",
                "phase": "issue760_learned_all_single_path_smoke",
                "error_type": type(error).__name__,
                "error": str(error),
                "official_test_evaluations": 0,
                "frozen_probe_runs": 0,
                "scientific_result": False,
            },
            args.output_dir / "completion_status.json",
        )
        log(
            "[MECHANISM_REPLAY_SMOKE_FAILED] "
            f"{type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()
