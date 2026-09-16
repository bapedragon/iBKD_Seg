#!/usr/bin/env python3
"""Run one controlled 300-epoch learned-all replay derived from issue 760."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .data import save_json
from .train_timing import file_sha256, format_duration


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/mechanism_analysis/configs/"
    "cub200_r50_224_b128_main_l0_ibkd_learned_all_replay_full_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "74a7d88e7cc23f428dbb82ac894699572fb6eed4970a2b8411aa04c9e30fae30"
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
EXPECTED_INITIAL_STUDENT_SHA256 = (
    "34e2122bcf137af67b7a52981980aa503446331c5e207cccfeb326a7140b684d"
)
EXPECTED_INITIAL_GUIDANCE_SHA256 = (
    "77ed222f04dd4355e8d36e0acc29e8e88c1584620af496b24b79d6c4e3d6f33f"
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
        raise RuntimeError("Mechanism replay full config SHA-256 changed")
    config = _load_json(path)
    smoke = config.get("smoke_gate", {})
    scope = config.get("scope", {})
    lineage = config.get("lineage", {})
    mechanism_info = lineage.get("issue760_mechanism_protocol", {})
    mechanism_path = _resolve_repository_path(str(mechanism_info.get("path", "")))
    base_info = lineage.get("base_main_l0_protocol", {})
    base_path = _resolve_repository_path(str(base_info.get("path", "")))
    identity = lineage.get("locked_initial_identity", {})
    dataset = config.get("dataset", {})
    teacher = config.get("teacher", {})
    student = config.get("student", {})
    trace = config.get("controlled_trace", {})
    test_policy = config.get("official_test_policy", {})
    execution = config.get("execution", {})
    checks = {
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_learned_all_replay_full_v1",
        "diagnostic_scope": config.get("scientific_result") is False
        and config.get("posthoc_mechanism_path_diagnostic") is True,
        "smoke_passed": smoke.get("source_h200_issue") == 767
        and smoke.get("status") == "passed"
        and smoke.get("execution_gates") == "11/11"
        and smoke.get("official_test_evaluations") == 0
        and smoke.get("frozen_probe_runs") == 0,
        "single_run": scope.get("source_h200_issue_under_diagnosis") == 760
        and scope.get("variants") == ["learned_all"]
        and scope.get("encoder_seeds") == [1]
        and scope.get("classification_runs") == 1
        and scope.get("classification_epochs") == 300
        and scope.get("frozen_probe") is False,
        "mechanism_lineage": mechanism_path.is_file()
        and mechanism_info.get("sha256") == EXPECTED_MECHANISM_PROTOCOL_SHA256
        and file_sha256(mechanism_path) == EXPECTED_MECHANISM_PROTOCOL_SHA256,
        "base_lineage": base_path.is_file()
        and base_info.get("sha256") == EXPECTED_BASE_SHA256
        and file_sha256(base_path) == EXPECTED_BASE_SHA256,
        "initial_identity": identity.get("student_state_sha256")
        == EXPECTED_INITIAL_STUDENT_SHA256
        and identity.get("guidance_state_sha256")
        == EXPECTED_INITIAL_GUIDANCE_SHA256,
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
        and student.get("epochs") == 300
        and student.get("guidance_controller_warmup_epochs") == 20,
        "controlled_trace": trace.get("torch_use_deterministic_algorithms") is True
        and trace.get("warn_only") is True
        and trace.get("formal_bitwise_determinism") is False
        and trace.get("scientific_model_path_changed") is False
        and trace.get("cudnn_benchmark") is False
        and trace.get("cudnn_deterministic") is True
        and trace.get("cuda_matmul_allow_tf32") is False
        and trace.get("cudnn_allow_tf32") is False
        and trace.get("float32_matmul_precision") == "highest"
        and trace.get("pythonhashseed") == 1
        and trace.get("cublas_workspace_config") == ":4096:8"
        and trace.get("omp_num_threads") == 1
        and trace.get("mkl_num_threads") == 1
        and trace.get("num_workers") == 0,
        "official_test": test_policy.get("selection_uses_official_test") is False
        and test_policy.get("total_evaluations") == 1
        and test_policy.get(
            "test_must_not_select_epoch_method_lambda_loader_or_followup"
        )
        is True,
        "execution": execution.get("requested_mig_slices") == 1
        and execution.get("ten_hour_limit_seconds") == 36000
        and execution.get("retain_selected_classification_checkpoint") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Invalid mechanism replay full contract: " + ", ".join(failures)
        )
    return config


def _validate_summary(
    summary: dict[str, Any], checkpoint_path: Path
) -> dict[str, bool]:
    history = summary.get("history", [])
    aggregation = summary.get("ibkd_aggregation", {})
    split = summary.get("split_manifest", {})
    controlled = summary.get("controlled_reproducibility", {})
    official = summary.get("official_test", {})
    metadata: dict[str, Any] = {}
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
    return {
        "run_complete": summary.get("status") == "complete",
        "diagnostic_identity": summary.get("scientific_result") is False
        and summary.get("confirmatory_main_result") is False
        and summary.get("canonical_phase1_result_replaced") is False
        and summary.get("mechanism_replay_full") is True
        and summary.get("posthoc_mechanism_path_diagnostic") is True
        and summary.get("controlled_aa_full") is False,
        "single_learned_all_path": summary.get("method") == "ibkd"
        and summary.get("fusion_ratio_lambda") == 0.25
        and summary.get("batch_size") == 128
        and summary.get("seed") == 1
        and summary.get("cub_loader_profile") == "l0_current_strong"
        and summary.get("ibkd_aggregation_mode") == "learned_all"
        and aggregation.get("mode") == "learned_all",
        "three_hundred_epochs": summary.get("epochs") == 300
        and len(history) == 300,
        "initial_identity": summary.get("initial_student_state_sha256")
        == EXPECTED_INITIAL_STUDENT_SHA256
        and summary.get("initial_guidance_state_sha256")
        == EXPECTED_INITIAL_GUIDANCE_SHA256,
        "teacher_identity": summary.get("teacher_checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and summary.get("teacher_model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "split_identity": split.get("validation_image_ids_sha256")
        == EXPECTED_VALIDATION_HASH,
        "validation_selection": summary.get("selection_metric")
        in {None, "validation_macro_top1"}
        and isinstance(summary.get("selected_epoch"), int)
        and 1 <= summary.get("selected_epoch", 0) <= 300
        and isinstance(summary.get("selected_validation"), dict),
        "one_post_selection_test": summary.get("official_test_evaluations") == 1
        and summary.get("official_test_accessed") is True
        and summary.get("official_test_policy") == "once_after_validation_selection"
        and summary.get("official_test_used_for_training_or_selection") is False
        and summary.get("selected_checkpoint_strict_reloaded") is True
        and isinstance(official.get("macro_top1"), (int, float)),
        "all_epoch_hashes": len(history) == 300
        and all(
            row.get("input_stream_sha256")
            and row.get("rng_state_sha256_after_epoch")
            and row.get("student_state_sha256_after_epoch")
            and row.get("guidance_state_sha256_after_epoch")
            for row in history
        ),
        "controlled_trace_disclosed": controlled.get("enabled") is True
        and controlled.get("formal_bitwise_determinism") is False
        and controlled.get("scientific_path_preserved") is True,
        "checkpoint_integrity": checkpoint_path.is_file()
        and summary.get("checkpoint_sha256") == file_sha256(checkpoint_path)
        and bool(summary.get("student_state_sha256"))
        and bool(summary.get("guidance_state_sha256")),
        "checkpoint_metadata": metadata.get("purpose")
        == "phase1_cub_main_l0_ibkd_learned_all_replay_full_v1"
        and metadata.get("mechanism_replay_full") is True
        and metadata.get("official_test_evaluations_at_checkpoint_write") == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.config = args.config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    config = validate_config(args.config)
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    if file_sha256(args.teacher_checkpoint) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("Audited issue-722 teacher checkpoint SHA-256 changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "student"
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse mechanism replay full: {run_dir}")
    environment = dict(os.environ)
    environment.update(CONTROLLED_ENVIRONMENT)
    command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_full",
        "--full-run",
        "--mechanism-replay-full",
        "--dataset",
        "cub",
        "--kind",
        "student",
        "--method",
        "ibkd",
        "--fusion-ratio",
        "0.25",
        "--ibkd-aggregation-mode",
        "learned_all",
        "--batch-size",
        "128",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--run-name",
        "student",
        "--teacher-checkpoint",
        str(args.teacher_checkpoint),
        "--teacher-architecture",
        "resnet50_224_scratch",
        "--scientific-cub-r50-teacher",
        "--cub-loader-profile",
        "l0_current_strong",
        "--protocol-config",
        str(args.config),
        "--batch-profile-role",
        "posthoc_issue760_learned_all_replay",
        "--eval-batch-size",
        "200",
        "--num-workers",
        "0",
        "--seed",
        "1",
    ]
    started = time.perf_counter()
    save_json(
        {
            "status": "running",
            "phase": "issue760_learned_all_single_replay_full",
            "classification_runs_complete": 0,
            "classification_runs_expected": 1,
            "official_test_evaluations": 0,
            "frozen_probe_runs": 0,
            "scientific_result": False,
        },
        args.output_dir / "completion_status.json",
    )
    log("[MECHANISM_REPLAY_FULL_START] variant=learned_all seed=1 epochs=300")
    subprocess.run(command, check=True, env=environment, cwd=REPOSITORY_ROOT)
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "student_best_validation.pt"
    summary = _load_json(summary_path)
    checks = _validate_summary(summary, checkpoint_path)
    failures = [name for name, passed in checks.items() if not passed]
    test_macro = float(summary["official_test"]["macro_top1"])
    prior_differences = [
        {
            "run": row["run"],
            "prior_test_macro_top1_percent": row["test_macro_top1_percent"],
            "replay_minus_prior_percentage_points": (
                test_macro - float(row["test_macro_top1_percent"])
            ),
        }
        for row in config["lineage"]["prior_seed1_observations"]
    ]
    elapsed = time.perf_counter() - started
    result = {
        "status": "complete" if not failures else "failed",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "config_sha256": file_sha256(args.config),
        "scientific_result": False,
        "posthoc_mechanism_path_diagnostic": True,
        "source_h200_issue_under_diagnosis": 760,
        "aggregation_mode": "learned_all",
        "encoder_seed": 1,
        "checks": checks,
        "failures": failures,
        "selected_epoch": summary["selected_epoch"],
        "controller_stop_epoch": (summary.get("controller_final") or {}).get(
            "stop_epoch"
        ),
        "selected_validation": summary["selected_validation"],
        "official_test": summary["official_test"],
        "official_test_evaluations": 1,
        "frozen_probe_runs": 0,
        "prior_observation_differences": prior_differences,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "student_state_sha256": summary["student_state_sha256"],
        "guidance_state_sha256": summary["guidance_state_sha256"],
        "elapsed_seconds": elapsed,
        "numerical_pass_threshold": None,
        "canonical_main_result_replaced": False,
    }
    save_json(result, args.output_dir / "full_summary.json")
    save_json(
        {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "scientific_result": False,
            "checkpoints": [
                {
                    "kind": "classification_encoder",
                    "aggregation_mode": "learned_all",
                    "encoder_seed": 1,
                    "path": str(checkpoint_path),
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "student_state_sha256": summary["student_state_sha256"],
                    "guidance_state_sha256": summary["guidance_state_sha256"],
                    "selected_epoch": summary["selected_epoch"],
                    "official_test": summary["official_test"],
                }
            ],
        },
        args.output_dir / "checkpoint_manifest.json",
    )
    save_json(
        {
            "status": result["status"],
            "phase": "issue760_learned_all_single_replay_full",
            "classification_runs_complete": int(not failures),
            "classification_runs_expected": 1,
            "execution_control_gates_passed": not failures,
            "official_test_evaluations": 1,
            "frozen_probe_runs": 0,
            "scientific_result": False,
            "elapsed_seconds": elapsed,
        },
        args.output_dir / "completion_status.json",
    )
    log(
        "[MECHANISM_REPLAY_FULL_RESULT] "
        f"status={result['status'].upper()} gates="
        f"{sum(checks.values())}/{len(checks)} variant=learned_all seed=1 "
        f"selected_epoch={summary['selected_epoch']} controller_stop_epoch="
        f"{result['controller_stop_epoch']} validation_macro="
        f"{summary['selected_validation']['macro_top1']:.6f} test_macro="
        f"{summary['official_test']['macro_top1']:.6f} test_overall="
        f"{summary['official_test']['overall_top1']:.6f} "
        "official_test_evaluations=1 frozen_probe_runs=0 "
        "numerical_pass_threshold=none canonical_main_result_replaced=false "
        f"elapsed={format_duration(elapsed)}"
    )
    if failures:
        raise RuntimeError(
            "Mechanism replay full gate failed: " + ", ".join(failures)
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
                "phase": "issue760_learned_all_single_replay_full",
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_result": False,
                "classification_runs_complete": 0,
                "classification_runs_expected": 1,
                "official_test_evaluations": 0,
                "frozen_probe_runs": 0,
            },
            args.output_dir / "completion_status.json",
        )
        log(
            "[MECHANISM_REPLAY_FULL_FAILED] "
            f"{type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()
