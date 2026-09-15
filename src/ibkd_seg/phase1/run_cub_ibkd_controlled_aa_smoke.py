#!/usr/bin/env python3
"""Run two tightly controlled CUB main-L0 iBKD smoke jobs and compare them."""

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
    / "phase1/phase1_cub/reproducibility/configs/"
    "cub200_r50_224_b128_main_l0_ibkd_controlled_aa_smoke_v2.json"
)
EXPECTED_CONFIG_SHA256 = (
    "f16efcd07b96e16ecbb2e9ee6ab5a3c59efd0970ef4f4224df8f8a78d80c8ec2"
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
EXACT_SUMMARY_FIELDS = (
    "dataset",
    "kind",
    "method",
    "batch_size",
    "seed",
    "fusion_ratio_lambda",
    "actual_epochs",
    "planned_epochs",
    "initial_student_state_sha256",
    "initial_guidance_state_sha256",
    "teacher_checkpoint_sha256",
    "teacher_model_state_sha256",
    "teacher_checkpoint_kind",
    "guidance_controller_warmup_epochs",
    "optimizer_contract",
    "split_manifest",
    "controlled_reproducibility",
)
RUNTIME_FIELDS = (
    "python",
    "platform",
    "torch",
    "torchvision",
    "timm",
    "device",
    "gpu_name",
    "cuda",
    "git_commit",
)
EXACT_EPOCH_CONTROL_FIELDS = (
    "epoch",
    "lr",
    "beta",
    "input_stream_sha256",
    "rng_state_sha256_after_epoch",
)
IGNORED_NUMERIC_EPOCH_FIELDS = {
    "seconds_including_validation",
    "peak_cuda_memory_bytes",
    "peak_cuda_memory_reserved_bytes",
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
        raise RuntimeError("Controlled A/A smoke config SHA-256 changed")
    config = _load_json(path)
    base_info = config.get("lineage", {}).get("base_protocol", {})
    base_path = _resolve_repository_path(str(base_info.get("path", "")))
    controlled = config.get("controlled_reproducibility", {})
    student = config.get("student", {})
    teacher = config.get("teacher", {})
    dataset = config.get("dataset", {})
    checks = {
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_controlled_aa_smoke_v2",
        "non_scientific": config.get("scientific_result") is False
        and config.get("posthoc_reproducibility_audit") is True,
        "base": base_path.is_file()
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
        and student.get("seed") == 1
        and student.get("precision") == "float32"
        and student.get("smoke_epochs") == 2
        and student.get("planned_full_epochs") == 300,
        "controlled": controlled.get("independent_fresh_processes") == 2
        and controlled.get("run_labels") == ["A", "B"]
        and controlled.get("torch_use_deterministic_algorithms") is True
        and controlled.get("warn_only") is True
        and controlled.get("formal_bitwise_determinism") is False
        and [
            row.get("kernel")
            for row in controlled.get("observed_nondeterministic_operations", [])
        ]
        == [
            "compute_grad_input",
            "adaptive_max_pool2d_backward_cuda",
            "memory_efficient_attention_backward_cuda",
        ]
        and controlled.get("scientific_model_path_changed") is False
        and controlled.get("cudnn_benchmark") is False
        and controlled.get("cudnn_deterministic") is True
        and controlled.get("cuda_matmul_allow_tf32") is False
        and controlled.get("cudnn_allow_tf32") is False
        and controlled.get("float32_matmul_precision") == "highest"
        and controlled.get("pythonhashseed") == 1
        and controlled.get("cublas_workspace_config") == ":4096:8"
        and controlled.get("omp_num_threads") == 1
        and controlled.get("mkl_num_threads") == 1
        and controlled.get("num_workers") == 0,
        "test_sealed": config.get("official_test_policy")
        == {
            "accessed": False,
            "evaluations": 0,
            "reason": (
                "A/A smoke validates controlled execution and measures numerical "
                "drift, not performance"
            ),
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Invalid controlled A/A smoke contract: " + ", ".join(failures)
        )
    return config


def exact_epoch_control_trace(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: row.get(key) for key in EXACT_EPOCH_CONTROL_FIELDS}
        for row in summary.get("epochs", [])
    ]


def _flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), nested, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def numeric_epoch_trace(summary: dict[str, Any]) -> list[dict[str, float]]:
    trace: list[dict[str, float]] = []
    for row in summary.get("epochs", []):
        filtered = {
            key: value
            for key, value in row.items()
            if key not in IGNORED_NUMERIC_EPOCH_FIELDS
            and key not in EXACT_EPOCH_CONTROL_FIELDS
            and not key.endswith("_sha256_after_epoch")
        }
        flattened: dict[str, float] = {}
        _flatten_numeric("", filtered, flattened)
        trace.append(flattened)
    return trace


def numeric_trace_delta(
    run_a: dict[str, Any], run_b: dict[str, Any]
) -> dict[str, Any]:
    trace_a = numeric_epoch_trace(run_a)
    trace_b = numeric_epoch_trace(run_b)
    deltas: list[dict[str, Any]] = []
    for index, (row_a, row_b) in enumerate(zip(trace_a, trace_b, strict=False), 1):
        for field in sorted(set(row_a) | set(row_b)):
            if field not in row_a or field not in row_b:
                deltas.append({"epoch": index, "field": field, "missing": True})
                continue
            deltas.append(
                {
                    "epoch": index,
                    "field": field,
                    "run_a": row_a[field],
                    "run_b": row_b[field],
                    "absolute_difference": abs(row_a[field] - row_b[field]),
                }
            )
    numeric = [row["absolute_difference"] for row in deltas if "absolute_difference" in row]
    return {
        "same_trace_length": len(trace_a) == len(trace_b),
        "all_reported_values_exact": bool(len(trace_a) == len(trace_b))
        and all(value == 0.0 for value in numeric)
        and not any(row.get("missing") for row in deltas),
        "max_absolute_difference": max(numeric, default=0.0),
        "deltas": deltas,
    }


def compare_summaries(
    run_a: dict[str, Any], run_b: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        f"summary.{field}": run_a.get(field) == run_b.get(field)
        for field in EXACT_SUMMARY_FIELDS
    }
    checks["epoch_input_rng_control"] = exact_epoch_control_trace(
        run_a
    ) == exact_epoch_control_trace(run_b)
    checks["runtime_contract"] = {
        field: run_a.get("runtime", {}).get(field) for field in RUNTIME_FIELDS
    } == {
        field: run_b.get("runtime", {}).get(field) for field in RUNTIME_FIELDS
    }
    checks["run_a_complete"] = run_a.get("status") == "complete"
    checks["run_b_complete"] = run_b.get("status") == "complete"
    checks["run_a_test_sealed"] = (
        run_a.get("official_test_accessed") is False
        and run_a.get("official_test") is None
    )
    checks["run_b_test_sealed"] = (
        run_b.get("official_test_accessed") is False
        and run_b.get("official_test") is None
    )
    checks["run_a_controlled"] = run_a.get("controlled_aa_smoke") is True
    checks["run_b_controlled"] = run_b.get("controlled_aa_smoke") is True
    checks["known_cuda_limitations_disclosed"] = all(
        [
            row.get("kernel")
            for row in summary.get("controlled_reproducibility", {}).get(
                "observed_nondeterministic_operations", []
            )
        ]
        == [
            "compute_grad_input",
            "adaptive_max_pool2d_backward_cuda",
            "memory_efficient_attention_backward_cuda",
        ]
        and summary.get("controlled_reproducibility", {}).get(
            "formal_bitwise_determinism"
        )
        is False
        for summary in (run_a, run_b)
    )
    checks["two_epoch_full_data_trace"] = (
        len(exact_epoch_control_trace(run_a)) == 2
        and len(exact_epoch_control_trace(run_b)) == 2
        and all(
            row.get("input_stream_sha256")
            and row.get("student_state_sha256_after_epoch")
            and row.get("guidance_state_sha256_after_epoch")
            and row.get("rng_state_sha256_after_epoch")
            for row in run_a.get("epochs", []) + run_b.get("epochs", [])
        )
    )
    failures = [name for name, passed in checks.items() if not passed]
    numerical_delta = numeric_trace_delta(run_a, run_b)
    return {
        "status": "pass" if not failures else "fail",
        "execution_control_gates_passed": not failures,
        "checks": checks,
        "failures": failures,
        "numerical_observation_not_a_smoke_gate": {
            "formal_bitwise_determinism_claimed": False,
            "reason": (
                "three disclosed CUDA backward paths prevent a formal bitwise "
                "determinism claim for the unchanged iBKD implementation"
            ),
            "epoch_metrics": numerical_delta,
            "checkpoint_file_sha256_equal": run_a.get("checkpoint_sha256")
            == run_b.get("checkpoint_sha256"),
            "final_student_state_sha256_equal": run_a.get("student_state_sha256")
            == run_b.get("student_state_sha256"),
            "final_guidance_state_sha256_equal": run_a.get("guidance_state_sha256")
            == run_b.get("guidance_state_sha256"),
            "controller_state_exact": run_a.get("controller")
            == run_b.get("controller"),
            "aggregation_state_exact": run_a.get("ibkd_aggregation")
            == run_b.get("ibkd_aggregation"),
        },
        "run_a": {
            "checkpoint_sha256": run_a.get("checkpoint_sha256"),
            "student_state_sha256": run_a.get("student_state_sha256"),
            "guidance_state_sha256": run_a.get("guidance_state_sha256"),
            "epoch_input_stream_sha256": [
                row.get("input_stream_sha256")
                for row in run_a.get("epochs", [])
            ],
        },
        "run_b": {
            "checkpoint_sha256": run_b.get("checkpoint_sha256"),
            "student_state_sha256": run_b.get("student_state_sha256"),
            "guidance_state_sha256": run_b.get("guidance_state_sha256"),
            "epoch_input_stream_sha256": [
                row.get("input_stream_sha256")
                for row in run_b.get("epochs", [])
            ],
        },
    }


def _run_once(
    *,
    label: str,
    data_dir: Path,
    output_dir: Path,
    teacher_checkpoint: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    run_name = f"run_{label}"
    run_dir = output_dir / run_name
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse existing A/A run directory: {run_dir}")
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
        "--controlled-aa-smoke",
        "--batch-size",
        "128",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--teacher-checkpoint",
        str(teacher_checkpoint),
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
    log(f"[CONTROLLED_AA_SMOKE_RUN_START] label={label}")
    subprocess.run(
        command,
        check=True,
        env=environment,
        cwd=REPOSITORY_ROOT,
    )
    summary = _load_json(run_dir / "summary.json")
    log(
        f"[CONTROLLED_AA_SMOKE_RUN_DONE] label={label} "
        f"student_state_sha256={summary.get('student_state_sha256')} "
        f"guidance_state_sha256={summary.get('guidance_state_sha256')}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = validate_config(args.config)
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(CONTROLLED_ENVIRONMENT)
    started = time.perf_counter()
    run_a = _run_once(
        label="A",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        environment=environment,
    )
    run_b = _run_once(
        label="B",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        environment=environment,
    )
    comparison = compare_summaries(run_a, run_b)
    comparison.update(
        {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "config_sha256": file_sha256(args.config),
            "scientific_result": False,
            "official_test_evaluations": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "estimated_controlled_full_aa_seconds": 300
            * (
                float(run_a["avg_epoch_seconds"])
                + float(run_b["avg_epoch_seconds"])
            ),
        }
    )
    save_json(comparison, args.output_dir / "aa_comparison.json")
    save_json(
        {
            "status": comparison["status"],
            "phase": "controlled_empirical_aa_smoke",
            "runs_complete": 2,
            "expected_runs": 2,
            "execution_control_gates_passed": comparison[
                "execution_control_gates_passed"
            ],
            "official_test_evaluations": 0,
            "scientific_result": False,
        },
        args.output_dir / "completion_status.json",
    )
    numerical = comparison["numerical_observation_not_a_smoke_gate"]
    epoch_metrics = numerical["epoch_metrics"]
    log(
        "[CONTROLLED_AA_NUMERICAL_OBSERVATION] "
        f"epoch_metrics_exact="
        f"{str(epoch_metrics['all_reported_values_exact']).lower()} "
        f"max_metric_abs_diff={epoch_metrics['max_absolute_difference']:.12g} "
        f"final_student_state_exact="
        f"{str(numerical['final_student_state_sha256_equal']).lower()} "
        f"final_guidance_state_exact="
        f"{str(numerical['final_guidance_state_sha256_equal']).lower()} "
        f"controller_state_exact={str(numerical['controller_state_exact']).lower()} "
        "smoke_gate=false"
    )
    log(
        "[CONTROLLED_AA_SMOKE_RESULT] "
        f"status={comparison['status'].upper()} runs=2/2 "
        f"execution_gates={sum(comparison['checks'].values())}/"
        f"{len(comparison['checks'])} "
        f"official_test_evaluations=0 scientific_result=false "
        "formal_bitwise_determinism=false estimated_full_aa="
        f"{format_duration(comparison['estimated_controlled_full_aa_seconds'])}"
    )
    if not comparison["execution_control_gates_passed"]:
        raise RuntimeError(
            "Controlled A/A smoke execution mismatch: "
            + ", ".join(comparison["failures"])
        )
    return comparison


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "status": "failed",
                "phase": "controlled_empirical_aa_smoke",
                "error_type": type(error).__name__,
                "error": str(error),
                "official_test_evaluations": 0,
                "scientific_result": False,
            },
            args.output_dir / "completion_status.json",
        )
        log(
            f"[CONTROLLED_AA_SMOKE_FAILED] {type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()
