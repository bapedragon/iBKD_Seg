#!/usr/bin/env python3
"""Run and compare two controlled 300-epoch CUB main-L0 iBKD jobs."""

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
    "cub200_r50_224_b128_main_l0_ibkd_controlled_aa_full_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "a6ed48c8cd79dade137f4c590f3e361d518c6474aa28f53a18021da798cc11e2"
)
EXPECTED_BASE_FULL_SHA256 = (
    "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633"
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
EXPECTED_NONDETERMINISTIC_KERNELS = (
    "compute_grad_input",
    "adaptive_max_pool2d_backward_cuda",
    "memory_efficient_attention_backward_cuda",
)
CONTROLLED_ENVIRONMENT = {
    "PYTHONHASHSEED": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NVIDIA_TF32_OVERRIDE": "0",
}
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
EXACT_SUMMARY_FIELDS = (
    "dataset",
    "num_classes",
    "kind",
    "method",
    "fusion_ratio_lambda",
    "batch_size",
    "epochs",
    "seed",
    "initial_student_state_sha256",
    "initial_guidance_state_sha256",
    "teacher_checkpoint_sha256",
    "teacher_model_state_sha256",
    "teacher_architecture",
    "protocol_config_sha256",
    "batch_profile_role",
    "cub_loader_profile",
    "ibkd_aggregation_mode",
    "guidance_controller_warmup_epochs",
    "optimizer_contract",
    "split_manifest",
    "controlled_reproducibility",
)
EXACT_EPOCH_CONTROL_FIELDS = (
    "epoch",
    "lr",
    "input_stream_sha256",
    "rng_state_sha256_after_epoch",
)
IGNORED_NUMERIC_HISTORY_FIELDS = {
    "epoch",
    "seconds_including_validation",
    "peak_cuda_memory_bytes",
    "peak_cuda_memory_reserved_bytes",
    "input_stream_sha256",
    "student_state_sha256_after_epoch",
    "guidance_state_sha256_after_epoch",
    "rng_state_sha256_after_epoch",
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


def _operation_kernels(value: dict[str, Any]) -> tuple[str | None, ...]:
    return tuple(
        row.get("kernel")
        for row in value.get("observed_nondeterministic_operations", [])
    )


def validate_config(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Controlled A/A full config SHA-256 changed")
    config = _load_json(path)
    base_info = config.get("lineage", {}).get("base_full_protocol", {})
    base_path = _resolve_repository_path(str(base_info.get("path", "")))
    dataset = config.get("dataset", {})
    teacher = config.get("teacher", {})
    student = config.get("student", {})
    controlled = config.get("controlled_reproducibility", {})
    test_policy = config.get("official_test_policy", {})
    checks = {
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_main_l0_ibkd_controlled_aa_full_v1",
        "diagnostic_scope": config.get("scientific_result") is False
        and config.get("posthoc_reproducibility_audit") is True,
        "smoke_gate": config.get("smoke_gate", {}).get("source_h200_issue") == 765
        and config.get("smoke_gate", {}).get("status") == "passed"
        and config.get("smoke_gate", {}).get("execution_control_gates") == "27/27",
        "base_full_protocol": base_path.is_file()
        and base_info.get("sha256") == EXPECTED_BASE_FULL_SHA256
        and file_sha256(base_path) == EXPECTED_BASE_FULL_SHA256,
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
        "controlled": controlled.get("independent_fresh_processes") == 2
        and controlled.get("run_labels") == ["A", "B"]
        and controlled.get("torch_use_deterministic_algorithms") is True
        and controlled.get("warn_only") is True
        and controlled.get("formal_bitwise_determinism") is False
        and _operation_kernels(controlled) == EXPECTED_NONDETERMINISTIC_KERNELS
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
        "official_test": test_policy.get("selection_uses_official_test") is False
        and test_policy.get("total_evaluations") == 2
        and test_policy.get("test_must_not_select_epoch_method_lambda_or_loader")
        is True,
        "execution": config.get("execution", {}).get("requested_mig_slices") == 1
        and config.get("execution", {}).get("runs") == "A_then_B_sequentially",
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Invalid controlled A/A full contract: " + ", ".join(failures)
        )
    return config


def _flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), nested, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def _numeric_history(summary: dict[str, Any]) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for row in summary.get("history", []):
        filtered = {
            key: value
            for key, value in row.items()
            if key not in IGNORED_NUMERIC_HISTORY_FIELDS
        }
        flattened: dict[str, float] = {}
        _flatten_numeric("", filtered, flattened)
        output.append(flattened)
    return output


def _history_numerical_difference(
    run_a: dict[str, Any], run_b: dict[str, Any]
) -> dict[str, Any]:
    history_a = _numeric_history(run_a)
    history_b = _numeric_history(run_b)
    max_difference = 0.0
    first_different_epoch: int | None = None
    maximum_at: dict[str, Any] | None = None
    compared_values = 0
    missing_fields: list[dict[str, Any]] = []
    for index, (row_a, row_b) in enumerate(
        zip(history_a, history_b, strict=False), 1
    ):
        for field in sorted(set(row_a) | set(row_b)):
            if field not in row_a or field not in row_b:
                missing_fields.append({"epoch": index, "field": field})
                continue
            compared_values += 1
            difference = abs(row_a[field] - row_b[field])
            if difference > 0.0 and first_different_epoch is None:
                first_different_epoch = index
            if difference > max_difference:
                max_difference = difference
                maximum_at = {
                    "epoch": index,
                    "field": field,
                    "run_a": row_a[field],
                    "run_b": row_b[field],
                    "absolute_difference": difference,
                }
    return {
        "same_history_length": len(history_a) == len(history_b),
        "history_lengths": {"A": len(history_a), "B": len(history_b)},
        "compared_numeric_values": compared_values,
        "first_epoch_with_numerical_difference": first_different_epoch,
        "maximum_absolute_difference": max_difference,
        "maximum_difference_location": maximum_at,
        "missing_fields": missing_fields,
    }


def _metric_differences(
    run_a: dict[str, Any], run_b: dict[str, Any], key: str
) -> dict[str, Any]:
    values_a = run_a.get(key) or {}
    values_b = run_b.get(key) or {}
    return {
        metric: {
            "run_a": values_a.get(metric),
            "run_b": values_b.get(metric),
            "absolute_difference": abs(
                float(values_a.get(metric)) - float(values_b.get(metric))
            ),
        }
        for metric in sorted(set(values_a) & set(values_b))
        if isinstance(values_a.get(metric), (int, float))
        and isinstance(values_b.get(metric), (int, float))
    }


def _control_trace(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {field: row.get(field) for field in EXACT_EPOCH_CONTROL_FIELDS}
        for row in summary.get("history", [])
    ]


def _runtime_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get("runtime", {}).get(field) for field in RUNTIME_FIELDS
    }


def compare_summaries(
    run_a: dict[str, Any], run_b: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        f"summary.{field}": run_a.get(field) == run_b.get(field)
        for field in EXACT_SUMMARY_FIELDS
    }
    checks.update(
        {
            "both_complete": run_a.get("status") == "complete"
            and run_b.get("status") == "complete",
            "both_300_epochs": len(run_a.get("history", [])) == 300
            and len(run_b.get("history", [])) == 300,
            "epoch_input_rng_lr_control": _control_trace(run_a)
            == _control_trace(run_b),
            "runtime_contract": _runtime_contract(run_a)
            == _runtime_contract(run_b),
            "both_controlled_diagnostics": all(
                summary.get("controlled_aa_full") is True
                and summary.get("posthoc_reproducibility_audit") is True
                and summary.get("scientific_result") is False
                and summary.get("confirmatory_main_result") is False
                and summary.get("eligible_locked_v3_matrix_cell") is False
                for summary in (run_a, run_b)
            ),
            "limitations_disclosed": all(
                _operation_kernels(
                    summary.get("controlled_reproducibility", {})
                )
                == EXPECTED_NONDETERMINISTIC_KERNELS
                and summary.get("controlled_reproducibility", {}).get(
                    "formal_bitwise_determinism"
                )
                is False
                for summary in (run_a, run_b)
            ),
            "one_post_selection_test_each": all(
                summary.get("official_test_evaluations") == 1
                and summary.get("official_test_accessed") is True
                and summary.get("official_test_used_for_training_or_selection")
                is False
                and summary.get("official_test_policy")
                == "once_after_validation_selection"
                and summary.get("selected_checkpoint_strict_reloaded") is True
                for summary in (run_a, run_b)
            ),
            "all_control_hashes_recorded": all(
                row.get("input_stream_sha256")
                and row.get("rng_state_sha256_after_epoch")
                and row.get("student_state_sha256_after_epoch")
                and row.get("guidance_state_sha256_after_epoch")
                for row in run_a.get("history", []) + run_b.get("history", [])
            ),
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    stop_a = (run_a.get("controller_final") or {}).get("stop_epoch")
    stop_b = (run_b.get("controller_final") or {}).get("stop_epoch")
    selected_a = run_a.get("selected_epoch")
    selected_b = run_b.get("selected_epoch")
    numerical = {
        "formal_bitwise_determinism_claimed": False,
        "no_posthoc_numerical_pass_threshold": True,
        "history": _history_numerical_difference(run_a, run_b),
        "selected_epoch": {
            "run_a": selected_a,
            "run_b": selected_b,
            "absolute_difference": abs(int(selected_a) - int(selected_b)),
        },
        "selected_validation": _metric_differences(
            run_a, run_b, "selected_validation"
        ),
        "official_test": _metric_differences(run_a, run_b, "official_test"),
        "controller_stop_epoch": {
            "run_a": stop_a,
            "run_b": stop_b,
            "absolute_difference": (
                None
                if stop_a is None or stop_b is None
                else abs(int(stop_a) - int(stop_b))
            ),
        },
        "selected_student_state_sha256_equal": run_a.get("student_state_sha256")
        == run_b.get("student_state_sha256"),
        "selected_guidance_state_sha256_equal": run_a.get(
            "guidance_state_sha256"
        )
        == run_b.get("guidance_state_sha256"),
        "controller_state_exact": run_a.get("controller_final")
        == run_b.get("controller_final"),
        "aggregation_state_exact": run_a.get("ibkd_aggregation")
        == run_b.get("ibkd_aggregation"),
        "checkpoint_file_sha256_equal": run_a.get("checkpoint_sha256")
        == run_b.get("checkpoint_sha256"),
    }
    return {
        "status": "complete" if not failures else "failed",
        "execution_control_gates_passed": not failures,
        "checks": checks,
        "failures": failures,
        "long_horizon_numerical_observation": numerical,
    }


def _run_once(
    *,
    label: str,
    data_dir: Path,
    output_dir: Path,
    teacher_checkpoint: Path,
    base_protocol: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    run_name = f"run_{label}"
    run_dir = output_dir / run_name
    if run_dir.exists():
        raise RuntimeError(f"Refusing to reuse existing A/A run directory: {run_dir}")
    command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_full",
        "--full-run",
        "--controlled-aa-full",
        "--dataset",
        "cub",
        "--kind",
        "student",
        "--method",
        "ibkd",
        "--fusion-ratio",
        "0.25",
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
        "--teacher-architecture",
        "resnet50_224_scratch",
        "--scientific-cub-r50-teacher",
        "--cub-loader-profile",
        "l0_current_strong",
        "--protocol-config",
        str(base_protocol),
        "--batch-profile-role",
        "locked_v3_partial_cell",
        "--eval-batch-size",
        "200",
        "--num-workers",
        "0",
        "--seed",
        "1",
    ]
    log(f"[CONTROLLED_AA_FULL_RUN_START] label={label} epochs=300")
    subprocess.run(command, check=True, env=environment, cwd=REPOSITORY_ROOT)
    summary = _load_json(run_dir / "summary.json")
    log(
        f"[CONTROLLED_AA_FULL_RUN_DONE] label={label} "
        f"selected_epoch={summary.get('selected_epoch')} "
        f"controller_stop_epoch="
        f"{(summary.get('controller_final') or {}).get('stop_epoch')} "
        f"validation_macro="
        f"{(summary.get('selected_validation') or {}).get('macro_top1')} "
        f"test_macro={(summary.get('official_test') or {}).get('macro_top1')}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = validate_config(args.config)
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    if file_sha256(args.teacher_checkpoint) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("Audited issue-722 teacher checkpoint SHA-256 changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_protocol = _resolve_repository_path(
        config["lineage"]["base_full_protocol"]["path"]
    )
    environment = dict(os.environ)
    environment.update(CONTROLLED_ENVIRONMENT)
    started = time.perf_counter()
    save_json(
        {
            "status": "running",
            "phase": "controlled_empirical_aa_full",
            "runs_complete": 0,
            "expected_runs": 2,
            "official_test_evaluations": 0,
            "scientific_result": False,
        },
        args.output_dir / "completion_status.json",
    )
    run_a = _run_once(
        label="A",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        base_protocol=base_protocol,
        environment=environment,
    )
    save_json(
        {
            "status": "running",
            "phase": "controlled_empirical_aa_full",
            "runs_complete": 1,
            "expected_runs": 2,
            "official_test_evaluations": 1,
            "scientific_result": False,
        },
        args.output_dir / "completion_status.json",
    )
    run_b = _run_once(
        label="B",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        base_protocol=base_protocol,
        environment=environment,
    )
    comparison = compare_summaries(run_a, run_b)
    comparison.update(
        {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "config_sha256": file_sha256(args.config),
            "scientific_result": False,
            "posthoc_reproducibility_audit": True,
            "official_test_evaluations": 2,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    save_json(comparison, args.output_dir / "aa_full_comparison.json")
    manifest = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scientific_result": False,
        "checkpoints": [
            {
                "run": label,
                "path": summary["checkpoint"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "student_state_sha256": summary["student_state_sha256"],
                "guidance_state_sha256": summary["guidance_state_sha256"],
                "selected_epoch": summary["selected_epoch"],
                "selected_validation": summary["selected_validation"],
                "official_test": summary["official_test"],
            }
            for label, summary in (("A", run_a), ("B", run_b))
        ],
    }
    save_json(manifest, args.output_dir / "checkpoint_manifest.json")
    save_json(
        {
            "status": comparison["status"],
            "phase": "controlled_empirical_aa_full",
            "runs_complete": 2,
            "expected_runs": 2,
            "execution_control_gates_passed": comparison[
                "execution_control_gates_passed"
            ],
            "official_test_evaluations": 2,
            "scientific_result": False,
            "elapsed_seconds": comparison["elapsed_seconds"],
        },
        args.output_dir / "completion_status.json",
    )
    numerical = comparison["long_horizon_numerical_observation"]
    for label, summary in (("A", run_a), ("B", run_b)):
        log(
            f"[CONTROLLED_AA_FULL_RUN_RESULT] label={label} "
            f"selected_epoch={summary['selected_epoch']} "
            f"controller_stop_epoch="
            f"{(summary.get('controller_final') or {}).get('stop_epoch')} "
            f"validation_macro={summary['selected_validation']['macro_top1']:.6f} "
            f"test_macro={summary['official_test']['macro_top1']:.6f} "
            f"test_overall={summary['official_test']['overall_top1']:.6f} "
            f"student_state_sha256={summary['student_state_sha256']}"
        )
    log(
        "[CONTROLLED_AA_FULL_COMPARISON] "
        f"status={comparison['status'].upper()} "
        f"execution_gates={sum(comparison['checks'].values())}/"
        f"{len(comparison['checks'])} official_test_evaluations=2 "
        f"test_macro_abs_diff="
        f"{numerical['official_test']['macro_top1']['absolute_difference']:.12g} "
        f"validation_macro_abs_diff="
        f"{numerical['selected_validation']['macro_top1']['absolute_difference']:.12g} "
        f"selected_epoch_abs_diff="
        f"{numerical['selected_epoch']['absolute_difference']} "
        f"controller_stop_epoch_abs_diff="
        f"{numerical['controller_stop_epoch']['absolute_difference']} "
        f"first_metric_diff_epoch="
        f"{numerical['history']['first_epoch_with_numerical_difference']} "
        f"max_epoch_metric_abs_diff="
        f"{numerical['history']['maximum_absolute_difference']:.12g} "
        f"selected_student_state_exact="
        f"{str(numerical['selected_student_state_sha256_equal']).lower()} "
        "formal_bitwise_determinism=false numerical_pass_threshold=none "
        f"elapsed={format_duration(comparison['elapsed_seconds'])}"
    )
    if not comparison["execution_control_gates_passed"]:
        raise RuntimeError(
            "Controlled A/A full execution mismatch: "
            + ", ".join(comparison["failures"])
        )
    return comparison


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        completed_runs = sum(
            int((args.output_dir / f"run_{label}/summary.json").is_file())
            for label in ("A", "B")
        )
        save_json(
            {
                "status": "failed",
                "phase": "controlled_empirical_aa_full",
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_result": False,
                "runs_complete": completed_runs,
                "expected_runs": 2,
                "official_test_evaluations": completed_runs,
            },
            args.output_dir / "completion_status.json",
        )
        log(f"[CONTROLLED_AA_FULL_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
