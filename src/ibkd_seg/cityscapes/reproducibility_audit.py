"""LG/ALG reproducibility gate and observational smoke for Cityscapes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


STEP_FIELDS = (
    "step",
    "epoch",
    "loss",
    "ce",
    "guidance",
    "beta",
    "lr",
    "grad_norm_unclipped",
    "guidance_parameter_norm",
)

NUMERICAL_STEP_FIELDS = (
    "loss",
    "ce",
    "guidance",
    "beta",
    "lr",
    "grad_norm_unclipped",
    "guidance_parameter_norm",
)

UPDATE_CONTROL_FIELDS = ("step", "epoch", "beta", "lr")


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_steps(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_trajectory(rows: list[dict]) -> list[dict]:
    return [
        {key: row.get(key) for key in STEP_FIELDS if key in row}
        for row in rows
    ]


def compare_runs(
    left: dict,
    right: dict,
    left_steps: list[dict],
    right_steps: list[dict],
    *,
    expected_steps: int | None = None,
    require_step_input_hashes: bool = False,
    require_step_gradient_hashes: bool = False,
) -> dict:
    left_trajectory = canonical_trajectory(left_steps)
    right_trajectory = canonical_trajectory(right_steps)
    mismatched_steps = []
    for index, (left_row, right_row) in enumerate(
        zip(left_trajectory, right_trajectory, strict=False), start=1
    ):
        if left_row != right_row:
            mismatched_steps.append(index)
    if len(left_trajectory) != len(right_trajectory):
        mismatched_steps.extend(
            range(min(len(left_trajectory), len(right_trajectory)) + 1,
                  max(len(left_trajectory), len(right_trajectory)) + 1)
        )
    left_input_hashes = [row.get("input_sha256") for row in left_steps]
    right_input_hashes = [row.get("input_sha256") for row in right_steps]
    input_hashes_present = (
        bool(left_input_hashes)
        and bool(right_input_hashes)
        and all(value is not None for value in left_input_hashes + right_input_hashes)
    )
    input_mismatched_steps = [
        index
        for index, (left_hash, right_hash) in enumerate(
            zip(left_input_hashes, right_input_hashes, strict=False), start=1
        )
        if left_hash != right_hash
    ]
    if len(left_input_hashes) != len(right_input_hashes):
        input_mismatched_steps.extend(
            range(
                min(len(left_input_hashes), len(right_input_hashes)) + 1,
                max(len(left_input_hashes), len(right_input_hashes)) + 1,
            )
        )
    left_gradient_hashes = [row.get("gradient_sha256") for row in left_steps]
    right_gradient_hashes = [row.get("gradient_sha256") for row in right_steps]
    gradient_hashes_present = (
        bool(left_gradient_hashes)
        and bool(right_gradient_hashes)
        and all(value is not None for value in left_gradient_hashes + right_gradient_hashes)
    )
    gradient_mismatched_steps = [
        index
        for index, (left_hash, right_hash) in enumerate(
            zip(left_gradient_hashes, right_gradient_hashes, strict=False), start=1
        )
        if left_hash != right_hash
    ]
    if len(left_gradient_hashes) != len(right_gradient_hashes):
        gradient_mismatched_steps.extend(
            range(
                min(len(left_gradient_hashes), len(right_gradient_hashes)) + 1,
                max(len(left_gradient_hashes), len(right_gradient_hashes)) + 1,
            )
        )
    max_abs_difference_by_field = {}
    for field in NUMERICAL_STEP_FIELDS:
        differences = []
        for left_row, right_row in zip(left_steps, right_steps, strict=False):
            left_value, right_value = left_row.get(field), right_row.get(field)
            if left_value is not None and right_value is not None:
                differences.append(abs(float(left_value) - float(right_value)))
        max_abs_difference_by_field[field] = max(differences, default=None)
    checks = {
        "nonempty_trajectories": bool(left_trajectory) and bool(right_trajectory),
        "completed_steps_equal": left.get("completed_steps") == right.get("completed_steps"),
        "input_stream_exact": left.get("input_stream_sha256") == right.get("input_stream_sha256"),
        "student_initial_state_exact": (
            left.get("student_initial_state_sha256")
            == right.get("student_initial_state_sha256")
        ),
        "teacher_state_exact": left.get("teacher_state_sha256") == right.get("teacher_state_sha256"),
        "trajectory_exact": left_trajectory == right_trajectory,
        "student_final_state_exact": (
            left.get("student_final_state_sha256")
            == right.get("student_final_state_sha256")
        ),
        "guidance_final_state_exact": (
            left.get("guidance_final_state_sha256")
            == right.get("guidance_final_state_sha256")
        ),
        "optimizer_final_state_exact": (
            left.get("optimizer_final_state_sha256")
            == right.get("optimizer_final_state_sha256")
        ),
        "validation_exact": left.get("diagnostic_validation") == right.get("diagnostic_validation"),
    }
    if require_step_input_hashes:
        checks["per_step_input_hashes_present"] = input_hashes_present
        checks["per_step_input_hashes_exact"] = (
            input_hashes_present
            and len(left_input_hashes) == len(right_input_hashes)
            and not input_mismatched_steps
        )
    if require_step_gradient_hashes:
        checks["per_step_gradient_hashes_present"] = gradient_hashes_present
        checks["per_step_gradient_hashes_exact"] = (
            gradient_hashes_present
            and len(left_gradient_hashes) == len(right_gradient_hashes)
            and not gradient_mismatched_steps
        )
    if expected_steps is not None:
        checks["expected_steps_completed"] = (
            len(left_trajectory) == expected_steps
            and len(right_trajectory) == expected_steps
            and left.get("completed_steps") == expected_steps
            and right.get("completed_steps") == expected_steps
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "left_trajectory_sha256": _json_hash(left_trajectory),
        "right_trajectory_sha256": _json_hash(right_trajectory),
        "left_input_hash_sequence_sha256": _json_hash(left_input_hashes),
        "right_input_hash_sequence_sha256": _json_hash(right_input_hashes),
        "left_gradient_hash_sequence_sha256": _json_hash(left_gradient_hashes),
        "right_gradient_hash_sequence_sha256": _json_hash(right_gradient_hashes),
        "first_input_mismatch_step": (
            input_mismatched_steps[0] if input_mismatched_steps else None
        ),
        "input_mismatched_step_count": len(input_mismatched_steps),
        "first_gradient_mismatch_step": (
            gradient_mismatched_steps[0] if gradient_mismatched_steps else None
        ),
        "gradient_mismatched_step_count": len(gradient_mismatched_steps),
        "first_trajectory_mismatch_step": mismatched_steps[0] if mismatched_steps else None,
        "trajectory_mismatched_step_count": len(mismatched_steps),
        "max_abs_difference_by_field": max_abs_difference_by_field,
        "mismatched_steps": mismatched_steps[:25],
    }


def compare_update_paths(
    left: dict,
    right: dict,
    left_steps: list[dict],
    right_steps: list[dict],
    *,
    expected_steps: int,
) -> dict:
    """Compare optimizer updates while reporting harmless scalar reduction drift separately."""
    exact = compare_runs(
        left,
        right,
        left_steps,
        right_steps,
        expected_steps=expected_steps,
        require_step_input_hashes=True,
        require_step_gradient_hashes=True,
    )
    left_controls = [
        {key: row.get(key) for key in UPDATE_CONTROL_FIELDS}
        for row in left_steps
    ]
    right_controls = [
        {key: row.get(key) for key in UPDATE_CONTROL_FIELDS}
        for row in right_steps
    ]
    optimizer_hashes_present = all(
        summary.get("optimizer_final_state_sha256") is not None
        for summary in (left, right)
    )
    checks = {
        key: exact["checks"][key]
        for key in (
            "nonempty_trajectories",
            "completed_steps_equal",
            "input_stream_exact",
            "student_initial_state_exact",
            "teacher_state_exact",
            "student_final_state_exact",
            "guidance_final_state_exact",
            "validation_exact",
            "per_step_input_hashes_present",
            "per_step_input_hashes_exact",
            "per_step_gradient_hashes_present",
            "per_step_gradient_hashes_exact",
            "expected_steps_completed",
        )
    }
    checks.update({
        "optimizer_controls_exact": left_controls == right_controls,
        "optimizer_final_state_hashes_present": optimizer_hashes_present,
        "optimizer_final_state_exact": (
            optimizer_hashes_present
            and exact["checks"]["optimizer_final_state_exact"]
        ),
    })
    scalar_fields = {}
    for field in NUMERICAL_STEP_FIELDS:
        mismatches = []
        differences = []
        for index, (left_row, right_row) in enumerate(
            zip(left_steps, right_steps, strict=False), start=1
        ):
            left_value, right_value = left_row.get(field), right_row.get(field)
            if left_value != right_value:
                mismatches.append(index)
            if left_value is not None and right_value is not None:
                differences.append(abs(float(left_value) - float(right_value)))
        scalar_fields[field] = {
            "exact": not mismatches and len(left_steps) == len(right_steps),
            "first_mismatch_step": mismatches[0] if mismatches else None,
            "mismatched_step_count": len(mismatches),
            "max_abs_difference": max(differences, default=None),
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scalar_values_exact": all(item["exact"] for item in scalar_fields.values()),
        "scalar_fields": scalar_fields,
        "exact_all_fields_passed": exact["passed"],
        "first_input_mismatch_step": exact["first_input_mismatch_step"],
        "first_gradient_mismatch_step": exact["first_gradient_mismatch_step"],
        "first_trajectory_mismatch_step": exact["first_trajectory_mismatch_step"],
        "input_mismatched_step_count": exact["input_mismatched_step_count"],
        "gradient_mismatched_step_count": exact["gradient_mismatched_step_count"],
        "trajectory_mismatched_step_count": exact["trajectory_mismatched_step_count"],
        "left_input_hash_sequence_sha256": exact["left_input_hash_sequence_sha256"],
        "right_input_hash_sequence_sha256": exact["right_input_hash_sequence_sha256"],
        "left_gradient_hash_sequence_sha256": exact["left_gradient_hash_sequence_sha256"],
        "right_gradient_hash_sequence_sha256": exact["right_gradient_hash_sequence_sha256"],
    }


def diagnose_observational_probe(comparisons: dict) -> dict:
    """Explain an observational repeat without claiming a CUDA root cause."""
    repeat = comparisons["lg_repeat"]
    cross = comparisons["lg_vs_alg_before_controller_action"]
    inputs_equal = all(
        item["checks"].get("input_stream_exact", False)
        and item["checks"].get("per_step_input_hashes_exact", False)
        for item in (repeat, cross)
    )
    initialization_equal = all(
        item["checks"].get("student_initial_state_exact", False)
        and item["checks"].get("teacher_state_exact", False)
        for item in (repeat, cross)
    )
    repeat_exact = repeat["passed"]
    cross_exact = cross["passed"]
    if not inputs_equal:
        code = "input_stream_mismatch"
        conclusion = "전체 배치 또는 augmentation 입력이 실행 간 달랐습니다."
        next_step = "DataLoader, sampler, worker 및 augmentation seed 경로를 수정합니다."
    elif not initialization_equal:
        code = "initialization_or_teacher_mismatch"
        conclusion = "Student 초기값 또는 teacher 상태가 실행 간 달랐습니다."
        next_step = "모델 초기화와 checkpoint 로딩 순서를 수정합니다."
    elif not repeat_exact:
        code = "same_method_numerical_nondeterminism_or_uncontrolled_state"
        conclusion = (
            "입력과 초기값이 같은 LG 반복 실행의 수치 궤적이 달랐습니다. "
            "GPU 비결정성 또는 통제되지 않은 실행 상태가 남아 있습니다."
        )
        next_step = "첫 수치 불일치 step의 연산과 RNG 상태를 좁혀서 확인합니다."
    elif not cross_exact:
        code = "lg_alg_execution_path_difference"
        conclusion = (
            "LG 반복은 일치하지만 controller가 동작하기 전 LG와 ALG 궤적이 달랐습니다."
        )
        next_step = "LG와 ALG의 forward, loss, optimizer 실행 경로 차이를 점검합니다."
    else:
        code = "exact_reproducibility_observed"
        conclusion = "25 step 범위에서 입력, 학습 궤적, 최종 상태와 검증 결과가 모두 일치했습니다."
        next_step = "동일한 진단을 2,000 step으로 확장합니다."
    return {
        "code": code,
        "input_streams_equal": inputs_equal,
        "initialization_and_teacher_equal": initialization_equal,
        "lg_repeat_exact": repeat_exact,
        "lg_vs_alg_pre_action_exact": cross_exact,
        "exact_reproducibility_observed": code == "exact_reproducibility_observed",
        "conclusion_ko": conclusion,
        "next_step_ko": next_step,
    }


def _determinism_ok(summary: dict) -> bool:
    environment = summary.get("environment", {})
    return (
        environment.get("strict_determinism_requested") is True
        and environment.get("torch_deterministic_algorithms") is True
        and environment.get("torch_deterministic_warn_only") is False
        and environment.get("cublas_workspace_config") in {":4096:8", ":16:8"}
        and environment.get("cudnn_benchmark") is False
        and environment.get("cudnn_deterministic") is True
        and environment.get("cuda_matmul_allow_tf32") is False
        and environment.get("cudnn_allow_tf32") is False
    )


def _baseline_controls_ok(summary: dict) -> bool:
    environment = summary.get("environment", {})
    return (
        environment.get("cublas_workspace_config") in {":4096:8", ":16:8"}
        and environment.get("cudnn_benchmark") is False
        and environment.get("cudnn_deterministic") is True
        and environment.get("cuda_matmul_allow_tf32") is False
        and environment.get("cudnn_allow_tf32") is False
    )


def run(args) -> int:
    from .data import save_json, verify_manifest
    from .full_data import prepare_labels

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    observational_probe = config.get("run_kind") == "engineering_reproducibility_smoke"
    manifest_data = json.loads(args.manifest.read_text())
    verify_manifest(args.data_dir, manifest_data)
    prepare_labels(
        args.data_dir,
        args.manifest,
        {"train": config["train_samples"], "val": config["val_samples"]},
        output,
    )
    save_json(output / "config.json", config)

    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONHASHSEED"] = str(config["seed"])
    plans = (("lg_a", "lg"), ("lg_b", "lg"), ("alg", "alg"))
    runs = {}
    returncodes = {}
    for run_id, method in plans:
        print(f"[L16_REPRO_START] run={run_id} method={method}", flush=True)
        run_output = output / run_id
        command = [
            sys.executable,
            "-u",
            "-m",
            "ibkd_seg.cityscapes.official_stability",
            "--cache-root",
            str(args.cache_root),
            "--data-dir",
            str(args.data_dir),
            "--manifest",
            str(args.manifest),
            "--labels-report",
            str(output / "labels.json"),
            "--output-dir",
            str(run_output),
            "--config",
            str(args.config),
            "--device",
            "cuda",
            "--method",
            method,
        ]
        completed = subprocess.run(command, check=False, env=environment)
        returncodes[run_id] = completed.returncode
        summary_path = run_output / "summary.json"
        if not summary_path.exists():
            runs[run_id] = {"status": "missing_summary", "method": method}
            continue
        runs[run_id] = json.loads(summary_path.read_text())

    comparisons = {}
    if all((output / run_id / "steps.jsonl").exists() for run_id, _ in plans):
        steps = {
            run_id: _read_steps(output / run_id / "steps.jsonl")
            for run_id, _ in plans
        }
        comparisons["lg_repeat"] = compare_runs(
            runs["lg_a"], runs["lg_b"], steps["lg_a"], steps["lg_b"],
            expected_steps=config["stability_steps"],
            require_step_input_hashes=observational_probe,
            require_step_gradient_hashes=observational_probe,
        )
        comparisons["lg_vs_alg_before_controller_action"] = compare_runs(
            runs["lg_a"], runs["alg"], steps["lg_a"], steps["alg"],
            expected_steps=config["stability_steps"],
            require_step_input_hashes=observational_probe,
            require_step_gradient_hashes=observational_probe,
        )

    deterministic_runs = {
        run_id: _determinism_ok(summary) for run_id, summary in runs.items()
    }
    baseline_controls = {
        run_id: _baseline_controls_ok(summary) for run_id, summary in runs.items()
    }
    all_completed = all(
        returncodes.get(run_id) == 0
        and summary.get("status") == "stable"
        and summary.get("completed_steps") == config["stability_steps"]
        for run_id, summary in runs.items()
    ) and len(runs) == len(plans)
    exact_gate_passed = (
        all_completed
        and all(deterministic_runs.values())
        and len(comparisons) == 2
        and all(comparison["passed"] for comparison in comparisons.values())
    )
    smoke_completed = (
        observational_probe
        and all_completed
        and all(baseline_controls.values())
        and len(comparisons) == 2
    )
    if observational_probe and len(comparisons) == 2:
        diagnosis = diagnose_observational_probe(comparisons)
    elif observational_probe:
        diagnosis = {
            "code": "incomplete_probe",
            "input_streams_equal": False,
            "initialization_and_teacher_equal": False,
            "lg_repeat_exact": False,
            "lg_vs_alg_pre_action_exact": False,
            "exact_reproducibility_observed": False,
            "conclusion_ko": "세 실행 또는 비교 결과가 완성되지 않았습니다.",
            "next_step_ko": "runtime error와 누락된 summary를 먼저 확인합니다.",
        }
    else:
        diagnosis = None
    run_status = {
        run_id: {
            "method": summary.get("method"),
            "status": summary.get("status"),
            "completed_steps": summary.get("completed_steps"),
            "input_stream_sha256": summary.get("input_stream_sha256"),
            "student_initial_state_sha256": summary.get("student_initial_state_sha256"),
            "teacher_state_sha256": summary.get("teacher_state_sha256"),
            "student_final_state_sha256": summary.get("student_final_state_sha256"),
            "guidance_final_state_sha256": summary.get("guidance_final_state_sha256"),
            "final_loss": (summary.get("final_step") or {}).get("loss"),
            "final_ce": (summary.get("final_step") or {}).get("ce"),
            "final_guidance": (summary.get("final_step") or {}).get("guidance"),
            "final_grad_norm": (summary.get("final_step") or {}).get("grad_norm_unclipped"),
            "final_gradient_sha256": (
                summary.get("final_step") or {}
            ).get("gradient_sha256"),
            "miou": (summary.get("diagnostic_validation") or {}).get("miou"),
            "pixel_accuracy": (
                summary.get("diagnostic_validation") or {}
            ).get("pixel_accuracy"),
            "validation_samples": summary.get("validation_samples"),
            "guidance_active_steps": summary.get("guidance_active_steps"),
            "guidance_stop_epoch": summary.get("guidance_stop_epoch"),
            "runtime_error": summary.get("runtime_error"),
            "stability_reasons": (summary.get("decision") or {}).get("reasons"),
            "strict_determinism_verified": deterministic_runs.get(run_id, False),
            "baseline_determinism_controls_verified": baseline_controls.get(run_id, False),
            "returncode": returncodes.get(run_id),
        }
        for run_id, summary in runs.items()
    }
    terminal = {
        "status": (
            "completed" if smoke_completed else "passed" if exact_gate_passed else "failed"
        ),
        "protocol_id": config["protocol_id"],
        "smoke_completed": smoke_completed if observational_probe else None,
        "reproducibility_gate_passed": exact_gate_passed,
        "strict_determinism_verified": all(deterministic_runs.values()),
        "strict_determinism_requested": bool(config.get("strict_determinism", False)),
        "baseline_determinism_controls_verified": all(baseline_controls.values()),
        "all_runs_completed": all_completed,
        "steps_per_run": config["stability_steps"],
        "guidance_beta": config["guidance_beta"],
        "seed": config["seed"],
        "test_used": config["test_used"],
        "validation_samples_per_run": config["val_samples"],
        "controller_action_observed": any(
            summary.get("guidance_stop_epoch") is not None for summary in runs.values()
        ),
        "run_status": run_status,
        "comparisons": comparisons,
        "diagnosis": diagnosis,
        "scientific_result": False,
        "full_2000_reproducibility_audit_authorized": (
            smoke_completed
            and diagnosis is not None
            and diagnosis["exact_reproducibility_observed"]
        ),
        "beta_sweep_authorized": exact_gate_passed,
        "full_training_authorized": False,
    }
    save_json(output / "reproducibility_summary.json", {
        "terminal_result": terminal,
        "runs": runs,
    })
    marker = (
        "[CITYSCAPES_REPRO_DIAGNOSTIC_SMOKE_DONE]"
        if observational_probe
        else "[CITYSCAPES_REPRODUCIBILITY_AUDIT_DONE]"
    )
    print(
        marker
        + " "
        + json.dumps(terminal, sort_keys=True, allow_nan=False, ensure_ascii=False),
        flush=True,
    )
    return 0 if smoke_completed or exact_gate_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache-root", "data-dir", "manifest", "output-dir", "config"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    for key in ("cache_root", "data_dir", "manifest", "output_dir", "config"):
        setattr(args, key, getattr(args, key).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
