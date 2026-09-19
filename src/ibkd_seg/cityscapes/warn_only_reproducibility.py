"""Warn-only 25-step update-path reproducibility gate for Cityscapes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .reproducibility_audit import _read_steps, compare_update_paths


PROTOCOL_ID = "cityscapes_segmenter_l16_crop512_warn25_repro_v9"
PLANS = (
    ("lg_a", "lg"),
    ("lg_b", "lg"),
    ("alg", "alg"),
    ("ibkd_a", "ibkd"),
    ("ibkd_b", "ibkd"),
)
COMPARISON_PLANS = (
    ("lg_repeat", "lg_a", "lg_b"),
    ("lg_vs_alg_before_controller_action", "lg_a", "alg"),
    ("ibkd_repeat", "ibkd_a", "ibkd_b"),
)


def warn_only_controls_ok(summary: dict) -> bool:
    environment = summary.get("environment", {})
    return (
        environment.get("strict_determinism_requested") is False
        and environment.get("determinism_warn_only_requested") is True
        and environment.get("torch_deterministic_algorithms") is True
        and environment.get("torch_deterministic_warn_only") is True
        and environment.get("cublas_workspace_config") in {":4096:8", ":16:8"}
        and environment.get("cudnn_benchmark") is False
        and environment.get("cudnn_deterministic") is True
        and environment.get("cuda_matmul_allow_tf32") is False
        and environment.get("cudnn_allow_tf32") is False
    )


def diagnose(comparisons: dict, all_completed: bool, controls_verified: bool) -> dict:
    failed_comparisons = [
        name for name, comparison in comparisons.items() if not comparison.get("passed", False)
    ]
    scalar_variation = any(
        not comparison.get("scalar_values_exact", False)
        for comparison in comparisons.values()
    )
    ce_scalar_variation = any(
        "ce" in comparison.get("scalar_fields", {})
        and not comparison["scalar_fields"]["ce"].get("exact", False)
        for comparison in comparisons.values()
    )
    if not all_completed:
        code = "incomplete_warn_only_runs"
        conclusion = "다섯 실행 중 하나 이상이 25 step과 검증을 완료하지 못했습니다."
        next_step = "각 실행의 runtime_error와 stability_reasons를 먼저 수정합니다."
    elif not controls_verified:
        code = "warn_only_controls_not_verified"
        conclusion = "한 실행 이상에서 요청한 warn-only 결정론 환경이 확인되지 않았습니다."
        next_step = "CUDA 결정론 환경 변수와 PyTorch warn-only 설정을 먼저 수정합니다."
    elif len(comparisons) != len(COMPARISON_PLANS):
        code = "missing_update_path_comparisons"
        conclusion = "필요한 LG·ALG·iBKD update 경로 비교가 모두 만들어지지 않았습니다."
        next_step = "누락된 steps.jsonl 또는 summary.json을 확인합니다."
    elif failed_comparisons:
        code = "update_path_not_reproducible"
        conclusion = "동일 입력 조건인데 gradient 또는 최종 학습 상태가 일치하지 않았습니다."
        next_step = "실패한 비교의 최초 gradient 불일치 step을 단일 step 분석으로 좁힙니다."
    else:
        code = "warn_only_update_path_reproducible"
        conclusion = (
            "LG 반복, 제어기 동작 전 LG·ALG, iBKD 반복에서 입력·gradient·최종 학습 상태가 "
            "모두 정확히 일치했습니다."
        )
        next_step = "같은 통과 기준으로 2,000-step 후보 실험을 다시 실행합니다."
    return {
        "code": code,
        "passed": code == "warn_only_update_path_reproducible",
        "failed_comparisons": failed_comparisons,
        "scalar_variation_observed": scalar_variation,
        "ce_scalar_variation_observed": ce_scalar_variation,
        "scalar_variation_affects_gate": False,
        "conclusion_ko": conclusion,
        "next_step_ko": next_step,
    }


def compact_run_status(summary: dict, returncode: int | None, controls_ok: bool) -> dict:
    final = summary.get("final_step") or {}
    validation = summary.get("diagnostic_validation") or {}
    return {
        "method": summary.get("method"),
        "status": summary.get("status"),
        "completed_steps": summary.get("completed_steps"),
        "expected_steps": summary.get("expected_steps"),
        "guidance_beta": summary.get("effective_guidance_beta"),
        "final_step": final,
        "final_loss": final.get("loss"),
        "final_ce": final.get("ce"),
        "final_guidance": final.get("guidance"),
        "final_grad_norm": final.get("grad_norm_unclipped"),
        "final_input_sha256": final.get("input_sha256"),
        "final_gradient_sha256": final.get("gradient_sha256"),
        "input_stream_sha256": summary.get("input_stream_sha256"),
        "student_initial_state_sha256": summary.get("student_initial_state_sha256"),
        "teacher_state_sha256": summary.get("teacher_state_sha256"),
        "student_final_state_sha256": summary.get("student_final_state_sha256"),
        "guidance_final_state_sha256": summary.get("guidance_final_state_sha256"),
        "optimizer_final_state_sha256": summary.get("optimizer_final_state_sha256"),
        "miou": validation.get("miou"),
        "pixel_accuracy": validation.get("pixel_accuracy"),
        "diagnostic_validation": summary.get("diagnostic_validation"),
        "validation_samples": summary.get("validation_samples"),
        "guidance_active_steps": summary.get("guidance_active_steps"),
        "guidance_stop_epoch": summary.get("guidance_stop_epoch"),
        "nondeterministic_operators": summary.get("nondeterministic_operators", []),
        "warn_only_controls_verified": controls_ok,
        "teacher_frozen_verified": summary.get("teacher_frozen_verified"),
        "parameters_finite": summary.get("parameters_finite"),
        "optimizer_state_finite": summary.get("optimizer_state_finite"),
        "train_seconds": summary.get("train_seconds"),
        "invocation_seconds": summary.get("invocation_seconds"),
        "peak_cuda_allocated_bytes": summary.get("peak_cuda_allocated_bytes"),
        "runtime_error": summary.get("runtime_error"),
        "stability_reasons": (summary.get("decision") or {}).get("reasons"),
        "stability_decision": summary.get("decision"),
        "returncode": returncode,
    }


def run(args) -> int:
    from .data import save_json, verify_manifest
    from .full_data import prepare_labels
    from .official_stability import validate_config

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    validate_config(config)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Unexpected protocol: {config.get('protocol_id')!r}")
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
    runs = {}
    returncodes = {}
    for run_id, method in PLANS:
        print(f"[L16_WARN25_REPRO_START] run={run_id} method={method}", flush=True)
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
            "--run-id",
            run_id,
        ]
        completed = subprocess.run(command, check=False, env=environment)
        returncodes[run_id] = completed.returncode
        summary_path = run_output / "summary.json"
        runs[run_id] = (
            json.loads(summary_path.read_text())
            if summary_path.exists()
            else {"status": "missing_summary", "method": method, "run_id": run_id}
        )

    steps = {}
    for run_id, _ in PLANS:
        path = output / run_id / "steps.jsonl"
        if path.exists():
            steps[run_id] = _read_steps(path)
    comparisons = {}
    for name, left_id, right_id in COMPARISON_PLANS:
        if left_id in steps and right_id in steps:
            comparisons[name] = compare_update_paths(
                runs[left_id],
                runs[right_id],
                steps[left_id],
                steps[right_id],
                expected_steps=config["stability_steps"],
            )

    controls = {
        run_id: warn_only_controls_ok(summary)
        for run_id, summary in runs.items()
    }
    all_completed = len(runs) == len(PLANS) and all(
        returncodes.get(run_id) == 0
        and summary.get("status") == "stable"
        and summary.get("completed_steps") == config["stability_steps"]
        for run_id, summary in runs.items()
    )
    controls_verified = len(controls) == len(PLANS) and all(controls.values())
    diagnosis = diagnose(comparisons, all_completed, controls_verified)
    gate_passed = diagnosis["passed"]
    controller_action_observed = any(
        summary.get("guidance_stop_epoch") is not None for summary in runs.values()
    )
    run_status = {
        run_id: compact_run_status(summary, returncodes.get(run_id), controls.get(run_id, False))
        for run_id, summary in runs.items()
    }
    terminal = {
        "status": "passed" if gate_passed else "failed",
        "protocol_id": config["protocol_id"],
        "all_runs_completed": all_completed,
        "warn_only_controls_verified": controls_verified,
        "update_path_reproducible": gate_passed,
        "steps_per_run": config["stability_steps"],
        "run_count": len(PLANS),
        "seed": config["seed"],
        "guidance_beta_by_method": config["guidance_beta_by_method"],
        "controller_action_observed": controller_action_observed,
        "acceptance_rule": (
            "input, gradient, optimizer controls, final student/guidance/optimizer state and "
            "validation must match exactly; logged scalar loss drift alone is reported but does not fail"
        ),
        "run_status": run_status,
        "comparisons": comparisons,
        "diagnosis": diagnosis,
        "nondeterministic_operators": sorted({
            operator
            for summary in runs.values()
            for operator in summary.get("nondeterministic_operators", [])
        }),
        "test_used": config["test_used"],
        "scientific_result": False,
        "full_2000_reproducibility_audit_authorized": gate_passed,
        "full_training_authorized": False,
    }
    save_json(output / "warn_only_reproducibility_summary.json", {
        "terminal_result": terminal,
        "runs": runs,
    })
    print(
        "[CITYSCAPES_L16_WARN25_REPRO_DONE] "
        + json.dumps(terminal, sort_keys=True, allow_nan=False, ensure_ascii=False),
        flush=True,
    )
    return 0 if gate_passed else 1


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
