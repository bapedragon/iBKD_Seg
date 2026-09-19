"""Optimizer reproducibility gates for the deterministic iBKD candidate."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from .reproducibility_audit import _read_steps, compare_update_paths
from .warn_only_reproducibility import compact_run_status, warn_only_controls_ok


PROTOCOL_ID = "cityscapes_segmenter_l16_crop512_ibkd_candidate_warn25_v11"
REPRO2000_PROTOCOL_ID = (
    "cityscapes_segmenter_l16_crop512_ibkd_candidate_repro2000_v12"
)
PROFILES = {
    PROTOCOL_ID: {
        "plans": (("ibkd_candidate_a", "ibkd"), ("ibkd_candidate_b", "ibkd")),
        "start_tag": "IBKD_CANDIDATE_WARN25_START",
        "done_tag": "CITYSCAPES_IBKD_CANDIDATE_WARN25_DONE",
        "summary_name": "ibkd_candidate_warn25_summary.json",
        "long_gate": False,
    },
    REPRO2000_PROTOCOL_ID: {
        "plans": (("ibkd_beta_0p5_a", "ibkd"), ("ibkd_beta_0p5_b", "ibkd")),
        "start_tag": "IBKD_CANDIDATE_REPRO2000_START",
        "done_tag": "CITYSCAPES_IBKD_CANDIDATE_REPRO2000_DONE",
        "summary_name": "ibkd_candidate_repro2000_summary.json",
        "long_gate": True,
    },
}
ALLOWED_NONDETERMINISTIC_OPERATORS = {"nll_loss2d_forward_out_cuda_template"}


def candidate_controls_ok(summary: dict) -> bool:
    contract = summary.get("ibkd_deterministic_candidate") or {}
    environment = summary.get("environment") or {}
    operators = set(summary.get("nondeterministic_operators") or [])
    stages = contract.get("stages") or []
    return (
        warn_only_controls_ok(summary)
        and contract.get("candidate_id") == "flatmax_cpu_deform_v1"
        and contract.get("applied") is True
        and len(stages) == 3
        and all(stage.get("applied") is True for stage in stages)
        and contract.get("cpu_threads") == 1
        and environment.get("cpu_threads") == 1
        and operators.issubset(ALLOWED_NONDETERMINISTIC_OPERATORS)
    )


def diagnose(comparison: dict | None, all_completed: bool, controls_verified: bool) -> dict:
    if not all_completed:
        code = "incomplete_ibkd_candidate_runs"
        conclusion = "iBKD 후보 A/B 중 하나 이상이 25 step과 검증을 완료하지 못했습니다."
        next_step = "각 실행의 runtime_error와 stability_reasons를 확인합니다."
    elif not controls_verified:
        code = "ibkd_candidate_controls_not_verified"
        conclusion = "결정적 iBKD 후보 또는 warn-only 실행 환경이 확인되지 않았습니다."
        next_step = "후보 적용 상태, CPU thread 수, 남은 비결정성 연산을 확인합니다."
    elif comparison is None:
        code = "missing_ibkd_candidate_comparison"
        conclusion = "A/B steps.jsonl 비교 결과가 만들어지지 않았습니다."
        next_step = "두 실행의 steps.jsonl 생성 여부를 확인합니다."
    elif not comparison.get("passed", False):
        code = "ibkd_candidate_update_path_not_reproducible"
        conclusion = "후보의 입력은 같지만 gradient 또는 최종 optimizer 경로가 다릅니다."
        next_step = "최초 gradient 불일치 step과 최종 state hash를 확인합니다."
    else:
        code = "ibkd_candidate_warn25_reproducible"
        conclusion = (
            "결정적 iBKD 후보의 25-step 입력·gradient·학생·guidance·optimizer 상태와 "
            "진단 검증 결과가 A/B에서 정확히 일치했습니다."
        )
        next_step = "이 후보를 고정한 2,000-step beta 후보 실험으로 진행합니다."
    return {
        "code": code,
        "passed": code == "ibkd_candidate_warn25_reproducible",
        "conclusion_ko": conclusion,
        "next_step_ko": next_step,
    }


def diagnose_repro2000(
    comparison: dict | None,
    all_completed: bool,
    controls_verified: bool,
) -> dict:
    if not all_completed:
        code = "incomplete_ibkd_candidate_repro2000_runs"
        conclusion = "iBKD beta=0.5 A/B 중 하나 이상이 2,000 step과 전체 검증을 완료하지 못했습니다."
        next_step = "각 실행의 runtime_error와 stability_reasons를 확인합니다."
    elif not controls_verified:
        code = "ibkd_candidate_repro2000_controls_not_verified"
        conclusion = "2,000-step 실행에서 결정적 iBKD 후보 또는 warn-only 환경이 확인되지 않았습니다."
        next_step = "후보 적용 상태, CPU thread 수, 남은 비결정성 연산을 확인합니다."
    elif comparison is None:
        code = "missing_ibkd_candidate_repro2000_comparison"
        conclusion = "2,000-step A/B의 steps.jsonl 비교 결과가 만들어지지 않았습니다."
        next_step = "두 실행의 steps.jsonl 생성 여부를 확인합니다."
    elif not comparison.get("passed", False):
        code = "ibkd_candidate_repro2000_update_path_not_reproducible"
        conclusion = "2,000-step A/B에서 gradient 또는 최종 학습 상태가 일치하지 않았습니다."
        next_step = "최초 gradient 불일치 step과 최종 state hash를 확인합니다."
    else:
        code = "ibkd_candidate_repro2000_reproducible"
        conclusion = (
            "결정적 iBKD beta=0.5의 2,000-step 입력·gradient·학생·guidance·optimizer "
            "상태와 전체 validation 결과가 A/B에서 정확히 일치했습니다."
        )
        next_step = "A 실행을 beta=0.5 결과로 채택하고 나머지 beta=0.1, 0.25, 1.0을 실행합니다."
    return {
        "code": code,
        "passed": code == "ibkd_candidate_repro2000_reproducible",
        "conclusion_ko": conclusion,
        "next_step_ko": next_step,
    }


def timing_summary(rows: list[dict]) -> dict:
    values = [float(row["seconds"]) for row in rows if math.isfinite(float(row["seconds"]))]
    return {
        "steps": len(values),
        "total_seconds": sum(values),
        "mean_step_seconds": statistics.mean(values) if values else None,
        "median_step_seconds": statistics.median(values) if values else None,
        "min_step_seconds": min(values, default=None),
        "max_step_seconds": max(values, default=None),
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
    protocol_id = config.get("protocol_id")
    profile = PROFILES.get(protocol_id)
    if profile is None:
        raise ValueError(f"Unexpected protocol: {protocol_id!r}")
    plans = profile["plans"]
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
    runs, returncodes = {}, {}
    for run_id, method in plans:
        print(f"[{profile['start_tag']}] run={run_id}", flush=True)
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
    for run_id, _ in plans:
        path = output / run_id / "steps.jsonl"
        if path.exists():
            steps[run_id] = _read_steps(path)
    comparison = None
    left_id, right_id = plans[0][0], plans[1][0]
    if all(run_id in steps for run_id, _ in plans):
        comparison = compare_update_paths(
            runs[left_id],
            runs[right_id],
            steps[left_id],
            steps[right_id],
            expected_steps=config["stability_steps"],
        )

    controls = {run_id: candidate_controls_ok(summary) for run_id, summary in runs.items()}
    all_completed = len(runs) == len(plans) and all(
        returncodes.get(run_id) == 0
        and summary.get("status") == "stable"
        and summary.get("completed_steps") == config["stability_steps"]
        for run_id, summary in runs.items()
    )
    controls_verified = len(controls) == len(plans) and all(controls.values())
    diagnosis = (
        diagnose_repro2000(comparison, all_completed, controls_verified)
        if profile["long_gate"]
        else diagnose(comparison, all_completed, controls_verified)
    )
    gate_passed = diagnosis["passed"]
    run_status = {}
    for run_id, summary in runs.items():
        status = compact_run_status(
            summary,
            returncodes.get(run_id),
            controls.get(run_id, False),
        )
        status["ibkd_deterministic_candidate"] = summary.get(
            "ibkd_deterministic_candidate"
        )
        status["environment"] = summary.get("environment")
        status["timing"] = timing_summary(steps.get(run_id, []))
        run_status[run_id] = status
    operators = sorted({
        operator
        for summary in runs.values()
        for operator in summary.get("nondeterministic_operators", [])
    })
    terminal = {
        "status": "passed" if gate_passed else "failed",
        "protocol_id": protocol_id,
        "all_runs_completed": all_completed,
        "candidate_controls_verified": controls_verified,
        "update_path_reproducible": gate_passed,
        "steps_per_run": config["stability_steps"],
        "run_count": len(plans),
        "seed": config["seed"],
        "guidance_beta": config["guidance_beta_by_method"]["ibkd"],
        "candidate_id": config["ibkd_deterministic_candidate_id"],
        "acceptance_rule": (
            "same inputs and gradients at every step; exact final student, guidance, optimizer "
            "and validation; only the known CE scalar reduction warning may remain"
        ),
        "run_status": run_status,
        "comparison": comparison,
        "diagnosis": diagnosis,
        "nondeterministic_operators": operators,
        "allowed_nondeterministic_operators": sorted(ALLOWED_NONDETERMINISTIC_OPERATORS),
        "test_used": config["test_used"],
        "scientific_result": False,
        "beta_grid_2000_authorized": gate_passed,
        "full_2000_reproducibility_audit_authorized": gate_passed,
        "full_2000_reproducibility_audit_passed": (
            gate_passed if profile["long_gate"] else False
        ),
        "beta_0p5_a_reusable_as_grid_result": (
            gate_passed if profile["long_gate"] else False
        ),
        "remaining_ibkd_beta_grid_authorized": (
            gate_passed if profile["long_gate"] else False
        ),
        "full_training_authorized": False,
    }
    save_json(
        output / profile["summary_name"],
        {"terminal_result": terminal, "runs": runs},
    )
    print(
        f"[{profile['done_tag']}] "
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
