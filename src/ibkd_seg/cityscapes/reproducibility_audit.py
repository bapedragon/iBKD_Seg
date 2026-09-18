"""Exact LG/ALG reproducibility gate before the Cityscapes beta sweep."""
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
        "validation_exact": left.get("diagnostic_validation") == right.get("diagnostic_validation"),
    }
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
        "mismatched_steps": mismatched_steps[:25],
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


def run(args) -> int:
    from .data import save_json, verify_manifest
    from .full_data import prepare_labels

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
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
        )
        comparisons["lg_vs_alg_before_controller_action"] = compare_runs(
            runs["lg_a"], runs["alg"], steps["lg_a"], steps["alg"],
            expected_steps=config["stability_steps"],
        )

    deterministic_runs = {
        run_id: _determinism_ok(summary) for run_id, summary in runs.items()
    }
    all_completed = all(
        returncodes.get(run_id) == 0
        and summary.get("status") == "stable"
        and summary.get("completed_steps") == config["stability_steps"]
        for run_id, summary in runs.items()
    ) and len(runs) == len(plans)
    passed = (
        all_completed
        and all(deterministic_runs.values())
        and len(comparisons) == 2
        and all(comparison["passed"] for comparison in comparisons.values())
    )
    terminal = {
        "status": "passed" if passed else "failed",
        "protocol_id": config["protocol_id"],
        "reproducibility_gate_passed": passed,
        "strict_determinism_verified": all(deterministic_runs.values()),
        "all_runs_completed": all_completed,
        "steps_per_run": config["stability_steps"],
        "guidance_beta": config["guidance_beta"],
        "run_status": {
            run_id: {
                "method": summary.get("method"),
                "status": summary.get("status"),
                "completed_steps": summary.get("completed_steps"),
                "final_loss": (summary.get("final_step") or {}).get("loss"),
                "final_ce": (summary.get("final_step") or {}).get("ce"),
                "final_guidance": (summary.get("final_step") or {}).get("guidance"),
                "runtime_error": summary.get("runtime_error"),
                "stability_reasons": (summary.get("decision") or {}).get("reasons"),
                "student_final_state_sha256": summary.get("student_final_state_sha256"),
                "guidance_final_state_sha256": summary.get("guidance_final_state_sha256"),
                "determinism_verified": deterministic_runs.get(run_id, False),
                "returncode": returncodes.get(run_id),
            }
            for run_id, summary in runs.items()
        },
        "comparisons": comparisons,
        "scientific_result": False,
        "beta_sweep_authorized": passed,
        "full_training_authorized": False,
    }
    save_json(output / "reproducibility_summary.json", {
        "terminal_result": terminal,
        "runs": runs,
    })
    print(
        "[CITYSCAPES_REPRODUCIBILITY_AUDIT_DONE] "
        + json.dumps(terminal, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0 if passed else 1


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
