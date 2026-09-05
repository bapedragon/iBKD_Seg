#!/usr/bin/env python3
"""Run one prespecified Phase 1 full-classification batch profile."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .data import prepare_and_audit_dataset, save_json
from .full_matrix import FullTask, build_full_tasks
from .train_timing import format_duration


PLANNED_EPOCHS = 300
EXPECTED_STUDENTS = 18
EXPECTED_TOTAL_TASKS = 19


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-suite", action="store_true", required=True)
    parser.add_argument("--batch-size", type=int, choices=(64, 128), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    return parser.parse_args()


def teacher_command(args: argparse.Namespace, run_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_full",
        "--full-run",
        "--kind",
        "teacher",
        "--batch-size",
        "128",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir / "teacher"),
        "--run-name",
        run_name,
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--seed",
        "1",
    ]


def student_command(
    task: FullTask,
    args: argparse.Namespace,
    teacher_checkpoint: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_full",
        "--full-run",
        "--kind",
        "student",
        "--method",
        task.method,
        "--batch-size",
        str(task.batch_size),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir / "students"),
        "--run-name",
        task.run_name,
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--seed",
        str(task.seed),
    ]
    if task.method != "vanilla":
        command.extend(("--teacher-checkpoint", str(teacher_checkpoint)))
    if task.fusion_ratio is not None:
        command.extend(("--fusion-ratio", str(task.fusion_ratio)))
    return command


def load_complete_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Process produced no summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Summary is not complete: {path}")
    return payload


def completed_row(
    *,
    index: int,
    payload: dict[str, Any],
    attempt_elapsed_seconds: float,
) -> dict[str, Any]:
    metrics = payload["official_test"]
    validation = payload["selected_validation"]
    return {
        "index": index,
        "kind": payload["kind"],
        "method": payload.get("method", "teacher"),
        "fusion_ratio_lambda": payload.get("fusion_ratio_lambda"),
        "batch_size": payload["batch_size"],
        "seed": payload["seed"],
        "selected_epoch": payload["selected_epoch"],
        "validation_macro_top1": validation["macro_top1"],
        "test_macro_top1": metrics["macro_top1"],
        "test_overall_top1": metrics["overall_top1"],
        "test_top5": metrics["top5"],
        "training_seconds": payload["training_seconds"],
        "attempt_elapsed_seconds": attempt_elapsed_seconds,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "initial_student_state_sha256": payload.get(
            "initial_student_state_sha256"
        ),
        "teacher_model_state_sha256": (
            payload.get("model_state_sha256")
            if payload["kind"] == "teacher"
            else payload.get("teacher_model_state_sha256")
        ),
        "validation_image_ids_sha256": payload["split_manifest"][
            "validation_image_ids_sha256"
        ],
        "official_test_evaluations": payload["official_test_evaluations"],
        "official_test_used_for_training_or_selection": payload[
            "official_test_used_for_training_or_selection"
        ],
        "strict_checkpoint_reload": payload["selected_checkpoint_strict_reloaded"],
        "status": "complete",
        "failure": None,
    }


def failed_row(
    *,
    index: int,
    kind: str,
    method: str,
    fusion_ratio: float | None,
    batch_size: int,
    seed: int,
    attempt_elapsed_seconds: float,
    failure: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "kind": kind,
        "method": method,
        "fusion_ratio_lambda": fusion_ratio,
        "batch_size": batch_size,
        "seed": seed,
        "selected_epoch": None,
        "validation_macro_top1": None,
        "test_macro_top1": None,
        "test_overall_top1": None,
        "test_top5": None,
        "training_seconds": None,
        "attempt_elapsed_seconds": attempt_elapsed_seconds,
        "checkpoint_sha256": None,
        "initial_student_state_sha256": None,
        "teacher_model_state_sha256": None,
        "validation_image_ids_sha256": None,
        "official_test_evaluations": 0,
        "official_test_used_for_training_or_selection": False,
        "strict_checkpoint_reload": False,
        "status": "failed",
        "failure": failure,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = tuple(failed_row(
        index=0,
        kind="",
        method="",
        fusion_ratio=None,
        batch_size=0,
        seed=0,
        attempt_elapsed_seconds=0.0,
        failure="",
    ).keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_students(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["kind"] == "student":
            grouped[(row["method"], row["fusion_ratio_lambda"], row["batch_size"])].append(row)
    aggregates: list[dict[str, Any]] = []
    for (method, fusion_ratio, batch_size), members in grouped.items():
        successful = sorted(
            (row for row in members if row["status"] == "complete"),
            key=lambda row: int(row["seed"]),
        )
        item: dict[str, Any] = {
            "method": method,
            "fusion_ratio_lambda": fusion_ratio,
            "batch_size": batch_size,
            "expected_seeds": [1, 2, 3],
            "completed_seeds": [row["seed"] for row in successful],
            "complete": len(successful) == 3,
        }
        for source, destination in (
            ("test_macro_top1", "test_macro_top1"),
            ("test_overall_top1", "test_overall_top1"),
            ("test_top5", "test_top5"),
        ):
            values = [float(row[source]) for row in successful]
            item[destination] = {
                "raw_by_seed": {
                    str(row["seed"]): row[source] for row in successful
                },
                "mean": statistics.mean(values) if values else None,
                "sample_standard_deviation": (
                    statistics.stdev(values) if len(values) >= 2 else None
                ),
            }
        aggregates.append(item)
    return aggregates


def variant_name(method: str, fusion_ratio: float | None) -> str:
    if fusion_ratio is None:
        return method
    return f"{method}_lambda{fusion_ratio:g}"


def metric_text(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def build_final_result_lines(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    *,
    profile_batch_size: int,
) -> list[str]:
    """Build a complete, grep-friendly result table for the final console log."""

    lines = [
        "=" * 96,
        f"[FINAL_RESULTS_BEGIN] profile_batch={profile_batch_size}",
    ]
    for row in rows:
        if row["status"] != "complete":
            failure = " ".join(str(row.get("failure", "unknown")).split())
            lines.append(
                "[FINAL_RESULT] "
                f"kind={row['kind']} "
                f"variant={variant_name(row['method'], row['fusion_ratio_lambda'])} "
                f"batch={row['batch_size']} seed={row['seed']} "
                f"status=failed failure={failure}"
            )
            continue
        lines.append(
            "[FINAL_RESULT] "
            f"kind={row['kind']} "
            f"variant={variant_name(row['method'], row['fusion_ratio_lambda'])} "
            f"batch={row['batch_size']} seed={row['seed']} "
            f"selected_epoch={row['selected_epoch']} "
            f"validation_macro_top1={metric_text(row['validation_macro_top1'])} "
            f"test_macro_top1={metric_text(row['test_macro_top1'])} "
            f"test_overall_top1={metric_text(row['test_overall_top1'])} "
            f"test_top5={metric_text(row['test_top5'])} status=complete"
        )
    lines.append(f"[FINAL_AGGREGATES_BEGIN] profile_batch={profile_batch_size}")
    for aggregate in aggregates:
        macro = aggregate["test_macro_top1"]
        overall = aggregate["test_overall_top1"]
        top5 = aggregate["test_top5"]
        seeds = ",".join(str(seed) for seed in aggregate["completed_seeds"])
        lines.append(
            "[FINAL_AGGREGATE] "
            f"variant={variant_name(aggregate['method'], aggregate['fusion_ratio_lambda'])} "
            f"batch={aggregate['batch_size']} seeds={seeds or 'none'} "
            f"n={len(aggregate['completed_seeds'])} "
            f"test_macro_top1_mean={metric_text(macro['mean'])} "
            f"test_macro_top1_sample_sd={metric_text(macro['sample_standard_deviation'])} "
            f"test_overall_top1_mean={metric_text(overall['mean'])} "
            f"test_overall_top1_sample_sd={metric_text(overall['sample_standard_deviation'])} "
            f"test_top5_mean={metric_text(top5['mean'])} "
            f"test_top5_sample_sd={metric_text(top5['sample_standard_deviation'])} "
            f"complete={aggregate['complete']}"
        )
    lines.extend(
        (
            f"[FINAL_AGGREGATES_END] profile_batch={profile_batch_size}",
            f"[FINAL_RESULTS_END] profile_batch={profile_batch_size}",
            "=" * 96,
        )
    )
    return lines


def write_text(lines: list[str], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def contract_checks(
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    students = [payload for payload in payloads if payload["kind"] == "student"]
    teacher_payloads = [payload for payload in payloads if payload["kind"] == "teacher"]
    initial_by_seed = {
        str(seed): sorted(
            {
                str(payload["initial_student_state_sha256"])
                for payload in students
                if payload["seed"] == seed
            }
        )
        for seed in (1, 2, 3)
    }
    split_hashes = sorted(
        {
            str(payload["split_manifest"]["validation_image_ids_sha256"])
            for payload in payloads
        }
    )
    teacher_hashes = {
        str(payload["model_state_sha256"])
        for payload in teacher_payloads
    }
    teacher_hashes.update(
        str(payload["teacher_model_state_sha256"])
        for payload in students
        if payload["method"] != "vanilla"
    )
    complete_rows = [row for row in rows if row["status"] == "complete"]
    all_test_once = all(row["official_test_evaluations"] == 1 for row in complete_rows)
    no_test_selection = all(
        not row["official_test_used_for_training_or_selection"]
        for row in complete_rows
    )
    all_strict_reloaded = all(row["strict_checkpoint_reload"] for row in complete_rows)
    complete_matrix = len(students) == EXPECTED_STUDENTS and len(teacher_payloads) == 1
    same_initial_state = complete_matrix and all(
        len(initial_by_seed[str(seed)]) == 1 for seed in (1, 2, 3)
    )
    checks = {
        "complete_1_teacher_18_student_matrix": complete_matrix,
        "student_initial_state_hashes_by_seed": initial_by_seed,
        "same_student_initial_state_within_each_seed": same_initial_state,
        "validation_split_hashes": split_hashes,
        "same_validation_split": len(split_hashes) == 1,
        "teacher_model_state_hashes": sorted(teacher_hashes),
        "same_teacher_for_all_guided_students": len(teacher_hashes) == 1,
        "every_completed_checkpoint_strict_reloaded": all_strict_reloaded,
        "every_completed_model_tested_once": all_test_once,
        "official_test_used_for_training_or_selection": not no_test_selection,
    }
    checks["all_passed"] = all(
        (
            checks["complete_1_teacher_18_student_matrix"],
            checks["same_student_initial_state_within_each_seed"],
            checks["same_validation_split"],
            checks["same_teacher_for_all_guided_students"],
            checks["every_completed_checkpoint_strict_reloaded"],
            checks["every_completed_model_tested_once"],
            no_test_selection,
        )
    )
    return checks


def failure_detail(summary_path: Path, returncode: int) -> str:
    path = summary_path.parent / "failure.json"
    if not path.is_file():
        return f"subprocess_exit_{returncode}; inspect H200 console log"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        f"{payload.get('failure_kind')}:"
        f"{payload.get('error_type')}:"
        f"{payload.get('error')}"
    )


def run(args: argparse.Namespace) -> None:
    if args.num_workers < 0 or args.eval_batch_size <= 0:
        raise ValueError("Invalid worker count or evaluation batch size")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 full suite requires a CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_audit = prepare_and_audit_dataset(args.data_dir)
    save_json(dataset_audit, args.output_dir / "dataset_audit.json")
    tasks = build_full_tasks(args.output_dir, batch_size=args.batch_size)
    teacher_run_name = "pet_teacher_resnet56_32_b128_full_300ep_seed1"
    teacher_dir = args.output_dir / "teacher" / teacher_run_name
    teacher_summary = teacher_dir / "summary.json"
    teacher_checkpoint = teacher_dir / "teacher_best_validation.pt"
    specs: list[tuple[int, str, str, float | None, int, int, list[str], Path]] = [
        (
            0,
            "teacher",
            "teacher",
            None,
            128,
            1,
            teacher_command(args, teacher_run_name),
            teacher_summary,
        )
    ]
    specs.extend(
        (
            task.index,
            "student",
            task.method,
            task.fusion_ratio,
            task.batch_size,
            task.seed,
            student_command(task, args, teacher_checkpoint),
            task.summary_path,
        )
        for task in tasks
    )

    status_path = args.output_dir / "sequence_status.json"
    csv_path = args.output_dir / "classification_summary.csv"
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    suite_start = time.perf_counter()
    log("=" * 96)
    log(f"OXFORD-IIIT PET PHASE 1 — FULL CLASSIFICATION BATCH {args.batch_size}")
    log("=" * 96)
    log(
        "[TASK_COUNT] teacher=1 student=18 total=19 "
        f"matrix=6_variants_x_3_seeds batch={args.batch_size}"
    )
    log(
        "[SELECTION_POLICY] validation_macro_top1; tie=earlier_epoch; "
        "official_test=once_after_selection"
    )
    log(
        "[REPORTING_POLICY] all_six_variants_and_all_three_seeds_reported; "
        "no_posthoc_batch_or_lambda_selection"
    )

    for sequence_index, spec in enumerate(specs, start=1):
        index, kind, method, fusion_ratio, batch_size, seed, command, summary_path = spec
        label = method.upper()
        if fusion_ratio is not None:
            label += f"/lambda={fusion_ratio:g}"
        label += f"/batch={batch_size}/seed={seed}"
        log("-" * 96)
        log(f"[TASK_START] {sequence_index:02d}/{EXPECTED_TOTAL_TASKS} {kind}/{label}")
        save_json(
            {
                "status": "running",
                "batch_size": args.batch_size,
                "completed_or_attempted": len(rows),
                "total_tasks": EXPECTED_TOTAL_TASKS,
                "active_task": label,
                "rows": rows,
            },
            status_path,
        )
        attempt_start = time.perf_counter()
        result = subprocess.run(command, check=False)
        attempt_elapsed = time.perf_counter() - attempt_start
        if result.returncode == 0:
            try:
                payload = load_complete_summary(summary_path)
                row = completed_row(
                    index=index,
                    payload=payload,
                    attempt_elapsed_seconds=attempt_elapsed,
                )
                payloads.append(payload)
                log(
                    f"[TASK_DONE] {sequence_index:02d}/{EXPECTED_TOTAL_TASKS} {label} "
                    f"selected_epoch={row['selected_epoch']} "
                    f"test_macro={row['test_macro_top1']:.3f} "
                    f"elapsed={format_duration(attempt_elapsed)}"
                )
            except Exception as error:
                row = failed_row(
                    index=index,
                    kind=kind,
                    method=method,
                    fusion_ratio=fusion_ratio,
                    batch_size=batch_size,
                    seed=seed,
                    attempt_elapsed_seconds=attempt_elapsed,
                    failure=f"invalid_summary:{type(error).__name__}:{error}",
                )
                log(f"[TASK_FAILED] {label} {row['failure']}")
        else:
            row = failed_row(
                index=index,
                kind=kind,
                method=method,
                fusion_ratio=fusion_ratio,
                batch_size=batch_size,
                seed=seed,
                attempt_elapsed_seconds=attempt_elapsed,
                failure=failure_detail(summary_path, result.returncode),
            )
            log(f"[TASK_FAILED] {label} {row['failure']} continuing={kind != 'teacher'}")
        rows.append(row)
        write_csv(rows, csv_path)
        if kind == "teacher" and row["status"] != "complete":
            break

    contracts = contract_checks(rows, payloads)
    completed = sum(row["status"] == "complete" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    status = "complete" if contracts["all_passed"] else "complete_with_failures"
    aggregates = aggregate_students(rows)
    results_text_path = args.output_dir / "classification_results.txt"
    final_result_lines = build_final_result_lines(
        rows,
        aggregates,
        profile_batch_size=args.batch_size,
    )
    summary = {
        "status": status,
        "scientific_result": status == "complete",
        "batch_size": args.batch_size,
        "epochs": PLANNED_EPOCHS,
        "expected_tasks": {
            "teacher": 1,
            "student": EXPECTED_STUDENTS,
            "total": EXPECTED_TOTAL_TASKS,
        },
        "completed_tasks": completed,
        "failed_tasks": failed,
        "suite_elapsed_seconds": time.perf_counter() - suite_start,
        "dataset_audit": dataset_audit,
        "contracts": contracts,
        "teacher_model_state_sha256": (
            payloads[0].get("model_state_sha256")
            if payloads and payloads[0]["kind"] == "teacher"
            else None
        ),
        "rows": rows,
        "aggregates": aggregates,
        "final_results_text": str(results_text_path.resolve()),
        "cross_issue_contract": (
            "batch64_and_batch128_teacher_model_state_sha256_must_match_before_"
            "cross_profile_interpretation"
        ),
        "reporting_policy": (
            "report_both_batch_profiles_and_both_ibkd_lambdas_separately; "
            "do_not_select_from_test_results"
        ),
    }
    summary_path = args.output_dir / "classification_summary.json"
    save_json(summary, summary_path)
    write_text(final_result_lines, results_text_path)
    save_json(
        {
            "status": status,
            "batch_size": args.batch_size,
            "completed_or_attempted": len(rows),
            "completed_tasks": completed,
            "failed_tasks": failed,
            "total_tasks": EXPECTED_TOTAL_TASKS,
            "active_task": None,
            "contracts": contracts,
            "rows": rows,
        },
        status_path,
    )
    log(
        "[CONTRACT_CHECK] "
        f"same_student_initial_state={contracts['same_student_initial_state_within_each_seed']} "
        f"same_validation_split={contracts['same_validation_split']} "
        f"same_teacher={contracts['same_teacher_for_all_guided_students']} "
        f"test_once={contracts['every_completed_model_tested_once']} "
        f"test_used_for_selection={contracts['official_test_used_for_training_or_selection']}"
    )
    for line in final_result_lines:
        log(line)
    log(f"[FINAL_RESULTS_FILE] path={results_text_path.resolve()}")
    log(
        f"[SEQUENCE_DONE] status={status} completed={completed}/{EXPECTED_TOTAL_TASKS} "
        f"failed={failed} summary={summary_path.resolve()}"
    )
    if status != "complete":
        raise RuntimeError("Full classification suite did not satisfy every contract")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
