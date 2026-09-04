#!/usr/bin/env python3
"""Run the 12-way Oxford-IIIT Pet Phase 1 H200 timing matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .data import save_json
from .timing_matrix import TimingTask, build_tasks


PLANNED_EPOCHS = 300
TIMING_EPOCHS = 2
POD_LIMIT_MINUTES = 600
SAFE_CHUNK_MINUTES = 540


def log(message: str = "") -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    return parser.parse_args()


def task_command(
    task: TimingTask,
    args: argparse.Namespace,
    teacher_checkpoint: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_timing",
        "--timing-run",
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
    ]
    if task.method != "vanilla":
        command.extend(("--teacher-checkpoint", str(teacher_checkpoint)))
    if task.fusion_ratio is not None:
        command.extend(("--fusion-ratio", str(task.fusion_ratio)))
    return command


def teacher_command(args: argparse.Namespace, run_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_timing",
        "--timing-run",
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
    ]


def load_complete_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Process produced no summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Summary is not complete: {path}")
    return payload


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "index",
        "kind",
        "method",
        "fusion_ratio_lambda",
        "batch_size",
        "avg_epoch_seconds",
        "estimated_300_epoch_seconds",
        "estimated_300_epoch_human",
        "peak_cuda_memory_bytes",
        "attempt_elapsed_seconds",
        "status",
        "failure",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def completed_row(
    *,
    index: int,
    kind: str,
    method: str,
    fusion_ratio: float | None,
    batch_size: int,
    payload: dict[str, Any],
    attempt_elapsed_seconds: float,
) -> dict[str, Any]:
    estimate = float(payload["estimated_planned_seconds"])
    epoch_rows = payload.get("epochs", [])
    peaks = [
        int(row["peak_cuda_memory_bytes"])
        for row in epoch_rows
        if row.get("peak_cuda_memory_bytes") is not None
    ]
    return {
        "index": index,
        "kind": kind,
        "method": method,
        "fusion_ratio_lambda": fusion_ratio,
        "batch_size": batch_size,
        "avg_epoch_seconds": float(payload["avg_epoch_seconds"]),
        "estimated_300_epoch_seconds": estimate,
        "estimated_300_epoch_human": format_duration(estimate),
        "peak_cuda_memory_bytes": max(peaks) if peaks else None,
        "attempt_elapsed_seconds": attempt_elapsed_seconds,
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
    returncode: int,
    attempt_elapsed_seconds: float,
    failure: str | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "kind": kind,
        "method": method,
        "fusion_ratio_lambda": fusion_ratio,
        "batch_size": batch_size,
        "avg_epoch_seconds": None,
        "estimated_300_epoch_seconds": None,
        "estimated_300_epoch_human": None,
        "peak_cuda_memory_bytes": None,
        "attempt_elapsed_seconds": attempt_elapsed_seconds,
        "status": "failed",
        "failure": failure or f"subprocess_exit_{returncode}; inspect H200 console log",
    }


def greedy_chunks(rows: list[dict[str, Any]], *, replicas: int) -> dict[str, Any]:
    capacity = SAFE_CHUNK_MINUTES * 60
    jobs: list[tuple[str, float]] = []
    infeasible: list[dict[str, Any]] = []
    for row in rows:
        if row["kind"] != "student" or row["status"] != "complete":
            continue
        seconds = float(row["estimated_300_epoch_seconds"])
        base = row["method"]
        if row["fusion_ratio_lambda"] is not None:
            base += f"_lambda{row['fusion_ratio_lambda']}"
        base += f"_b{row['batch_size']}"
        if seconds > capacity:
            infeasible.append(
                {"job": base, "seconds": seconds, "human": format_duration(seconds)}
            )
            continue
        for replica in range(1, replicas + 1):
            jobs.append((f"{base}_seed{replica}", seconds))
    jobs.sort(key=lambda item: item[1], reverse=True)
    chunks: list[dict[str, Any]] = []
    for name, seconds in jobs:
        destination = next(
            (
                chunk
                for chunk in chunks
                if float(chunk["estimated_seconds"]) + seconds <= capacity
            ),
            None,
        )
        if destination is None:
            destination = {"jobs": [], "estimated_seconds": 0.0}
            chunks.append(destination)
        destination["jobs"].append(name)
        destination["estimated_seconds"] += seconds
    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk"] = index
        chunk["estimated_human"] = format_duration(float(chunk["estimated_seconds"]))
    return {
        "replicas": replicas,
        "safety_capacity_minutes": SAFE_CHUNK_MINUTES,
        "chunks": chunks,
        "jobs_exceeding_safe_capacity": infeasible,
    }


def run(args: argparse.Namespace) -> None:
    if args.num_workers < 0 or args.eval_batch_size <= 0:
        raise ValueError("Invalid worker count or evaluation batch size")
    if not torch.cuda.is_available():
        raise RuntimeError("This timing suite requires a CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.output_dir)
    rows: list[dict[str, Any]] = []
    status_path = args.output_dir / "sequence_status.json"
    csv_path = args.output_dir / "timing_summary.csv"
    suite_start = time.perf_counter()

    teacher_run_name = "pet_teacher_resnet56_32_b128_timing_2ep_seed1"
    teacher_dir = args.output_dir / "teacher" / teacher_run_name
    teacher_summary_path = teacher_dir / "summary.json"
    teacher_checkpoint = teacher_dir / "timing_teacher_latest.pt"
    all_specs: list[tuple[int, str, str, float | None, int, list[str], Path]] = [
        (
            0,
            "teacher",
            "teacher",
            None,
            128,
            teacher_command(args, teacher_run_name),
            teacher_summary_path,
        )
    ]
    all_specs.extend(
        (
            task.index,
            "student",
            task.method,
            task.fusion_ratio,
            task.batch_size,
            task_command(task, args, teacher_checkpoint),
            task.summary_path,
        )
        for task in tasks
    )

    log("=" * 96)
    log("OXFORD-IIIT PET PHASE 1 — 12-WAY FULL-DATA TIMING SMOKE")
    log("=" * 96)
    log(
        "[TASK_COUNT] teacher=1 student=12 total=13 "
        "matrix=6_variants_x_2_batches"
    )
    log(
        "[MATRIX] variants=Vanilla,KD,LG,ALG,iBKD-lambda0.25,iBKD-lambda0.5 "
        "batches=64,128"
    )
    log(
        "[DATA_POLICY] train=2940 validation=740 test_accessed=False "
        "timing_epochs=2 planned_epochs=300"
    )
    log(
        "[DECISION_POLICY] timing_accuracy_must_not_select_batch_lambda_method_or_checkpoint"
    )

    total_specs = len(all_specs)
    for sequence_index, spec in enumerate(all_specs, start=1):
        index, kind, method, fusion_ratio, batch_size, command, summary_path = spec
        label = method.upper()
        if fusion_ratio is not None:
            label += f"/lambda={fusion_ratio:g}"
        label += f"/batch={batch_size}"
        log("-" * 96)
        log(f"[TASK_START] {sequence_index:02d}/{total_specs} {kind}/{label}")
        save_json(
            {
                "status": "running",
                "completed_or_attempted": len(rows),
                "total_tasks": total_specs,
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
                    kind=kind,
                    method=method,
                    fusion_ratio=fusion_ratio,
                    batch_size=batch_size,
                    payload=payload,
                    attempt_elapsed_seconds=attempt_elapsed,
                )
                log(
                    f"[TASK_DONE] {sequence_index:02d}/{total_specs} {label} "
                    f"avg_epoch={row['avg_epoch_seconds']:.2f}s "
                    f"estimated_300={row['estimated_300_epoch_human']}"
                )
            except Exception as error:
                row = failed_row(
                    index=index,
                    kind=kind,
                    method=method,
                    fusion_ratio=fusion_ratio,
                    batch_size=batch_size,
                    returncode=0,
                    attempt_elapsed_seconds=attempt_elapsed,
                )
                row["failure"] = f"invalid_summary:{type(error).__name__}:{error}"
                log(f"[TASK_FAILED] {label} {row['failure']}")
        else:
            failure_path = summary_path.parent / "failure.json"
            failure_detail = None
            if failure_path.is_file():
                failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
                failure_detail = (
                    f"{failure_payload.get('failure_kind')}:"
                    f"{failure_payload.get('error_type')}:"
                    f"{failure_payload.get('error')}"
                )
            row = failed_row(
                index=index,
                kind=kind,
                method=method,
                fusion_ratio=fusion_ratio,
                batch_size=batch_size,
                returncode=result.returncode,
                attempt_elapsed_seconds=attempt_elapsed,
                failure=failure_detail,
            )
            log(f"[TASK_FAILED] {label} {row['failure']} continuing=True")
        rows.append(row)
        write_csv(rows, csv_path)
        save_json(
            {
                "status": "running",
                "completed_or_attempted": len(rows),
                "total_tasks": total_specs,
                "active_task": None,
                "rows": rows,
            },
            status_path,
        )

    completed = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in rows if row["status"] == "failed"]
    student_completed = [row for row in completed if row["kind"] == "student"]
    teacher_completed = [row for row in completed if row["kind"] == "teacher"]
    one_seed_students = sum(
        float(row["estimated_300_epoch_seconds"]) for row in student_completed
    )
    teacher_seconds = sum(
        float(row["estimated_300_epoch_seconds"]) for row in teacher_completed
    )

    initial_hashes: set[str] = set()
    split_hashes: set[str] = set()
    completed_student_indices = {
        int(row["index"]) for row in student_completed
    }
    for task in tasks:
        if task.index in completed_student_indices and task.summary_path.is_file():
            payload = json.loads(task.summary_path.read_text(encoding="utf-8"))
            if payload.get("status") == "complete":
                initial_hashes.add(str(payload["initial_student_state_sha256"]))
                split_hashes.add(str(payload["split_manifest"]["validation_image_ids_sha256"]))
    contracts = {
        "completed_student_initial_hashes": sorted(initial_hashes),
        "same_student_initial_state": len(initial_hashes) <= 1,
        "completed_validation_split_hashes": sorted(split_hashes),
        "same_validation_split": len(split_hashes) <= 1,
    }
    overall_status = "complete" if not failed else "complete_with_failures"
    summary = {
        "status": overall_status,
        "purpose": "runtime_memory_and_job_partitioning_only",
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "timing_epochs": TIMING_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
        "requested_student_tasks": 12,
        "teacher_tasks": 1,
        "completed_tasks": len(completed),
        "failed_tasks": len(failed),
        "rows": rows,
        "contracts": contracts,
        "estimates": {
            "teacher_300_epoch_seconds": teacher_seconds,
            "students_one_seed_seconds": one_seed_students,
            "teacher_plus_students_one_seed_seconds": teacher_seconds + one_seed_students,
            "teacher_plus_students_three_seeds_seconds": teacher_seconds + 3 * one_seed_students,
            "teacher_plus_students_one_seed_human": format_duration(
                teacher_seconds + one_seed_students
            ),
            "teacher_plus_students_three_seeds_human": format_duration(
                teacher_seconds + 3 * one_seed_students
            ),
        },
        "recommended_partition_one_seed": greedy_chunks(rows, replicas=1),
        "recommended_partition_three_seeds": greedy_chunks(rows, replicas=3),
        "pod_limit_minutes": POD_LIMIT_MINUTES,
        "suite_elapsed_seconds": time.perf_counter() - suite_start,
    }
    summary_path = args.output_dir / "timing_summary.json"
    save_json(summary, summary_path)
    save_json(summary, status_path)

    log("=" * 96)
    log(
        f"[FINAL_TOTAL_ESTIMATE] teacher={format_duration(teacher_seconds)} "
        f"students_one_seed={format_duration(one_seed_students)} "
        f"teacher_plus_3seeds={format_duration(teacher_seconds + 3 * one_seed_students)}"
    )
    log(
        f"[CONTRACT_CHECK] same_student_initial_state={contracts['same_student_initial_state']} "
        f"same_validation_split={contracts['same_validation_split']} "
        "official_test_accessed=False"
    )
    log(
        f"[SEQUENCE_DONE] status={overall_status} completed={len(completed)}/{total_specs} "
        f"failed={len(failed)} summary={summary_path}"
    )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
