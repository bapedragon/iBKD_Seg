#!/usr/bin/env python3
"""Verify imported checkpoints and generate small Git-trackable Phase 1 reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ibkd_seg.phase1.models import ResNet56, create_student
from ibkd_seg.phase1.train_timing import state_dict_sha256


VARIANT_ORDER = (
    "vanilla",
    "kd",
    "lg",
    "alg",
    "ibkd_lambda0.25",
    "ibkd_lambda0.5",
)

TRACKED_SOURCE_ARTIFACTS = {
    "classification_summary.json": "h200_classification_summary.json",
    "classification_summary.csv": "h200_classification_summary.csv",
    "sequence_status.json": "h200_sequence_status.json",
    "dataset_audit.json": "h200_dataset_audit.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def copy_source_artifact(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if source.suffix == ".csv":
        temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        transformation = "line_endings_normalized_to_lf"
    else:
        shutil.copyfile(source, temporary)
        transformation = "none_byte_identical"
    temporary.replace(destination)
    return transformation


def variant(row: dict[str, Any]) -> str:
    fusion_ratio = row.get("fusion_ratio_lambda")
    if fusion_ratio is None:
        return row["method"]
    return f"{row['method']}_lambda{fusion_ratio:g}"


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "raw": values,
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def paired_difference(
    groups: dict[str, list[dict[str, Any]]],
    left: str,
    right: str,
) -> dict[str, Any]:
    left_by_seed = {row["seed"]: row["test_macro_top1"] for row in groups[left]}
    right_by_seed = {row["seed"]: row["test_macro_top1"] for row in groups[right]}
    raw = [left_by_seed[seed] - right_by_seed[seed] for seed in (1, 2, 3)]
    return {
        "contrast": f"{left}_minus_{right}",
        "metric": "test_macro_top1_percentage_points",
        "raw_by_seed": {str(seed): value for seed, value in zip((1, 2, 3), raw)},
        "mean": statistics.mean(raw),
        "sample_standard_deviation": statistics.stdev(raw),
    }


def checkpoint_audit(raw_dir: Path, suite: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for summary_path in sorted(raw_dir.glob("teacher/*/summary.json")) + sorted(
        raw_dir.glob("students/*/summary.json")
    ):
        run = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint_path = summary_path.parent / Path(run["checkpoint"]).name
        actual_file_hash = file_sha256(checkpoint_path)
        if actual_file_hash != run["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint file hash mismatch: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if run["kind"] == "teacher":
            model = ResNet56(num_classes=37)
            state = payload["model"]
            expected_state_hash = run["model_state_sha256"]
        else:
            model = create_student(num_classes=37, drop_path_rate=0.1)
            state = payload["student"]
            expected_state_hash = run["student_state_sha256"]
        model.load_state_dict(state, strict=True)
        actual_state_hash = state_dict_sha256(model)
        if actual_state_hash != expected_state_hash:
            raise RuntimeError(f"Model-state hash mismatch: {checkpoint_path}")
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in state.values()
            if tensor.is_floating_point()
        ):
            raise RuntimeError(f"Non-finite model state: {checkpoint_path}")
        entries.append(
            {
                "kind": run["kind"],
                "method": run.get("method", "teacher"),
                "fusion_ratio_lambda": run.get("fusion_ratio_lambda"),
                "batch_size": run["batch_size"],
                "seed": run["seed"],
                "selected_epoch": run["selected_epoch"],
                "path_under_ignored_raw_root": checkpoint_path.relative_to(raw_dir).as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": actual_file_hash,
                "model_state_sha256": actual_state_hash,
                "strict_load": True,
                "all_floating_tensors_finite": True,
            }
        )
    if len(entries) != 19:
        raise RuntimeError(f"Expected 19 checkpoints, audited {len(entries)}")
    teacher_hashes = {
        entry["model_state_sha256"] for entry in entries if entry["kind"] == "teacher"
    }
    if teacher_hashes != {suite["teacher_model_state_sha256"]}:
        raise RuntimeError("Teacher model-state hash disagrees with suite summary")
    return {
        "schema_version": 1,
        "status": "pass",
        "safe_torch_load": "weights_only_true",
        "checkpoint_count": 19,
        "teacher_checkpoint_count": 1,
        "student_checkpoint_count": 18,
        "all_checkpoint_file_hashes_match": True,
        "all_model_state_hashes_match": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    suite = json.loads((raw_dir / "classification_summary.json").read_text())
    artifact_import = json.loads((raw_dir / "artifact_manifest.json").read_text())
    if suite["status"] != "complete" or not suite["contracts"]["all_passed"]:
        raise RuntimeError("Cannot curate an incomplete or contract-failing suite")

    teacher = next(row for row in suite["rows"] if row["kind"] == "teacher")
    student_rows = [row for row in suite["rows"] if row["kind"] == "student"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in student_rows:
        groups[variant(row)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["seed"])
    if set(groups) != set(VARIANT_ORDER) or any(
        [row["seed"] for row in rows] != [1, 2, 3] for rows in groups.values()
    ):
        raise RuntimeError("Expected six complete three-seed student groups")

    controller_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    for summary_path in sorted(raw_dir.glob("students/*/summary.json")):
        run = json.loads(summary_path.read_text(encoding="utf-8"))
        controller = run.get("controller_final")
        if controller is not None:
            controller_by_run[(variant(run), int(run["seed"]))] = controller

    method_summaries: list[dict[str, Any]] = []
    for method in VARIANT_ORDER:
        rows = groups[method]
        method_summary: dict[str, Any] = {
            "variant": method,
            "method": rows[0]["method"],
            "fusion_ratio_lambda": rows[0]["fusion_ratio_lambda"],
            "selected_epoch_by_seed": {
                str(row["seed"]): row["selected_epoch"] for row in rows
            },
            "validation_macro_top1": summarize(
                [row["validation_macro_top1"] for row in rows]
            ),
            "test_macro_top1": summarize(
                [row["test_macro_top1"] for row in rows]
            ),
            "test_overall_top1": summarize(
                [row["test_overall_top1"] for row in rows]
            ),
            "test_top5": summarize([row["test_top5"] for row in rows]),
            "validation_minus_test_macro_top1": summarize(
                [
                    row["validation_macro_top1"] - row["test_macro_top1"]
                    for row in rows
                ]
            ),
        }
        controllers = [
            controller_by_run.get((method, int(row["seed"]))) for row in rows
        ]
        if any(controller is not None for controller in controllers):
            if not all(controller is not None for controller in controllers):
                raise RuntimeError(f"Incomplete controller metadata for {method}")
            complete_controllers = [
                controller for controller in controllers if controller is not None
            ]
            method_summary["guidance_controller"] = {
                "kind_by_seed": {
                    str(row["seed"]): controller["kind"]
                    for row, controller in zip(rows, complete_controllers)
                },
                "stop_epoch_by_seed": {
                    str(row["seed"]): controller.get("stop_epoch")
                    for row, controller in zip(rows, complete_controllers)
                },
                "warmup_epochs_by_seed": {
                    str(row["seed"]): controller.get("warmup_epochs")
                    for row, controller in zip(rows, complete_controllers)
                },
                "smoothing_window_by_seed": {
                    str(row["seed"]): controller.get("smoothing_window")
                    for row, controller in zip(rows, complete_controllers)
                },
                "threshold_by_seed": {
                    str(row["seed"]): controller.get("threshold")
                    for row, controller in zip(rows, complete_controllers)
                },
            }
        method_summaries.append(method_summary)

    contrasts = [
        paired_difference(groups, method, "vanilla")
        for method in VARIANT_ORDER
        if method != "vanilla"
    ]
    contrasts.extend(
        paired_difference(groups, method, baseline)
        for baseline in ("kd", "lg", "alg")
        for method in ("ibkd_lambda0.25", "ibkd_lambda0.5")
    )
    contrasts.append(
        paired_difference(groups, "ibkd_lambda0.5", "ibkd_lambda0.25")
    )

    audit = checkpoint_audit(raw_dir, suite)
    import_manifest_path = raw_dir / "artifact_manifest.json"
    source_archive = dict(artifact_import["source_archive"])
    retained_archive_path = (
        repository_root
        / "phase1/results/raw/archives/oxford_iiit_pet/full_classification_v1"
        / f"batch{suite['batch_size']}"
        / source_archive["canonical_filename"]
    )
    source_archive["retained_local_archive"] = retained_archive_path.is_file()
    if retained_archive_path.is_file():
        if retained_archive_path.stat().st_size != source_archive["bytes"]:
            raise RuntimeError("Retained source archive byte size mismatch")
        if file_sha256(retained_archive_path) != source_archive["sha256"]:
            raise RuntimeError("Retained source archive SHA-256 mismatch")
        source_archive["stored_at"] = retained_archive_path.relative_to(
            repository_root
        ).as_posix()
    tracked_source_artifacts = []
    for source_name, destination_name in TRACKED_SOURCE_ARTIFACTS.items():
        source_path = raw_dir / source_name
        destination_path = report_dir / destination_name
        transformation = copy_source_artifact(source_path, destination_path)
        tracked_source_artifacts.append(
            {
                "path": destination_name,
                "bytes": destination_path.stat().st_size,
                "sha256": file_sha256(destination_path),
                "copied_from": source_name,
                "transformation": transformation,
            }
        )
    report = {
        "schema_version": 1,
        "status": "classification_complete",
        "dataset": "Oxford-IIIT Pet",
        "task": "37_way_breed_classification",
        "batch_size": suite["batch_size"],
        "encoder_seeds": [1, 2, 3],
        "h200_issue_id": artifact_import["h200_issue_id"],
        "runtime_git_commit": artifact_import["runtime_git_commits"][0],
        "source_archive": source_archive,
        "ignored_raw_artifact_root": raw_dir.relative_to(repository_root).as_posix(),
        "raw_import_manifest_sha256": file_sha256(import_manifest_path),
        "tracked_source_artifacts": tracked_source_artifacts,
        "suite_elapsed_seconds": suite["suite_elapsed_seconds"],
        "protocol_contracts": suite["contracts"],
        "dataset_audit": suite["dataset_audit"],
        "teacher": {
            "selected_epoch": teacher["selected_epoch"],
            "validation_macro_top1": teacher["validation_macro_top1"],
            "test_macro_top1": teacher["test_macro_top1"],
            "test_overall_top1": teacher["test_overall_top1"],
            "test_top5": teacher["test_top5"],
            "model_state_sha256": suite["teacher_model_state_sha256"],
            "checkpoint_sha256": teacher["checkpoint_sha256"],
        },
        "methods": method_summaries,
        "classification_ranking_by_test_macro_top1": [
            summary["variant"]
            for summary in sorted(
                method_summaries,
                key=lambda summary: summary["test_macro_top1"]["mean"],
                reverse=True,
            )
        ],
        "paired_test_macro_top1_contrasts": contrasts,
        "checkpoint_audit": {
            key: value for key, value in audit.items() if key != "entries"
        },
        "interpretation_scope": {
            "classification_result_only": True,
            "frozen_segmentation_probe_required_for_spatial_conclusion": True,
            "no_spatial_information_conclusion_from_this_report": True,
            "posthoc_batch_or_lambda_selection_forbidden": True,
        },
    }

    per_seed = [
        {
            "variant": variant(row),
            "method": row["method"],
            "fusion_ratio_lambda": row["fusion_ratio_lambda"],
            "batch_size": row["batch_size"],
            "seed": row["seed"],
            "selected_epoch": row["selected_epoch"],
            "validation_macro_top1": row["validation_macro_top1"],
            "test_macro_top1": row["test_macro_top1"],
            "test_overall_top1": row["test_overall_top1"],
            "test_top5": row["test_top5"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        for method in VARIANT_ORDER
        for row in groups[method]
    ]
    save_json(report, report_dir / "summary.json")
    save_json(audit, report_dir / "checkpoint_manifest.json")
    write_csv(per_seed, report_dir / "per_seed.csv")
    print(
        f"[CURATION_DONE] batch={suite['batch_size']} methods={len(groups)} "
        f"checkpoints={audit['checkpoint_count']} report={report_dir}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
