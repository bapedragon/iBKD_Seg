#!/usr/bin/env python3
"""Import only reusable Phase 1 artifacts from an H200 result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


TOP_LEVEL_FILES = (
    "classification_summary.json",
    "classification_summary.csv",
    "sequence_status.json",
    "dataset_audit.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_to_file(source: BinaryIO, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    temporary.replace(destination)
    return size, digest.hexdigest()


def save_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_members(infos: list[zipfile.ZipInfo]) -> None:
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise RuntimeError("Archive contains duplicate member names")
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe archive path: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError(f"Archive symlink is not allowed: {info.filename}")


def result_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload["kind"],
        payload.get("method", "teacher"),
        payload.get("fusion_ratio_lambda"),
        payload["batch_size"],
        payload["seed"],
    )


def validate_suite(summary: dict[str, Any], batch_size: int) -> None:
    expected = {"teacher": 1, "student": 18, "total": 19}
    if summary.get("status") != "complete" or not summary.get("scientific_result"):
        raise RuntimeError("Classification suite is not marked complete/scientific")
    if summary.get("batch_size") != batch_size:
        raise RuntimeError("Requested batch does not match archive summary")
    if summary.get("expected_tasks") != expected:
        raise RuntimeError("Unexpected task matrix in suite summary")
    if summary.get("completed_tasks") != 19 or summary.get("failed_tasks") != 0:
        raise RuntimeError("Archive does not contain a successful 19/19 suite")
    contracts = summary.get("contracts", {})
    if not contracts.get("all_passed"):
        raise RuntimeError("Suite contract checks did not all pass")
    if contracts.get("official_test_used_for_training_or_selection"):
        raise RuntimeError("Official test was used for training or selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(64, 128), required=True)
    parser.add_argument("--issue-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-all-crc", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    archive = args.archive.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_sha256 = file_sha256(archive)
    imported: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as source_zip:
        infos = source_zip.infolist()
        validate_members(infos)
        if args.verify_all_crc:
            corrupt = source_zip.testzip()
            if corrupt is not None:
                raise RuntimeError(f"CRC failure in archive member: {corrupt}")
        names = {info.filename for info in infos}
        suffix = f"/phase1_pet_full_b{args.batch_size}_v1/classification_summary.json"
        summary_members = [name for name in names if name.endswith(suffix)]
        if len(summary_members) != 1:
            raise RuntimeError(f"Expected one suite summary, found {summary_members}")
        summary_member = summary_members[0]
        suite_root = summary_member.rsplit("/", 1)[0] + "/"
        issue_root = suite_root[: -len(f"phase1_pet_full_b{args.batch_size}_v1/")]
        suite_summary = json.loads(source_zip.read(summary_member))
        validate_suite(suite_summary, args.batch_size)

        top_rows = {result_key(row): row for row in suite_summary["rows"]}
        if len(top_rows) != 19:
            raise RuntimeError("Suite summary rows are not 19 unique tasks")

        selected: list[tuple[str, Path, str | None]] = []
        for filename in TOP_LEVEL_FILES:
            member = suite_root + filename
            if member not in names:
                raise RuntimeError(f"Missing required artifact: {member}")
            selected.append((member, Path(filename), None))

        log_candidates = sorted(
            name
            for name in names
            if name.startswith(issue_root)
            and "/phase1_pet_full_" not in name
            and name.endswith("_result.txt")
        )
        if len(log_candidates) != 1:
            raise RuntimeError(f"Expected one H200 result log, found {log_candidates}")
        selected.append(
            (log_candidates[0], Path(f"h200_issue_{args.issue_id}.log"), None)
        )

        individual_summaries = sorted(
            name
            for name in names
            if name.startswith(suite_root)
            and ("/students/" in name or "/teacher/" in name)
            and name.endswith("/summary.json")
        )
        if len(individual_summaries) != 19:
            raise RuntimeError(
                f"Expected 19 individual summaries, found {len(individual_summaries)}"
            )
        seen_keys: set[tuple[Any, ...]] = set()
        runtime_commits: set[str] = set()
        for member in individual_summaries:
            payload = json.loads(source_zip.read(member))
            key = result_key(payload)
            if key in seen_keys or key not in top_rows:
                raise RuntimeError(f"Unexpected or duplicate individual result: {key}")
            seen_keys.add(key)
            top_row = top_rows[key]
            if payload.get("checkpoint_sha256") != top_row.get("checkpoint_sha256"):
                raise RuntimeError(f"Checkpoint hash disagrees with suite row: {key}")
            if payload.get("official_test_evaluations") != 1:
                raise RuntimeError(f"Test evaluation count is not one: {key}")
            if payload.get("official_test_used_for_training_or_selection"):
                raise RuntimeError(f"Test leakage flag is set: {key}")
            runtime_commits.add(str(payload["runtime"]["git_commit"]))
            relative_summary = Path(PurePosixPath(member).relative_to(suite_root))
            selected.append((member, relative_summary, None))
            checkpoint_name = Path(payload["checkpoint"]).name
            checkpoint_member = member.rsplit("/", 1)[0] + "/" + checkpoint_name
            if checkpoint_member not in names:
                raise RuntimeError(f"Missing checkpoint for {key}: {checkpoint_member}")
            selected.append(
                (
                    checkpoint_member,
                    relative_summary.parent / checkpoint_name,
                    payload["checkpoint_sha256"],
                )
            )
        if seen_keys != set(top_rows):
            raise RuntimeError("Individual summaries do not cover every suite row")
        if len(runtime_commits) != 1:
            raise RuntimeError(f"Multiple runtime commits found: {runtime_commits}")

        for member, relative_path, expected_sha256 in selected:
            destination = output_dir / relative_path
            with source_zip.open(member) as source:
                size, actual_sha256 = stream_to_file(source, destination)
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Checkpoint SHA-256 mismatch after import: {relative_path}"
                )
            imported.append(
                {
                    "path": relative_path.as_posix(),
                    "source_member": member,
                    "bytes": size,
                    "sha256": actual_sha256,
                    "checkpoint": expected_sha256 is not None,
                }
            )

    manifest = {
        "schema_version": 1,
        "status": "pass",
        "dataset": "Oxford-IIIT Pet",
        "experiment": "phase1_full_classification",
        "batch_size": args.batch_size,
        "h200_issue_id": args.issue_id,
        "source_archive": {
            "original_filename": archive.name,
            "canonical_filename": (
                f"phase1_pet_b{args.batch_size}_full_classification_v1_"
                f"h200_issue{args.issue_id}.zip"
            ),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "all_member_crc_verified": args.verify_all_crc,
        },
        "runtime_git_commits": sorted(runtime_commits),
        "import_policy": {
            "kept": [
                "suite summaries",
                "individual summaries",
                "teacher checkpoint",
                "18 student checkpoints",
                "H200 console log",
            ],
            "omitted": [
                "downloaded Oxford-IIIT Pet dataset",
                "duplicate training_status.json files",
                "duplicate validation_split.json files",
                "__MACOSX metadata",
            ],
        },
        "imported_file_count_excluding_this_manifest": len(imported),
        "imported_bytes_excluding_this_manifest": sum(item["bytes"] for item in imported),
        "files": sorted(imported, key=lambda item: item["path"]),
    }
    save_json(manifest, output_dir / "artifact_manifest.json")
    print(
        f"[IMPORT_DONE] batch={args.batch_size} issue={args.issue_id} "
        f"files={len(imported)} bytes={manifest['imported_bytes_excluding_this_manifest']} "
        f"output={output_dir}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
