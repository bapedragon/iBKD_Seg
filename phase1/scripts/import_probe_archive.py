#!/usr/bin/env python3
"""Import a complete Phase 1 frozen-probe result from an H200 ZIP bundle."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from import_classification_archive import (
    file_sha256,
    save_json,
    stream_to_file,
    validate_members,
)


EXPECTED_PROTOCOL_SHA256 = (
    "38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143"
)
EXPECTED_VARIANTS = {
    "vanilla",
    "kd",
    "lg",
    "alg",
    "ibkd_lambda_0.25",
    "ibkd_lambda_0.5",
}


def _validate_summary(summary: dict[str, Any], batch_size: int) -> None:
    matrix = summary.get("matrix", {})
    test_policy = summary.get("test_policy", {})
    contracts = summary.get("contracts", {})
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "batch": summary.get("classification_batch_size") == batch_size,
        "protocol": summary.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "variants": set(matrix.get("variants", [])) == EXPECTED_VARIANTS,
        "encoder_seeds": matrix.get("encoder_seeds") == [1, 2, 3],
        "probe_seeds": matrix.get("probe_seeds") == [1, 2, 3, 4, 5],
        "lr_grid": matrix.get("learning_rates") == [0.01, 0.03, 0.1],
        "epochs": matrix.get("epochs_per_lr_candidate") == 100,
        "candidate_count": matrix.get("lr_candidate_count") == 270,
        "selected_count": matrix.get("selected_probe_count") == 90,
        "test_count": test_policy.get("official_test_evaluations") == 90,
        "test_expected": test_policy.get("expected_official_test_evaluations")
        == 90,
        "test_not_selected": test_policy.get(
            "official_test_used_for_training_or_selection"
        )
        is False,
        "contracts": contracts.get("all_passed") is True,
        "raw_count": len(summary.get("raw_results", [])) == 90,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("probe suite validation failed: " + ", ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(64, 128), required=True)
    parser.add_argument("--issue-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-bundle-filename")
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
        suffix = (
            f"/phase1_pet_probe_b{args.batch_size}_full_v1/probe_summary.json"
        )
        summary_members = [name for name in names if name.endswith(suffix)]
        if len(summary_members) != 1:
            raise RuntimeError(f"Expected one probe summary, found {summary_members}")
        summary_member = summary_members[0]
        suite_root = summary_member.rsplit("/", 1)[0] + "/"
        issue_root = suite_root[: -len(f"phase1_pet_probe_b{args.batch_size}_full_v1/")]
        summary = json.loads(source_zip.read(summary_member))
        _validate_summary(summary, args.batch_size)

        selected_members = sorted(
            name
            for name in names
            if name.startswith(suite_root) and not name.endswith("/")
        )
        expected_probe_paths = {
            suite_root + str(result["probe_artifact"]["relative_path"])
            for result in summary["raw_results"]
        }
        observed_probe_paths = {
            name for name in selected_members if "/probes/" in name
        }
        if observed_probe_paths != expected_probe_paths:
            raise RuntimeError("probe checkpoint members do not match final summary")
        if len(observed_probe_paths) != 90:
            raise RuntimeError("expected exactly 90 selected probe checkpoints")

        log_candidates = sorted(
            name
            for name in names
            if name.startswith(issue_root)
            and "/phase1_pet_probe_" not in name
            and name.endswith("_result.txt")
        )
        if len(log_candidates) != 1:
            raise RuntimeError(f"Expected one H200 result log, found {log_candidates}")

        for member in selected_members:
            relative = Path(PurePosixPath(member).relative_to(suite_root))
            expected_sha256 = None
            if member in expected_probe_paths:
                result = next(
                    result
                    for result in summary["raw_results"]
                    if suite_root + result["probe_artifact"]["relative_path"] == member
                )
                expected_sha256 = result["probe_artifact"]["sha256"]
            with source_zip.open(member) as source:
                size, actual_sha256 = stream_to_file(source, output_dir / relative)
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise RuntimeError(f"probe checkpoint SHA-256 mismatch: {relative}")
            imported.append(
                {
                    "path": relative.as_posix(),
                    "source_member": member,
                    "bytes": size,
                    "sha256": actual_sha256,
                    "probe_checkpoint": expected_sha256 is not None,
                }
            )

        log_relative = Path(f"h200_issue_{args.issue_id}.log")
        with source_zip.open(log_candidates[0]) as source:
            log_size, log_sha256 = stream_to_file(source, output_dir / log_relative)
        imported.append(
            {
                "path": log_relative.as_posix(),
                "source_member": log_candidates[0],
                "bytes": log_size,
                "sha256": log_sha256,
                "probe_checkpoint": False,
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "pass",
        "dataset": "Oxford-IIIT Pet",
        "experiment": "phase1_frozen_spatial_probe",
        "classification_batch_size": args.batch_size,
        "h200_issue_id": args.issue_id,
        "source_archive": {
            "original_filename": archive.name,
            "canonical_bundle_filename": (
                args.canonical_bundle_filename
                or f"phase1_pet_b{args.batch_size}_frozen_probe_v1_"
                f"h200_issue{args.issue_id}.zip"
            ),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "all_member_crc_verified": args.verify_all_crc,
        },
        "runtime_git_commit": summary["runtime"]["git_commit"],
        "import_policy": {
            "kept": [
                "full probe summary and raw CSV",
                "pre-test selection record and final sequence status",
                "90 selected probe checkpoints",
                "fixed qualitative panels and masks",
                "H200 console log",
            ],
            "omitted": [
                "unrelated batch-128 classification output in the shared ZIP",
                "__MACOSX metadata",
            ],
        },
        "imported_file_count_excluding_this_manifest": len(imported),
        "imported_bytes_excluding_this_manifest": sum(
            item["bytes"] for item in imported
        ),
        "files": sorted(imported, key=lambda item: item["path"]),
    }
    save_json(manifest, output_dir / "artifact_manifest.json")
    print(
        f"[PROBE_IMPORT_DONE] batch={args.batch_size} issue={args.issue_id} "
        f"files={len(imported)} "
        f"bytes={manifest['imported_bytes_excluding_this_manifest']} "
        f"output={output_dir}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
