#!/usr/bin/env python3
"""Import the Phase 1 batch-128 ALG controller-warm-up-20 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
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


EXPERIMENT_ID = "oxford_iiit_pet_alg_controller_warmup20_posthoc_full_v1"
DIAGNOSTIC_ID = "oxford_iiit_pet_alg_controller_warmup20_posthoc_v1"
VARIANT = "alg_controller_warmup20"
EXPECTED_PROTOCOL_SHA256 = (
    "38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143"
)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_final(summary: dict[str, Any]) -> None:
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "posthoc": summary.get("posthoc_diagnostic") is True,
        "not_confirmatory": summary.get("confirmatory_main_result") is False,
        "canonical_retained": summary.get("canonical_phase1_result_replaced")
        is False,
        "experiment": summary.get("experiment_id") == EXPERIMENT_ID,
        "diagnostic": summary.get("diagnostic_id") == DIAGNOSTIC_ID,
        "single_change": summary.get("changed_field")
        == "alg_controller_warmup_epochs:0_to_20",
        "classification_count": summary.get("classification", {}).get(
            "official_test_evaluations"
        )
        == 3,
        "probe_count": summary.get("probe", {}).get(
            "official_test_evaluations"
        )
        == 15,
        "contracts": summary.get("contracts", {}).get("all_passed") is True,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("ALG warm-up-20 final summary failed: " + ", ".join(failures))


def _validate_classification(summary: dict[str, Any]) -> None:
    rows = summary.get("rows", [])
    checks = {
        "status": summary.get("status") == "complete",
        "batch": summary.get("batch_size") == 128,
        "epochs": summary.get("epochs") == 300,
        "completed": summary.get("completed_tasks") == 3,
        "failed": summary.get("failed_tasks") == 0,
        "row_count": len(rows) == 3,
        "seeds": {row.get("seed") for row in rows} == {1, 2, 3},
        "methods": all(row.get("method") == "alg" for row in rows),
        "test_once": all(row.get("official_test_evaluations") == 1 for row in rows),
        "test_not_selected": all(
            row.get("official_test_used_for_training_or_selection") is False
            for row in rows
        ),
        "contracts": summary.get("contracts", {}).get("all_passed") is True,
        "posthoc": summary.get("posthoc_diagnostic") is True,
        "not_confirmatory": summary.get("confirmatory_main_result") is False,
        "canonical_retained": summary.get("canonical_phase1_result_replaced")
        is False,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "ALG warm-up-20 classification summary failed: " + ", ".join(failures)
        )


def _probe_key(result: dict[str, Any]) -> tuple[int, int]:
    return int(result["encoder_seed"]), int(result["probe_seed"])


def _validate_probe(summary: dict[str, Any]) -> None:
    matrix = summary.get("matrix", {})
    policy = summary.get("probe_test_policy", {})
    results = summary.get("raw_results", [])
    expected_keys = {(encoder, probe) for encoder in (1, 2, 3) for probe in range(1, 6)}
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "posthoc": summary.get("posthoc_diagnostic") is True,
        "not_confirmatory": summary.get("confirmatory_main_result") is False,
        "canonical_retained": summary.get("canonical_phase1_result_replaced")
        is False,
        "experiment": summary.get("experiment_id") == EXPERIMENT_ID,
        "diagnostic": summary.get("diagnostic_id") == DIAGNOSTIC_ID,
        "protocol": summary.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "batch": summary.get("classification_batch_size") == 128,
        "variant": matrix.get("variant") == VARIANT,
        "encoder_seeds": matrix.get("encoder_seeds") == [1, 2, 3],
        "probe_seeds": matrix.get("probe_seeds") == [1, 2, 3, 4, 5],
        "learning_rates": matrix.get("learning_rates") == [0.01, 0.03, 0.1],
        "epochs": matrix.get("epochs_per_lr_candidate") == 100,
        "candidates": matrix.get("lr_candidate_count") == 45,
        "selections": matrix.get("selected_probe_count") == 15,
        "raw_count": len(results) == 15,
        "raw_keys": {_probe_key(result) for result in results} == expected_keys,
        "test_once": all(result.get("official_test_evaluations") == 1 for result in results),
        "test_count": policy.get("official_test_evaluations") == 15,
        "test_expected": policy.get("expected_official_test_evaluations") == 15,
        "test_not_selected": policy.get(
            "official_test_used_for_training_or_selection"
        )
        is False,
        "contracts": summary.get("contracts", {}).get("all_passed") is True,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("ALG warm-up-20 probe summary failed: " + ", ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--issue-id", default="712")
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

    imported: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as source_zip:
        infos = source_zip.infolist()
        validate_members(infos)
        if args.verify_all_crc:
            corrupt = source_zip.testzip()
            if corrupt is not None:
                raise RuntimeError(f"CRC failure in archive member: {corrupt}")
        names = {info.filename for info in infos}
        suffix = "/phase1_pet_alg_warmup20_b128_full_v1/alg_warmup20_full_summary.json"
        final_members = [name for name in names if name.endswith(suffix)]
        if len(final_members) != 1:
            raise RuntimeError(f"Expected one final summary, found {final_members}")
        final_member = final_members[0]
        suite_root = final_member.rsplit("/", 1)[0] + "/"
        issue_root = suite_root[: -len("phase1_pet_alg_warmup20_b128_full_v1/")]

        final_payload = source_zip.read(final_member)
        final_summary = json.loads(final_payload)
        _validate_final(final_summary)
        classification_member = suite_root + "classification/classification_summary.json"
        probe_member = suite_root + "probe/probe_summary.json"
        classification_payload = source_zip.read(classification_member)
        probe_payload = source_zip.read(probe_member)
        classification = json.loads(classification_payload)
        probe = json.loads(probe_payload)
        _validate_classification(classification)
        _validate_probe(probe)
        if _bytes_sha256(classification_payload) != final_summary["classification"][
            "summary_sha256"
        ]:
            raise RuntimeError("classification summary hash differs from final summary")
        if _bytes_sha256(probe_payload) != final_summary["probe"]["summary_sha256"]:
            raise RuntimeError("probe summary hash differs from final summary")

        expected_checkpoint_hashes: dict[str, str] = {}
        runtime_commits: set[str] = set()
        for row in classification["rows"]:
            seed = int(row["seed"])
            run_root = (
                suite_root
                + "classification/students/"
                + f"pet_alg_controller_warmup20_b128_full_300ep_seed{seed}/"
            )
            summary_member = run_root + "summary.json"
            checkpoint_member = run_root + "student_best_validation.pt"
            individual = json.loads(source_zip.read(summary_member))
            if individual.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
                raise RuntimeError(f"classification checkpoint hash mismatch for seed {seed}")
            runtime_commits.add(str(individual["runtime"]["git_commit"]))
            expected_checkpoint_hashes[checkpoint_member] = str(row["checkpoint_sha256"])

        for result in probe["raw_results"]:
            member = suite_root + "probe/" + str(result["probe_artifact"]["relative_path"])
            expected_checkpoint_hashes[member] = str(result["probe_artifact"]["sha256"])
        expected_checkpoint_hashes[
            suite_root + "classification/reference_teacher/teacher_best_validation.pt"
        ] = str(classification["reference_audit"]["teacher_checkpoint_sha256"])

        selected_members = sorted(
            name
            for name in names
            if name.startswith(suite_root)
            and not name.endswith("/")
            and not name.endswith("/training_status.json")
            and not name.endswith("/validation_split.json")
        )
        expected_required = {
            final_member,
            classification_member,
            probe_member,
            suite_root + "sequence_status.json",
            suite_root + "dataset_audit.json",
            suite_root + "probe/selection_complete_before_test.json",
            *expected_checkpoint_hashes,
        }
        missing = expected_required - set(selected_members)
        if missing:
            raise RuntimeError(f"Missing required diagnostic artifacts: {sorted(missing)}")

        log_candidates = sorted(
            name
            for name in names
            if name.startswith(issue_root)
            and "/phase1_pet_alg_warmup20_" not in name
            and name.endswith("_result.txt")
        )
        if len(log_candidates) != 1:
            raise RuntimeError(f"Expected one H200 result log, found {log_candidates}")

        for member in selected_members:
            relative = Path(PurePosixPath(member).relative_to(suite_root))
            with source_zip.open(member) as source:
                size, actual_sha256 = stream_to_file(source, output_dir / relative)
            expected_sha256 = expected_checkpoint_hashes.get(member)
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise RuntimeError(f"checkpoint SHA-256 mismatch: {relative}")
            imported.append(
                {
                    "path": relative.as_posix(),
                    "source_member": member,
                    "bytes": size,
                    "sha256": actual_sha256,
                    "checkpoint": expected_sha256 is not None,
                }
            )

        log_relative = Path(f"h200_issue_{args.issue_id}.log")
        with source_zip.open(log_candidates[0]) as source:
            size, actual_sha256 = stream_to_file(source, output_dir / log_relative)
        imported.append(
            {
                "path": log_relative.as_posix(),
                "source_member": log_candidates[0],
                "bytes": size,
                "sha256": actual_sha256,
                "checkpoint": False,
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "pass",
        "dataset": "Oxford-IIIT Pet",
        "experiment": "phase1_alg_controller_warmup20_posthoc_full_v1",
        "classification_batch_size": 128,
        "h200_issue_id": str(args.issue_id),
        "source_archive": {
            "original_filename": archive.name,
            "canonical_bundle_filename": (
                args.canonical_bundle_filename
                or "phase1_pet_b128_alg_controller_warmup20_issue712_v1.zip"
            ),
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
            "all_member_crc_verified": args.verify_all_crc,
        },
        "runtime_git_commits": sorted(runtime_commits),
        "import_policy": {
            "kept": [
                "three ALG warm-up-20 classification checkpoints and summaries",
                "reference teacher checkpoint and audit",
                "fifteen selected probe checkpoints and full selection histories",
                "classification/probe summaries and raw CSV files",
                "pre-test selection record and sequence status",
                "fixed qualitative panels and prediction masks",
                "H200 console log",
            ],
            "omitted": [
                "duplicate training_status.json files",
                "duplicate validation_split.json files",
                "unrelated issue 710 output in the shared ZIP",
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
        f"[ALG_W20_IMPORT_DONE] issue={args.issue_id} files={len(imported)} "
        f"bytes={manifest['imported_bytes_excluding_this_manifest']} "
        f"output={output_dir}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
