#!/usr/bin/env python3
"""Safely import the reusable artifacts from a Phase 1 CUB H200 ZIP.

Archive contents are treated only as untrusted data.  This importer rejects
unsafe paths and symlinks, verifies every CRC, validates the expected completed
experiment contract, and copies only the explicitly selected artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


KINDS = (
    "resnet56-v2-guided",
    "resnet50-v3-teacher",
    "resnet50-v4-guided-seed1",
    "resnet50-v5-guided-seeds2-3",
    "resnet50-direct-spatial-v2-seed1",
    "resnet50-direct-spatial-v2-seeds2-3",
)
V2_CONFIG_SHA256 = (
    "0cf751c28168872a4108274644f80dadc7466d5c1210995e7da3abfc0737e575"
)
V3_CONFIG_SHA256 = (
    "e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc"
)
V4_CONFIG_SHA256 = (
    "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633"
)
V5_CONFIG_SHA256 = (
    "f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9"
)
DIRECT_V2_CONFIG_SHA256 = (
    "90f7dc92b7e1ad27b6a4a4b68e91bb5e72ea021389304d87dda5950fac8e6017"
)
DIRECT_SEED23_V2_CONFIG_SHA256 = (
    "54980771cf910543a3aba24c0a5ff86de6a0dce34866c662025409ab3abd0691"
)
R50_TEACHER_CHECKPOINT_SHA256 = (
    "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
)
R50_TEACHER_STATE_SHA256 = (
    "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_to_file(source: BinaryIO, destination: Path) -> tuple[int, str]:
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


def _save_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_members(infos: list[zipfile.ZipInfo]) -> None:
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise RuntimeError("archive contains duplicate member names")
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive path: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError(f"archive symlink is not allowed: {info.filename}")
        if info.flag_bits & 0x1:
            raise RuntimeError(f"encrypted archive member is not allowed: {info.filename}")


def _unique_suffix(names: set[str], suffix: str) -> str:
    matches = sorted(name for name in names if name.endswith(suffix))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix}, found {matches}")
    return matches[0]


def _unique_root_suffix(names: set[str], root: str, suffix: str) -> str:
    matches = sorted(
        name for name in names if name.startswith(root) and name.endswith(suffix)
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix} under {root}, found {matches}")
    return matches[0]


def _read_json(source_zip: zipfile.ZipFile, member: str) -> dict[str, Any]:
    value = json.loads(source_zip.read(member))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {member}")
    return value


def _validate_v2(summary: dict[str, Any]) -> None:
    expected_counts = {
        "classification_students": 9,
        "probe_lr_candidates": 135,
        "probe_official_test_evaluations": 45,
        "retained_checkpoints": 55,
        "selected_probes": 45,
        "teacher": 1,
    }
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "protocol": summary.get("protocol_id")
        == "cub200_phase1_b128_frozen_spatial_probe_full_v2",
        "shard": summary.get("shard") == "guided",
        "config": summary.get("config_sha256") == V2_CONFIG_SHA256,
        "variants": summary.get("variants")
        == ["alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5"],
        "counts": summary.get("counts") == expected_counts,
        "contracts": summary.get("contracts", {}).get("all_passed") is True
        and all(summary.get("contracts", {}).values()),
        "aggregates": len(summary.get("probe_aggregates", [])) == 3,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid completed CUB v2 guided result: " + ", ".join(failures))


def _validate_v3_teacher(summary: dict[str, Any]) -> None:
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "protocol": summary.get("protocol_id")
        == "cub200_phase1_resnet50_224_scratch_b128_full_v3",
        "dataset": summary.get("dataset") == "CUB-200-2011",
        "architecture": summary.get("architecture") == "torchvision_resnet50",
        "scratch": summary.get("initialization") == "scratch"
        and summary.get("external_pretraining") is False,
        "input": summary.get("input_size") == 224,
        "training": summary.get("epochs") == 200
        and summary.get("seed") == 1
        and summary.get("train_batch_size") == 128,
        "config": summary.get("full_config_sha256") == V3_CONFIG_SHA256,
        "test_once": summary.get("official_test_evaluations") == 1
        and summary.get("official_test_used_for_training_or_selection") is False,
        "strict_reload": summary.get("selected_checkpoint_strict_reloaded") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid completed CUB v3 teacher result: " + ", ".join(failures))


def _validate_v4_guided_seed1(summary: dict[str, Any]) -> None:
    expected_counts = {
        "classification_official_test_evaluations": 8,
        "classification_students": 8,
        "new_checkpoints": 48,
        "probe_lr_candidates": 120,
        "probe_official_test_evaluations": 40,
        "selected_probes": 40,
        "teacher_reused": 1,
    }
    expected_batch_counts = {
        "classification_students": 4,
        "new_checkpoints": 24,
        "probe_lr_candidates": 60,
        "probe_official_test_evaluations": 20,
        "selected_probes": 20,
    }
    batches = summary.get("batches", [])
    by_batch = {
        batch.get("batch_size"): batch
        for batch in batches
        if isinstance(batch, dict)
    }
    teacher = summary.get("teacher", {})
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "protocol": summary.get("protocol_id")
        == "cub200_phase1_r50_224_guided_b128_b64_seed1_full_v4",
        "config": summary.get("config_sha256") == V4_CONFIG_SHA256,
        "batches": summary.get("batch_order") == [128, 64]
        and set(by_batch) == {64, 128},
        "variants": summary.get("variants")
        == ["lg", "alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5"],
        "seeds": summary.get("encoder_seeds") == [1],
        "counts": summary.get("counts") == expected_counts,
        "initialization": summary.get("same_initial_student_state_across_variants_and_batches")
        is True,
        "partial": summary.get("final_confirmatory_matrix_complete") is False,
        "roles": by_batch.get(128, {}).get("batch_profile_role")
        == "locked_v3_partial_cell"
        and by_batch.get(128, {}).get("confirmatory_main_result") is True
        and by_batch.get(64, {}).get("batch_profile_role") == "batch64_sensitivity"
        and by_batch.get(64, {}).get("confirmatory_main_result") is False,
        "batch_counts": all(
            by_batch.get(batch_size, {}).get("counts") == expected_batch_counts
            for batch_size in (128, 64)
        ),
        "batch_selection_contracts": all(
            all(by_batch.get(batch_size, {}).get("selection_contracts", {}).values())
            for batch_size in (128, 64)
        ),
        "teacher": teacher.get("reused") is True
        and teacher.get("source_h200_issue") == 722
        and teacher.get("checkpoint_sha256") == R50_TEACHER_CHECKPOINT_SHA256
        and teacher.get("model_state_sha256") == R50_TEACHER_STATE_SHA256,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid completed CUB v4 guided seed-1 result: " + ", ".join(failures)
        )


def _validate_v5_guided_seeds2_3(summary: dict[str, Any]) -> None:
    expected_counts = {
        "classification_official_test_evaluations": 8,
        "classification_students": 8,
        "new_checkpoints": 48,
        "probe_lr_candidates": 120,
        "probe_official_test_evaluations": 40,
        "selected_probes": 40,
        "teacher_reused": 1,
    }
    expected_profile_counts = {
        "classification_students": 4,
        "new_checkpoints": 24,
        "probe_lr_candidates": 60,
        "probe_official_test_evaluations": 20,
        "selected_probes": 20,
    }
    profiles = summary.get("profiles", [])
    by_seed = {
        profile.get("encoder_seed"): profile
        for profile in profiles
        if isinstance(profile, dict)
    }
    teacher = summary.get("teacher", {})
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "protocol": summary.get("protocol_id")
        == "cub200_phase1_r50_224_guided_b128_s23_b64_s2_full_v5",
        "config": summary.get("config_sha256") == V5_CONFIG_SHA256,
        "partition": summary.get("partition") == "batch128_encoder_seeds_2_3",
        "seeds": summary.get("encoder_seeds") == [2, 3] and set(by_seed) == {2, 3},
        "variants": summary.get("variants")
        == ["lg", "alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5"],
        "counts": summary.get("counts") == expected_counts,
        "profiles": all(
            by_seed.get(seed, {}).get("batch_size") == 128
            and by_seed.get(seed, {}).get("batch_profile_role")
            == "locked_v3_confirmatory_continuation"
            and by_seed.get(seed, {}).get("confirmatory_main_result") is True
            and by_seed.get(seed, {}).get("counts") == expected_profile_counts
            and all(by_seed.get(seed, {}).get("selection_contracts", {}).values())
            for seed in (2, 3)
        ),
        "teacher": teacher.get("reused") is True
        and teacher.get("source_h200_issue") == 722
        and teacher.get("checkpoint_sha256") == R50_TEACHER_CHECKPOINT_SHA256
        and teacher.get("model_state_sha256") == R50_TEACHER_STATE_SHA256,
        "partial": summary.get("final_confirmatory_matrix_complete") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid completed CUB v5 guided seed-2/3 result: " + ", ".join(failures)
        )


def _validate_direct_spatial_v2(summary: dict[str, Any]) -> None:
    gate = summary.get("completion_gate", {})
    expected_gate = {
        "attention_metric_rows": 4,
        "attention_official_test_evaluations": 4,
        "checkpoint_strict_loads": 4,
        "official_test_evaluations": 24,
        "part_probe_lr_candidates": 60,
        "part_probe_official_test_evaluations": 20,
        "part_probe_validation_selections": 20,
        "qualitative_pngs": 32,
        "spatial_cka_values": 48,
    }
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "protocol": summary.get("protocol_id")
        == "cub200_phase1_r50_224_b128_seed1_direct_spatial_full_v2",
        "config": summary.get("config_sha256") == DIRECT_V2_CONFIG_SHA256,
        "batch_seed": summary.get("student_batch_size") == 128
        and summary.get("encoder_seed") == 1
        and summary.get("independent_encoder_seed_n") == 1,
        "variants": summary.get("variants")
        == ["lg", "alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5"],
        "scope": summary.get("final_encoder_seed_inference") is False
        and summary.get("seed2_3_settings_remain_locked_regardless_of_seed1_results")
        is True,
        "gate": gate == expected_gate,
        "part_rows": len(summary.get("part_probe_aggregates", [])) == 4,
        "cka_rows": len(summary.get("spatial_cka_rows", [])) == 48,
        "attention_rows": len(summary.get("attention_gt_rows", [])) == 4,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid completed CUB direct-spatial v2 result: " + ", ".join(failures)
        )


def _validate_direct_spatial_seed23_v2(summary: dict[str, Any]) -> None:
    gate = summary.get("completion_gate", {})
    expected_gate = {
        "attention_metric_rows": 8,
        "attention_official_test_evaluations": 8,
        "checkpoint_strict_loads": 8,
        "official_test_evaluations": 48,
        "part_probe_lr_candidates": 120,
        "part_probe_official_test_evaluations": 40,
        "part_probe_validation_selections": 40,
        "qualitative_pngs": 64,
        "spatial_cka_values": 96,
    }
    checks = {
        "status": summary.get("status") == "complete",
        "scientific": summary.get("scientific_result") is True,
        "protocol": summary.get("protocol_id")
        == "cub200_phase1_r50_224_b128_seed2_3_direct_spatial_full_v2",
        "config": summary.get("config_sha256")
        == DIRECT_SEED23_V2_CONFIG_SHA256,
        "batch_seeds": summary.get("student_batch_size") == 128
        and summary.get("encoder_seeds") == [2, 3]
        and summary.get("independent_encoder_seed_n") == 2,
        "variants": summary.get("variants")
        == ["lg", "alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5"],
        "scope": summary.get("final_encoder_seed_inference") is False
        and summary.get("settings_unchanged_after_seed1_and_smoke") is True,
        "test_policy": summary.get("official_test_accessed") is True
        and summary.get("official_test_evaluations") == 48
        and summary.get("official_test_used_for_selection") is False,
        "gate": gate == expected_gate,
        "part_rows": len(summary.get("part_probe_aggregates", [])) == 8,
        "cka_rows": len(summary.get("spatial_cka_rows", [])) == 96,
        "attention_rows": len(summary.get("attention_gt_rows", [])) == 8,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid completed CUB direct-spatial seed-2/3 v2 result: "
            + ", ".join(failures)
        )


def _v2_selection(
    source_zip: zipfile.ZipFile, names: set[str]
) -> tuple[str, list[tuple[str, Path]], list[str]]:
    summary_member = _unique_suffix(
        names, "/phase1_cub_b128_full_v2_guided/combined_full_summary.json"
    )
    _validate_v2(_read_json(source_zip, summary_member))
    suite_root = summary_member.rsplit("/", 1)[0] + "/"
    issue_root = suite_root[: -len("phase1_cub_b128_full_v2_guided/")]

    top_level = (
        "checkpoint_manifest.json",
        "classification_results.csv",
        "combined_full_summary.json",
        "dataset_audit.json",
        "probe/qualitative_manifest.json",
        "probe/raw_results.csv",
        "probe/results.json",
        "probe/selection_complete_before_test.json",
        "probe/validation_selections.json",
        "run.log",
        "sequence_status.json",
    )
    selected: list[tuple[str, Path]] = []
    for relative in top_level:
        member = suite_root + relative
        if member not in names:
            raise RuntimeError(f"required v2 artifact is missing: {member}")
        selected.append((member, Path(relative)))

    summaries = sorted(
        name
        for name in names
        if name.startswith(suite_root + "classification/")
        and name.endswith("/summary.json")
    )
    classification_checkpoints = sorted(
        name
        for name in names
        if name.startswith(suite_root + "classification/")
        and (name.endswith("/student_best_validation.pt") or name.endswith("/teacher_best_validation.pt"))
    )
    probe_checkpoints = sorted(
        name
        for name in names
        if name.startswith(suite_root + "probe/checkpoints/")
        and name.endswith("_best_validation.pt")
    )
    if (len(summaries), len(classification_checkpoints), len(probe_checkpoints)) != (
        10,
        10,
        45,
    ):
        raise RuntimeError(
            "unexpected v2 reusable artifact counts: "
            f"summaries={len(summaries)} classification_checkpoints="
            f"{len(classification_checkpoints)} probe_checkpoints={len(probe_checkpoints)}"
        )
    for member in summaries + classification_checkpoints + probe_checkpoints:
        selected.append((member, Path(PurePosixPath(member).relative_to(suite_root))))

    teacher_split = _unique_suffix(
        names,
        "/classification/teacher/cub_teacher_resnet56_32_b128_full_300ep_seed1/validation_split.json",
    )
    selected.append(
        (teacher_split, Path(PurePosixPath(teacher_split).relative_to(suite_root)))
    )
    issue_log = _unique_suffix(names, "_result.txt")
    if not issue_log.startswith(issue_root):
        raise RuntimeError("v2 H200 issue log is outside the expected archive root")
    selected.append((issue_log, Path("h200_issue.log")))
    omitted = [
        "duplicate per-run training_status.json",
        "duplicate per-run validation_split.json except the teacher copy",
        "directory entries",
    ]
    return suite_root, selected, omitted


def _v3_teacher_selection(
    source_zip: zipfile.ZipFile, names: set[str]
) -> tuple[str, list[tuple[str, Path]], list[str]]:
    summary_member = _unique_suffix(
        names, "/phase1_cub_r50_224_teacher_full_v3/teacher_summary.json"
    )
    _validate_v3_teacher(_read_json(source_zip, summary_member))
    suite_root = summary_member.rsplit("/", 1)[0] + "/"
    issue_root = suite_root[: -len("phase1_cub_r50_224_teacher_full_v3/")]
    relative_files = (
        "protocol_snapshot.json",
        "run.log",
        "teacher_best_validation.pt",
        "teacher_history.csv",
        "teacher_summary.json",
        "validation_split.json",
    )
    selected: list[tuple[str, Path]] = []
    for relative in relative_files:
        member = suite_root + relative
        if member not in names:
            raise RuntimeError(f"required v3 teacher artifact is missing: {member}")
        selected.append((member, Path(relative)))
    issue_log = _unique_suffix(names, "_result.txt")
    if not issue_log.startswith(issue_root):
        raise RuntimeError("v3 H200 issue log is outside the expected archive root")
    selected.append((issue_log, Path("h200_issue.log")))
    omitted = ["duplicate training_status.json", "directory entries"]
    return suite_root, selected, omitted


def _v4_guided_seed1_selection(
    source_zip: zipfile.ZipFile, names: set[str]
) -> tuple[str, list[tuple[str, Path]], list[str]]:
    suite_name = "phase1_cub_r50_224_b128_b64_guided_probe_seed1_full_v4/"
    summary_member = _unique_suffix(names, "/" + suite_name + "combined_full_summary.json")
    _validate_v4_guided_seed1(_read_json(source_zip, summary_member))
    suite_root = summary_member.rsplit("/", 1)[0] + "/"
    issue_root = suite_root[: -len(suite_name)]

    selected: list[tuple[str, Path]] = []
    root_files = (
        "checkpoint_manifest.json",
        "combined_full_summary.json",
        "dataset_audit.json",
        "run.log",
        "sequence_status.json",
    )
    batch_files = (
        "batch_full_summary.json",
        "checkpoint_manifest.json",
        "classification_results.csv",
        "probe/raw_results.csv",
        "probe/results.json",
        "probe/selection_complete_before_test.json",
        "probe/validation_selections.json",
    )
    for relative in root_files:
        member = suite_root + relative
        if member not in names:
            raise RuntimeError(f"required v4 artifact is missing: {member}")
        selected.append((member, Path(relative)))
    for batch_size in (128, 64):
        for relative in batch_files:
            canonical = Path(f"batch{batch_size}") / relative
            member = suite_root + canonical.as_posix()
            if member not in names:
                raise RuntimeError(f"required v4 artifact is missing: {member}")
            selected.append((member, canonical))

    summaries = sorted(
        name
        for name in names
        if name.startswith(suite_root)
        and "/classification/students/" in name
        and name.endswith("/summary.json")
    )
    classification_checkpoints = sorted(
        name
        for name in names
        if name.startswith(suite_root)
        and "/classification/students/" in name
        and name.endswith("/student_best_validation.pt")
    )
    probe_checkpoints = sorted(
        name
        for name in names
        if name.startswith(suite_root)
        and "/probe/checkpoints/" in name
        and name.endswith("_best_validation.pt")
    )
    if (len(summaries), len(classification_checkpoints), len(probe_checkpoints)) != (
        8,
        8,
        40,
    ):
        raise RuntimeError(
            "unexpected v4 reusable artifact counts: "
            f"summaries={len(summaries)} classification_checkpoints="
            f"{len(classification_checkpoints)} probe_checkpoints={len(probe_checkpoints)}"
        )
    for member in summaries + classification_checkpoints + probe_checkpoints:
        selected.append((member, Path(PurePosixPath(member).relative_to(suite_root))))

    validation_split = _unique_suffix(
        names,
        "/batch128/classification/students/"
        "cub_r50_224_lg_deit_tiny_b128_full_300ep_seed1/validation_split.json",
    )
    selected.append((validation_split, Path("validation_split.json")))
    issue_log = _unique_suffix(names, "_result.txt")
    if not issue_log.startswith(issue_root):
        raise RuntimeError("v4 H200 issue log is outside the expected archive root")
    selected.append((issue_log, Path("h200_issue.log")))
    omitted = [
        "duplicate per-student training_status.json",
        "duplicate per-student validation_split.json except one canonical copy",
        "directory entries",
    ]
    return suite_root, selected, omitted


def _v5_guided_seeds2_3_selection(
    source_zip: zipfile.ZipFile, names: set[str]
) -> tuple[str, list[tuple[str, Path]], list[str]]:
    suite_name = "phase1_cub_r50_224_b128_guided_probe_seeds2_3_full_v5/"
    summary_member = _unique_suffix(names, "/" + suite_name + "combined_full_summary.json")
    _validate_v5_guided_seeds2_3(_read_json(source_zip, summary_member))
    suite_root = summary_member.rsplit("/", 1)[0] + "/"
    issue_root = suite_root[: -len(suite_name)]
    selected: list[tuple[str, Path]] = []

    root_files = (
        "checkpoint_manifest.json",
        "combined_full_summary.json",
        "dataset_audit.json",
        "run.log",
        "sequence_status.json",
    )
    profile_files = (
        "checkpoint_manifest.json",
        "classification_results.csv",
        "probe/raw_results.csv",
        "probe/results.json",
        "probe/selection_complete_before_test.json",
        "probe/validation_selections.json",
        "profile_full_summary.json",
    )
    for relative in root_files:
        member = suite_root + relative
        if member not in names:
            raise RuntimeError(f"required v5 artifact is missing: {member}")
        selected.append((member, Path(relative)))
    for seed in (2, 3):
        for relative in profile_files:
            canonical = Path(f"batch128_seed{seed}") / relative
            member = suite_root + canonical.as_posix()
            if member not in names:
                raise RuntimeError(f"required v5 artifact is missing: {member}")
            selected.append((member, canonical))

    summaries = sorted(
        name
        for name in names
        if name.startswith(suite_root)
        and "/classification/students/" in name
        and name.endswith("/summary.json")
    )
    classification_checkpoints = sorted(
        name
        for name in names
        if name.startswith(suite_root)
        and "/classification/students/" in name
        and name.endswith("/student_best_validation.pt")
    )
    probe_checkpoints = sorted(
        name
        for name in names
        if name.startswith(suite_root)
        and "/probe/checkpoints/" in name
        and name.endswith("_best_validation.pt")
    )
    if (len(summaries), len(classification_checkpoints), len(probe_checkpoints)) != (
        8,
        8,
        40,
    ):
        raise RuntimeError(
            "unexpected v5 reusable artifact counts: "
            f"summaries={len(summaries)} classification_checkpoints="
            f"{len(classification_checkpoints)} probe_checkpoints={len(probe_checkpoints)}"
        )
    for member in summaries + classification_checkpoints + probe_checkpoints:
        selected.append((member, Path(PurePosixPath(member).relative_to(suite_root))))

    validation_split = _unique_suffix(
        names,
        "/batch128_seed2/classification/students/"
        "cub_r50_224_lg_deit_tiny_b128_full_300ep_seed2/validation_split.json",
    )
    selected.append((validation_split, Path("validation_split.json")))
    issue_log = _unique_root_suffix(names, issue_root, "_result.txt")
    selected.append((issue_log, Path("h200_issue.log")))
    omitted = [
        "duplicate per-student training_status.json",
        "duplicate per-student validation_split.json except one canonical copy",
        "directory entries",
    ]
    return suite_root, selected, omitted


def _direct_spatial_v2_selection(
    source_zip: zipfile.ZipFile, names: set[str]
) -> tuple[str, list[tuple[str, Path]], list[str]]:
    suite_name = "phase1_cub_r50_224_b128_seed1_direct_spatial_full_v2/"
    summary_member = _unique_suffix(names, "/" + suite_name + "summary.json")
    _validate_direct_spatial_v2(_read_json(source_zip, summary_member))
    suite_root = summary_member.rsplit("/", 1)[0] + "/"
    issue_root = suite_root[: -len(suite_name)]
    selected: list[tuple[str, Path]] = []

    fixed_files = (
        "attention_gt/results.csv",
        "attention_gt/results.json",
        "checkpoint_audit.json",
        "dataset_audit.json",
        "part_probe/annotation_validity_audit.json",
        "part_probe/candidates.csv",
        "part_probe/official_test_results.csv",
        "part_probe/results.json",
        "part_probe/selection_complete_before_test.json",
        "run.log",
        "sequence_status.json",
        "spatial_cka/layerwise_heatmap.png",
        "spatial_cka/results.csv",
        "spatial_cka/results.json",
        "summary.json",
    )
    for relative in fixed_files:
        member = suite_root + relative
        if member not in names:
            raise RuntimeError(f"required direct-spatial artifact is missing: {member}")
        selected.append((member, Path(relative)))

    checkpoint_members = sorted(
        name
        for name in names
        if name.startswith(suite_root + "part_probe/checkpoints/")
        and name.endswith("_best_validation.pt")
    )
    history_members = sorted(
        name
        for name in names
        if name.startswith(suite_root + "part_probe/histories/")
        and name.endswith(".json")
    )
    qualitative_members = sorted(
        name
        for name in names
        if name.startswith(suite_root + "attention_gt/qualitative/")
        and name.endswith(".png")
    )
    if (len(checkpoint_members), len(history_members), len(qualitative_members)) != (
        20,
        60,
        32,
    ):
        raise RuntimeError(
            "unexpected direct-spatial reusable artifact counts: "
            f"checkpoints={len(checkpoint_members)} histories={len(history_members)} "
            f"qualitative={len(qualitative_members)}"
        )
    for member in checkpoint_members + history_members + qualitative_members:
        selected.append((member, Path(PurePosixPath(member).relative_to(suite_root))))

    issue_log = _unique_root_suffix(names, issue_root, "_result.txt")
    selected.append((issue_log, Path("h200_issue.log")))
    omitted = ["directory entries"]
    return suite_root, selected, omitted


def _direct_spatial_seed23_v2_selection(
    source_zip: zipfile.ZipFile, names: set[str]
) -> tuple[str, list[tuple[str, Path]], list[str]]:
    suite_name = "phase1_cub_r50_224_b128_seed2_3_direct_spatial_full_v2/"
    summary_member = _unique_suffix(names, "/" + suite_name + "summary.json")
    _validate_direct_spatial_seed23_v2(_read_json(source_zip, summary_member))
    suite_root = summary_member.rsplit("/", 1)[0] + "/"
    issue_root = suite_root[: -len(suite_name)]
    selected: list[tuple[str, Path]] = []

    fixed_files = (
        "attention_gt/results.csv",
        "attention_gt/results.json",
        "checkpoint_audit.json",
        "dataset_audit.json",
        "part_probe/annotation_validity_audit.json",
        "part_probe/candidates.csv",
        "part_probe/official_test_results.csv",
        "part_probe/results.json",
        "part_probe/selection_complete_before_test.json",
        "run.log",
        "sequence_status.json",
        "spatial_cka/layerwise_heatmap.png",
        "spatial_cka/results.csv",
        "spatial_cka/results.json",
        "summary.json",
    )
    for relative in fixed_files:
        member = suite_root + relative
        if member not in names:
            raise RuntimeError(
                f"required direct-spatial seed-2/3 artifact is missing: {member}"
            )
        selected.append((member, Path(relative)))

    checkpoint_members = sorted(
        name
        for name in names
        if name.startswith(suite_root + "part_probe/checkpoints/")
        and name.endswith("_best_validation.pt")
    )
    history_members = sorted(
        name
        for name in names
        if name.startswith(suite_root + "part_probe/histories/")
        and name.endswith(".json")
    )
    qualitative_members = sorted(
        name
        for name in names
        if name.startswith(suite_root + "attention_gt/qualitative/")
        and name.endswith(".png")
    )
    if (len(checkpoint_members), len(history_members), len(qualitative_members)) != (
        40,
        120,
        64,
    ):
        raise RuntimeError(
            "unexpected direct-spatial seed-2/3 reusable artifact counts: "
            f"checkpoints={len(checkpoint_members)} histories={len(history_members)} "
            f"qualitative={len(qualitative_members)}"
        )
    for member in checkpoint_members + history_members + qualitative_members:
        selected.append((member, Path(PurePosixPath(member).relative_to(suite_root))))

    issue_log = _unique_root_suffix(names, issue_root, "_result.txt")
    selected.append((issue_log, Path("h200_issue.log")))
    omitted = ["directory entries"]
    return suite_root, selected, omitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--issue-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-bundle-filename", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source_zip:
        infos = source_zip.infolist()
        _validate_members(infos)
        corrupt = source_zip.testzip()
        if corrupt is not None:
            raise RuntimeError(f"CRC failure in archive member: {corrupt}")
        names = {info.filename for info in infos}
        if args.kind == "resnet56-v2-guided":
            suite_root, selected, omitted = _v2_selection(source_zip, names)
        elif args.kind == "resnet50-v3-teacher":
            suite_root, selected, omitted = _v3_teacher_selection(source_zip, names)
        elif args.kind == "resnet50-v4-guided-seed1":
            suite_root, selected, omitted = _v4_guided_seed1_selection(
                source_zip, names
            )
        elif args.kind == "resnet50-v5-guided-seeds2-3":
            suite_root, selected, omitted = _v5_guided_seeds2_3_selection(
                source_zip, names
            )
        elif args.kind == "resnet50-direct-spatial-v2-seed1":
            suite_root, selected, omitted = _direct_spatial_v2_selection(
                source_zip, names
            )
        else:
            suite_root, selected, omitted = _direct_spatial_seed23_v2_selection(
                source_zip, names
            )
        if len({str(relative) for _, relative in selected}) != len(selected):
            raise RuntimeError("selected archive artifacts collide after canonicalization")

        imported: list[dict[str, Any]] = []
        for member, relative in selected:
            destination = output_dir / relative
            size, digest = _stream_to_file(source_zip.open(member), destination)
            imported.append(
                {
                    "path": relative.as_posix(),
                    "source_member": member,
                    "bytes": size,
                    "sha256": digest,
                    "checkpoint": relative.suffix == ".pt",
                }
            )

    manifest = {
        "schema_version": 1,
        "status": "pass",
        "dataset": "CUB-200-2011",
        "experiment_kind": args.kind,
        "h200_issue_id": str(args.issue_id),
        "source_archive": {
            "original_filename": archive.name,
            "canonical_filename": args.canonical_bundle_filename,
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
            "all_member_crc_verified": True,
            "unsafe_paths_or_symlinks": False,
        },
        "source_suite_root": suite_root,
        "import_policy": {
            "kept": [
                "scientific summaries and CSV/JSON evidence",
                "classification checkpoints needed for reproducibility",
                "selected probe checkpoints when present",
                "H200 execution logs",
            ],
            "omitted": omitted,
        },
        "imported_file_count_excluding_this_manifest": len(imported),
        "imported_bytes_excluding_this_manifest": sum(item["bytes"] for item in imported),
        "checkpoint_count": sum(bool(item["checkpoint"]) for item in imported),
        "files": sorted(imported, key=lambda item: item["path"]),
    }
    _save_json(manifest, output_dir / "artifact_manifest.json")
    print(
        f"[CUB_IMPORT_DONE] kind={args.kind} issue={args.issue_id} "
        f"files={len(imported)} checkpoints={manifest['checkpoint_count']} "
        f"bytes={manifest['imported_bytes_excluding_this_manifest']} "
        f"output={output_dir}",
        flush=True,
    )
    return manifest


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
