#!/usr/bin/env python3
"""Audit and summarize the two independent L1/L2 full-pilot shards."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_cub_loader_pilot_full import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_TEACHER_STATE_SHA256,
    EXPECTED_VARIANTS,
)
from .train_timing import file_sha256, format_duration


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROFILES = ("l1_matched_weak", "l2_conservative_spatial")
JOB_ID = "cub200_phase1_r50_224_b128_seed1_loader_pilot_l1_l2_combined_job_v1"
PROTOCOL_ID = "cub200_phase1_r50_224_b128_seed1_loader_pilot_full_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_job_config(config: dict[str, Any], path: Path) -> None:
    scientific = config.get("scientific_protocol", {})
    runtime = config.get("runtime_evidence", {})
    execution = config.get("execution", {})
    policy = config.get("policy", {})
    protocol_path = Path(str(scientific.get("path", "")))
    if not protocol_path.is_absolute():
        protocol_path = REPOSITORY_ROOT / protocol_path
    checks = {
        "job_id": config.get("job_id") == JOB_ID,
        "locked": config.get("status")
        == "locked_after_l1_l2_subset_smoke_pass_before_combined_full_results_2026-09-12",
        "operational_only": config.get("operational_packaging_only") is True,
        "scientific_result": config.get("scientific_result") is True,
        "profiles": tuple(config.get("profiles", ())) == PROFILES,
        "variants": tuple(config.get("variants", ())) == EXPECTED_VARIANTS,
        "protocol_id": scientific.get("protocol_id") == PROTOCOL_ID,
        "protocol_hash": scientific.get("sha256") == EXPECTED_CONFIG_SHA256,
        "protocol_file": protocol_path.is_file()
        and file_sha256(protocol_path) == EXPECTED_CONFIG_SHA256,
        "settings_unchanged": scientific.get("scientific_settings_changed") is False,
        "runtime_evidence": runtime
        == {
            "l0_full_log_seconds": 12254,
            "l1_l2_subset_smoke_seconds": 418.26,
            "l1_l2_linear_conservative_upper_seconds": 36259,
            "l1_l2_linear_conservative_upper_display": "10h 04m 19s",
            "l0_upper_to_actual_calibration_ratio": 0.6852317844,
            "l1_l2_calibrated_expected_seconds": 24846,
            "l1_l2_calibrated_expected_display": "6h 54m 06s",
            "runtime_values_used_only_for_job_partitioning": True,
            "smoke_metrics_used_for_scientific_selection": False,
        },
        "execution": execution
        == {
            "order": list(PROFILES),
            "one_h200_issue": True,
            "sequential_not_parallel": True,
            "mig_slices": 7,
            "profile_outputs_are_independent": True,
            "profile_summaries_are_independent": True,
            "profile_checkpoints_are_independent": True,
            "l1_artifacts_remain_if_l2_fails": True,
        },
        "official_test_closed": policy.get("official_test_accessed") is False,
        "selection_deferred": policy.get("loader_selection_in_this_job") is False
        and policy.get("all_l0_l1_l2_archives_required_before_loader_selection")
        is True
        and policy.get("classification_cka_attention_used_for_loader_selection")
        is False,
        "gate": config.get("completion_gate")
        == {
            "loader_profiles": 2,
            "classification_students": 8,
            "classification_checkpoints": 8,
            "segmentation_probe_lr_candidates": 120,
            "segmentation_probe_validation_selections": 40,
            "part_probe_lr_candidates": 120,
            "part_probe_validation_selections": 40,
            "spatial_cka_values": 96,
            "attention_metric_rows": 8,
            "attention_qualitative_pngs": 32,
            "new_checkpoints": 88,
            "official_test_evaluations": 0,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            f"invalid L1/L2 combined job config {path}: {','.join(failures)}"
        )


def _profile_rows(
    profile_dir: Path, profile: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_json(profile_dir / "summary.json")
    segmentation = _load_json(profile_dir / "segmentation_probe/results.json")
    part = _load_json(profile_dir / "part_probe/results.json")
    cka = _load_json(profile_dir / "spatial_cka/results.json")
    attention = _load_json(profile_dir / "attention_gt/results.json")
    checkpoint_audit = _load_json(profile_dir / "checkpoint_audit.json")
    manifest = _load_json(profile_dir / "checkpoint_manifest.json")
    seg_candidates = _read_csv_rows(
        profile_dir / "segmentation_probe/candidates.csv"
    )
    part_candidates = _read_csv_rows(profile_dir / "part_probe/candidates.csv")
    qualitative_pngs = sorted(
        (profile_dir / "attention_gt/qualitative").glob("*.png")
    )

    classifications = summary.get("classification", [])
    seg_selections = segmentation.get("selections", [])
    part_selections = part.get("selections", [])
    cka_rows = cka.get("rows", [])
    attention_rows = attention.get("rows", [])
    expected_gate = {
        "loader_profiles": 1,
        "classification_students": 4,
        "classification_checkpoints": 4,
        "segmentation_probe_lr_candidates": 60,
        "segmentation_probe_validation_selections": 20,
        "part_probe_lr_candidates": 60,
        "part_probe_validation_selections": 20,
        "spatial_cka_values": 48,
        "attention_metric_rows": 4,
        "attention_qualitative_pngs": 16,
        "new_checkpoints": 44,
        "official_test_evaluations": 0,
    }
    checks = {
        "summary": summary.get("status") == "complete"
        and summary.get("protocol_id") == PROTOCOL_ID
        and summary.get("config_sha256") == EXPECTED_CONFIG_SHA256
        and summary.get("loader_profile") == profile
        and summary.get("scientific_result") is True
        and summary.get("exploratory_loader_pilot") is True
        and summary.get("confirmatory_main_result") is False
        and summary.get("all_three_shards_required_before_loader_selection") is True
        and summary.get("official_test_accessed") is False
        and summary.get("official_test_evaluations") == 0
        and summary.get("this_shard_alone_must_not_select_loader") is True,
        "gate": summary.get("completion_gate") == expected_gate,
        "classification": len(classifications) == 4
        and tuple(row.get("variant") for row in classifications)
        == EXPECTED_VARIANTS
        and all(row.get("profile") == profile for row in classifications)
        and all(row.get("official_test_evaluations") == 0 for row in classifications),
        "segmentation": segmentation.get("status") == "complete"
        and segmentation.get("profile") == profile
        and segmentation.get("scientific_result") is True
        and segmentation.get("exploratory_loader_pilot") is True
        and len(seg_selections) == 20
        and len(seg_candidates) == 60
        and segmentation.get("official_test_evaluations") == 0,
        "part": part.get("status") == "complete"
        and part.get("profile") == profile
        and part.get("scientific_result") is True
        and part.get("exploratory_loader_pilot") is True
        and len(part_selections) == 20
        and len(part_candidates) == 60
        and part.get("official_test_evaluations") == 0,
        "cka": cka.get("status") == "complete"
        and cka.get("profile") == profile
        and cka.get("scientific_result") is True
        and cka.get("exploratory_loader_pilot") is True
        and len(cka_rows) == 48
        and cka.get("official_test_used") is False,
        "attention": attention.get("status") == "complete"
        and attention.get("profile") == profile
        and attention.get("scientific_result") is True
        and attention.get("exploratory_loader_pilot") is True
        and len(attention_rows) == 4
        and len(qualitative_pngs) == 16
        and attention.get("official_test_evaluations") == 0,
        "teacher": checkpoint_audit.get("status") == "pass"
        and checkpoint_audit.get("teacher", {}).get("checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and checkpoint_audit.get("teacher", {}).get("model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "manifest": manifest.get("count") == 45
        and manifest.get("external_shared_teacher_count") == 1
        and manifest.get("new_checkpoint_count") == 44
        and len(manifest.get("entries", [])) == 45,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"invalid full shard {profile}: {','.join(failures)}")

    seg_by_variant: dict[str, list[float]] = defaultdict(list)
    part_by_variant: dict[str, list[float]] = defaultdict(list)
    for row in seg_selections:
        if (
            row.get("profile") != profile
            or row.get("official_test_evaluations") != 0
        ):
            raise RuntimeError(f"invalid segmentation selection row: {profile}")
        seg_by_variant[str(row["variant"])].append(
            float(row["validation"]["input_224"]["mean_iou"])
        )
    for row in part_selections:
        if (
            row.get("profile") != profile
            or row.get("official_test_evaluations") != 0
        ):
            raise RuntimeError(f"invalid part selection row: {profile}")
        part_by_variant[str(row["variant"])].append(
            float(row["validation"]["micro_pck_at_0.1"])
        )
    cka_lookup = {
        (str(row["variant"]), int(row["student_block"])): row for row in cka_rows
    }
    attention_lookup = {str(row["variant"]): row for row in attention_rows}
    expected_cka_keys = {
        (variant, block)
        for variant in EXPECTED_VARIANTS
        for block in range(12)
    }
    if (
        set(seg_by_variant) != set(EXPECTED_VARIANTS)
        or any(len(values) != 5 for values in seg_by_variant.values())
        or set(part_by_variant) != set(EXPECTED_VARIANTS)
        or any(len(values) != 5 for values in part_by_variant.values())
        or set(cka_lookup) != expected_cka_keys
        or tuple(attention_lookup) != EXPECTED_VARIANTS
    ):
        raise RuntimeError(f"incomplete per-variant result grid: {profile}")

    manifest_entries = manifest["entries"]
    manifest_kind_counts = {
        kind: sum(entry.get("kind") == kind for entry in manifest_entries)
        for kind in (
            "external_shared_teacher",
            "classification_encoder",
            "selected_segmentation_probe",
            "selected_part_probe",
        )
    }
    if manifest_kind_counts != {
        "external_shared_teacher": 1,
        "classification_encoder": 4,
        "selected_segmentation_probe": 20,
        "selected_part_probe": 20,
    }:
        raise RuntimeError(f"checkpoint kind counts changed: {profile}")
    for entry in manifest_entries:
        checkpoint = Path(str(entry.get("path", "")))
        if (
            not checkpoint.is_file()
            or file_sha256(checkpoint) != entry.get("sha256")
        ):
            raise RuntimeError(f"checkpoint manifest verification failed: {checkpoint}")

    rows: list[dict[str, Any]] = []
    for classification in classifications:
        variant = str(classification["variant"])
        values = {
            "profile": profile,
            "variant": variant,
            "classification_validation_macro_top1": float(
                classification["validation_macro_top1"]
            ),
            "segmentation_validation_input224_miou_probe_seed_mean": statistics.mean(
                seg_by_variant[variant]
            ),
            "part_validation_pck_at_0.1_probe_seed_mean": statistics.mean(
                part_by_variant[variant]
            ),
            "cka_block11": float(
                cka_lookup[(variant, 11)]["centered_linear_cka"]
            ),
            "attention_patch_ap": float(
                attention_lookup[variant]["global_micro_patch_average_precision"]
            ),
            "attention_pointing": float(
                attention_lookup[variant]["pointing_game_peak_inside_mask"]
            ),
        }
        rows.append(values)

    primary = statistics.mean(
        row["part_validation_pck_at_0.1_probe_seed_mean"] for row in rows
    )
    tie_break = statistics.mean(
        row["segmentation_validation_input224_miou_probe_seed_mean"] for row in rows
    )
    scores = summary.get("loader_selection_scores", {})
    if (
        abs(
            primary
            - float(scores["primary_mean_validation_part_micro_pck_at_0.1"])
        )
        > 1e-12
    ):
        raise RuntimeError(f"part score changed while combining {profile}")
    if abs(
        tie_break
        - float(
            scores[
                "tie_break_1_mean_validation_frozen_segmentation_input224_miou"
            ]
        )
    ) > 1e-12:
        raise RuntimeError(f"segmentation score changed while combining {profile}")
    return summary, rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.combined_root.expanduser().resolve()
    config_path = args.job_config.expanduser().resolve()
    config = _load_json(config_path)
    _validate_job_config(config, config_path)

    summaries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for profile in PROFILES:
        summary, rows = _profile_rows(root / profile, profile)
        summaries.append(summary)
        results.extend(rows)

    elapsed_sum = sum(
        float(summary.get("runtime", {}).get("elapsed_seconds", 0.0))
        for summary in summaries
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "job_id": JOB_ID,
        "job_config_path": str(config_path),
        "job_config_sha256": file_sha256(config_path),
        "scientific_protocol_sha256": EXPECTED_CONFIG_SHA256,
        "profiles": list(PROFILES),
        "variants": list(EXPECTED_VARIANTS),
        "results": results,
        "profile_loader_selection_scores": {
            str(summary["loader_profile"]): summary["loader_selection_scores"]
            for summary in summaries
        },
        "completion_gate": config["completion_gate"],
        "official_test_accessed": False,
        "official_test_evaluations": 0,
        "loader_selection_deferred_until_l0_l1_l2_archives_audited": True,
        "summed_profile_elapsed_seconds": elapsed_sum,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = root / "combined_summary.json"
    _write_json(payload, output)

    print("[LOADER_PILOT_L1_L2_FULL_RESULTS]", flush=True)
    for row in results:
        print(
            "[LOADER_PILOT_L1_L2_FULL_RESULT] "
            f"profile={row['profile']} variant={row['variant']} "
            "class_val_macro="
            f"{row['classification_validation_macro_top1']:.4f} "
            "seg_val_miou_probe_seed_mean="
            f"{row['segmentation_validation_input224_miou_probe_seed_mean']:.6f} "
            "part_val_pck_probe_seed_mean="
            f"{row['part_validation_pck_at_0.1_probe_seed_mean']:.6f} "
            f"cka_block11={row['cka_block11']:.6f} "
            f"attention_patch_ap={row['attention_patch_ap']:.6f} "
            f"attention_pointing={row['attention_pointing']:.6f}",
            flush=True,
        )
    for summary in summaries:
        profile = summary["loader_profile"]
        scores = summary["loader_selection_scores"]
        print(
            "[LOADER_PILOT_L1_L2_FULL_SCORE] "
            f"profile={profile} "
            "primary_part_pck="
            f"{scores['primary_mean_validation_part_micro_pck_at_0.1']:.6f} "
            "tie1_seg_miou="
            f"{scores['tie_break_1_mean_validation_frozen_segmentation_input224_miou']:.6f}",
            flush=True,
        )
    print(
        "[LOADER_PILOT_L1_L2_FULL_DONE] status=complete profiles=2/2 "
        "classification=8/8 segmentation_candidates=120/120 "
        "segmentation_selections=40/40 part_candidates=120/120 "
        "part_selections=40/40 cka_values=96/96 attention_rows=8/8 "
        "qualitative_pngs=32/32 new_checkpoints=88/88 official_test=0 "
        f"profile_elapsed_sum={format_duration(elapsed_sum)} summary={output}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--job-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
