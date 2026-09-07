#!/usr/bin/env python3
"""Audit and curate the batch-128 ALG controller-warm-up-20 diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from curate_probe_results import (
    METRIC_NAMES,
    _assert_number_equal,
    _audit_metric_block,
    _nested_equal,
    file_sha256,
    save_json,
    write_csv,
)
from ibkd_seg.phase1.probe import probe_from_state
from ibkd_seg.phase1.run_alg_warmup20_full import (
    DIAGNOSTIC_ID,
    ENCODER_SEEDS,
    EXPERIMENT_ID,
    PROBE_SEEDS,
    VARIANT,
    _aggregate_probe,
    _validate_classification_result,
)
from ibkd_seg.phase1.run_probe_full import LOCKED_PROTOCOL_SHA256
from ibkd_seg.phase1.train_full import load_full_teacher


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPOSITORY_ROOT / "phase1/phase1_pet/configs/oxford_iiit_pet_phase1_v1.json"
FULL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "phase1/phase1_pet/configs/oxford_iiit_pet_alg_warmup20_full_v1.json"
)
CANONICAL_CLASSIFICATION_PATH = (
    REPOSITORY_ROOT / "phase1/phase1_pet/reports/classification/batch128/summary.json"
)
CANONICAL_PROBE_PATH = (
    REPOSITORY_ROOT / "phase1/phase1_pet/reports/frozen_probe/batch128/summary.json"
)
PRIMARY_METRIC = "input_224_mean_iou"


def _key(result: dict[str, Any]) -> tuple[int, int]:
    return int(result["encoder_seed"]), int(result["probe_seed"])


def _audit_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    initial_hashes = {seed: set() for seed in PROBE_SEEDS}
    order_hashes = {
        seed: {epoch: set() for epoch in range(1, 101)} for seed in PROBE_SEEDS
    }
    count = 0
    for result in results:
        candidates = result["candidate_audits"]
        if [float(candidate["learning_rate"]) for candidate in candidates] != [
            0.01,
            0.03,
            0.1,
        ]:
            raise RuntimeError(f"candidate LR grid mismatch: {_key(result)}")
        for candidate in candidates:
            count += 1
            history = candidate["history"]
            orders = candidate["batch_order_sha256_by_epoch"]
            if len(history) != 100 or len(orders) != 100:
                raise RuntimeError(f"candidate epoch count mismatch: {_key(result)}")
            if [int(row["epoch"]) for row in history] != list(range(1, 101)):
                raise RuntimeError(f"candidate epoch sequence mismatch: {_key(result)}")
            best_index = max(
                range(100),
                key=lambda index: float(history[index]["validation_grid_mean_iou"]),
            )
            if int(candidate["best_epoch"]) != best_index + 1:
                raise RuntimeError(f"candidate best epoch mismatch: {_key(result)}")
            _assert_number_equal(
                candidate["best_validation_grid_mean_iou"],
                history[best_index]["validation_grid_mean_iou"],
            )
            if candidate["gradient_contract"] != {
                "cached_feature_gradient_tensor_count": 0,
                "probe_gradient_tensor_count": 2,
            }:
                raise RuntimeError(f"gradient contract mismatch: {_key(result)}")
            probe_seed = int(result["probe_seed"])
            initial_hashes[probe_seed].add(candidate["initial_probe_state_sha256"])
            for epoch, digest in enumerate(orders, start=1):
                order_hashes[probe_seed][epoch].add(digest)

        selected_index = max(
            range(3),
            key=lambda index: float(
                candidates[index]["best_validation_grid_mean_iou"]
            ),
        )
        selected = candidates[selected_index]
        reported = result["selection"]
        if (
            float(reported["learning_rate"]) != float(selected["learning_rate"])
            or int(reported["epoch"]) != int(selected["best_epoch"])
        ):
            raise RuntimeError(f"validation selection mismatch: {_key(result)}")
        _assert_number_equal(
            reported["validation_grid_mean_iou"],
            selected["best_validation_grid_mean_iou"],
        )

    if count != 45:
        raise RuntimeError(f"expected 45 candidates, found {count}")
    if not all(len(values) == 1 for values in initial_hashes.values()):
        raise RuntimeError("probe initialization differs within a probe seed")
    if not all(
        len(values) == 1
        for epoch_map in order_hashes.values()
        for values in epoch_map.values()
    ):
        raise RuntimeError("probe batch order differs within a probe seed/epoch")
    return {
        "candidate_count": count,
        "all_candidate_histories_have_100_epochs": True,
        "validation_best_epoch_recomputed": True,
        "validation_lr_selection_recomputed": True,
        "same_initial_probe_state_per_probe_seed": True,
        "same_batch_order_per_probe_seed_and_epoch": True,
        "encoder_feature_gradient_tensor_count": 0,
        "probe_gradient_tensor_count": 2,
    }


def _audit_classification(
    raw_dir: Path,
    suite: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = suite["reference_audit"]
    validation_hash = reference["teacher_metadata"]["validation_image_ids_sha256"]
    teacher_path = raw_dir / "classification/reference_teacher/teacher_best_validation.pt"
    teacher, teacher_metadata, checkpoint_hash, state_hash = load_full_teacher(
        teacher_path,
        validation_hash=validation_hash,
        device=torch.device("cpu"),
    )
    del teacher
    if checkpoint_hash != reference["teacher_checkpoint_sha256"]:
        raise RuntimeError("reference teacher checkpoint hash mismatch")
    if state_hash != reference["teacher_model_state_sha256"]:
        raise RuntimeError("reference teacher model-state hash mismatch")
    if teacher_metadata != reference["teacher_metadata"]:
        raise RuntimeError("reference teacher metadata mismatch")

    rows_by_seed = {int(row["seed"]): row for row in suite["rows"]}
    audits_by_seed = {
        int(audit["seed"]): audit for audit in suite["classification_audits"]
    }
    output_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = [
        {
            "kind": "reference_teacher",
            "seed": 1,
            "path_under_ignored_raw_root": teacher_path.relative_to(raw_dir).as_posix(),
            "bytes": teacher_path.stat().st_size,
            "sha256": checkpoint_hash,
            "safe_load_weights_only": True,
            "strict_load": True,
            "official_test_evaluations_at_checkpoint_write": 0,
        }
    ]
    for seed in ENCODER_SEEDS:
        run_dir = (
            raw_dir
            / "classification/students"
            / f"pet_alg_controller_warmup20_b128_full_300ep_seed{seed}"
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        checkpoint_path = run_dir / "student_best_validation.pt"
        audit = _validate_classification_result(
            summary,
            checkpoint_path,
            seed=seed,
            validation_hash=validation_hash,
            teacher_checkpoint_sha256=reference["teacher_checkpoint_sha256"],
            teacher_state_sha256=reference["teacher_model_state_sha256"],
            expected_initial_hash=reference[
                "canonical_alg_initial_student_state_sha256_by_seed"
            ][str(seed)],
        )
        expected_audit = {
            key: value
            for key, value in audits_by_seed[seed].items()
            if key != "seed"
        }
        if audit != expected_audit:
            raise RuntimeError(f"classification audit summary mismatch for seed {seed}")
        history = summary["history"]
        selected_index = max(
            range(len(history)),
            key=lambda index: float(history[index]["validation"]["macro_top1"]),
        )
        if selected_index + 1 != int(summary["selected_epoch"]):
            raise RuntimeError(f"classification best epoch mismatch for seed {seed}")
        if summary["selected_validation"] != history[selected_index]["validation"]:
            raise RuntimeError(f"classification validation metric mismatch for seed {seed}")
        row = rows_by_seed[seed]
        if row["checkpoint_sha256"] != summary["checkpoint_sha256"]:
            raise RuntimeError(f"classification suite row mismatch for seed {seed}")
        output_rows.append(
            {
                "encoder_seed": seed,
                "selected_epoch": summary["selected_epoch"],
                "controller_stop_epoch": summary["controller_final"]["stop_epoch"],
                "validation_macro_top1": summary["selected_validation"]["macro_top1"],
                "test_macro_top1": summary["official_test"]["macro_top1"],
                "test_overall_top1": summary["official_test"]["overall_top1"],
                "test_top5": summary["official_test"]["top5"],
            }
        )
        entries.append(
            {
                "kind": "classification_student",
                "seed": seed,
                "path_under_ignored_raw_root": checkpoint_path.relative_to(raw_dir).as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": file_sha256(checkpoint_path),
                "safe_load_weights_only": True,
                "strict_load": True,
                "student_state_sha256": summary["student_state_sha256"],
                "all_floating_tensors_finite": True,
                "official_test_evaluations_at_checkpoint_write": 0,
            }
        )
    return output_rows, entries


def _audit_probe_artifacts(
    raw_dir: Path,
    results: list[dict[str, Any]],
    probe_config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_paths = {
        raw_dir / "probe" / result["probe_artifact"]["relative_path"]
        for result in results
    }
    observed_paths = set((raw_dir / "probe/probes").glob("**/*.pt"))
    if observed_paths != expected_paths or len(observed_paths) != 15:
        raise RuntimeError("diagnostic probe checkpoint set is incomplete")
    entries = []
    for result in results:
        path = raw_dir / "probe" / result["probe_artifact"]["relative_path"]
        digest = file_sha256(path)
        if digest != result["probe_artifact"]["sha256"]:
            raise RuntimeError(f"probe checkpoint hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checks = {
            "purpose": payload.get("purpose")
            == "phase1_posthoc_alg_warmup20_full_frozen_probe",
            "scientific": payload.get("scientific_result") is True,
            "not_confirmatory": payload.get("confirmatory_main_result") is False,
            "posthoc": payload.get("posthoc_diagnostic") is True,
            "canonical_retained": payload.get("canonical_phase1_result_replaced")
            is False,
            "experiment": payload.get("experiment_id") == EXPERIMENT_ID,
            "diagnostic": payload.get("diagnostic_id") == DIAGNOSTIC_ID,
            "protocol": payload.get("protocol_sha256") == LOCKED_PROTOCOL_SHA256,
            "batch": payload.get("classification_batch_size") == 128,
            "variant": payload.get("variant") == VARIANT,
            "encoder_seed": payload.get("encoder_seed") == result["encoder_seed"],
            "probe_seed": payload.get("probe_seed") == result["probe_seed"],
            "test_zero": payload.get("official_test_evaluations_at_checkpoint_write")
            == 0,
            "selection": payload.get("selection") == result["selection"],
            "candidate_audits": payload.get("candidate_audits")
            == result["candidate_audits"],
        }
        if not all(checks.values()):
            failures = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"probe payload mismatch {path}: {failures}")
        probe = probe_from_state(
            probe_config,
            int(result["probe_seed"]),
            payload["model"],
            torch.device("cpu"),
        )
        if sum(parameter.numel() for parameter in probe.parameters()) != 386:
            raise RuntimeError("diagnostic probe parameter count changed")
        if not all(bool(torch.isfinite(value).all()) for value in payload["model"].values()):
            raise RuntimeError(f"non-finite diagnostic probe state: {path}")
        entries.append(
            {
                "kind": "frozen_probe",
                "encoder_seed": result["encoder_seed"],
                "probe_seed": result["probe_seed"],
                "selected_learning_rate": result["selection"]["learning_rate"],
                "selected_epoch": result["selection"]["epoch"],
                "path_under_ignored_raw_root": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "safe_load_weights_only": True,
                "strict_load": True,
                "parameter_count": 386,
                "all_floating_tensors_finite": True,
                "official_test_evaluations_at_checkpoint_write": 0,
            }
        )
    return entries


def _audit_probe_csv(raw_dir: Path, results: list[dict[str, Any]]) -> None:
    with (raw_dir / "probe/probe_raw_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 15:
        raise RuntimeError("diagnostic probe raw CSV does not have 15 rows")
    by_key = {_key(result): result for result in results}
    csv_keys = {(int(row["encoder_seed"]), int(row["probe_seed"])) for row in rows}
    if csv_keys != set(by_key):
        raise RuntimeError("diagnostic probe CSV matrix differs from JSON")
    for row in rows:
        result = by_key[(int(row["encoder_seed"]), int(row["probe_seed"]))]
        if int(row["official_test_evaluations"]) != 1:
            raise RuntimeError("diagnostic probe CSV test count is not one")
        _assert_number_equal(
            float(row["test_input_224_mean_iou"]),
            result["test"]["input_224"]["mean_iou"],
        )


def _copy_qualitative(
    raw_dir: Path,
    report_dir: Path,
    qualitative: dict[str, Any],
) -> dict[str, Any]:
    source_root = raw_dir / "probe/qualitative"
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest != qualitative:
        raise RuntimeError("diagnostic qualitative manifest mismatch")
    if qualitative["test_image_ids"] != [Path(name).stem for name in qualitative["panels"]]:
        raise RuntimeError("diagnostic qualitative IDs and panels differ")
    if qualitative["posthoc_example_selection"] is not False:
        raise RuntimeError("diagnostic qualitative examples were selected post hoc")
    entries = []
    for filename in qualitative["panels"]:
        source = source_root / filename
        with Image.open(source) as image:
            image.load()
            if image.size != (672, 252) or image.mode != "RGB":
                raise RuntimeError(f"invalid diagnostic qualitative panel: {source}")
        destination = report_dir / "figures" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entries.append(
            {
                "image_id": source.stem,
                "path": destination.relative_to(report_dir).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
                "width": 672,
                "height": 252,
                "mode": "RGB",
            }
        )
    masks = sorted(source_root.glob("masks/*.png"))
    if len(masks) != 8:
        raise RuntimeError(f"expected eight diagnostic masks, found {len(masks)}")
    for path in masks:
        with Image.open(path) as image:
            if image.size != (224, 224) or image.mode != "RGB":
                raise RuntimeError(f"invalid diagnostic mask: {path}")
    return {
        "schema_version": 1,
        "status": "pass",
        "fixed_before_results": True,
        "posthoc_example_selection": False,
        "encoder_seed": qualitative["encoder_seed"],
        "probe_seed": qualitative["probe_seed"],
        "test_image_ids": qualitative["test_image_ids"],
        "panel_count": len(entries),
        "source_mask_count_audited_but_not_committed": len(masks),
        "panels": entries,
    }


def _classification_comparisons(
    warmup_mean: float,
    canonical: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for method in canonical["methods"]:
        baseline = float(method["test_macro_top1"]["mean"])
        rows.append(
            {
                "variant": method["variant"],
                "canonical_test_macro_top1": baseline,
                "alg_warmup20_test_macro_top1": warmup_mean,
                "warmup20_minus_canonical_percentage_points": warmup_mean - baseline,
            }
        )
    return rows


def _probe_comparisons(
    warmup_mean: float,
    canonical: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for method in canonical["methods"]:
        baseline = float(method["test_input_224_mean_iou"]["mean"])
        rows.append(
            {
                "variant": method["variant"],
                "canonical_test_input_224_mean_iou": baseline,
                "alg_warmup20_test_input_224_mean_iou": warmup_mean,
                "warmup20_minus_canonical": warmup_mean - baseline,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    final = json.loads((raw_dir / "alg_warmup20_full_summary.json").read_text(encoding="utf-8"))
    classification = json.loads(
        (raw_dir / "classification/classification_summary.json").read_text(
            encoding="utf-8"
        )
    )
    probe = json.loads((raw_dir / "probe/probe_summary.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (raw_dir / "probe/selection_complete_before_test.json").read_text(
            encoding="utf-8"
        )
    )
    status = json.loads((raw_dir / "sequence_status.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((raw_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    full_config = json.loads(FULL_CONFIG_PATH.read_text(encoding="utf-8"))
    canonical_classification = json.loads(
        CANONICAL_CLASSIFICATION_PATH.read_text(encoding="utf-8")
    )
    canonical_probe = json.loads(CANONICAL_PROBE_PATH.read_text(encoding="utf-8"))

    checks = {
        "final_complete": final.get("status") == "complete",
        "final_contracts": final.get("contracts", {}).get("all_passed") is True,
        "posthoc": final.get("posthoc_diagnostic") is True,
        "not_confirmatory": final.get("confirmatory_main_result") is False,
        "canonical_retained": final.get("canonical_phase1_result_replaced") is False,
        "experiment": final.get("experiment_id") == EXPERIMENT_ID,
        "diagnostic": final.get("diagnostic_id") == DIAGNOSTIC_ID,
        "single_change": final.get("changed_field")
        == "alg_controller_warmup_epochs:0_to_20",
        "protocol_hash": file_sha256(PROTOCOL_PATH) == LOCKED_PROTOCOL_SHA256,
        "reported_protocol_hash": probe.get("protocol_sha256")
        == LOCKED_PROTOCOL_SHA256,
        "full_config_hash": file_sha256(FULL_CONFIG_PATH)
        == probe.get("full_config_sha256"),
        "reported_final_config_hash": final.get("full_config", {}).get("sha256")
        == file_sha256(FULL_CONFIG_PATH),
        "classification_hash": final.get("classification", {}).get("summary_sha256")
        == file_sha256(raw_dir / "classification/classification_summary.json"),
        "probe_hash": final.get("probe", {}).get("summary_sha256")
        == file_sha256(raw_dir / "probe/probe_summary.json"),
        "source_crc": source_manifest.get("source_archive", {}).get(
            "all_member_crc_verified"
        )
        is True,
        "status_complete": status.get("status") == "complete",
        "classification_count": status.get("classification_complete") == 3,
        "classification_test_count": status.get(
            "classification_official_test_evaluations"
        )
        == 3,
        "probe_selection_count": status.get("probe_selections_complete") == 15,
        "probe_test_count": status.get("probe_test_evaluations_complete") == 15,
        "no_failure": status.get("failure") is None,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("diagnostic top-level audit failed: " + ", ".join(failures))
    if full_config["changed_field"] != {
        "name": "alg_controller_warmup_epochs",
        "canonical_value": 0,
        "diagnostic_value": 20,
    }:
        raise RuntimeError("diagnostic config did not change exactly the controller warm-up")

    classification_rows, classification_checkpoints = _audit_classification(
        raw_dir, classification
    )
    results = probe["raw_results"]
    expected_keys = {(encoder, seed) for encoder in ENCODER_SEEDS for seed in PROBE_SEEDS}
    if {_key(result) for result in results} != expected_keys or len(results) != 15:
        raise RuntimeError("diagnostic probe matrix is incomplete")
    for result in results:
        for split in ("validation", "test"):
            for resolution in ("grid_14x14", "input_224"):
                _audit_metric_block(result[split][resolution])
                if not all(
                    math.isfinite(float(result[split][resolution][name]))
                    for name in METRIC_NAMES
                ):
                    raise RuntimeError("non-finite diagnostic probe metric")
    if (
        selection.get("status") != "complete"
        or selection.get("completed_probe_selections") != 15
        or selection.get("probe_official_test_accessed") is not False
        or selection.get("probe_official_test_evaluations") != 0
        or not all(selection.get("selection_gates", {}).values())
    ):
        raise RuntimeError("diagnostic pre-test selection record failed")
    selected_by_key = {_key(result): result for result in selection["selected_probes"]}
    if set(selected_by_key) != expected_keys:
        raise RuntimeError("diagnostic pre-test selection matrix differs")
    for result in results:
        selected = selected_by_key[_key(result)]
        if selected.get("test") is not None or selected.get("official_test_evaluations") != 0:
            raise RuntimeError("diagnostic pre-test record contains test results")
        for field in (
            "variant",
            "method",
            "fusion_ratio_lambda",
            "encoder_seed",
            "probe_seed",
            "classification_checkpoint",
            "probe_artifact",
            "selection",
            "candidate_audits",
            "validation",
        ):
            if selected[field] != result[field]:
                raise RuntimeError(f"diagnostic pre/post-test mismatch: {_key(result)} {field}")

    candidate_audit = _audit_candidates(results)
    probe_checkpoints = _audit_probe_artifacts(
        raw_dir,
        results,
        protocol["frozen_spatial_probe"]["probe"],
    )
    _audit_probe_csv(raw_dir, results)
    recomputed = _aggregate_probe(results)
    if not _nested_equal(recomputed, probe["aggregate"]):
        raise RuntimeError("diagnostic probe aggregate recomputation mismatch")
    qualitative_audit = _copy_qualitative(raw_dir, report_dir, probe["qualitative"])

    classification_mean = float(
        classification["aggregates"][0]["test_macro_top1"]["mean"]
    )
    probe_primary = probe["aggregate"]["across_encoder_seed_means"]["test"][
        PRIMARY_METRIC
    ]
    probe_mean = float(probe_primary["mean"])
    classification_comparisons = _classification_comparisons(
        classification_mean, canonical_classification
    )
    probe_comparisons = _probe_comparisons(probe_mean, canonical_probe)

    probe_encoder_rows = []
    for seed in ENCODER_SEEDS:
        value = probe["aggregate"]["by_encoder_seed"][str(seed)]["test"][PRIMARY_METRIC]
        probe_encoder_rows.append(
            {
                "encoder_seed": seed,
                "probe_seed_1": value["values"][0],
                "probe_seed_2": value["values"][1],
                "probe_seed_3": value["values"][2],
                "probe_seed_4": value["values"][3],
                "probe_seed_5": value["values"][4],
                "probe_seed_mean": value["mean"],
                "probe_seed_sample_standard_deviation": value[
                    "sample_standard_deviation"
                ],
            }
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(classification_rows, report_dir / "classification_per_seed.csv")
    write_csv(probe_encoder_rows, report_dir / "probe_per_encoder_seed.csv")
    shutil.copyfile(
        raw_dir / "probe/probe_raw_results.csv",
        report_dir / "probe_raw_results.csv",
    )
    shutil.copyfile(raw_dir / "sequence_status.json", report_dir / "h200_sequence_status.json")
    shutil.copyfile(raw_dir / "artifact_manifest.json", report_dir / "source_manifest.json")
    save_json(qualitative_audit, report_dir / "figures/manifest.json")
    checkpoint_entries = classification_checkpoints + probe_checkpoints
    checkpoint_manifest = {
        "schema_version": 1,
        "status": "pass",
        "checkpoint_count": len(checkpoint_entries),
        "reference_teacher_count": 1,
        "classification_student_count": 3,
        "selected_probe_count": 15,
        "all_file_hashes_match": True,
        "all_safe_loads_use_weights_only": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "all_written_before_official_test": True,
        "entries": checkpoint_entries,
    }
    save_json(checkpoint_manifest, report_dir / "checkpoint_manifest.json")

    summary = {
        "schema_version": 1,
        "status": "complete_posthoc_diagnostic",
        "dataset": "Oxford-IIIT Pet",
        "experiment": EXPERIMENT_ID,
        "diagnostic_id": DIAGNOSTIC_ID,
        "h200_issue_id": source_manifest["h200_issue_id"],
        "runtime_git_commit": final["runtime"]["git_commit"],
        "classification_batch_size": 128,
        "posthoc_diagnostic": True,
        "confirmatory_main_result": False,
        "canonical_phase1_result_replaced": False,
        "changed_field": "alg_controller_warmup_epochs:0_to_20",
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "full_config_sha256": file_sha256(FULL_CONFIG_PATH),
        "source_archive": source_manifest["source_archive"],
        "ignored_raw_artifact_root": raw_dir.relative_to(REPOSITORY_ROOT).as_posix(),
        "data_counts": {
            "train": 2940,
            "validation": 740,
            "test": 3669,
        },
        "classification": {
            "encoder_seeds": list(ENCODER_SEEDS),
            "epochs": 300,
            "controller_stop_epoch_by_seed": {
                str(row["encoder_seed"]): row["controller_stop_epoch"]
                for row in classification_rows
            },
            "selected_epoch_by_seed": {
                str(row["encoder_seed"]): row["selected_epoch"]
                for row in classification_rows
            },
            "test_macro_top1": classification["aggregates"][0]["test_macro_top1"],
            "test_overall_top1": classification["aggregates"][0]["test_overall_top1"],
            "test_top5": classification["aggregates"][0]["test_top5"],
            "elapsed_seconds": classification["elapsed_seconds"],
            "official_test_evaluations": 3,
            "canonical_comparisons": classification_comparisons,
        },
        "frozen_probe": {
            "matrix": probe["matrix"],
            "test_input_224_mean_iou": probe_primary,
            "test_input_224_foreground_iou": probe["aggregate"][
                "across_encoder_seed_means"
            ]["test"]["input_224_foreground_iou"],
            "test_input_224_background_iou": probe["aggregate"][
                "across_encoder_seed_means"
            ]["test"]["input_224_background_iou"],
            "test_input_224_foreground_dice": probe["aggregate"][
                "across_encoder_seed_means"
            ]["test"]["input_224_foreground_dice"],
            "test_grid_14x14_mean_iou": probe["aggregate"][
                "across_encoder_seed_means"
            ]["test"]["grid_14x14_mean_iou"],
            "test_policy": probe["probe_test_policy"],
            "timing": probe["timing"],
            "peak_cuda_memory_bytes": probe["peak_cuda_memory_bytes"],
            "canonical_comparisons": probe_comparisons,
        },
        "total_elapsed_seconds": final["elapsed_seconds"],
        "candidate_audit": candidate_audit,
        "checkpoint_audit": {
            key: value for key, value in checkpoint_manifest.items() if key != "entries"
        },
        "qualitative_audit": {
            key: value for key, value in qualitative_audit.items() if key != "panels"
        },
        "curation_gates": {
            "source_zip_all_member_crc_verified": True,
            "single_declared_change_is_controller_warmup_0_to_20": True,
            "canonical_phase1_result_retained": True,
            "classification_best_epoch_recomputed_for_three_seeds": True,
            "classification_test_exactly_once_per_seed": True,
            "all_45_probe_candidates_recomputed": True,
            "all_15_probe_checkpoints_hash_and_strict_load_passed": True,
            "all_probe_metrics_recomputed_from_global_confusions": True,
            "probe_aggregate_exactly_recomputed": True,
            "pre_test_probe_selection_contains_no_test_results": True,
            "probe_test_exactly_once_per_selected_probe": True,
            "fixed_qualitative_panels_decoded": True,
        },
        "interpretation": {
            "canonical_alg_epoch2_shutdown_avoided": True,
            "warmup20_stop_epochs": [
                row["controller_stop_epoch"] for row in classification_rows
            ],
            "warmup20_improves_over_canonical_alg_classification": (
                classification_mean
                > next(
                    item["canonical_test_macro_top1"]
                    for item in classification_comparisons
                    if item["variant"] == "alg"
                )
            ),
            "warmup20_improves_over_canonical_alg_probe": (
                probe_mean
                > next(
                    item["canonical_test_input_224_mean_iou"]
                    for item in probe_comparisons
                    if item["variant"] == "alg"
                )
            ),
            "warmup20_probe_exceeds_both_ibkd_lambdas": all(
                item["warmup20_minus_canonical"] > 0
                for item in probe_comparisons
                if item["variant"] in {"ibkd_lambda_0.25", "ibkd_lambda_0.5"}
            ),
            "lg_remains_highest_probe": (
                probe_mean
                < next(
                    item["canonical_test_input_224_mean_iou"]
                    for item in probe_comparisons
                    if item["variant"] == "lg"
                )
            ),
            "causal_scope": "controller_warmup_effect_within_batch128_alg_only",
            "formal_p_value_reported": False,
            "diagnostic_must_not_replace_canonical_result": True,
        },
    }
    save_json(summary, report_dir / "summary.json")
    print(
        f"[ALG_W20_CURATION_DONE] classification=3 probes=15 "
        f"checkpoints={len(checkpoint_entries)} panels={qualitative_audit['panel_count']} "
        f"report={report_dir}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
