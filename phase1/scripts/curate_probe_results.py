#!/usr/bin/env python3
"""Audit raw Phase 1 probe outputs and create compact Git-trackable reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ibkd_seg.phase1.probe import Confusion, probe_from_state
from ibkd_seg.phase1.run_probe_full import (
    EXPECTED_ENCODER_SEEDS,
    EXPECTED_PROBE_SEEDS,
    EXPECTED_VARIANTS,
    LOCKED_PROTOCOL_SHA256,
    _aggregate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_phase1_v1.json"
PRIMARY_METRIC = "input_224_mean_iou"
METRIC_NAMES = (
    "foreground_iou",
    "background_iou",
    "mean_iou",
    "foreground_dice",
    "pixel_accuracy",
)


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
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _assert_number_equal(actual: int | float, expected: int | float) -> None:
    if isinstance(expected, int):
        if actual != expected:
            raise RuntimeError(f"metric count mismatch: {actual} != {expected}")
    elif not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"metric mismatch: {actual} != {expected}")


def _nested_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _nested_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _nested_equal(left, right) for left, right in zip(actual, expected)
        )
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15)
    return actual == expected


def _audit_metric_block(metrics: dict[str, Any]) -> None:
    confusion = Confusion(
        true_positive=int(metrics["true_positive"]),
        true_negative=int(metrics["true_negative"]),
        false_positive=int(metrics["false_positive"]),
        false_negative=int(metrics["false_negative"]),
        ignored=int(metrics["ignored"]),
    )
    recomputed = confusion.metrics()
    for name, expected in recomputed.items():
        _assert_number_equal(metrics[name], expected)
    if not all(math.isfinite(float(metrics[name])) for name in METRIC_NAMES):
        raise RuntimeError("non-finite probe metric")
    if not all(0.0 <= float(metrics[name]) <= 1.0 for name in METRIC_NAMES):
        raise RuntimeError("probe metric outside [0, 1]")


def _key(result: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(result["variant"]),
        int(result["encoder_seed"]),
        int(result["probe_seed"]),
    )


def _audit_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    initial_hashes: dict[int, set[str]] = {
        seed: set() for seed in EXPECTED_PROBE_SEEDS
    }
    order_hashes: dict[int, dict[int, set[str]]] = {
        seed: {epoch: set() for epoch in range(1, 101)}
        for seed in EXPECTED_PROBE_SEEDS
    }
    candidate_count = 0
    for result in results:
        candidates = result["candidate_audits"]
        if [float(candidate["learning_rate"]) for candidate in candidates] != [
            0.01,
            0.03,
            0.1,
        ]:
            raise RuntimeError(f"candidate LR grid mismatch: {_key(result)}")
        for candidate in candidates:
            candidate_count += 1
            history = candidate["history"]
            if len(history) != 100 or len(candidate["batch_order_sha256_by_epoch"]) != 100:
                raise RuntimeError(f"candidate epoch count mismatch: {_key(result)}")
            if [int(row["epoch"]) for row in history] != list(range(1, 101)):
                raise RuntimeError(f"candidate epoch sequence mismatch: {_key(result)}")
            if not all(
                math.isfinite(float(row["train_loss"]))
                and math.isfinite(float(row["validation_grid_mean_iou"]))
                for row in history
            ):
                raise RuntimeError(f"non-finite candidate history: {_key(result)}")
            best_index = max(
                range(len(history)),
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
            for epoch, digest in enumerate(
                candidate["batch_order_sha256_by_epoch"], start=1
            ):
                order_hashes[probe_seed][epoch].add(digest)

        selected_index = max(
            range(len(candidates)),
            key=lambda index: float(
                candidates[index]["best_validation_grid_mean_iou"]
            ),
        )
        selected = candidates[selected_index]
        selection = result["selection"]
        if (
            float(selection["learning_rate"]) != float(selected["learning_rate"])
            or int(selection["epoch"]) != int(selected["best_epoch"])
        ):
            raise RuntimeError(f"validation selection mismatch: {_key(result)}")
        _assert_number_equal(
            selection["validation_grid_mean_iou"],
            selected["best_validation_grid_mean_iou"],
        )

    if candidate_count != 270:
        raise RuntimeError(f"expected 270 candidates, found {candidate_count}")
    if not all(len(values) == 1 for values in initial_hashes.values()):
        raise RuntimeError("probe initial states differ within a probe seed")
    if not all(
        len(values) == 1
        for epoch_map in order_hashes.values()
        for values in epoch_map.values()
    ):
        raise RuntimeError("probe batch order differs within a probe seed/epoch")
    return {
        "candidate_count": candidate_count,
        "all_candidate_histories_have_100_epochs": True,
        "validation_best_epoch_recomputed": True,
        "validation_lr_selection_recomputed": True,
        "lower_lr_and_earlier_epoch_tie_break_verified": True,
        "same_initial_probe_state_per_probe_seed": True,
        "same_batch_order_per_probe_seed_and_epoch": True,
        "encoder_feature_gradient_tensor_count": 0,
        "probe_gradient_tensor_count": 2,
    }


def _audit_artifacts(
    raw_dir: Path,
    results: list[dict[str, Any]],
    probe_config: dict[str, Any],
) -> dict[str, Any]:
    expected_paths = {
        raw_dir / result["probe_artifact"]["relative_path"] for result in results
    }
    observed_paths = set(raw_dir.glob("probes/**/*.pt"))
    if observed_paths != expected_paths or len(observed_paths) != 90:
        raise RuntimeError("selected probe artifact set is incomplete or unexpected")
    entries: list[dict[str, Any]] = []
    for result in results:
        path = raw_dir / result["probe_artifact"]["relative_path"]
        digest = file_sha256(path)
        if digest != result["probe_artifact"]["sha256"]:
            raise RuntimeError(f"probe artifact hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checks = {
            "purpose": payload.get("purpose")
            == "phase1_scientific_full_frozen_probe",
            "scientific": payload.get("scientific_result") is True,
            "protocol": payload.get("protocol_sha256") == LOCKED_PROTOCOL_SHA256,
            "batch": payload.get("classification_batch_size") == 64,
            "variant": payload.get("variant") == result["variant"],
            "encoder_seed": payload.get("encoder_seed") == result["encoder_seed"],
            "probe_seed": payload.get("probe_seed") == result["probe_seed"],
            "test_zero_at_write": payload.get(
                "official_test_evaluations_at_checkpoint_write"
            )
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
            raise RuntimeError("probe parameter count changed")
        if not all(
            bool(torch.isfinite(tensor).all()) for tensor in payload["model"].values()
        ):
            raise RuntimeError(f"non-finite probe state: {path}")
        entries.append(
            {
                "variant": result["variant"],
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
    return {
        "schema_version": 1,
        "status": "pass",
        "checkpoint_count": len(entries),
        "all_file_hashes_match": True,
        "all_safe_loads_use_weights_only": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "all_written_before_official_test": True,
        "entries": entries,
    }


def _audit_raw_csv(raw_dir: Path, results: list[dict[str, Any]]) -> None:
    with (raw_dir / "probe_raw_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90:
        raise RuntimeError("raw result CSV does not have 90 rows")
    result_by_key = {_key(result): result for result in results}
    csv_keys = {
        (row["variant"], int(row["encoder_seed"]), int(row["probe_seed"]))
        for row in rows
    }
    if csv_keys != set(result_by_key):
        raise RuntimeError("raw CSV matrix differs from JSON matrix")
    for row in rows:
        result = result_by_key[
            (row["variant"], int(row["encoder_seed"]), int(row["probe_seed"]))
        ]
        checks = {
            "lr": float(row["selected_learning_rate"])
            == float(result["selection"]["learning_rate"]),
            "epoch": int(row["selected_epoch"]) == int(result["selection"]["epoch"]),
            "test_once": int(row["official_test_evaluations"]) == 1,
            "scientific": row["scientific_result"] == "True",
        }
        if not all(checks.values()):
            raise RuntimeError(f"raw CSV contract mismatch: {_key(result)}")
        _assert_number_equal(
            float(row["test_input_224_mean_iou"]),
            result["test"]["input_224"]["mean_iou"],
        )


def _copy_qualitative(
    raw_dir: Path,
    report_dir: Path,
    qualitative: dict[str, Any],
) -> dict[str, Any]:
    source_manifest = json.loads(
        (raw_dir / "qualitative/manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest != qualitative:
        raise RuntimeError("qualitative manifest differs from final summary")
    source_masks = sorted((raw_dir / "qualitative").glob("*/masks/*.png"))
    if len(source_masks) != 80:
        raise RuntimeError(f"expected 80 qualitative masks, found {len(source_masks)}")
    for path in source_masks:
        with Image.open(path) as image:
            if image.size != (224, 224) or image.mode != "RGB":
                raise RuntimeError(f"invalid qualitative mask: {path}")

    entries: list[dict[str, Any]] = []
    for source_set, destination_set in (
        ("ibkd_0.25", "ibkd_lambda_0.25"),
        ("ibkd_0.5", "ibkd_lambda_0.5"),
    ):
        panels = qualitative["panel_sets"][source_set]["panels"]
        if len(panels) != 8:
            raise RuntimeError(f"expected eight panels in {source_set}")
        for relative_source in panels:
            source_path = raw_dir / relative_source
            with Image.open(source_path) as image:
                image.load()
                size = image.size
                mode = image.mode
            if size != (1568, 252) or mode != "RGB":
                raise RuntimeError(f"invalid qualitative panel: {source_path}")
            destination = (
                report_dir / "figures" / destination_set / source_path.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            entries.append(
                {
                    "ibkd_panel_variant": destination_set,
                    "image_id": source_path.stem,
                    "path": destination.relative_to(report_dir).as_posix(),
                    "source_path_under_ignored_raw_root": relative_source,
                    "bytes": destination.stat().st_size,
                    "sha256": file_sha256(destination),
                    "width": size[0],
                    "height": size[1],
                    "mode": mode,
                }
            )
    return {
        "schema_version": 1,
        "status": "pass",
        "fixed_before_results": True,
        "posthoc_example_selection": False,
        "encoder_seed": qualitative["encoder_seed"],
        "probe_seed": qualitative["probe_seed"],
        "test_image_ids": qualitative["test_image_ids"],
        "panel_count": len(entries),
        "source_mask_count_audited_but_not_committed": len(source_masks),
        "panels": entries,
    }


def _paired_primary(
    aggregates: dict[str, Any],
    left: str,
    right: str,
) -> dict[str, Any]:
    rows = []
    for seed in EXPECTED_ENCODER_SEEDS:
        left_value = aggregates[left]["by_encoder_seed"][str(seed)]["test"][
            PRIMARY_METRIC
        ]["mean"]
        right_value = aggregates[right]["by_encoder_seed"][str(seed)]["test"][
            PRIMARY_METRIC
        ]["mean"]
        rows.append(
            {
                "encoder_seed": seed,
                "left": left_value,
                "right": right_value,
                "difference": left_value - right_value,
            }
        )
    differences = [row["difference"] for row in rows]
    return {
        "contrast": f"{left}_minus_{right}",
        "metric": "test_input_224_mean_iou",
        "per_encoder_seed": rows,
        "mean_difference": sum(differences) / len(differences),
        "all_encoder_seed_differences_positive": all(
            difference > 0 for difference in differences
        ),
        "all_encoder_seed_differences_negative": all(
            difference < 0 for difference in differences
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    suite = json.loads((raw_dir / "probe_summary.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (raw_dir / "selection_complete_before_test.json").read_text(encoding="utf-8")
    )
    status = json.loads((raw_dir / "sequence_status.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (raw_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    if suite.get("status") != "complete" or suite.get("scientific_result") is not True:
        raise RuntimeError("probe suite is not complete/scientific")
    if suite.get("protocol_sha256") != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError("probe suite protocol hash mismatch")
    if file_sha256(PROTOCOL_PATH) != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError("committed protocol hash changed")
    if not suite.get("contracts", {}).get("all_passed"):
        raise RuntimeError("probe suite contracts did not all pass")
    if status != {
        "active": None,
        "completed_official_test_evaluations": 90,
        "completed_selections": 90,
        "expected_official_test_evaluations": 90,
        "expected_selections": 90,
        "failure": None,
        "official_test_accessed": True,
        "phase": "complete",
        "scientific_result": True,
        "status": "complete",
        "updated_at_utc": status["updated_at_utc"],
    }:
        raise RuntimeError("final sequence status contract mismatch")

    results = suite["raw_results"]
    expected_keys = {
        (variant, encoder_seed, probe_seed)
        for variant in EXPECTED_VARIANTS
        for encoder_seed in EXPECTED_ENCODER_SEEDS
        for probe_seed in EXPECTED_PROBE_SEEDS
    }
    if {_key(result) for result in results} != expected_keys or len(results) != 90:
        raise RuntimeError("final result matrix is incomplete or duplicated")
    if any(int(result["official_test_evaluations"]) != 1 for result in results):
        raise RuntimeError("a selected probe was not tested exactly once")
    for result in results:
        for split in ("validation", "test"):
            for resolution in ("grid_14x14", "input_224"):
                _audit_metric_block(result[split][resolution])

    if (
        selection.get("status") != "complete"
        or selection.get("completed_selections") != 90
        or selection.get("official_test_accessed") is not False
        or selection.get("official_test_evaluations") != 0
        or not all(selection.get("selection_gates", {}).values())
    ):
        raise RuntimeError("pre-test selection record failed its contract")
    selected_by_key = {_key(result): result for result in selection["selected_probes"]}
    if set(selected_by_key) != expected_keys:
        raise RuntimeError("pre-test selection matrix differs from final matrix")
    for result in results:
        selected = selected_by_key[_key(result)]
        if selected.get("test") is not None or selected.get("official_test_evaluations") != 0:
            raise RuntimeError("pre-test record contains official test results")
        for field in (
            "variant",
            "method",
            "fusion_ratio_lambda",
            "encoder_seed",
            "probe_seed",
            "probe_artifact",
            "selection",
            "candidate_audits",
            "validation",
        ):
            if selected[field] != result[field]:
                raise RuntimeError(f"pre/post-test record mismatch: {_key(result)} {field}")

    candidate_audit = _audit_candidates(results)
    checkpoint_audit = _audit_artifacts(
        raw_dir,
        results,
        protocol["frozen_spatial_probe"]["probe"],
    )
    _audit_raw_csv(raw_dir, results)
    recomputed_aggregates = _aggregate(results)
    if not _nested_equal(recomputed_aggregates, suite["aggregates"]):
        raise RuntimeError("hierarchical aggregate recomputation mismatch")

    qualitative_audit = _copy_qualitative(
        raw_dir,
        report_dir,
        suite["qualitative"],
    )
    aggregates = suite["aggregates"]["variants"]
    method_summaries: list[dict[str, Any]] = []
    per_encoder_rows: list[dict[str, Any]] = []
    for variant in EXPECTED_VARIANTS:
        variant_aggregate = aggregates[variant]
        across = variant_aggregate["across_encoder_seed_means"]["test"]
        method_summaries.append(
            {
                "variant": variant,
                "test_input_224_mean_iou": across[PRIMARY_METRIC],
                "test_input_224_foreground_iou": across[
                    "input_224_foreground_iou"
                ],
                "test_input_224_background_iou": across[
                    "input_224_background_iou"
                ],
                "test_input_224_foreground_dice": across[
                    "input_224_foreground_dice"
                ],
                "test_input_224_pixel_accuracy": across[
                    "input_224_pixel_accuracy"
                ],
                "test_grid_14x14_mean_iou": across["grid_14x14_mean_iou"],
            }
        )
        for encoder_seed in EXPECTED_ENCODER_SEEDS:
            value = variant_aggregate["by_encoder_seed"][str(encoder_seed)]["test"][
                PRIMARY_METRIC
            ]
            per_encoder_rows.append(
                {
                    "variant": variant,
                    "encoder_seed": encoder_seed,
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

    ranking = [
        summary["variant"]
        for summary in sorted(
            method_summaries,
            key=lambda summary: summary["test_input_224_mean_iou"]["mean"],
            reverse=True,
        )
    ]
    contrasts = [
        _paired_primary(aggregates, left, right)
        for left in ("ibkd_lambda_0.25", "ibkd_lambda_0.5")
        for right in ("vanilla", "kd", "lg", "alg")
    ]
    primary_contrasts = [
        contrast for contrast in contrasts if contrast["contrast"].endswith("_minus_alg")
    ]
    primary_supported = all(
        contrast["all_encoder_seed_differences_positive"]
        for contrast in primary_contrasts
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "raw_results.csv").write_text(
        (raw_dir / "probe_raw_results.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    shutil.copyfile(
        raw_dir / "sequence_status.json", report_dir / "h200_sequence_status.json"
    )
    shutil.copyfile(
        raw_dir / "qualitative/manifest.json",
        report_dir / "h200_qualitative_manifest.json",
    )
    shutil.copyfile(
        raw_dir / "artifact_manifest.json", report_dir / "source_manifest.json"
    )
    save_json(checkpoint_audit, report_dir / "checkpoint_manifest.json")
    save_json(qualitative_audit, report_dir / "figures/manifest.json")
    write_csv(per_encoder_rows, report_dir / "per_encoder_seed.csv")

    summary = {
        "schema_version": 1,
        "status": "complete_primary_hypothesis_not_supported",
        "dataset": "Oxford-IIIT Pet",
        "experiment": "phase1_batch64_frozen_spatial_probe",
        "classification_batch_size": 64,
        "h200_issue_id": source_manifest["h200_issue_id"],
        "runtime_git_commit": suite["runtime"]["git_commit"],
        "protocol_sha256": suite["protocol_sha256"],
        "source_archive": source_manifest["source_archive"],
        "ignored_raw_artifact_root": raw_dir.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "source_full_summary_sha256": file_sha256(raw_dir / "probe_summary.json"),
        "matrix": suite["matrix"],
        "data_counts": suite["data"]["counts"],
        "test_policy": suite["test_policy"],
        "protocol_contracts": suite["contracts"],
        "timing": suite["timing"],
        "peak_cuda_memory_bytes": suite["peak_cuda_memory_bytes"],
        "methods": method_summaries,
        "ranking_by_test_input_224_mean_iou": ranking,
        "paired_primary_contrasts_from_source": suite["aggregates"][
            "paired_primary_contrasts"
        ],
        "paired_contrasts": contrasts,
        "baselines": suite["baselines"],
        "candidate_audit": candidate_audit,
        "checkpoint_audit": {
            key: value for key, value in checkpoint_audit.items() if key != "entries"
        },
        "qualitative_audit": {
            key: value for key, value in qualitative_audit.items() if key != "panels"
        },
        "curation_gates": {
            "source_zip_all_member_crc_verified": source_manifest["source_archive"][
                "all_member_crc_verified"
            ],
            "complete_6_variant_3_encoder_5_probe_matrix": True,
            "all_270_candidates_recomputed": True,
            "all_90_probe_artifacts_hash_and_strict_load_passed": True,
            "all_metrics_recomputed_from_global_confusions": True,
            "hierarchical_aggregates_exactly_recomputed": True,
            "pre_test_selection_record_contains_no_test_results": True,
            "official_test_exactly_once_per_selected_probe": True,
            "fixed_qualitative_panels_decoded": True,
        },
        "interpretation": {
            "primary_claim": "ibkd_test_input_224_mean_iou_exceeds_matched_alg",
            "primary_claim_supported": primary_supported,
            "all_ibkd_minus_alg_encoder_seed_differences_negative": all(
                contrast["all_encoder_seed_differences_negative"]
                for contrast in primary_contrasts
            ),
            "decision_scope": "no_go_for_batch64_v1_ibkd_greater_than_alg_claim",
            "ibkd_exceeds_kd_for_both_lambdas_and_all_encoder_seeds": all(
                contrast["all_encoder_seed_differences_positive"]
                for contrast in contrasts
                if contrast["contrast"].endswith("_minus_kd")
            ),
            "lg_is_highest": ranking[0] == "lg",
            "formal_p_value_reported": False,
            "posthoc_protocol_change_forbidden": True,
        },
    }
    save_json(summary, report_dir / "summary.json")
    print(
        f"[PROBE_CURATION_DONE] results={len(results)} "
        f"checkpoints={checkpoint_audit['checkpoint_count']} "
        f"panels={qualitative_audit['panel_count']} "
        f"primary_supported={primary_supported} report={report_dir}",
        flush=True,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
