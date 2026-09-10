#!/usr/bin/env python3
"""Run a locked CUB direct-spatial v2 full encoder-seed shard."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .cub_data import DATASET_NAME, NUM_CLASSES, resolve_dataset_root
from .cub_direct_spatial import (
    GRID_SIZE,
    assert_probability,
    attention_gt_metrics,
    attention_rollout,
    build_part_probe,
    evaluate_part_probe,
    load_mask_views,
    load_part_supervision,
    load_spatial_annotations,
    save_attention_triptych,
    summarize_part_supervision,
    train_part_candidate,
)
from .cub_probe_data import (
    CubImageDataset,
    CubProbeRecord,
    load_official_test_records,
    load_train_validation_records,
)
from .run_cub_combined_smoke import _atomic_json_save, _atomic_torch_save, _runtime
from .run_cub_direct_spatial_smoke import (
    EXPECTED_VALIDATION_SHA256,
    EXPECTED_VARIANTS,
    _cka_smoke,
    _extract_last_features,
    _load_json,
    _load_student,
    _resolve_repository_path,
    _save_cka_heatmap,
    _validate_config as _validate_metric_protocol,
    _write_csv,
    file_digest_from_ids,
)
from .run_cub_r50_teacher_full import load_scientific_teacher
from .train_timing import file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_seed1_direct_spatial_full_v2.json"
)
EXPECTED_CONFIG_SHA256 = (
    "90f7dc92b7e1ad27b6a4a4b68e91bb5e72ea021389304d87dda5950fac8e6017"
)
SEED23_CONFIG_SHA256 = (
    "54980771cf910543a3aba24c0a5ff86de6a0dce34866c662025409ab3abd0691"
)
EXPECTED_METRIC_CONFIG_SHA256 = (
    "55ac0598c11a4065f3b1416022e8fbb3de35b21cad94780ace2e2036d5430bc6"
)
SEED23_METRIC_CONFIG_SHA256 = (
    "bd71b02ebcca3240c7278819b4a34416a131914bd442887afcce6251a7e366d1"
)
QUALITATIVE_TEST_IDS = (787, 2285, 3735, 5205, 6691, 8139, 9597, 11064)


def log(message: str = "") -> None:
    print(message, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_part_validity_audit(split: str, audit: dict[str, Any]) -> None:
    log(
        "[CUB_PART_VALIDITY_AUDIT] "
        f"split={split} records={audit['records']} "
        f"official_visible={audit['official_visible_keypoints']} "
        f"valid_visible={audit['valid_visible_keypoints']} "
        "excluded_out_of_frame_visible="
        f"{audit['excluded_out_of_frame_visible_keypoints']} "
        f"affected_images={audit['affected_image_count']} "
        f"images_without_valid={len(audit['images_without_valid_keypoints'])} "
        "coordinate_clipping=false"
    )
    for row in audit["excluded_landmarks"]:
        log(
            "[CUB_PART_VALIDITY_EXCLUSION] "
            f"split={split} image_id={row['image_id']} "
            f"part_ids={','.join(str(value) for value in row['part_ids'])} "
            f"coordinates={json.dumps(row['coordinates'], separators=(',', ':'))} "
            f"image_size={json.dumps(row['image_size'], separators=(',', ':'))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-release-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--cka-batch-size", type=int, default=8)
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _validate_seed1_config(config: dict[str, Any], path: Path) -> dict[str, Any]:
    scope = config.get("scope", {})
    dataset = config.get("dataset", {})
    part = config.get("part_localization_probe", {})
    cka = config.get("spatial_cka", {})
    attention = config.get("attention_gt", {})
    policy = config.get("official_test_policy", {})
    statistics_config = config.get("statistics", {})
    execution = config.get("execution", {})
    gate = config.get("completion_gate", {})
    checks = {
        "config_hash": file_sha256(path) == EXPECTED_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_seed1_direct_spatial_full_v2",
        "locked_status": config.get("status")
        == "locked_after_v1_annotation_preflight_failure_before_any_metric_2026-09-10",
        "scientific": config.get("scientific_result") is True,
        "partial_scope": config.get("result_scope")
        == "precommitted_encoder_seed1_shard_pending_encoder_seeds2_3_for_final_statistics",
        "revision": config.get("revision")
        == {
            "supersedes": (
                "cub200_phase1_r50_224_b128_seed1_direct_spatial_full_v1"
            ),
            "reason": (
                "v1 stopped before probe training because official-visible CUB "
                "image 5007 part 4 is outside the 500x333 image at coordinate "
                "405,344"
            ),
            "v1_probe_training_started": False,
            "v1_official_test_accessed": False,
            "metric_or_method_result_used_to_choose_revision": False,
        },
        "scope": scope
        == {
            "student_batch_size": 128,
            "encoder_seeds_in_this_run": [1],
            "eventual_encoder_seeds": [1, 2, 3],
            "variants": list(EXPECTED_VARIANTS),
            "method_lambda_or_protocol_selection_from_smoke": False,
            "seed1_alone_is_not_final_encoder_seed_inference": True,
        },
        "dataset": dataset.get("name") == DATASET_NAME
        and dataset.get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_SHA256,
        }
        and dataset.get("input_size") == 224
        and dataset.get("part_count") == 15
        and dataset.get("visible_parts_only") is True
        and dataset.get("part_coordinate_validity")
        == {
            "coordinate_domain": (
                "continuous_original_image_0_le_x_le_width_and_0_le_y_le_height"
            ),
            "validity": "official_visible_and_in_image_bounds",
            "out_of_frame_visible_action": (
                "exclude_from_part_probe_loss_validation_selection_and_pck"
            ),
            "coordinate_clipping": False,
            "audit": (
                "per_split_counts_image_ids_part_ids_coordinates_and_image_sizes"
            ),
            "train_validation_audited_before_probe_training": True,
            "official_test_audited_only_after_all_validation_selections": True,
        },
        "part": part.get("head") == "Conv2d(192,15,1,bias=True)"
        and part.get("target")
        == "valid_visible_part_gaussian_heatmap_sigma_1_grid_pixel"
        and part.get("loss") == "valid_visible_part_masked_mean_squared_error"
        and part.get("learning_rates") == [0.01, 0.03, 0.1]
        and part.get("epochs") == 100
        and part.get("batch_size") == 64
        and part.get("probe_seeds") == [1, 2, 3, 4, 5]
        and part.get("official_test_metric")
        == "valid_visible_keypoint_micro_PCK_at_0.1"
        and part.get("official_test_once_per_validation_selected_probe") is True,
        "cka": cka.get("split") == "fixed_validation_600"
        and cka.get("student_blocks") == list(range(12))
        and cka.get("teacher_feature") == "resnet50_layer3_1024x14x14"
        and cka.get("metric") == "centered_linear_CKA"
        and cka.get("accumulator_dtype") == "float64"
        and cka.get("official_test_used") is False,
        "attention": attention.get("split") == "official_test_5794"
        and attention.get("primary_metric")
        == "global_micro_patch_average_precision"
        and attention.get("qualitative_test_image_ids")
        == list(QUALITATIVE_TEST_IDS)
        and attention.get("official_test_once_per_encoder") is True,
        "test_policy": policy
        == {
            "all_20_part_probe_validation_selections_complete_before_test_open": True,
            "selection_uses_official_test": False,
            "no_method_lambda_lr_epoch_or_protocol_change_from_test": True,
            "seed2_3_will_use_identical_locked_settings_regardless_of_seed1_results": True,
        },
        "statistics": statistics_config
        == {
            "probe_seed_is_nested_within_encoder_seed": True,
            "report_probe_seed_values_mean_and_sample_standard_deviation": True,
            "independent_encoder_seed_n_in_this_run": 1,
            "encoder_seed_standard_deviation_not_estimable": True,
            "final_mean_sd_and_claim_wait_for_encoder_seeds_2_3": True,
        },
        "execution": execution.get("requested_mig_slices") == 1
        and execution.get("feature_batch_size") == 16
        and execution.get("cka_batch_size") == 8
        and execution.get("attention_batch_size") == 16
        and execution.get("num_workers") == 4,
        "gate": gate
        == {
            "checkpoint_strict_loads": 4,
            "part_probe_lr_candidates": 60,
            "part_probe_validation_selections": 20,
            "part_probe_official_test_evaluations": 20,
            "spatial_cka_values": 48,
            "attention_metric_rows": 4,
            "attention_official_test_evaluations": 4,
            "qualitative_pngs": 32,
            "official_test_evaluations": 24,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB direct-spatial seed-1 full config: "
            + ", ".join(failures)
        )

    metric_source = config["locked_metric_protocol"]
    metric_path = _resolve_repository_path(metric_source["path"])
    if (
        metric_source.get("sha256") != EXPECTED_METRIC_CONFIG_SHA256
        or not metric_path.is_file()
        or file_sha256(metric_path) != EXPECTED_METRIC_CONFIG_SHA256
    ):
        raise RuntimeError("locked direct-spatial metric protocol changed")
    metric_config = _load_json(metric_path)
    _validate_metric_protocol(metric_config, metric_path)

    for source_name in ("students", "teacher"):
        source = config["checkpoint_sources"][source_name]
        manifest_path = _resolve_repository_path(source["manifest_path"])
        if (
            not manifest_path.is_file()
            or file_sha256(manifest_path) != source["manifest_sha256"]
        ):
            raise RuntimeError(
                f"direct-spatial full checkpoint manifest changed: {source_name}"
            )
    return metric_config


def _validate_seed23_config(config: dict[str, Any], path: Path) -> dict[str, Any]:
    scope = config.get("scope", {})
    dataset = config.get("dataset", {})
    part = config.get("part_localization_probe", {})
    cka = config.get("spatial_cka", {})
    attention = config.get("attention_gt", {})
    policy = config.get("official_test_policy", {})
    statistics_config = config.get("statistics", {})
    execution = config.get("execution", {})
    gate = config.get("completion_gate", {})
    checks = {
        "config_hash": file_sha256(path) == SEED23_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_seed2_3_direct_spatial_full_v2",
        "locked_status": config.get("status")
        == "locked_after_seed2_3_smoke_pass_before_full_results_2026-09-10",
        "scientific": config.get("scientific_result") is True,
        "partial_scope": config.get("result_scope")
        == (
            "precommitted_encoder_seeds2_3_shard_to_complete_three_seed_"
            "inference_with_issue737_seed1"
        ),
        "inheritance": config.get("protocol_inheritance")
        == {
            "seed1_full_v2_path": (
                "phase1/phase1_cub/configs/"
                "cub200_r50_224_b128_seed1_direct_spatial_full_v2.json"
            ),
            "seed1_full_v2_sha256": EXPECTED_CONFIG_SHA256,
            "seed2_3_smoke_v2_path": (
                "phase1/phase1_cub/configs/"
                "cub200_r50_224_b128_seed2_3_direct_spatial_smoke_v2.json"
            ),
            "seed2_3_smoke_v2_sha256": SEED23_METRIC_CONFIG_SHA256,
            "source_smoke_h200_issue": 738,
            "source_smoke_status": "pass",
            "method_lambda_metric_or_protocol_changed_after_seed1_or_smoke": False,
        },
        "scope": scope
        == {
            "student_batch_size": 128,
            "encoder_seeds_in_this_run": [2, 3],
            "eventual_encoder_seeds": [1, 2, 3],
            "variants": list(EXPECTED_VARIANTS),
            "method_lambda_or_protocol_selection_from_smoke": False,
            "this_shard_alone_is_not_final_encoder_seed_inference": True,
            "combine_only_with_audited_issue737_seed1": True,
        },
        "dataset": dataset.get("name") == DATASET_NAME
        and dataset.get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_SHA256,
        }
        and dataset.get("input_size") == 224
        and dataset.get("part_count") == 15
        and dataset.get("visible_parts_only") is True
        and dataset.get("part_coordinate_validity")
        == {
            "coordinate_domain": (
                "continuous_original_image_0_le_x_le_width_and_0_le_y_le_height"
            ),
            "validity": "official_visible_and_in_image_bounds",
            "out_of_frame_visible_action": (
                "exclude_from_part_probe_loss_validation_selection_and_pck"
            ),
            "coordinate_clipping": False,
            "audit": (
                "per_split_counts_image_ids_part_ids_coordinates_and_image_sizes"
            ),
            "train_validation_audited_before_probe_training": True,
            "official_test_audited_only_after_all_validation_selections": True,
        },
        "part": part.get("head") == "Conv2d(192,15,1,bias=True)"
        and part.get("target")
        == "valid_visible_part_gaussian_heatmap_sigma_1_grid_pixel"
        and part.get("loss") == "valid_visible_part_masked_mean_squared_error"
        and part.get("learning_rates") == [0.01, 0.03, 0.1]
        and part.get("epochs") == 100
        and part.get("batch_size") == 64
        and part.get("probe_seeds") == [1, 2, 3, 4, 5]
        and part.get("official_test_metric")
        == "valid_visible_keypoint_micro_PCK_at_0.1"
        and part.get("official_test_once_per_validation_selected_probe") is True,
        "cka": cka.get("split") == "fixed_validation_600"
        and cka.get("student_blocks") == list(range(12))
        and cka.get("teacher_feature") == "resnet50_layer3_1024x14x14"
        and cka.get("metric") == "centered_linear_CKA"
        and cka.get("accumulator_dtype") == "float64"
        and cka.get("official_test_used") is False,
        "attention": attention.get("split") == "official_test_5794"
        and attention.get("primary_metric")
        == "global_micro_patch_average_precision"
        and attention.get("qualitative_test_image_ids")
        == list(QUALITATIVE_TEST_IDS)
        and attention.get("official_test_once_per_encoder") is True,
        "test_policy": policy
        == {
            "all_40_part_probe_validation_selections_complete_before_test_open": True,
            "selection_uses_official_test": False,
            "no_method_lambda_lr_epoch_or_protocol_change_from_test": True,
            "identical_v2_settings_to_seed1_regardless_of_seed1_or_smoke_results": True,
        },
        "statistics": statistics_config
        == {
            "probe_seed_is_nested_within_encoder_seed": True,
            "report_probe_seed_values_mean_and_sample_standard_deviation": True,
            "independent_encoder_seed_n_in_this_run": 2,
            "report_encoder_seed_values_mean_and_sample_standard_deviation_for_this_shard": True,
            "final_three_seed_statistics_require_audited_issue737_seed1": True,
        },
        "execution": execution.get("requested_mig_slices") == 1
        and execution.get("feature_batch_size") == 16
        and execution.get("cka_batch_size") == 8
        and execution.get("attention_batch_size") == 16
        and execution.get("num_workers") == 4,
        "gate": gate
        == {
            "checkpoint_strict_loads": 8,
            "part_probe_lr_candidates": 120,
            "part_probe_validation_selections": 40,
            "part_probe_official_test_evaluations": 40,
            "spatial_cka_values": 96,
            "attention_metric_rows": 8,
            "attention_official_test_evaluations": 8,
            "qualitative_pngs": 64,
            "official_test_evaluations": 48,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB direct-spatial seed-2/3 full config: "
            + ", ".join(failures)
        )

    inheritance = config["protocol_inheritance"]
    for path_key, digest_key in (
        ("seed1_full_v2_path", "seed1_full_v2_sha256"),
        ("seed2_3_smoke_v2_path", "seed2_3_smoke_v2_sha256"),
    ):
        source_path = _resolve_repository_path(inheritance[path_key])
        if (
            not source_path.is_file()
            or file_sha256(source_path) != inheritance[digest_key]
        ):
            raise RuntimeError(f"direct-spatial inherited protocol changed: {path_key}")

    metric_source = config["locked_metric_protocol"]
    metric_path = _resolve_repository_path(metric_source["path"])
    if (
        metric_source.get("sha256") != SEED23_METRIC_CONFIG_SHA256
        or not metric_path.is_file()
        or file_sha256(metric_path) != SEED23_METRIC_CONFIG_SHA256
    ):
        raise RuntimeError("locked seed-2/3 direct-spatial metric protocol changed")
    metric_config = _load_json(metric_path)
    _validate_metric_protocol(metric_config, metric_path)
    inputs = metric_config.get("checkpoint_inputs", [])
    if (
        len(inputs) != 8
        or [(item.get("encoder_seed"), item.get("variant")) for item in inputs]
        != [(seed, variant) for seed in (2, 3) for variant in EXPECTED_VARIANTS]
    ):
        raise RuntimeError("seed-2/3 full checkpoint input inventory changed")

    for source_name in ("students", "teacher"):
        source = config["checkpoint_sources"][source_name]
        manifest_path = _resolve_repository_path(source["manifest_path"])
        if (
            not manifest_path.is_file()
            or file_sha256(manifest_path) != source["manifest_sha256"]
        ):
            raise RuntimeError(
                f"direct-spatial full checkpoint manifest changed: {source_name}"
            )
    return metric_config


def _validate_config(config: dict[str, Any], path: Path) -> dict[str, Any]:
    digest = file_sha256(path)
    if digest == EXPECTED_CONFIG_SHA256:
        return _validate_seed1_config(config, path)
    if digest == SEED23_CONFIG_SHA256:
        return _validate_seed23_config(config, path)
    raise RuntimeError(f"unsupported CUB direct-spatial full config hash: {digest}")


def _validate_cli(args: argparse.Namespace, config: dict[str, Any]) -> None:
    execution = config["execution"]
    actual = {
        "feature_batch_size": args.feature_batch_size,
        "cka_batch_size": args.cka_batch_size,
        "attention_batch_size": args.attention_batch_size,
        "num_workers": args.num_workers,
    }
    expected = {key: execution[key] for key in actual}
    if actual != expected:
        raise RuntimeError(
            f"direct-spatial full CLI changed locked values: {actual} != {expected}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUB direct-spatial full experiment requires CUDA")


def _learning_rate_slug(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _select_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("cannot select from an empty part-probe candidate list")
    return min(
        candidates,
        key=lambda value: (
            -value["best_validation"]["micro_pck_at_0.1"],
            value["learning_rate"],
            value["best_epoch"],
        ),
    )


def _run_part_validation(
    *,
    variant: str,
    encoder_seed: int,
    model: torch.nn.Module,
    train_records: Sequence[CubProbeRecord],
    validation_records: Sequence[CubProbeRecord],
    train_supervision: dict[str, torch.Tensor],
    validation_supervision: dict[str, torch.Tensor],
    config: dict[str, Any],
    config_sha256: str,
    output_dir: Path,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    train_features = _extract_last_features(
        model,
        train_records,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        device=device,
    )
    validation_features = _extract_last_features(
        model,
        validation_records,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        device=device,
    )
    protocol = config["part_localization_probe"]
    candidate_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    initial_hashes: dict[int, str] = {}
    for probe_seed in protocol["probe_seeds"]:
        candidates: list[dict[str, Any]] = []
        one_seed_initial_hashes: set[str] = set()
        for learning_rate in protocol["learning_rates"]:
            result = train_part_candidate(
                train_features=train_features,
                train_supervision=train_supervision,
                validation_features=validation_features,
                validation_supervision=validation_supervision,
                learning_rate=float(learning_rate),
                epochs=int(protocol["epochs"]),
                seed=int(probe_seed),
                batch_size=int(protocol["batch_size"]),
                device=device,
            )
            one_seed_initial_hashes.add(result["initial_probe_state_sha256"])
            history_stem = (
                f"{variant}_probe_seed{probe_seed}"
                if config["scope"]["encoder_seeds_in_this_run"] == [1]
                else f"{variant}_encoder_seed{encoder_seed}_probe_seed{probe_seed}"
            )
            history_relative = (
                Path("part_probe/histories")
                / f"{history_stem}_lr{_learning_rate_slug(float(learning_rate))}.json"
            )
            _atomic_json_save(
                {
                    "variant": variant,
                    "encoder_seed": encoder_seed,
                    "probe_seed": probe_seed,
                    "learning_rate": learning_rate,
                    "epochs": protocol["epochs"],
                    "initial_probe_state_sha256": result[
                        "initial_probe_state_sha256"
                    ],
                    "history": result["history"],
                    "scientific_result": True,
                    "config_sha256": config_sha256,
                },
                output_dir / history_relative,
            )
            candidates.append(result)
            log(
                "[DIRECT_PART_FULL_CANDIDATE] "
                f"variant={variant} encoder_seed={encoder_seed} "
                f"probe_seed={probe_seed} "
                f"lr={learning_rate} best_epoch={result['best_epoch']} "
                "val_pck="
                f"{result['best_validation']['micro_pck_at_0.1']:.6f}"
            )
        if len(one_seed_initial_hashes) != 1:
            raise RuntimeError(
                f"part-probe initialization changed across LR: {variant}/{probe_seed}"
            )
        initial_hashes[int(probe_seed)] = next(iter(one_seed_initial_hashes))
        selected = _select_candidate(candidates)
        probe = build_part_probe(int(probe_seed))
        incompatible = probe.load_state_dict(selected["probe_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("selected full part probe strict reload failed")
        probe.to(device).eval()
        validation_metrics = evaluate_part_probe(
            probe,
            validation_features,
            validation_supervision,
            device=device,
            batch_size=int(protocol["batch_size"]),
        )
        if validation_metrics != selected["best_validation"]:
            raise RuntimeError("selected full part probe validation changed")

        checkpoint_relative = (
            Path("part_probe/checkpoints")
            / (
                f"{variant}_encoder_seed{encoder_seed}_"
                f"probe_seed{probe_seed}_best_validation.pt"
            )
        )
        checkpoint = output_dir / checkpoint_relative
        _atomic_torch_save(
            {
                "probe": selected["probe_state"],
                "metadata": {
                    "purpose": (
                        f"cub_direct_spatial_part_probe_seed{encoder_seed}_full_v2"
                    ),
                    "variant": variant,
                    "encoder_seed": encoder_seed,
                    "probe_seed": probe_seed,
                    "learning_rate": selected["learning_rate"],
                    "selected_epoch": selected["best_epoch"],
                    "selection_metric": "validation_micro_PCK_at_0.1",
                    "official_test_evaluations_at_checkpoint_write": 0,
                    "config_sha256": config_sha256,
                },
            },
            checkpoint,
        )
        checkpoint_hash = file_sha256(checkpoint)
        for candidate in candidates:
            candidate_rows.append(
                {
                    "variant": variant,
                    "encoder_seed": encoder_seed,
                    "probe_seed": probe_seed,
                    "learning_rate": candidate["learning_rate"],
                    "best_epoch": candidate["best_epoch"],
                    "validation_micro_pck_at_0.1": candidate["best_validation"][
                        "micro_pck_at_0.1"
                    ],
                    "validation_mean_normalized_error": candidate[
                        "best_validation"
                    ]["mean_normalized_localization_error"],
                    "selected": candidate is selected,
                    "scientific_result": True,
                }
            )
        selections.append(
            {
                "variant": variant,
                "encoder_seed": encoder_seed,
                "probe_seed": probe_seed,
                "selected_learning_rate": selected["learning_rate"],
                "selected_epoch": selected["best_epoch"],
                "validation": validation_metrics,
                "checkpoint_relative_path": checkpoint_relative.as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "initial_probe_state_sha256": initial_hashes[int(probe_seed)],
                "official_test_evaluations_at_selection": 0,
                "scientific_result": True,
            }
        )
        del probe, candidates
    del train_features, validation_features
    return candidate_rows, selections, initial_hashes


def _evaluate_part_test(
    *,
    variant: str,
    encoder_seed: int,
    model: torch.nn.Module,
    test_records: Sequence[CubProbeRecord],
    test_supervision: dict[str, torch.Tensor],
    selections: Sequence[dict[str, Any]],
    config: dict[str, Any],
    config_sha256: str,
    output_dir: Path,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    test_features = _extract_last_features(
        model,
        test_records,
        batch_size=feature_batch_size,
        num_workers=num_workers,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    for selection in selections:
        checkpoint = output_dir / selection["checkpoint_relative_path"]
        if (
            not checkpoint.is_file()
            or file_sha256(checkpoint) != selection["checkpoint_sha256"]
        ):
            raise RuntimeError("selected part-probe checkpoint byte audit failed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        if (
            metadata.get("config_sha256") != config_sha256
            or metadata.get("encoder_seed") != encoder_seed
            or metadata.get("official_test_evaluations_at_checkpoint_write") != 0
        ):
            raise RuntimeError("selected part-probe checkpoint metadata changed")
        probe = build_part_probe(int(selection["probe_seed"]))
        incompatible = probe.load_state_dict(payload["probe"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("selected part-probe official-test reload failed")
        probe.to(device).eval()
        metrics = evaluate_part_probe(
            probe,
            test_features,
            test_supervision,
            device=device,
            batch_size=int(config["part_localization_probe"]["batch_size"]),
        )
        rows.append(
            {
                "variant": variant,
                "encoder_seed": encoder_seed,
                "probe_seed": selection["probe_seed"],
                "selected_learning_rate": selection["selected_learning_rate"],
                "selected_epoch": selection["selected_epoch"],
                "test": metrics,
                "official_test_evaluations": 1,
                "scientific_result": True,
            }
        )
        log(
            "[DIRECT_PART_FULL_TEST] "
            f"variant={variant} encoder_seed={encoder_seed} "
            f"probe_seed={selection['probe_seed']} "
            f"selected_lr={selection['selected_learning_rate']} "
            f"selected_epoch={selection['selected_epoch']} "
            f"test_pck={metrics['micro_pck_at_0.1']:.6f}"
        )
        del probe, payload
    del test_features
    return rows


@torch.no_grad()
def _run_attention_test(
    *,
    variant: str,
    encoder_seed: int,
    student: torch.nn.Module,
    records: Sequence[CubProbeRecord],
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        CubImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    qualitative_ids = set(QUALITATIVE_TEST_IDS)
    available_ids = {record.image_id for record in records}
    if not qualitative_ids.issubset(available_ids):
        raise RuntimeError("locked qualitative image is not in official CUB test")
    saved_ids: set[int] = set()
    rollout_values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    occupancies: list[torch.Tensor] = []
    offset = 0
    for images, _image_ids in loader:
        batch_rollout = attention_rollout(
            student, images.to(device, non_blocking=True)
        ).cpu()
        batch_records = records[offset : offset + len(batch_rollout)]
        for record, one_rollout in zip(
            batch_records, batch_rollout, strict=True
        ):
            mask, occupancy = load_mask_views(record)
            masks.append(mask)
            occupancies.append(occupancy)
            if record.image_id in qualitative_ids:
                save_attention_triptych(
                    record=record,
                    rollout=one_rollout,
                    destination=(
                        output_dir
                        / "attention_gt/qualitative"
                        / (
                            f"{variant}_encoder_seed{encoder_seed}_"
                            f"image{record.image_id}.png"
                        )
                    ),
                )
                saved_ids.add(record.image_id)
        rollout_values.append(batch_rollout)
        offset += len(batch_rollout)
    if offset != len(records) or saved_ids != qualitative_ids:
        raise RuntimeError("official-test attention record or qualitative count mismatch")
    rollout = torch.cat(rollout_values)
    metrics = attention_gt_metrics(
        rollout,
        torch.stack(masks),
        torch.stack(occupancies),
    )
    for key in (
        "global_micro_patch_average_precision",
        "pointing_game_peak_inside_mask",
        "foreground_attention_mass_mean",
    ):
        assert_probability(metrics[key], name=f"{variant}/{key}")
    result = {
        "variant": variant,
        "encoder_seed": encoder_seed,
        **metrics,
        "attention_rollout_layers": 12,
        "qualitative_image_ids": list(QUALITATIVE_TEST_IDS),
        "qualitative_png_count": len(QUALITATIVE_TEST_IDS),
        "official_test_evaluations": 1,
        "scientific_result": True,
    }
    log(
        "[DIRECT_ATTENTION_FULL_TEST] "
        f"variant={variant} encoder_seed={encoder_seed} "
        f"patch_ap={metrics['global_micro_patch_average_precision']:.6f} "
        f"pointing={metrics['pointing_game_peak_inside_mask']:.6f} "
        f"foreground_mass={metrics['foreground_attention_mass_mean']:.6f}"
    )
    return result


def _part_aggregates(test_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    encoder_seeds = sorted({int(row["encoder_seed"]) for row in test_rows})
    for encoder_seed in encoder_seeds:
        for variant in EXPECTED_VARIANTS:
            rows = [
                row
                for row in test_rows
                if row["variant"] == variant
                and int(row["encoder_seed"]) == encoder_seed
            ]
            if [row["probe_seed"] for row in rows] != [1, 2, 3, 4, 5]:
                raise RuntimeError(
                    "part-probe seed inventory mismatch: "
                    f"encoder_seed={encoder_seed} variant={variant}"
                )
            pck_values = [row["test"]["micro_pck_at_0.1"] for row in rows]
            error_values = [
                row["test"]["mean_normalized_localization_error"] for row in rows
            ]
            aggregates.append(
                {
                    "variant": variant,
                    "encoder_seed": encoder_seed,
                    "probe_seeds": [1, 2, 3, 4, 5],
                    "probe_seed_pck_values": pck_values,
                    "test_micro_pck_at_0.1_probe_seed_mean": statistics.fmean(
                        pck_values
                    ),
                    "test_micro_pck_at_0.1_probe_seed_sample_sd": statistics.stdev(
                        pck_values
                    ),
                    "test_mean_normalized_error_probe_seed_mean": statistics.fmean(
                        error_values
                    ),
                    "independent_encoder_seed_n": 1,
                    "encoder_seed_standard_deviation_not_estimable": True,
                }
            )
    return aggregates


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
    metric_config = _validate_config(config, config_path)
    _validate_cli(args, config)
    config_sha256 = file_sha256(config_path)
    checkpoint_inputs = metric_config["checkpoint_inputs"]
    encoder_seeds = list(config["scope"]["encoder_seeds_in_this_run"])
    expected_gate = config["completion_gate"]
    expected_selections = expected_gate["part_probe_validation_selections"]

    import timm

    if timm.__version__ != "1.0.27":
        raise RuntimeError(f"expected timm==1.0.27, found {timm.__version__}")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.cuda.reset_peak_memory_stats(device)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "sequence_status.json"
    started_at = _utc_now()
    started = time.perf_counter()
    _atomic_json_save(
        {
            "status": "running",
            "phase": "dataset_and_checkpoint_audit",
            "started_at_utc": started_at,
            "config_sha256": config_sha256,
            "official_test_accessed": False,
        },
        status_path,
    )

    splits, split_manifest, train_source = load_train_validation_records(
        args.data_dir.expanduser().resolve(), download=True
    )
    if (
        len(splits["train"]) != 5394
        or len(splits["validation"]) != 600
        or split_manifest.get("validation_image_ids_sha256")
        != EXPECTED_VALIDATION_SHA256
    ):
        raise RuntimeError("direct-spatial full CUB split changed")
    train_records = splits["train"]
    validation_records = splits["validation"]
    dataset_root = resolve_dataset_root(args.data_dir.expanduser().resolve())
    part_names, annotations = load_spatial_annotations(dataset_root)
    if len(annotations) != 11788 or len(part_names) != 15:
        raise RuntimeError("official CUB spatial annotation inventory changed")
    train_supervision = load_part_supervision(train_records, annotations)
    validation_supervision = load_part_supervision(validation_records, annotations)
    train_part_audit = summarize_part_supervision(
        train_records, train_supervision
    )
    validation_part_audit = summarize_part_supervision(
        validation_records, validation_supervision
    )
    if (
        train_part_audit["images_without_valid_keypoints"]
        or validation_part_audit["images_without_valid_keypoints"]
    ):
        raise RuntimeError("CUB train/validation image has no valid visible part")
    if 5007 not in train_part_audit["affected_image_ids"]:
        raise RuntimeError("expected audited CUB image-5007 annotation edge case is missing")
    part_annotation_audit: dict[str, Any] = {
        "status": "pretest_complete",
        "rule": config["dataset"]["part_coordinate_validity"],
        "train": train_part_audit,
        "validation": validation_part_audit,
        "official_test": None,
        "official_test_accessed": False,
        "coordinate_clipping": False,
        "config_sha256": config_sha256,
    }
    _atomic_json_save(
        part_annotation_audit,
        output_dir / "part_probe/annotation_validity_audit.json",
    )
    _log_part_validity_audit("train", train_part_audit)
    _log_part_validity_audit("validation", validation_part_audit)

    teacher, _teacher_metadata, teacher_hash, teacher_state_hash = (
        load_scientific_teacher(
            args.teacher_checkpoint.expanduser().resolve(), device=device
        )
    )
    teacher_expected = config["checkpoint_sources"]["teacher"]
    if (
        teacher_hash != teacher_expected["checkpoint_sha256"]
        or teacher_state_hash != teacher_expected["model_state_sha256"]
    ):
        raise RuntimeError("direct-spatial full teacher identity changed")

    checkpoint_audits: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    matched_initial_hashes: dict[int, set[str]] = {
        seed: set() for seed in config["part_localization_probe"]["probe_seeds"]
    }
    student_root = args.student_release_dir.expanduser().resolve()
    for index, item in enumerate(checkpoint_inputs, 1):
        variant = item["variant"]
        encoder_seed = int(item["encoder_seed"])
        _atomic_json_save(
            {
                "status": "running",
                "phase": "validation_selection_and_cka",
                "active_variant": variant,
                "active_encoder_seed": encoder_seed,
                "variants_complete": index - 1,
                "variants_expected": len(checkpoint_inputs),
                "official_test_accessed": False,
            },
            status_path,
        )
        student, audit = _load_student(
            student_root,
            item,
            config=metric_config,
            device=device,
        )
        checkpoint_audits.append(audit)
        rows, variant_selections, initial_hashes = _run_part_validation(
            variant=variant,
            encoder_seed=encoder_seed,
            model=student,
            train_records=train_records,
            validation_records=validation_records,
            train_supervision=train_supervision,
            validation_supervision=validation_supervision,
            config=config,
            config_sha256=config_sha256,
            output_dir=output_dir,
            device=device,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        candidate_rows.extend(rows)
        selections.extend(variant_selections)
        for seed, digest in initial_hashes.items():
            matched_initial_hashes[seed].add(digest)
        variant_cka = _cka_smoke(
            variant=variant,
            encoder_seed=encoder_seed,
            student=student,
            teacher=teacher,
            records=validation_records,
            batch_size=args.cka_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        for row in variant_cka:
            row["scientific_result"] = True
        cka_rows.extend(variant_cka)
        del student
        torch.cuda.empty_cache()

    if (
        len(candidate_rows) != expected_gate["part_probe_lr_candidates"]
        or len(selections) != expected_selections
        or len(cka_rows) != expected_gate["spatial_cka_values"]
        or any(len(values) != 1 for values in matched_initial_hashes.values())
    ):
        raise RuntimeError("pre-test direct-spatial full completion gate failed")
    _write_csv(candidate_rows, output_dir / "part_probe/candidates.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "part_probe_validation_selections": len(selections),
            "expected_selections": expected_selections,
            "all_variants_and_probe_seeds_selected": True,
            "selection_uses_official_test": False,
            "official_test_accessed": False,
            "config_sha256": config_sha256,
        },
        output_dir / "part_probe/selection_complete_before_test.json",
    )
    _atomic_json_save(
        {
            "status": "running",
            "phase": "official_test_evaluation",
            "part_probe_validation_selections_complete": expected_selections,
            "official_test_accessed": True,
        },
        status_path,
    )

    test_records, test_source = load_official_test_records(
        args.data_dir.expanduser().resolve(), download=False
    )
    if len(test_records) != 5794:
        raise RuntimeError("official CUB test count changed")
    test_supervision = load_part_supervision(test_records, annotations)
    test_part_audit = summarize_part_supervision(test_records, test_supervision)
    if test_part_audit["images_without_valid_keypoints"]:
        raise RuntimeError("CUB official-test image has no valid visible part")
    part_annotation_audit.update(
        {
            "status": "complete",
            "official_test": test_part_audit,
            "official_test_accessed": True,
        }
    )
    _atomic_json_save(
        part_annotation_audit,
        output_dir / "part_probe/annotation_validity_audit.json",
    )
    _log_part_validity_audit("official_test", test_part_audit)
    part_test_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    for item in checkpoint_inputs:
        variant = item["variant"]
        encoder_seed = int(item["encoder_seed"])
        student, _audit = _load_student(
            student_root,
            item,
            config=metric_config,
            device=device,
        )
        variant_selections = [
            selection
            for selection in selections
            if selection["variant"] == variant
            and int(selection["encoder_seed"]) == encoder_seed
        ]
        part_test_rows.extend(
            _evaluate_part_test(
                variant=variant,
                encoder_seed=encoder_seed,
                model=student,
                test_records=test_records,
                test_supervision=test_supervision,
                selections=variant_selections,
                config=config,
                config_sha256=config_sha256,
                output_dir=output_dir,
                device=device,
                feature_batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
        )
        attention_rows.append(
            _run_attention_test(
                variant=variant,
                encoder_seed=encoder_seed,
                student=student,
                records=test_records,
                batch_size=args.attention_batch_size,
                num_workers=args.num_workers,
                output_dir=output_dir,
                device=device,
            )
        )
        del student
        torch.cuda.empty_cache()

    qualitative_pngs = sorted(
        (output_dir / "attention_gt/qualitative").glob("*.png")
    )
    gate = {
        "checkpoint_strict_loads": len(checkpoint_audits),
        "part_probe_lr_candidates": len(candidate_rows),
        "part_probe_validation_selections": len(selections),
        "part_probe_official_test_evaluations": len(part_test_rows),
        "spatial_cka_values": len(cka_rows),
        "attention_metric_rows": len(attention_rows),
        "attention_official_test_evaluations": sum(
            row["official_test_evaluations"] for row in attention_rows
        ),
        "qualitative_pngs": len(qualitative_pngs),
        "official_test_evaluations": len(part_test_rows)
        + sum(row["official_test_evaluations"] for row in attention_rows),
    }
    if gate != config["completion_gate"]:
        raise RuntimeError(f"direct-spatial full completion gate failed: {gate}")

    aggregates = _part_aggregates(part_test_rows)
    _atomic_json_save(
        {
            "status": "complete",
            "encoder_seeds": encoder_seeds,
            "candidates": candidate_rows,
            "selections": selections,
            "official_test_rows": part_test_rows,
            "aggregates": aggregates,
            "matched_initialization_across_variants_by_probe_seed": {
                str(seed): next(iter(values))
                for seed, values in matched_initial_hashes.items()
            },
            "part_annotation_validity_audit": part_annotation_audit,
            "official_test_evaluations": len(part_test_rows),
            "scientific_result": True,
            "final_encoder_seed_inference": False,
        },
        output_dir / "part_probe/results.json",
    )
    _write_csv(
        [
            {
                "variant": row["variant"],
                "encoder_seed": row["encoder_seed"],
                "probe_seed": row["probe_seed"],
                "selected_learning_rate": row["selected_learning_rate"],
                "selected_epoch": row["selected_epoch"],
                "test_micro_pck_at_0.1": row["test"]["micro_pck_at_0.1"],
                "test_mean_normalized_error": row["test"][
                    "mean_normalized_localization_error"
                ],
                "official_test_evaluations": 1,
            }
            for row in part_test_rows
        ],
        output_dir / "part_probe/official_test_results.csv",
    )
    _write_csv(cka_rows, output_dir / "spatial_cka/results.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "rows": cka_rows,
            "official_test_used": False,
            "scientific_result": True,
        },
        output_dir / "spatial_cka/results.json",
    )
    _save_cka_heatmap(cka_rows, output_dir / "spatial_cka/layerwise_heatmap.png")
    _write_csv(attention_rows, output_dir / "attention_gt/results.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "rows": attention_rows,
            "official_test_evaluations": sum(
                row["official_test_evaluations"] for row in attention_rows
            ),
            "scientific_result": True,
        },
        output_dir / "attention_gt/results.json",
    )
    _atomic_json_save(
        {
            "status": "pass",
            "teacher": {
                "checkpoint_sha256": teacher_hash,
                "model_state_sha256": teacher_state_hash,
                "strict_load": True,
                "eval_mode": True,
                "trainable_parameters": 0,
            },
            "students": checkpoint_audits,
        },
        output_dir / "checkpoint_audit.json",
    )
    _atomic_json_save(
        {
            "status": "pass",
            "dataset": DATASET_NAME,
            "train_source": train_source,
            "official_test_source": test_source,
            "split_manifest": split_manifest,
            "counts": {"train": 5394, "validation": 600, "official_test": 5794},
            "train_image_ids_sha256": file_digest_from_ids(train_records),
            "validation_image_ids_sha256": file_digest_from_ids(
                validation_records
            ),
            "part_names": list(part_names),
            "part_annotation_images": len(annotations),
            "part_coordinate_validity": config["dataset"][
                "part_coordinate_validity"
            ],
            "part_annotation_validity_audit": {
                "train": train_part_audit,
                "validation": validation_part_audit,
                "official_test": test_part_audit,
            },
            "official_test_opened_after_all_validation_selections": True,
            "validation_selections_complete_before_test": expected_selections,
        },
        output_dir / "dataset_audit.json",
    )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    part_lookup = {
        (int(row["encoder_seed"]), row["variant"]): row for row in aggregates
    }
    cka_lookup = {
        (int(row["encoder_seed"]), row["variant"], row["student_block"]): row
        for row in cka_rows
    }
    attention_lookup = {
        (int(row["encoder_seed"]), row["variant"]): row
        for row in attention_rows
    }
    summary = {
        "schema_version": 2,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "scientific_result": True,
        "result_scope": config["result_scope"],
        "student_batch_size": 128,
        "encoder_seeds": encoder_seeds,
        "independent_encoder_seed_n": len(encoder_seeds),
        "final_encoder_seed_inference": False,
        "settings_unchanged_after_seed1_and_smoke": True,
        "variants": list(EXPECTED_VARIANTS),
        "part_annotation_validity_audit": {
            "train": train_part_audit,
            "validation": validation_part_audit,
            "official_test": test_part_audit,
        },
        "part_probe_aggregates": aggregates,
        "spatial_cka_rows": cka_rows,
        "attention_gt_rows": attention_rows,
        "runtime": {
            **_runtime(device),
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "completion_gate": gate,
        "official_test_accessed": True,
        "official_test_used_for_selection": False,
        "official_test_evaluations": gate["official_test_evaluations"],
    }
    if encoder_seeds == [1]:
        summary["encoder_seed"] = 1
        summary["seed2_3_settings_remain_locked_regardless_of_seed1_results"] = True
    _atomic_json_save(summary, output_dir / "summary.json")
    _atomic_json_save(
        {
            "status": "complete",
            "phase": "complete",
            "completion_gate": gate,
            "finished_at_utc": summary["runtime"]["finished_at_utc"],
            "official_test_accessed": True,
            "official_test_used_for_selection": False,
            "failure": None,
        },
        status_path,
    )

    log("")
    results_marker = (
        "DIRECT_SPATIAL_FULL_SEED1_RESULTS"
        if encoder_seeds == [1]
        else "DIRECT_SPATIAL_FULL_SEED23_RESULTS"
    )
    log(f"[{results_marker}]")
    for item in checkpoint_inputs:
        variant = item["variant"]
        encoder_seed = int(item["encoder_seed"])
        part = part_lookup[(encoder_seed, variant)]
        attention = attention_lookup[(encoder_seed, variant)]
        log(
            f"[DIRECT_SPATIAL_FULL_RESULT] variant={variant} "
            f"encoder_seed={encoder_seed} "
            "part_test_pck_probe_seed_mean="
            f"{part['test_micro_pck_at_0.1_probe_seed_mean']:.6f} "
            "part_test_pck_probe_seed_sd="
            f"{part['test_micro_pck_at_0.1_probe_seed_sample_sd']:.6f} "
            "part_test_normalized_error_probe_seed_mean="
            f"{part['test_mean_normalized_error_probe_seed_mean']:.6f} "
            "cka_block11="
            f"{cka_lookup[(encoder_seed, variant, 11)]['centered_linear_cka']:.6f} "
            "attention_patch_ap="
            f"{attention['global_micro_patch_average_precision']:.6f} "
            "attention_pointing="
            f"{attention['pointing_game_peak_inside_mask']:.6f} "
            "attention_foreground_mass="
            f"{attention['foreground_attention_mass_mean']:.6f}"
        )
    done_marker = (
        "DIRECT_SPATIAL_FULL_SEED1_DONE"
        if encoder_seeds == [1]
        else "DIRECT_SPATIAL_FULL_SEED23_DONE"
    )
    seed_marker = (
        "encoder_seed=1 "
        if encoder_seeds == [1]
        else f"encoder_seeds={','.join(str(seed) for seed in encoder_seeds)} "
    )
    log(
        f"[{done_marker}] status=complete {seed_marker}"
        f"strict_loads={gate['checkpoint_strict_loads']} "
        f"part_candidates={gate['part_probe_lr_candidates']} "
        f"part_selections={gate['part_probe_validation_selections']} "
        f"part_test={gate['part_probe_official_test_evaluations']} "
        f"cka_values={gate['spatial_cka_values']} "
        f"attention_rows={gate['attention_metric_rows']} "
        f"qualitative_pngs={gate['qualitative_pngs']} "
        f"official_test={gate['official_test_evaluations']} "
        "final_encoder_seed_inference=false "
        "excluded_oob_train="
        f"{train_part_audit['excluded_out_of_frame_visible_keypoints']} "
        "excluded_oob_validation="
        f"{validation_part_audit['excluded_out_of_frame_visible_keypoints']} "
        "excluded_oob_test="
        f"{test_part_audit['excluded_out_of_frame_visible_keypoints']} "
        f"elapsed_seconds={elapsed:.2f} "
        f"peak_allocated_bytes={summary['runtime']['peak_allocated_bytes']} "
        f"peak_reserved_bytes={summary['runtime']['peak_reserved_bytes']}"
    )
    return summary


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "sequence_status.json"
        official_test_accessed = False
        if status_path.is_file():
            try:
                official_test_accessed = bool(
                    _load_json(status_path).get("official_test_accessed", False)
                )
            except (OSError, ValueError, RuntimeError):
                pass
        _atomic_json_save(
            {
                "status": "failed",
                "phase": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
                "official_test_accessed": official_test_accessed,
            },
            status_path,
        )
        raise


if __name__ == "__main__":
    main()
