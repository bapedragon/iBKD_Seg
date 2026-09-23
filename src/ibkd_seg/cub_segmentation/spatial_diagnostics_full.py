#!/usr/bin/env python3
"""Run the locked seed-1 full spatial diagnostics for CUB segmentation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from ibkd_seg.cityscapes.data import json_hash, save_json, sha256
from ibkd_seg.cityscapes.runtime import state_hash
from ibkd_seg.phase1.cub_direct_spatial import (
    GRID_SIZE,
    evaluate_part_probe,
    load_mask_views,
    load_part_supervision,
    load_spatial_annotations,
    summarize_part_supervision,
)
from ibkd_seg.phase1.cub_probe_data import (
    ids_sha256,
    load_train_validation_records,
)

from .spatial_diagnostics import (
    METHODS,
    PROBE_KINDS,
    ProbeCandidateDiverged,
    _load_checkpoint_model,
    _repository_path,
    _select_candidate,
    build_probe,
    extract_block11_features,
    log,
    probe_parameter_count,
    run_attention,
    run_cka,
    train_probe_candidate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub_Seg/configs/spatial_diagnostics_seed1_full_v1.json"
)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id",
        "status",
        "scientific_result",
        "result_scope",
        "full_training_authorized",
        "scope",
        "checkpoint_source",
        "smoke_source",
        "dataset",
        "part_probe",
        "spatial_cka",
        "attention_gt",
        "test_miou_reference",
        "execution",
        "completion_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise RuntimeError("CUB segmentation spatial-full config keys changed")

    scope = config["scope"]
    checkpoint_source = config["checkpoint_source"]
    smoke_source = config["smoke_source"]
    dataset = config["dataset"]
    probe = config["part_probe"]
    cka = config["spatial_cka"]
    attention = config["attention_gt"]
    checks = {
        "protocol": config["protocol_id"]
        == "cub200_direct_segmentation_seed1_spatial_diagnostics_full_v1",
        "locked": config["status"]
        == "locked_after_smoke_pass_before_full_results_2026-09-23",
        "exploratory": config["scientific_result"] is False
        and config["result_scope"] == "exploratory_encoder_seed1_only"
        and config["full_training_authorized"] is True,
        "scope": scope
        == {
            "encoder_seed": 1,
            "probe_seeds": [1],
            "methods": list(METHODS),
            "checkpoint_selection": "job783_validation_selected_best",
            "method_or_protocol_selection_from_full_results": False,
        },
        "checkpoint_source": checkpoint_source["h200_job_id"] == 783
        and checkpoint_source["training_protocol_id"]
        == "cub200_direct_binary_segmentation_window30_exploratory_full_v1"
        and checkpoint_source["training_config_sha256"]
        == "be94f411f97ff162e1adacac4951cf7185b18d00bb8d09486b068105a38f8d93",
        "smoke_source": smoke_source
        == {
            "config_path": "phase1/phase1_cub_Seg/configs/spatial_diagnostics_seed1_smoke_v1.json",
            "config_sha256": "d1632a1c3272847bb8ebb0d855e0f9fc4eeda760066649e369aae24010997671",
            "status": "passed",
            "elapsed_seconds": 123.04131173714995,
            "detected_issue": "vanilla_linear_lr_0.1_loss_divergence",
            "full_protocol_response": "isolate_diverged_candidate_and_continue_prespecified_lr_grid",
        },
        "dataset": dataset["split"]
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
        }
        and dataset["part_probe_train_images"] == 5394
        and dataset["part_probe_validation_images"] == 600
        and dataset["input_size"] == 224
        and dataset["part_count"] == 15
        and dataset["official_test_images_accessed"] is False,
        "probe": probe["student_feature"]
        == "pre_final_norm_block11_192x14x14"
        and probe["target"]
        == "visible_part_gaussian_heatmap_sigma_1_grid_pixel"
        and probe["loss"] == "visible_part_masked_mean_squared_error"
        and probe["probe_seeds"] == [1]
        and probe["learning_rates"] == [0.01, 0.03, 0.1]
        and probe["epochs"] == 100
        and probe["batch_size"] == 64
        and probe["candidate_eligibility"]
        == "completed_all_epochs_with_finite_batch_loss_at_most_10.0"
        and probe["divergence_loss_ceiling"] == 10.0
        and probe["divergence_action"]
        == "record_failed_candidate_exclude_from_selection_and_continue"
        and probe["minimum_valid_candidates_per_method_and_probe_kind"] == 1
        and probe["linear"]["parameter_count"] == 2895
        and probe["nonlinear"]["parameter_count"] == 111695,
        "cka": cka["split"] == "fixed_validation_600"
        and cka["student_blocks"] == list(range(12))
        and cka["teacher_feature"] == "resnet50_layer3_1024x14x14"
        and cka["metric"] == "centered_linear_CKA"
        and cka["accumulator_dtype"] == "float64"
        and cka["reported_primary"] == "student_block11",
        "attention": attention["split"] == "fixed_validation_600"
        and attention["rollout_layers"] == 12
        and attention["head_fusion"] == "arithmetic_mean"
        and attention["reported_metrics"]
        == [
            "global_micro_patch_average_precision",
            "pointing_game_peak_inside_mask",
            "foreground_attention_mass_mean",
        ],
        "test_reference": config["test_miou_reference"]
        == {
            "source": "audited_job783_full_summary_only",
            "metric": "two_class_miou",
            "official_test_images_reevaluated": False,
        },
        "execution": config["execution"]
        == {
            "requested_mig_slices": 1,
            "feature_batch_size": 16,
            "cka_batch_size": 8,
            "attention_batch_size": 16,
            "num_workers": 4,
            "precision": "fp32",
        },
        "gate": config["completion_gate"]
        == {
            "checkpoint_strict_loads": 5,
            "linear_probe_candidate_attempts": 12,
            "nonlinear_probe_candidate_attempts": 12,
            "probe_validation_selections": 8,
            "spatial_cka_values": 48,
            "attention_metric_rows": 4,
            "test_miou_references": 4,
            "official_test_image_evaluations": 0,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB segmentation spatial-full config: " + ", ".join(failures)
        )

    for source, path_key, digest_key in (
        (checkpoint_source, "manifest_path", "manifest_sha256"),
        (smoke_source, "config_path", "config_sha256"),
    ):
        source_path = _repository_path(source[path_key])
        if not source_path.is_file() or sha256(source_path) != source[digest_key]:
            raise RuntimeError(f"spatial-full provenance changed: {path_key}")
    return config


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _lr_slug(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _mean_and_sample_sd(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        raise ValueError("cannot aggregate an empty metric list")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_standard_deviation": (
            float(statistics.stdev(values)) if len(values) > 1 else None
        ),
        "n": len(values),
    }


def _run_probe_kind(
    *,
    method: str,
    kind: str,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    train_supervision: dict[str, torch.Tensor],
    validation_supervision: dict[str, torch.Tensor],
    protocol: dict[str, Any],
    config_sha256: str,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], int, int]:
    seed_results: list[dict[str, Any]] = []
    attempt_count = 0
    failed_count = 0
    for probe_seed in protocol["probe_seeds"]:
        valid_candidates: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        initial_hashes: set[str] = set()
        for learning_rate in protocol["learning_rates"]:
            attempt_count += 1
            stem = (
                f"{method}_{kind}_probe_seed{probe_seed}_"
                f"lr{_lr_slug(float(learning_rate))}"
            )

            def progress(row: dict[str, Any]) -> None:
                epoch = int(row["epoch"])
                if epoch == 1 or epoch % 10 == 0 or epoch == protocol["epochs"]:
                    log(
                        "[CUB_SEG_SPATIAL_FULL_PROBE_PROGRESS] "
                        f"method={method} kind={kind} probe_seed={probe_seed} "
                        f"lr={learning_rate} epoch={epoch}/{protocol['epochs']} "
                        "train_loss="
                        f"{row['train_visible_weighted_loss']:.8f} "
                        "val_pck="
                        f"{row['validation']['micro_pck_at_0.1']:.6f}"
                    )

            try:
                candidate = train_probe_candidate(
                    kind=kind,
                    train_features=train_features,
                    train_supervision=train_supervision,
                    validation_features=validation_features,
                    validation_supervision=validation_supervision,
                    learning_rate=float(learning_rate),
                    epochs=int(protocol["epochs"]),
                    seed=int(probe_seed),
                    batch_size=int(protocol["batch_size"]),
                    device=device,
                    divergence_loss_ceiling=float(
                        protocol["divergence_loss_ceiling"]
                    ),
                    progress_callback=progress,
                )
            except ProbeCandidateDiverged as error:
                failed_count += 1
                failure = {
                    "status": "failed_diverged",
                    "method": method,
                    "kind": kind,
                    "probe_seed": int(probe_seed),
                    "learning_rate": float(learning_rate),
                    "failed_epoch": error.epoch,
                    "failed_batch_start": error.batch_start,
                    "observed_batch_loss": (
                        error.observed_loss
                        if math.isfinite(error.observed_loss)
                        else None
                    ),
                    "observed_batch_loss_was_nonfinite": not math.isfinite(
                        error.observed_loss
                    ),
                    "divergence_loss_ceiling": protocol[
                        "divergence_loss_ceiling"
                    ],
                    "eligible_for_selection": False,
                }
                candidate_rows.append(failure)
                save_json(
                    output_dir / "probe_histories" / f"{stem}.json",
                    {**failure, "history": error.history},
                )
                log(
                    "[CUB_SEG_SPATIAL_FULL_PROBE_DIVERGED] "
                    f"method={method} kind={kind} probe_seed={probe_seed} "
                    f"lr={learning_rate} epoch={error.epoch} "
                    f"batch_loss={error.observed_loss} action=excluded_and_continue"
                )
                gc.collect()
                torch.cuda.empty_cache()
                continue

            initial_hashes.add(candidate["initial_probe_state_sha256"])
            valid_candidates.append(candidate)
            row = {
                "status": "complete",
                "method": method,
                "kind": kind,
                "probe_seed": int(probe_seed),
                "learning_rate": float(learning_rate),
                "epochs_completed": int(protocol["epochs"]),
                "best_epoch": candidate["best_epoch"],
                "best_validation": candidate["best_validation"],
                "initial_probe_state_sha256": candidate[
                    "initial_probe_state_sha256"
                ],
                "final_probe_state_sha256": candidate[
                    "final_probe_state_sha256"
                ],
                "first_epoch_train_loss": candidate["history"][0][
                    "train_visible_weighted_loss"
                ],
                "last_epoch_train_loss": candidate["history"][-1][
                    "train_visible_weighted_loss"
                ],
                "eligible_for_selection": True,
                "selected": False,
            }
            candidate_rows.append(row)
            save_json(
                output_dir / "probe_histories" / f"{stem}.json",
                {**row, "history": candidate["history"]},
            )
            log(
                "[CUB_SEG_SPATIAL_FULL_PROBE_CANDIDATE] "
                f"method={method} kind={kind} probe_seed={probe_seed} "
                f"lr={learning_rate} best_epoch={candidate['best_epoch']} "
                "val_pck="
                f"{candidate['best_validation']['micro_pck_at_0.1']:.6f}"
            )

        if len(valid_candidates) < protocol[
            "minimum_valid_candidates_per_method_and_probe_kind"
        ]:
            raise RuntimeError(
                f"no valid full probe candidate remains: {method}/{kind}/seed{probe_seed}"
            )
        if len(initial_hashes) != 1:
            raise RuntimeError(
                f"probe initialization changed across valid LR: {method}/{kind}"
            )
        selected = _select_candidate(valid_candidates)
        for row in candidate_rows:
            if (
                row["status"] == "complete"
                and row["learning_rate"] == selected["learning_rate"]
            ):
                row["selected"] = True

        selected_probe = build_probe(kind, int(probe_seed))
        incompatible = selected_probe.load_state_dict(
            selected["probe_state"], strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("selected full probe strict reload failed")
        selected_probe.to(device).eval()
        validation = evaluate_part_probe(
            selected_probe,
            validation_features,
            validation_supervision,
            device=device,
            batch_size=int(protocol["batch_size"]),
        )
        if (
            validation["micro_pck_at_0.1"]
            != selected["best_validation"]["micro_pck_at_0.1"]
            or validation["mean_normalized_localization_error"]
            != selected["best_validation"][
                "mean_normalized_localization_error"
            ]
        ):
            raise RuntimeError("selected full probe validation changed")

        checkpoint = (
            output_dir
            / "probe_checkpoints"
            / f"{method}_{kind}_probe_seed{probe_seed}_best_validation.pt"
        )
        _atomic_torch_save(
            {
                "probe": selected["probe_state"],
                "metadata": {
                    "protocol_id": (
                        "cub200_direct_segmentation_seed1_"
                        "spatial_diagnostics_full_v1"
                    ),
                    "scientific_result": False,
                    "method": method,
                    "kind": kind,
                    "encoder_seed": 1,
                    "probe_seed": int(probe_seed),
                    "selected_learning_rate": selected["learning_rate"],
                    "selected_epoch": selected["best_epoch"],
                    "selection_metric": "validation_micro_PCK_at_0.1",
                    "official_test_evaluations_at_checkpoint_write": 0,
                    "config_sha256": config_sha256,
                },
            },
            checkpoint,
        )
        checkpoint_hash = sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reloaded_probe = build_probe(kind, int(probe_seed))
        reloaded = reloaded_probe.load_state_dict(payload["probe"], strict=True)
        if reloaded.missing_keys or reloaded.unexpected_keys:
            raise RuntimeError("saved selected full probe strict reload failed")
        if state_hash(reloaded_probe) != state_hash(selected_probe.cpu()):
            raise RuntimeError("saved selected full probe state changed")

        seed_results.append(
            {
                "probe_seed": int(probe_seed),
                "selected_learning_rate": selected["learning_rate"],
                "selected_epoch": selected["best_epoch"],
                "validation": validation,
                "checkpoint_relative_path": checkpoint.relative_to(
                    output_dir
                ).as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "strict_reload": True,
                "candidates": candidate_rows,
            }
        )
        del selected_probe, reloaded_probe, payload, valid_candidates
        gc.collect()
        torch.cuda.empty_cache()

    pck_values = [
        row["validation"]["micro_pck_at_0.1"] for row in seed_results
    ]
    error_values = [
        row["validation"]["mean_normalized_localization_error"]
        for row in seed_results
    ]
    return (
        {
            "kind": kind,
            "architecture": protocol[kind]["architecture"],
            "parameter_count": probe_parameter_count(kind),
            "probe_seed_results": seed_results,
            "validation_micro_pck_at_0.1": _mean_and_sample_sd(pck_values),
            "validation_mean_normalized_localization_error": (
                _mean_and_sample_sd(error_values)
            ),
            "candidate_attempts": attempt_count,
            "candidate_failures": failed_count,
            "candidate_completions": attempt_count - failed_count,
        },
        attempt_count,
        failed_count,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(args.config)
    expected_execution = config["execution"]
    actual_execution = {
        "feature_batch_size": args.feature_batch_size,
        "cka_batch_size": args.cka_batch_size,
        "attention_batch_size": args.attention_batch_size,
        "num_workers": args.num_workers,
    }
    if actual_execution != {
        key: expected_execution[key] for key in actual_execution
    }:
        raise RuntimeError("spatial-full CLI differs from locked execution settings")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUB segmentation spatial full experiment requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty full output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config_hash = sha256(args.config)
    release_manifest_path = _repository_path(
        config["checkpoint_source"]["manifest_path"]
    )
    release_manifest = json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    if (
        release_manifest.get("asset_type")
        != "phase1_cub_direct_segmentation_window30_seed1_v1"
        or release_manifest.get("source", {}).get("h200_job_id") != 783
        or set(release_manifest.get("checkpoints", {}))
        != {"teacher", *METHODS}
    ):
        raise RuntimeError("unexpected CUB segmentation checkpoint release manifest")

    full_summary_path = args.release_dir / "full_summary.json"
    training_config_path = args.release_dir / "config.json"
    if (
        not full_summary_path.is_file()
        or not training_config_path.is_file()
        or sha256(full_summary_path)
        != release_manifest["source"]["full_summary_sha256"]
    ):
        raise RuntimeError("audited issue-783 summary is missing or changed")
    full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if (
        full_summary.get("status") != "complete"
        or full_summary.get("scientific_result") is not False
        or full_summary.get("config") != training_config
        or full_summary.get("config_sha256")
        != config["checkpoint_source"]["training_config_sha256"]
    ):
        raise RuntimeError("audited issue-783 completion contract failed")

    records, split_manifest, data_source = load_train_validation_records(
        args.data_dir, download=True
    )
    if {key: len(value) for key, value in records.items()} != {
        "train": 5394,
        "validation": 600,
    }:
        raise RuntimeError("CUB train/validation counts changed")
    full_identity = full_summary["data_identity"]
    if (
        json_hash(split_manifest) != full_identity["split_manifest_sha256"]
        or ids_sha256(records["train"])
        != full_identity["train_image_ids_sha256"]
        or ids_sha256(records["validation"])
        != full_identity["validation_image_ids_sha256"]
    ):
        raise RuntimeError("CUB data identity differs from issue 783")

    _part_names, annotations = load_spatial_annotations(
        Path(data_source["dataset_root"])
    )
    train_supervision = load_part_supervision(records["train"], annotations)
    validation_supervision = load_part_supervision(
        records["validation"], annotations
    )
    dataset_audit = {
        "train_count": len(records["train"]),
        "validation_count": len(records["validation"]),
        "train_ids_sha256": ids_sha256(records["train"]),
        "validation_ids_sha256": ids_sha256(records["validation"]),
        "train_part_supervision": summarize_part_supervision(
            records["train"], train_supervision
        ),
        "validation_part_supervision": summarize_part_supervision(
            records["validation"], validation_supervision
        ),
        "official_test_images_accessed": False,
    }
    save_json(args.output_dir / "dataset_audit.json", dataset_audit)
    mask_views = [load_mask_views(record) for record in records["validation"]]
    validation_masks = torch.stack([value[0] for value in mask_views])
    validation_occupancies = torch.stack([value[1] for value in mask_views])

    teacher, teacher_audit = _load_checkpoint_model(
        args.release_dir,
        release_manifest,
        "teacher",
        training_config,
        device=device,
    )
    checkpoint_audits = [teacher_audit]
    method_results: dict[str, Any] = {}
    counts = {
        "checkpoint_strict_loads": 1,
        "linear_probe_candidate_attempts": 0,
        "nonlinear_probe_candidate_attempts": 0,
        "probe_validation_selections": 0,
        "spatial_cka_values": 0,
        "attention_metric_rows": 0,
        "test_miou_references": 0,
        "official_test_image_evaluations": 0,
    }
    candidate_failures = {"linear": 0, "nonlinear": 0}
    full_result_methods = full_summary["final_results"]["methods"]

    for method in METHODS:
        method_started = time.monotonic()
        log(f"[CUB_SEG_SPATIAL_FULL_METHOD_START] method={method}")
        student, checkpoint_audit = _load_checkpoint_model(
            args.release_dir,
            release_manifest,
            method,
            training_config,
            device=device,
        )
        checkpoint_audits.append(checkpoint_audit)
        counts["checkpoint_strict_loads"] += 1
        train_features = extract_block11_features(
            student,
            records["train"],
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        validation_features = extract_block11_features(
            student,
            records["validation"],
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        probe_results: dict[str, Any] = {}
        for kind in PROBE_KINDS:
            probe_result, attempts, failures = _run_probe_kind(
                method=method,
                kind=kind,
                train_features=train_features,
                validation_features=validation_features,
                train_supervision=train_supervision,
                validation_supervision=validation_supervision,
                protocol=config["part_probe"],
                config_sha256=config_hash,
                output_dir=args.output_dir,
                device=device,
            )
            probe_results[kind] = probe_result
            counts[f"{kind}_probe_candidate_attempts"] += attempts
            counts["probe_validation_selections"] += len(
                probe_result["probe_seed_results"]
            )
            candidate_failures[kind] += failures

        cka_values = run_cka(
            student,
            teacher,
            records["validation"],
            batch_size=args.cka_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        counts["spatial_cka_values"] += len(cka_values)
        attention_metrics = run_attention(
            student,
            records["validation"],
            validation_masks,
            validation_occupancies,
            batch_size=args.attention_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        counts["attention_metric_rows"] += 1
        reference = full_result_methods[method]
        test_metrics = reference["test"]
        counts["test_miou_references"] += 1
        linear_pck = probe_results["linear"][
            "validation_micro_pck_at_0.1"
        ]["mean"]
        linear_error = probe_results["linear"][
            "validation_mean_normalized_localization_error"
        ]["mean"]
        nonlinear_pck = probe_results["nonlinear"][
            "validation_micro_pck_at_0.1"
        ]["mean"]
        nonlinear_error = probe_results["nonlinear"][
            "validation_mean_normalized_localization_error"
        ]["mean"]
        table_row = {
            "method": method,
            "part_probe_primary": "linear",
            "part_pck_at_0.1": linear_pck,
            "normalized_localization_error": linear_error,
            "nonlinear_part_pck_at_0.1": nonlinear_pck,
            "nonlinear_normalized_localization_error": nonlinear_error,
            "cka_block11": cka_values[11],
            "attention_ap": attention_metrics[
                "global_micro_patch_average_precision"
            ],
            "pointing": attention_metrics["pointing_game_peak_inside_mask"],
            "foreground_mass": attention_metrics[
                "foreground_attention_mass_mean"
            ],
            "test_miou_reference": test_metrics["two_class_miou"],
        }
        method_result = {
            "table_row": table_row,
            "checkpoint": checkpoint_audit,
            "linear_probe": probe_results["linear"],
            "nonlinear_probe": probe_results["nonlinear"],
            "spatial_cka": {
                "validation_images": len(records["validation"]),
                "spatial_observations": len(records["validation"])
                * GRID_SIZE
                * GRID_SIZE,
                "teacher_feature": "resnet50_layer3",
                "student_blocks_0_through_11": cka_values,
                "block11": cka_values[11],
            },
            "attention": attention_metrics,
            "test_miou_reference": test_metrics["two_class_miou"],
            "test_metrics_reference": test_metrics,
            "test_reference_selected_epoch": reference["selected_epoch"],
            "official_test_images_reevaluated": False,
            "elapsed_seconds": time.monotonic() - method_started,
        }
        method_results[method] = method_result
        save_json(args.output_dir / "methods" / method / "summary.json", method_result)
        log(
            "[CUB_SEG_SPATIAL_FULL_METHOD_DONE] "
            f"method={method} linear_pck={linear_pck:.6f} "
            f"nonlinear_pck={nonlinear_pck:.6f} "
            f"cka_block11={cka_values[11]:.6f} "
            "attention_ap="
            f"{attention_metrics['global_micro_patch_average_precision']:.6f} "
            f"test_miou_reference={test_metrics['two_class_miou']:.6f}"
        )
        del student, train_features, validation_features, probe_results
        gc.collect()
        torch.cuda.empty_cache()

    expected_counts = config["completion_gate"]
    if counts != expected_counts:
        raise RuntimeError(f"full completion gate failed: {counts} != {expected_counts}")
    for kind in PROBE_KINDS:
        if probe_parameter_count(kind) != config["part_probe"][kind][
            "parameter_count"
        ]:
            raise RuntimeError(f"{kind} part-probe parameter count changed")

    summary = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "scientific_result": False,
        "result_scope": config["result_scope"],
        "config_sha256": config_hash,
        "checkpoint_release_manifest_sha256": sha256(release_manifest_path),
        "source_h200_job_id": 783,
        "encoder_seed": 1,
        "probe_seeds": config["scope"]["probe_seeds"],
        "dataset": dataset_audit,
        "checkpoint_audits": checkpoint_audits,
        "methods": method_results,
        "table_rows": [method_results[method]["table_row"] for method in METHODS],
        "candidate_failures": candidate_failures,
        "completion": {
            "actual": counts,
            "expected": expected_counts,
            "passed": True,
        },
        "official_test_images_accessed": False,
        "official_test_image_evaluations": 0,
        "test_miou_source": "audited_job783_full_summary_only",
        "elapsed_seconds": time.monotonic() - started,
    }
    save_json(args.output_dir / "full_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--cka-batch-size", type=int, default=8)
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    log(
        "[CUB_SEG_SPATIAL_FULL_FINAL_RESULTS] "
        + json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


if __name__ == "__main__":
    main()
