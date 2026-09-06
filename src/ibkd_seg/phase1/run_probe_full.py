#!/usr/bin/env python3
"""Run a locked Phase 1 frozen spatial probe experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .data import (
    ANNOTATION_HASHES,
    ARCHIVE_CONTRACTS,
    OFFICIAL_TEST_COUNT,
    OFFICIAL_TRAINVAL_COUNT,
    _parse_annotation_split,
    file_digest,
)
from .probe import (
    confusion_from_tensors,
    evaluate_probe_both_resolutions,
    probe_from_state,
    train_candidate,
)
from .probe_data import (
    PetRecord,
    load_official_test_records,
    load_train_validation_records,
)
from .run_probe_smoke import (
    DEFAULT_PROTOCOL,
    EXPECTED_VARIANTS,
    _atomic_json_save,
    _atomic_torch_save,
    _device,
    _feature_cache,
    _finite_metrics,
    _load_encoder,
    _load_json,
    _runtime,
    _synchronize,
    _target_cache,
    _validate_classification_input,
    log,
)
from .train_timing import file_sha256


LOCKED_PROTOCOL_SHA256 = (
    "38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143"
)
EXPECTED_ENCODER_SEEDS = (1, 2, 3)
EXPECTED_PROBE_SEEDS = (1, 2, 3, 4, 5)
EXPECTED_SELECTIONS = 90
EXPECTED_LR_CANDIDATES = 270

METRIC_PATHS: dict[str, tuple[str, str]] = {
    "input_224_mean_iou": ("input_224", "mean_iou"),
    "input_224_foreground_iou": ("input_224", "foreground_iou"),
    "input_224_background_iou": ("input_224", "background_iou"),
    "input_224_foreground_dice": ("input_224", "foreground_dice"),
    "input_224_pixel_accuracy": ("input_224", "pixel_accuracy"),
    "grid_14x14_mean_iou": ("grid_14x14", "mean_iou"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sequence_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    completed_selections: int,
    completed_test_evaluations: int,
    active: dict[str, Any] | None,
    official_test_accessed: bool,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "completed_selections": completed_selections,
            "expected_selections": EXPECTED_SELECTIONS,
            "completed_official_test_evaluations": completed_test_evaluations,
            "expected_official_test_evaluations": EXPECTED_SELECTIONS,
            "active": active,
            "scientific_result": True,
            "official_test_accessed": official_test_accessed,
            "failure": failure,
            "updated_at_utc": _utc_now(),
        },
        output_dir / "sequence_status.json",
    )


def _validate_protocol(protocol: dict[str, Any], protocol_path: Path) -> None:
    actual_sha256 = file_sha256(protocol_path)
    if actual_sha256 != LOCKED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Phase 1 protocol changed after lock: "
            f"{actual_sha256} != {LOCKED_PROTOCOL_SHA256}"
        )
    if protocol.get("protocol_id") != (
        "oxford_iiit_pet_phase1_frozen_spatial_probe_v1"
    ):
        raise RuntimeError("unexpected Phase 1 protocol id")
    probe = protocol["frozen_spatial_probe"]["probe"]
    expected = {
        "learning_rates": probe.get("learning_rates") == [0.01, 0.03, 0.1],
        "epochs": probe.get("epochs") == 100,
        "batch_size": probe.get("batch_size") == 64,
        "probe_seeds": tuple(probe.get("probe_seeds", ()))
        == EXPECTED_PROBE_SEEDS,
        "parameter_count": probe.get("parameter_count") == 386,
        "test_policy": protocol["frozen_spatial_probe"]["selection"].get(
            "test_policy"
        )
        == "once_after_validation_selection",
        "primary_metric": protocol["frozen_spatial_probe"]["evaluation"].get(
            "primary_metric"
        )
        == "input_224_two_class_mean_iou",
    }
    if not all(expected.values()):
        failures = [name for name, passed in expected.items() if not passed]
        raise RuntimeError("locked full-probe contract failed: " + ", ".join(failures))


def _verify_dataset_identity(
    data_dir: Path,
    classification_suite: dict[str, Any],
) -> dict[str, Any]:
    """Verify bytes and official split manifests without decoding test pixels."""

    source_audit = classification_suite.get("dataset_audit")
    if not isinstance(source_audit, dict) or source_audit.get("status") != "pass":
        raise RuntimeError("classification suite has no passing dataset audit")
    source_archives = source_audit.get("archives", {})
    base = data_dir / "oxford-iiit-pet"
    archives: dict[str, Any] = {}
    for filename, public_contract in ARCHIVE_CONTRACTS.items():
        path = base / filename
        if not path.is_file():
            raise RuntimeError(f"dataset archive is missing: {path}")
        source = source_archives.get(filename)
        if not isinstance(source, dict) or not source.get("sha256"):
            raise RuntimeError(f"classification audit lacks {filename} SHA-256")
        observed = {
            "bytes": path.stat().st_size,
            "md5": file_digest(path, "md5"),
            "sha256": file_digest(path, "sha256"),
        }
        if (
            observed["bytes"] != int(public_contract["bytes"])
            or observed["md5"] != public_contract["md5"]
            or observed != source
        ):
            raise RuntimeError(f"dataset archive identity mismatch: {filename}")
        if (
            public_contract["sha256"] is not None
            and observed["sha256"] != public_contract["sha256"]
        ):
            raise RuntimeError(f"public SHA-256 mismatch: {filename}")
        archives[filename] = observed

    annotation_dir = base / "annotations"
    annotation_hashes: dict[str, str] = {}
    for filename, expected_sha256 in ANNOTATION_HASHES.items():
        actual_sha256 = file_digest(annotation_dir / filename)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"official annotation hash mismatch: {filename}")
        if source_audit.get("annotation_hashes", {}).get(filename) != actual_sha256:
            raise RuntimeError(f"classification annotation identity mismatch: {filename}")
        annotation_hashes[filename] = actual_sha256

    trainval_manifest = _parse_annotation_split(annotation_dir / "trainval.txt")
    test_manifest = _parse_annotation_split(annotation_dir / "test.txt")
    trainval_ids = {image_id for image_id, _ in trainval_manifest}
    test_ids = {image_id for image_id, _ in test_manifest}
    manifest_checks = {
        "trainval_count": len(trainval_manifest) == OFFICIAL_TRAINVAL_COUNT,
        "test_count": len(test_manifest) == OFFICIAL_TEST_COUNT,
        "trainval_unique": len(trainval_ids) == OFFICIAL_TRAINVAL_COUNT,
        "test_unique": len(test_ids) == OFFICIAL_TEST_COUNT,
        "official_splits_disjoint": not bool(trainval_ids & test_ids),
    }
    if not all(manifest_checks.values()):
        failures = [name for name, passed in manifest_checks.items() if not passed]
        raise RuntimeError("official split-manifest audit failed: " + ", ".join(failures))
    return {
        "status": "pass",
        "matched_classification_dataset_audit": True,
        "archives": archives,
        "annotation_hashes": annotation_hashes,
        "manifest_checks": manifest_checks,
        "test_manifest_metadata_read": True,
        "test_rgb_or_trimap_pixels_decoded_before_selection": False,
        "official_test_used_for_training_or_selection": False,
    }


def _select_candidate(
    candidates: Sequence[tuple[dict[str, torch.Tensor], dict[str, Any]]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not candidates:
        raise ValueError("no probe candidates to select")
    learning_rates = [float(candidate[1]["learning_rate"]) for candidate in candidates]
    if learning_rates != sorted(learning_rates) or len(set(learning_rates)) != len(
        learning_rates
    ):
        raise RuntimeError("LR candidates must be unique and ordered low to high")
    # train_candidate keeps the earlier epoch on an exact within-LR tie. max()
    # keeps the earlier (therefore lower-LR) item on an exact across-LR tie.
    index = max(
        range(len(candidates)),
        key=lambda candidate_index: float(
            candidates[candidate_index][1]["best_validation_grid_mean_iou"]
        ),
    )
    return candidates[index]


def _baseline_templates(
    train_targets: dict[str, Any],
    *,
    ignore_index: int,
    input_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    train_grid = train_targets["grid_targets"]
    valid_counts = train_grid.ne(ignore_index).sum(dim=0)
    foreground_counts = train_grid.eq(1).sum(dim=0)
    train_mean_score = torch.zeros_like(valid_counts, dtype=torch.float32)
    usable = valid_counts.gt(0)
    train_mean_score[usable] = (
        foreground_counts[usable].float() / valid_counts[usable].float()
    )
    train_mean_grid = train_mean_score.ge(0.5)
    train_mean_input = F.interpolate(
        train_mean_score[None, None],
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].ge(0.5)
    return {
        "all_background": (
            torch.zeros_like(train_mean_grid),
            torch.zeros_like(train_mean_input),
        ),
        "train_mean_mask": (train_mean_grid, train_mean_input),
    }


def _evaluate_baselines(
    templates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    targets: dict[str, Any],
    *,
    ignore_index: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, (grid_template, input_template) in templates.items():
        report[name] = {
            "grid_14x14": confusion_from_tensors(
                grid_template.expand_as(targets["grid_targets"]),
                targets["grid_targets"],
                ignore_index=ignore_index,
            ).metrics(),
            "input_224": confusion_from_tensors(
                input_template.expand_as(targets["input_targets"]),
                targets["input_targets"],
                ignore_index=ignore_index,
            ).metrics(),
        }
    return report


def _summary_stat(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.mean(numeric),
        "sample_standard_deviation": statistics.stdev(numeric)
        if len(numeric) > 1
        else 0.0,
        "values": numeric,
        "count": len(numeric),
    }


def _metric(result: dict[str, Any], split: str, metric_name: str) -> float:
    resolution, name = METRIC_PATHS[metric_name]
    return float(result[split][resolution][name])


def _aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for variant in EXPECTED_VARIANTS:
        variant_rows = [row for row in results if row["variant"] == variant]
        if len(variant_rows) != 15:
            raise RuntimeError(f"{variant} has {len(variant_rows)} rows, expected 15")
        variant_report: dict[str, Any] = {"by_encoder_seed": {}}
        for encoder_seed in EXPECTED_ENCODER_SEEDS:
            seed_rows = sorted(
                (
                    row
                    for row in variant_rows
                    if int(row["encoder_seed"]) == encoder_seed
                ),
                key=lambda row: int(row["probe_seed"]),
            )
            if [int(row["probe_seed"]) for row in seed_rows] != list(
                EXPECTED_PROBE_SEEDS
            ):
                raise RuntimeError(
                    f"{variant} encoder seed {encoder_seed} probe matrix is incomplete"
                )
            seed_report: dict[str, Any] = {}
            for split in ("validation", "test"):
                seed_report[split] = {
                    metric_name: _summary_stat(
                        [_metric(row, split, metric_name) for row in seed_rows]
                    )
                    for metric_name in METRIC_PATHS
                }
            variant_report["by_encoder_seed"][str(encoder_seed)] = seed_report

        across: dict[str, Any] = {}
        for split in ("validation", "test"):
            across[split] = {}
            for metric_name in METRIC_PATHS:
                encoder_means = [
                    variant_report["by_encoder_seed"][str(seed)][split][metric_name][
                        "mean"
                    ]
                    for seed in EXPECTED_ENCODER_SEEDS
                ]
                across[split][metric_name] = _summary_stat(encoder_means)
        variant_report["across_encoder_seed_means"] = across
        aggregates[variant] = variant_report

    paired: dict[str, Any] = {}
    alg = aggregates["alg"]["by_encoder_seed"]
    for variant in ("ibkd_lambda_0.25", "ibkd_lambda_0.5"):
        rows: list[dict[str, Any]] = []
        for encoder_seed in EXPECTED_ENCODER_SEEDS:
            ibkd_value = aggregates[variant]["by_encoder_seed"][str(encoder_seed)][
                "test"
            ]["input_224_mean_iou"]["mean"]
            alg_value = alg[str(encoder_seed)]["test"]["input_224_mean_iou"][
                "mean"
            ]
            rows.append(
                {
                    "encoder_seed": encoder_seed,
                    "ibkd_mean_over_probe_seeds": ibkd_value,
                    "alg_mean_over_probe_seeds": alg_value,
                    "ibkd_minus_alg": ibkd_value - alg_value,
                }
            )
        paired[variant + "_minus_alg"] = {
            "metric": "test_input_224_mean_iou",
            "per_encoder_seed": rows,
            "difference_summary": _summary_stat(
                [row["ibkd_minus_alg"] for row in rows]
            ),
            "formal_p_value_reported": False,
        }
    return {"variants": aggregates, "paired_primary_contrasts": paired}


def _write_raw_csv(results: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "method",
        "fusion_ratio_lambda",
        "encoder_seed",
        "probe_seed",
        "selected_learning_rate",
        "selected_epoch",
        "validation_grid_mean_iou",
        "validation_input_224_mean_iou",
        "test_grid_mean_iou",
        "test_input_224_mean_iou",
        "test_input_224_foreground_iou",
        "test_input_224_background_iou",
        "test_input_224_foreground_dice",
        "test_input_224_pixel_accuracy",
        "official_test_evaluations",
        "scientific_result",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "variant": result["variant"],
                    "method": result["method"],
                    "fusion_ratio_lambda": result["fusion_ratio_lambda"],
                    "encoder_seed": result["encoder_seed"],
                    "probe_seed": result["probe_seed"],
                    "selected_learning_rate": result["selection"][
                        "learning_rate"
                    ],
                    "selected_epoch": result["selection"]["epoch"],
                    "validation_grid_mean_iou": result["validation"][
                        "grid_14x14"
                    ]["mean_iou"],
                    "validation_input_224_mean_iou": result["validation"][
                        "input_224"
                    ]["mean_iou"],
                    "test_grid_mean_iou": result["test"]["grid_14x14"][
                        "mean_iou"
                    ],
                    "test_input_224_mean_iou": result["test"]["input_224"][
                        "mean_iou"
                    ],
                    "test_input_224_foreground_iou": result["test"]["input_224"][
                        "foreground_iou"
                    ],
                    "test_input_224_background_iou": result["test"]["input_224"][
                        "background_iou"
                    ],
                    "test_input_224_foreground_dice": result["test"]["input_224"][
                        "foreground_dice"
                    ],
                    "test_input_224_pixel_accuracy": result["test"]["input_224"][
                        "pixel_accuracy"
                    ],
                    "official_test_evaluations": result[
                        "official_test_evaluations"
                    ],
                    "scientific_result": True,
                }
            )
    temporary.replace(path)


def _mask_image(mask: torch.Tensor, *, ground_truth: bool) -> Image.Image:
    values = mask.detach().cpu().numpy()
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[values == 1] = (255, 255, 255)
    if ground_truth:
        rgb[values == 255] = (127, 127, 127)
    return Image.fromarray(rgb, mode="RGB")


def _save_qualitative_panels(
    records: Sequence[PetRecord],
    test_targets: dict[str, Any],
    predictions: dict[tuple[str, str], torch.Tensor],
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    qualitative = protocol["frozen_spatial_probe"]["qualitative"]
    image_ids = list(qualitative["test_image_ids"])
    record_by_id = {record.image_id: record for record in records}
    index_by_id = {record.image_id: index for index, record in enumerate(records)}
    if not set(image_ids).issubset(record_by_id):
        raise RuntimeError("a predeclared qualitative test id is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "encoder_seed": int(qualitative["encoder_seed"]),
        "probe_seed": int(qualitative["probe_seed"]),
        "test_image_ids": image_ids,
        "selection_rule": qualitative["selection_rule"],
        "test_inference_reused_from_metric_pass": True,
        "posthoc_example_selection": False,
        "panel_sets": {},
    }
    shared_variants = ["vanilla", "kd", "lg", "alg"]
    for ibkd_variant in ("ibkd_lambda_0.25", "ibkd_lambda_0.5"):
        set_name = ibkd_variant.replace("ibkd_lambda_", "ibkd_")
        labels = ["input", "ground truth", "vanilla", "KD", "LG", "ALG", set_name]
        variants = shared_variants + [ibkd_variant]
        set_dir = output_dir / set_name
        mask_dir = set_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        panel_paths: list[str] = []
        for image_id in image_ids:
            with Image.open(record_by_id[image_id].image_path) as handle:
                rgb_input = handle.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            ground_truth = _mask_image(
                test_targets["input_targets"][index_by_id[image_id]],
                ground_truth=True,
            )
            images = [rgb_input, ground_truth]
            for variant in variants:
                key = (variant, image_id)
                if key not in predictions:
                    raise RuntimeError(f"missing qualitative prediction: {key}")
                prediction_image = _mask_image(predictions[key], ground_truth=False)
                prediction_path = mask_dir / f"{image_id}_{variant}.png"
                prediction_image.save(prediction_path)
                images.append(prediction_image)

            header_height = 28
            panel = Image.new(
                "RGB",
                (224 * len(images), 224 + header_height),
                color=(255, 255, 255),
            )
            draw = ImageDraw.Draw(panel)
            for column, (label, image) in enumerate(zip(labels, images, strict=True)):
                left = column * 224
                draw.text((left + 6, 7), label, fill=(0, 0, 0))
                panel.paste(image, (left, header_height))
            panel_path = set_dir / f"{image_id}.png"
            panel.save(panel_path)
            panel_paths.append(str(panel_path.relative_to(output_dir.parent)))
        manifest["panel_sets"][set_name] = {
            "panel_order": labels,
            "panels": panel_paths,
        }
    _atomic_json_save(manifest, output_dir / "manifest.json")
    return manifest


def _artifact_path(output_dir: Path, variant: str, encoder_seed: int, probe_seed: int) -> Path:
    return (
        output_dir
        / "probes"
        / variant
        / f"encoder_seed_{encoder_seed}"
        / f"probe_seed_{probe_seed}.pt"
    )


def _remove_feature_caches(paths: Sequence[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _target_values_are_valid(
    target_payloads: Sequence[dict[str, Any]],
    *,
    ignore_index: int,
) -> bool:
    allowed = {0, 1, ignore_index}
    return all(
        {
            int(value)
            for key in ("input_targets", "grid_targets")
            for value in torch.unique(payload[key])
        }.issubset(allowed)
        for payload in target_payloads
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    protocol = _load_json(args.protocol_config)
    _validate_protocol(protocol, args.protocol_config)
    protocol_sha256 = file_sha256(args.protocol_config)
    classification_batch_size = int(args.classification_batch_size)
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _sequence_status(
        args.output_dir,
        status="running",
        phase="preflight",
        completed_selections=0,
        completed_test_evaluations=0,
        active=None,
        official_test_accessed=False,
    )
    log(
        f"[PROBE_FULL_MODE] scientific_result=true "
        f"classification_batch={classification_batch_size} "
        "selection=validation_grid_miou official_test=sealed_until_all_90_selections"
    )

    entries, classification_suite, classification_split_hash = (
        _validate_classification_input(
            args.classification_root,
            expected_batch_size=classification_batch_size,
            encoder_seeds=EXPECTED_ENCODER_SEEDS,
        )
    )
    if len(entries) != 18:
        raise RuntimeError(
            f"expected all 18 batch-{classification_batch_size} encoder checkpoints"
        )
    log(
        f"[PROBE_FULL_INPUT] classification_batch={classification_batch_size} "
        "classification_suite=pass checkpoints=18 strict_load=pending"
    )

    records, split_manifest = load_train_validation_records(args.data_dir, download=True)
    if split_manifest["validation_image_ids_sha256"] != classification_split_hash:
        raise RuntimeError("probe validation split differs from classification")
    if {split: len(rows) for split, rows in records.items()} != {
        "train": 2940,
        "validation": 740,
    }:
        raise RuntimeError("full-probe train/validation counts changed")
    dataset_audit = _verify_dataset_identity(args.data_dir, classification_suite)
    log(
        "[PROBE_FULL_DATA] archive_and_manifest_audit=pass train=2940 "
        "validation=740 test_pixels_accessed=false"
    )

    target_started = time.monotonic()
    targets: dict[str, dict[str, Any]] = {}
    target_cache_audits: list[bool] = []
    for split in ("train", "validation"):
        targets[split], reloaded = _target_cache(
            records[split],
            split=split,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
        target_cache_audits.append(reloaded)
    target_cache_seconds = time.monotonic() - target_started

    probe_config = protocol["frozen_spatial_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    epochs = int(probe_config["epochs"])
    probe_seeds = tuple(int(value) for value in probe_config["probe_seeds"])
    input_size = int(protocol["frozen_spatial_probe"]["image_input"]["size"])
    ignore_index = int(protocol["frozen_spatial_probe"]["mask"]["ignore_index"])
    if not _target_values_are_valid(
        [targets["train"], targets["validation"]],
        ignore_index=ignore_index,
    ):
        raise RuntimeError("train/validation mapped-target contract failed")
    baseline_templates = _baseline_templates(
        targets["train"],
        ignore_index=ignore_index,
        input_size=input_size,
    )
    validation_baselines = _evaluate_baselines(
        baseline_templates,
        targets["validation"],
        ignore_index=ignore_index,
    )

    selection_started = time.monotonic()
    results: list[dict[str, Any]] = []
    feature_cache_audits: list[bool] = []
    encoder_audits: list[dict[str, Any]] = []
    initial_hashes: dict[int, set[str]] = {seed: set() for seed in probe_seeds}
    batch_order_hashes: dict[int, dict[int, set[str]]] = {
        seed: {epoch: set() for epoch in range(1, epochs + 1)}
        for seed in probe_seeds
    }
    gradient_contracts: list[dict[str, int]] = []
    peak_cuda_memory_bytes = 0
    candidate_count = 0

    log(
        "[PROBE_FULL_TASK_COUNT] encoders=18 probe_selections=90 "
        "lr_candidates=270 epochs_per_candidate=100"
    )
    for encoder_index, entry in enumerate(entries, start=1):
        variant = str(entry["variant"])
        encoder_seed = int(entry["encoder_seed"])
        active = {
            "stage": "selection",
            "variant": variant,
            "encoder_seed": encoder_seed,
            "encoder_index": encoder_index,
        }
        _sequence_status(
            args.output_dir,
            status="running",
            phase="validation_selection",
            completed_selections=len(results),
            completed_test_evaluations=0,
            active=active,
            official_test_accessed=False,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, encoder_audit = _load_encoder(entry, device)
        encoder_audits.append(
            {
                "variant": variant,
                "encoder_seed": encoder_seed,
                **encoder_audit,
            }
        )

        feature_started = time.monotonic()
        features: dict[str, dict[str, Any]] = {}
        feature_paths: list[Path] = []
        for split in ("train", "validation"):
            cache_path = (
                args.cache_dir
                / "features"
                / variant
                / f"encoder_seed_{encoder_seed}"
                / f"{split}.pt"
            )
            feature_paths.append(cache_path)
            features[split], reloaded = _feature_cache(
                model,
                entry,
                records[split],
                split=split,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                cache_path=cache_path,
                device=device,
                feature_batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
            feature_cache_audits.append(reloaded)
        _synchronize(device)
        feature_seconds = time.monotonic() - feature_started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for probe_seed in probe_seeds:
            candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
            probe_started = time.monotonic()
            for learning_rate in learning_rates:
                log(
                    "[PROBE_CANDIDATE_START] "
                    f"variant={variant} encoder_seed={encoder_seed} "
                    f"probe_seed={probe_seed} lr={learning_rate:g} epochs={epochs}"
                )
                candidate_started = time.monotonic()
                state, candidate = train_candidate(
                    features["train"]["features"],
                    targets["train"]["grid_targets"],
                    features["validation"]["features"],
                    targets["validation"]["grid_targets"],
                    probe_config=probe_config,
                    learning_rate=learning_rate,
                    seed=probe_seed,
                    device=device,
                    epochs=epochs,
                )
                _synchronize(device)
                candidate["elapsed_seconds"] = time.monotonic() - candidate_started
                candidates.append((state, candidate))
                candidate_count += 1
                initial_hashes[probe_seed].add(candidate["initial_probe_state_sha256"])
                gradient_contracts.append(candidate["gradient_contract"])
                for epoch, digest in enumerate(
                    candidate["batch_order_sha256_by_epoch"], start=1
                ):
                    batch_order_hashes[probe_seed][epoch].add(digest)
                log(
                    "[PROBE_CANDIDATE_DONE] "
                    f"variant={variant} encoder_seed={encoder_seed} "
                    f"probe_seed={probe_seed} lr={learning_rate:g} "
                    f"best_epoch={candidate['best_epoch']} "
                    "best_val_grid_miou="
                    f"{candidate['best_validation_grid_mean_iou']:.6f} "
                    f"seconds={candidate['elapsed_seconds']:.2f}"
                )

            selected_state, selected = _select_candidate(candidates)
            probe = probe_from_state(probe_config, probe_seed, selected_state, device)
            validation_metrics, _ = evaluate_probe_both_resolutions(
                probe,
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                targets["validation"]["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=input_size,
                ignore_index=ignore_index,
            )
            if not all(_finite_metrics(metrics) for metrics in validation_metrics.values()):
                raise RuntimeError("validation probe metrics are non-finite")
            artifact_path = _artifact_path(
                args.output_dir,
                variant,
                encoder_seed,
                probe_seed,
            )
            candidate_reports = [candidate for _, candidate in candidates]
            selection = {
                "split": "validation",
                "metric": "grid_14x14_two_class_mean_iou",
                "learning_rate": float(selected["learning_rate"]),
                "epoch": int(selected["best_epoch"]),
                "validation_grid_mean_iou": float(
                    selected["best_validation_grid_mean_iou"]
                ),
                "tie_break": ["lower_learning_rate", "earlier_epoch"],
            }
            if not math.isclose(
                validation_metrics["grid_14x14"]["mean_iou"],
                selection["validation_grid_mean_iou"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("selected validation metric changed after strict reload")
            _atomic_torch_save(
                {
                    "purpose": "phase1_scientific_full_frozen_probe",
                    "scientific_result": True,
                    "protocol_sha256": protocol_sha256,
                    "classification_batch_size": classification_batch_size,
                    "variant": variant,
                    "method": entry["method"],
                    "fusion_ratio_lambda": entry["fusion_ratio_lambda"],
                    "encoder_seed": encoder_seed,
                    "encoder_checkpoint_sha256": encoder_audit[
                        "checkpoint_sha256"
                    ],
                    "probe_seed": probe_seed,
                    "selection": selection,
                    "candidate_audits": candidate_reports,
                    "official_test_evaluations_at_checkpoint_write": 0,
                    "model": selected_state,
                },
                artifact_path,
            )
            artifact_payload = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=True,
            )
            metadata_checks = {
                "purpose": artifact_payload.get("purpose")
                == "phase1_scientific_full_frozen_probe",
                "protocol": artifact_payload.get("protocol_sha256")
                == protocol_sha256,
                "classification_batch_size": artifact_payload.get(
                    "classification_batch_size"
                )
                == classification_batch_size,
                "variant": artifact_payload.get("variant") == variant,
                "encoder_seed": artifact_payload.get("encoder_seed")
                == encoder_seed,
                "probe_seed": artifact_payload.get("probe_seed") == probe_seed,
                "test_count_at_write": artifact_payload.get(
                    "official_test_evaluations_at_checkpoint_write"
                )
                == 0,
            }
            if not all(metadata_checks.values()):
                raise RuntimeError("saved probe artifact metadata failed strict audit")
            strict_probe = probe_from_state(
                probe_config,
                probe_seed,
                artifact_payload["model"],
                device,
            )
            del strict_probe, artifact_payload, probe
            probe_seconds = time.monotonic() - probe_started
            checkpoint_relative = entry["checkpoint_path"].relative_to(
                args.classification_root
            )
            results.append(
                {
                    "variant": variant,
                    "method": entry["method"],
                    "fusion_ratio_lambda": entry["fusion_ratio_lambda"],
                    "encoder_seed": encoder_seed,
                    "probe_seed": probe_seed,
                    "classification_checkpoint": {
                        **encoder_audit,
                        "relative_path": str(checkpoint_relative),
                    },
                    "probe_artifact": {
                        "relative_path": str(artifact_path.relative_to(args.output_dir)),
                        "sha256": file_sha256(artifact_path),
                        "strict_reloaded": True,
                        "official_test_evaluations_at_write": 0,
                    },
                    "selection": selection,
                    "candidate_audits": candidate_reports,
                    "validation": validation_metrics,
                    "test": None,
                    "official_test_evaluations": 0,
                    "timing": {
                        "feature_cache_seconds_for_encoder": feature_seconds,
                        "probe_seed_total_seconds": probe_seconds,
                    },
                }
            )
            log(
                "[PROBE_SELECTION_DONE] "
                f"completed={len(results)}/{EXPECTED_SELECTIONS} variant={variant} "
                f"encoder_seed={encoder_seed} probe_seed={probe_seed} "
                f"lr={selection['learning_rate']:g} epoch={selection['epoch']} "
                "val_grid_miou="
                f"{selection['validation_grid_mean_iou']:.6f} test_accessed=false"
            )
            _sequence_status(
                args.output_dir,
                status="running",
                phase="validation_selection",
                completed_selections=len(results),
                completed_test_evaluations=0,
                active=active,
                official_test_accessed=False,
            )
            del candidates, selected_state

        if device.type == "cuda":
            peak_cuda_memory_bytes = max(
                peak_cuda_memory_bytes,
                int(torch.cuda.max_memory_allocated(device)),
            )
        del features
        _remove_feature_caches(feature_paths)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selection_seconds = time.monotonic() - selection_started
    selection_gates = {
        "all_18_classification_checkpoints_loaded_strictly": len(encoder_audits) == 18
        and all(audit["strict_load"] for audit in encoder_audits),
        "all_270_lr_candidates_completed": candidate_count == EXPECTED_LR_CANDIDATES,
        "all_90_probes_selected_by_validation": len(results) == EXPECTED_SELECTIONS,
        "no_official_test_evaluation_before_selection_complete": all(
            result["official_test_evaluations"] == 0 for result in results
        ),
        "same_initial_probe_state_per_probe_seed_across_all_candidates": all(
            len(values) == 1 for values in initial_hashes.values()
        ),
        "same_batch_order_per_probe_seed_and_epoch_across_all_candidates": all(
            len(values) == 1
            for epoch_map in batch_order_hashes.values()
            for values in epoch_map.values()
        ),
        "only_two_probe_parameter_gradients": all(
            contract
            == {
                "cached_feature_gradient_tensor_count": 0,
                "probe_gradient_tensor_count": 2,
            }
            for contract in gradient_contracts
        ),
        "target_cache_safe_reload": all(target_cache_audits),
        "feature_cache_safe_reload": all(feature_cache_audits),
        "dataset_identity_passed": dataset_audit["status"] == "pass",
        "trimaps_mapped_only_to_background_foreground_ignore": (
            _target_values_are_valid(
                [targets["train"], targets["validation"]],
                ignore_index=ignore_index,
            )
        ),
        "classification_validation_split_reused": split_manifest[
            "validation_image_ids_sha256"
        ]
        == classification_split_hash,
    }
    if not all(selection_gates.values()):
        failures = [name for name, passed in selection_gates.items() if not passed]
        raise RuntimeError("selection gates failed: " + ", ".join(failures))

    selection_complete = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "scientific_result": True,
        "protocol_sha256": protocol_sha256,
        "classification_batch_size": classification_batch_size,
        "selection_split": "validation",
        "completed_selections": len(results),
        "official_test_accessed": False,
        "official_test_evaluations": 0,
        "selection_gates": selection_gates,
        "selected_probes": results,
    }
    _atomic_json_save(
        selection_complete,
        args.output_dir / "selection_complete_before_test.json",
    )
    log(
        "[PROBE_SELECTION_ALL_DONE] completed=90/90 gates=pass "
        "official_test_evaluations=0 opening_test_now=true"
    )

    # This is the first construction of the official test dataset in this process.
    selection_count_when_test_opened = len(results)
    selection_record_existed_when_test_opened = (
        args.output_dir / "selection_complete_before_test.json"
    ).is_file()
    test_started = time.monotonic()
    test_records = load_official_test_records(args.data_dir)
    if len(test_records) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("official test count changed")
    records["test"] = test_records
    targets["test"], test_target_reloaded = _target_cache(
        test_records,
        split="test",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        cache_path=args.cache_dir / "targets" / "test.pt",
    )
    if not _target_values_are_valid([targets["test"]], ignore_index=ignore_index):
        raise RuntimeError("official-test mapped-target contract failed")
    test_baselines = _evaluate_baselines(
        baseline_templates,
        targets["test"],
        ignore_index=ignore_index,
    )
    qualitative_config = protocol["frozen_spatial_probe"]["qualitative"]
    qualitative_ids = list(qualitative_config["test_image_ids"])
    test_index_by_id = {
        record.image_id: index for index, record in enumerate(test_records)
    }
    if not set(qualitative_ids).issubset(test_index_by_id):
        raise RuntimeError("predeclared qualitative IDs are absent from official test")
    capture_indices = {test_index_by_id[image_id] for image_id in qualitative_ids}
    qualitative_predictions: dict[tuple[str, str], torch.Tensor] = {}
    result_by_key = {
        (result["variant"], result["encoder_seed"], result["probe_seed"]): result
        for result in results
    }
    test_evaluations = 0

    for encoder_index, entry in enumerate(entries, start=1):
        variant = str(entry["variant"])
        encoder_seed = int(entry["encoder_seed"])
        active = {
            "stage": "official_test",
            "variant": variant,
            "encoder_seed": encoder_seed,
            "encoder_index": encoder_index,
        }
        _sequence_status(
            args.output_dir,
            status="running",
            phase="official_test",
            completed_selections=EXPECTED_SELECTIONS,
            completed_test_evaluations=test_evaluations,
            active=active,
            official_test_accessed=True,
        )
        model, encoder_audit = _load_encoder(entry, device)
        cache_path = (
            args.cache_dir
            / "features"
            / variant
            / f"encoder_seed_{encoder_seed}"
            / "test.pt"
        )
        test_features, reloaded = _feature_cache(
            model,
            entry,
            test_records,
            split="test",
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            cache_path=cache_path,
            device=device,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        feature_cache_audits.append(reloaded)
        del model
        if device.type == "cuda":
            peak_cuda_memory_bytes = max(
                peak_cuda_memory_bytes,
                int(torch.cuda.max_memory_allocated(device)),
            )
            torch.cuda.empty_cache()

        for probe_seed in probe_seeds:
            result = result_by_key[(variant, encoder_seed, probe_seed)]
            artifact_path = args.output_dir / result["probe_artifact"]["relative_path"]
            if file_sha256(artifact_path) != result["probe_artifact"]["sha256"]:
                raise RuntimeError("selected probe artifact changed before test")
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
            checks = {
                "purpose": artifact.get("purpose")
                == "phase1_scientific_full_frozen_probe",
                "protocol": artifact.get("protocol_sha256") == protocol_sha256,
                "variant": artifact.get("variant") == variant,
                "encoder_seed": artifact.get("encoder_seed") == encoder_seed,
                "probe_seed": artifact.get("probe_seed") == probe_seed,
                "encoder_checkpoint": artifact.get("encoder_checkpoint_sha256")
                == encoder_audit["checkpoint_sha256"],
                "test_zero_at_write": artifact.get(
                    "official_test_evaluations_at_checkpoint_write"
                )
                == 0,
            }
            if not all(checks.values()):
                raise RuntimeError("selected probe failed pre-test strict audit")
            probe = probe_from_state(
                probe_config,
                probe_seed,
                artifact["model"],
                device,
            )
            should_capture = (
                encoder_seed == int(qualitative_config["encoder_seed"])
                and probe_seed == int(qualitative_config["probe_seed"])
            )
            test_metrics, captured = evaluate_probe_both_resolutions(
                probe,
                test_features["features"],
                targets["test"]["grid_targets"],
                targets["test"]["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=input_size,
                ignore_index=ignore_index,
                capture_indices=capture_indices if should_capture else None,
            )
            test_evaluations += 1
            result["test"] = test_metrics
            result["official_test_evaluations"] = 1
            if not all(_finite_metrics(metrics) for metrics in test_metrics.values()):
                raise RuntimeError("official test probe metrics are non-finite")
            if should_capture:
                for image_id in qualitative_ids:
                    qualitative_predictions[(variant, image_id)] = captured[
                        test_index_by_id[image_id]
                    ]
            log(
                "[PROBE_TEST_ONCE] "
                f"completed={test_evaluations}/{EXPECTED_SELECTIONS} "
                f"variant={variant} encoder_seed={encoder_seed} "
                f"probe_seed={probe_seed} input_miou="
                f"{test_metrics['input_224']['mean_iou']:.6f}"
            )
            _sequence_status(
                args.output_dir,
                status="running",
                phase="official_test",
                completed_selections=EXPECTED_SELECTIONS,
                completed_test_evaluations=test_evaluations,
                active=active,
                official_test_accessed=True,
            )
            del probe, artifact, captured

        del test_features
        _remove_feature_caches([cache_path])
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    test_seconds = time.monotonic() - test_started
    expected_qualitative_keys = {
        (variant, image_id)
        for variant in EXPECTED_VARIANTS
        for image_id in qualitative_ids
    }
    qualitative_complete = set(qualitative_predictions) == expected_qualitative_keys
    qualitative_manifest = _save_qualitative_panels(
        test_records,
        targets["test"],
        qualitative_predictions,
        protocol,
        args.output_dir / "qualitative",
    )

    final_gates = {
        **selection_gates,
        "official_test_opened_only_after_all_selections": (
            selection_count_when_test_opened == EXPECTED_SELECTIONS
            and selection_record_existed_when_test_opened
        ),
        "official_test_evaluated_exactly_once_per_selected_probe": test_evaluations
        == EXPECTED_SELECTIONS
        and all(result["official_test_evaluations"] == 1 for result in results),
        "all_test_metrics_finite": all(
            all(_finite_metrics(metrics) for metrics in result["test"].values())
            for result in results
        ),
        "test_target_cache_safe_reload": test_target_reloaded,
        "test_trimap_mapped_only_to_background_foreground_ignore": (
            _target_values_are_valid([targets["test"]], ignore_index=ignore_index)
        ),
        "all_feature_caches_safe_reload": all(feature_cache_audits),
        "both_non_learned_baselines_reported": set(validation_baselines)
        == {"all_background", "train_mean_mask"}
        and set(test_baselines) == {"all_background", "train_mean_mask"},
        "eight_fixed_qualitative_ids_reported_for_all_variants": qualitative_complete,
        "qualitative_inference_reused_from_test_metric_pass": qualitative_manifest[
            "test_inference_reused_from_metric_pass"
        ],
        "no_posthoc_qualitative_selection": not qualitative_manifest[
            "posthoc_example_selection"
        ],
    }
    if not all(final_gates.values()):
        failures = [name for name, passed in final_gates.items() if not passed]
        raise RuntimeError("final full-probe gates failed: " + ", ".join(failures))

    aggregates = _aggregate(results)
    elapsed_seconds = time.monotonic() - started
    summary = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "scientific_result": True,
        "experiment": (
            "Phase 1 Oxford-IIIT Pet "
            f"batch-{classification_batch_size} frozen spatial probe"
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "classification_batch_size": classification_batch_size,
        "runtime": _runtime(device),
        "matrix": {
            "variants": list(EXPECTED_VARIANTS),
            "encoder_seeds": list(EXPECTED_ENCODER_SEEDS),
            "probe_seeds": list(EXPECTED_PROBE_SEEDS),
            "learning_rates": learning_rates,
            "epochs_per_lr_candidate": epochs,
            "lr_candidate_count": candidate_count,
            "selected_probe_count": len(results),
        },
        "data": {
            "counts": {"train": 2940, "validation": 740, "test": 3669},
            "split_manifest": split_manifest,
            "dataset_audit": dataset_audit,
        },
        "probe_contract": protocol["frozen_spatial_probe"],
        "test_policy": {
            "selection_uses": "validation_grid_14x14_two_class_mean_iou",
            "official_test_used_for_training_or_selection": False,
            "official_test_opened_after_selection_complete": True,
            "official_test_evaluations": test_evaluations,
            "expected_official_test_evaluations": EXPECTED_SELECTIONS,
            "one_forward_pass_per_selected_probe_for_all_metrics": True,
        },
        "baselines": {
            "validation": validation_baselines,
            "test": test_baselines,
        },
        "raw_results": results,
        "aggregates": aggregates,
        "qualitative": qualitative_manifest,
        "contracts": {"all_passed": True, **final_gates},
        "statistics_policy": protocol["statistics"],
        "timing": {
            "target_cache_seconds_before_selection": target_cache_seconds,
            "validation_selection_seconds": selection_seconds,
            "official_test_and_qualitative_seconds": test_seconds,
            "total_seconds": elapsed_seconds,
        },
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }
    _atomic_json_save(summary, args.output_dir / "probe_summary.json")
    _write_raw_csv(results, args.output_dir / "probe_raw_results.csv")

    for result in results:
        log(
            "[PROBE_FULL_RAW] "
            f"variant={result['variant']} encoder_seed={result['encoder_seed']} "
            f"probe_seed={result['probe_seed']} "
            f"lr={result['selection']['learning_rate']:g} "
            f"epoch={result['selection']['epoch']} "
            "test_input_miou="
            f"{result['test']['input_224']['mean_iou']:.6f}"
        )
    for variant in EXPECTED_VARIANTS:
        metric = aggregates["variants"][variant]["across_encoder_seed_means"][
            "test"
        ]["input_224_mean_iou"]
        log(
            "[PROBE_FULL_RESULT] "
            f"variant={variant} test_input_miou_mean={metric['mean']:.6f} "
            f"sample_sd={metric['sample_standard_deviation']:.6f} "
            "encoder_seed_means="
            + ",".join(f"{value:.6f}" for value in metric["values"])
        )
    for name, contrast in aggregates["paired_primary_contrasts"].items():
        difference = contrast["difference_summary"]
        log(
            "[PROBE_FULL_CONTRAST] "
            f"name={name} mean={difference['mean']:.6f} "
            f"sample_sd={difference['sample_standard_deviation']:.6f} "
            "formal_p_value=false"
        )

    _sequence_status(
        args.output_dir,
        status="complete",
        phase="complete",
        completed_selections=EXPECTED_SELECTIONS,
        completed_test_evaluations=EXPECTED_SELECTIONS,
        active=None,
        official_test_accessed=True,
    )
    log(
        f"[PROBE_FULL_DONE] status=pass "
        f"classification_batch={classification_batch_size} selections=90/90 "
        f"test_once=90/90 seconds={elapsed_seconds:.2f} "
        f"output={args.output_dir}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-probe",
        action="store_true",
        help="required acknowledgement that this is the scientific full probe",
    )
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--classification-batch-size",
        type=int,
        choices=(64, 128),
        default=64,
        help="classification checkpoint profile to validate and probe",
    )
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.full_probe:
        parser.error("--full-probe is required")
    if args.feature_batch_size <= 0 or args.num_workers < 0:
        parser.error("invalid loader settings")
    return args


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        try:
            previous = _load_json(args.output_dir / "sequence_status.json")
            _sequence_status(
                args.output_dir,
                status="failed",
                phase="failed",
                completed_selections=int(previous.get("completed_selections", 0)),
                completed_test_evaluations=int(
                    previous.get("completed_official_test_evaluations", 0)
                ),
                active=previous.get("active"),
                official_test_accessed=bool(
                    previous.get("official_test_accessed", False)
                ),
                failure=f"{type(error).__name__}: {error}",
            )
        except Exception:
            pass
        log(f"[PROBE_FULL_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
