#!/usr/bin/env python3
"""Audit CUB train-time crop damage without training or opening official test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

from .cub_data import DERIVED_TRAIN_COUNT, DERIVED_VALIDATION_COUNT
from .cub_direct_spatial import load_spatial_annotations
from .cub_loader_profiles import (
    LOADER_PROFILE_ORDER,
    crop_geometry,
    loader_profile_contract,
    sample_random_resized_crop,
)
from .cub_probe_data import CubProbeRecord, load_train_validation_records
from .data import save_json
from .train_timing import file_sha256, runtime_metadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/image_loader_experiment/configs/"
    "cub200_r50_224_loader_damage_audit_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "6eeff24ba01b188d00f3f8ec7e4de533cd4b531eb95bda352b36fdd9fb9695bf"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty loader-audit CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    profiles = config.get("loader_profiles", [])
    ids = tuple(item.get("id") for item in profiles)
    paired = config.get("paired_sampling", {})
    gate = config.get("completion_gate", {})
    checks = {
        "config_sha256": file_sha256(config_path) == EXPECTED_CONFIG_SHA256,
        "audit_id": config.get("audit_id")
        == "cub200_phase1_r50_224_student_loader_damage_audit_v1",
        "locked": config.get("status")
        == "locked_before_loader_audit_results_2026-09-12",
        "posthoc": config.get("scientific_role")
        == "posthoc_exploratory_loader_diagnostic",
        "no_training": config.get("training_performed") is False,
        "test_closed": config.get("official_test_accessed") is False,
        "dataset": config.get("dataset", {}).get("name") == "CUB-200-2011",
        "counts": config.get("dataset", {}).get("train_samples")
        == DERIVED_TRAIN_COUNT
        and config.get("dataset", {}).get("validation_samples_not_used_for_audit")
        == DERIVED_VALIDATION_COUNT
        and config.get("dataset", {}).get("official_test_samples_not_opened") == 5794,
        "profiles": ids == LOADER_PROFILE_ORDER,
        "draws": paired.get("draws_per_train_image") == 5
        and gate.get("draws_per_profile") == DERIVED_TRAIN_COUNT * 5,
        "paired_geometry": paired.get("same_geometry_id_uses_identical_crop_draws")
        is True
        and paired.get("l0_l1_geometry_must_match_exactly") is True,
        "test_gate": gate.get("official_test_images_or_masks_decoded") == 0,
        "mig": config.get("execution", {}).get("requested_mig_slices") == 1,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid CUB loader audit config: " + ", ".join(failures))

    expected_contracts = {
        item["id"]: item for item in config["loader_profiles"]
    }
    for profile in LOADER_PROFILE_ORDER:
        actual = loader_profile_contract(profile)
        expected = expected_contracts[profile]
        pairs = {
            "scale": actual["random_resized_crop"]["scale"]
            == expected["random_resized_crop_scale"],
            "ratio": actual["random_resized_crop"]["ratio"]
            == expected["random_resized_crop_ratio"],
            "flip": actual["horizontal_flip_probability"]
            == expected["horizontal_flip_probability"],
            "jitter": actual["color_jitter_argument"]
            == expected["color_jitter_argument"],
            "auto_augment": actual["auto_augment"] == expected["auto_augment"],
            "erasing": actual["random_erasing_probability"]
            == expected["random_erasing_probability"],
        }
        mismatch = [name for name, matched in pairs.items() if not matched]
        if mismatch:
            raise RuntimeError(
                f"loader profile/config mismatch for {profile}: {mismatch}"
            )


def _paired_seed(base_seed: int, geometry_id: str, image_id: int, draw: int) -> int:
    value = f"{base_seed}:{geometry_id}:{image_id}:{draw}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _inside_crop(
    x: np.ndarray,
    y: np.ndarray,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    return (
        (x >= left)
        & (x <= left + width)
        & (y >= top)
        & (y <= top + height)
    )


def _box_coverage(
    box: tuple[float, float, float, float],
    *,
    top: int,
    left: int,
    height: int,
    width: int,
) -> float:
    x, y, box_width, box_height = box
    intersection_width = max(0.0, min(x + box_width, left + width) - max(x, left))
    intersection_height = max(0.0, min(y + box_height, top + height) - max(y, top))
    return intersection_width * intersection_height / (box_width * box_height)


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("loader audit metric is empty or non-finite")
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _select_qualitative(records: list[CubProbeRecord], count: int) -> list[CubProbeRecord]:
    by_class: dict[int, CubProbeRecord] = {}
    for record in sorted(records, key=lambda item: item.image_id):
        by_class.setdefault(record.label, record)
    if len(by_class) != 200 or count <= 0 or count > len(by_class):
        raise RuntimeError("invalid result-independent qualitative selection")
    return [by_class[label] for label in sorted(by_class)[:count]]


def _draw_overlay(
    record: CubProbeRecord,
    *,
    annotation: Any,
    crop: tuple[int, int, int, int],
    destination: Path,
) -> None:
    top, left, height, width = crop
    with Image.open(record.image_path) as handle:
        image = handle.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (left, top, left + width, top + height),
        outline=(255, 215, 0),
        width=max(2, min(image.size) // 120),
    )
    x, y, box_width, box_height = annotation.bounding_box
    draw.rectangle(
        (x, y, x + box_width, y + box_height),
        outline=(0, 220, 255),
        width=max(2, min(image.size) // 120),
    )
    for (point_x, point_y), visible in zip(
        annotation.points, annotation.visible, strict=True
    ):
        if visible and 0 <= point_x <= image.width and 0 <= point_y <= image.height:
            radius = max(2, min(image.size) // 100)
            draw.ellipse(
                (
                    point_x - radius,
                    point_y - radius,
                    point_x + radius,
                    point_y + radius,
                ),
                fill=(255, 64, 64),
            )
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config = _load_json(args.config)
    _validate_config(config, args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = file_sha256(args.config)
    save_json(
        {
            "status": "running",
            "phase": "dataset_setup",
            "official_test_images_or_masks_decoded": 0,
        },
        args.output_dir / "sequence_status.json",
    )

    partitions, split_manifest, source = load_train_validation_records(
        args.data_dir, download=True
    )
    train_records = partitions["train"]
    if len(train_records) != DERIVED_TRAIN_COUNT:
        raise RuntimeError("CUB derived-train count changed")
    if len(partitions["validation"]) != DERIVED_VALIDATION_COUNT:
        raise RuntimeError("CUB derived-validation count changed")
    if (
        split_manifest["validation_image_ids_sha256"]
        != config["dataset"]["validation_image_ids_sha256"]
    ):
        raise RuntimeError("CUB validation split hash changed")
    dataset_root = Path(source["dataset_root"])
    _, annotations = load_spatial_annotations(dataset_root)

    base_seed = int(config["paired_sampling"]["base_seed"])
    draws_per_image = int(config["paired_sampling"]["draws_per_train_image"])
    near_empty_threshold = float(
        config["metrics"]["near_empty_foreground_fraction_threshold"]
    )
    by_geometry: dict[str, list[str]] = defaultdict(list)
    for profile in LOADER_PROFILE_ORDER:
        by_geometry[crop_geometry(profile).identity].append(profile)

    geometry_metrics: dict[str, dict[str, list[float]]] = {}
    qualitative_crops: dict[tuple[str, int], tuple[int, int, int, int]] = {}
    qualitative_records = _select_qualitative(
        train_records, int(config["qualitative"]["examples"])
    )
    qualitative_ids = {record.image_id for record in qualitative_records}
    completed_draws = 0
    expected_geometry_draws = len(by_geometry) * len(train_records) * draws_per_image
    for geometry_id, profiles in by_geometry.items():
        representative = profiles[0]
        metrics: dict[str, list[float]] = defaultdict(list)
        for record_index, record in enumerate(train_records, start=1):
            annotation = annotations[record.image_id]
            with Image.open(record.image_path) as image_handle:
                image = image_handle.convert("RGB")
                width, height = image.size
            with Image.open(record.mask_path) as mask_handle:
                if mask_handle.size != (width, height):
                    raise RuntimeError(f"image/mask size mismatch: {record.image_id}")
                mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8) > 0
            foreground_total = int(mask.sum())
            if foreground_total <= 0:
                raise RuntimeError(f"empty official CUB mask: {record.image_id}")
            points = np.asarray(annotation.points, dtype=np.float64)
            visible = np.asarray(annotation.visible, dtype=np.bool_)
            visible &= (
                (points[:, 0] >= 0)
                & (points[:, 0] <= width)
                & (points[:, 1] >= 0)
                & (points[:, 1] <= height)
            )
            visible_count = int(visible.sum())
            if visible_count <= 0:
                raise RuntimeError(f"no valid visible CUB part: {record.image_id}")

            for draw in range(draws_per_image):
                generator = torch.Generator().manual_seed(
                    _paired_seed(base_seed, geometry_id, record.image_id, draw)
                )
                crop = sample_random_resized_crop(
                    image, profile=representative, generator=generator
                )
                top, left, crop_height, crop_width = crop
                retained = _inside_crop(
                    points[:, 0],
                    points[:, 1],
                    top=top,
                    left=left,
                    height=crop_height,
                    width=crop_width,
                ) & visible
                crop_foreground = int(
                    mask[top : top + crop_height, left : left + crop_width].sum()
                )
                crop_pixels = crop_height * crop_width
                foreground_fraction = crop_foreground / crop_pixels
                metrics["visible_part_retention"].append(
                    float(retained.sum() / visible_count)
                )
                metrics["all_visible_parts_retained"].append(
                    float(retained.sum() == visible_count)
                )
                metrics["foreground_mask_retention"].append(
                    crop_foreground / foreground_total
                )
                metrics["bounding_box_coverage"].append(
                    _box_coverage(
                        annotation.bounding_box,
                        top=top,
                        left=left,
                        height=crop_height,
                        width=crop_width,
                    )
                )
                metrics["crop_foreground_fraction"].append(foreground_fraction)
                metrics["crop_area_fraction"].append(crop_pixels / (width * height))
                metrics["near_empty_crop"].append(
                    float(foreground_fraction < near_empty_threshold)
                )
                metrics["background_only_crop"].append(float(crop_foreground == 0))
                if record.image_id in qualitative_ids and draw == 0:
                    qualitative_crops[(geometry_id, record.image_id)] = crop
                completed_draws += 1
            if record_index % 500 == 0:
                print(
                    f"[CUB_LOADER_AUDIT_PROGRESS] geometry={geometry_id} "
                    f"records={record_index}/{len(train_records)} "
                    f"draws={completed_draws}/{expected_geometry_draws}",
                    flush=True,
                )
        geometry_metrics[geometry_id] = dict(metrics)

    rows: list[dict[str, Any]] = []
    profile_summaries: dict[str, Any] = {}
    for profile in LOADER_PROFILE_ORDER:
        geometry_id = crop_geometry(profile).identity
        values = geometry_metrics[geometry_id]
        summary = {
            "profile_contract": loader_profile_contract(profile),
            "records": len(train_records),
            "draws_per_image": draws_per_image,
            "total_draws": len(train_records) * draws_per_image,
            "metrics": {name: _summary(metric) for name, metric in values.items()},
        }
        profile_summaries[profile] = summary
        row: dict[str, Any] = {
            "profile": profile,
            "geometry_id": geometry_id,
            "records": len(train_records),
            "total_draws": summary["total_draws"],
        }
        for metric_name, metric_summary in summary["metrics"].items():
            row[f"{metric_name}_mean"] = metric_summary["mean"]
            row[f"{metric_name}_p05"] = metric_summary["p05"]
            row[f"{metric_name}_median"] = metric_summary["median"]
        rows.append(row)

    l0_metrics = profile_summaries[LOADER_PROFILE_ORDER[0]]["metrics"]
    l1_metrics = profile_summaries[LOADER_PROFILE_ORDER[1]]["metrics"]
    l0_l1_equal = l0_metrics == l1_metrics
    if not l0_l1_equal:
        raise RuntimeError("L0/L1 paired crop geometry metrics diverged")

    qualitative_manifest: list[dict[str, Any]] = []
    for profile in LOADER_PROFILE_ORDER:
        geometry_id = crop_geometry(profile).identity
        for record in qualitative_records:
            destination = (
                args.output_dir
                / "qualitative"
                / profile
                / f"image_{record.image_id}_crop_overlay.png"
            )
            crop = qualitative_crops[(geometry_id, record.image_id)]
            _draw_overlay(
                record,
                annotation=annotations[record.image_id],
                crop=crop,
                destination=destination,
            )
            qualitative_manifest.append(
                {
                    "profile": profile,
                    "image_id": record.image_id,
                    "relative_path": record.relative_path,
                    "crop_top_left_height_width": list(crop),
                    "file": str(destination.resolve()),
                    "legend": {
                        "yellow": "sampled_random_resized_crop",
                        "cyan": "official_bounding_box",
                        "red": "valid_visible_part",
                    },
                }
            )

    _write_csv(rows, args.output_dir / "loader_damage_summary.csv")
    save_json(
        {"selection_rule": config["qualitative"]["selection"], "items": qualitative_manifest},
        args.output_dir / "qualitative_manifest.json",
    )
    elapsed = time.monotonic() - started
    result = {
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_id": config["audit_id"],
        "scientific_role": config["scientific_role"],
        "training_performed": False,
        "official_test_accessed": False,
        "official_test_images_or_masks_decoded": 0,
        "config": {"path": str(args.config.resolve()), "sha256": config_hash},
        "source": source,
        "split_manifest": split_manifest,
        "profile_summaries": profile_summaries,
        "contracts": {
            "all_passed": True,
            "derived_train_only": True,
            "finite_metrics": all(
                math.isfinite(value)
                for summary in profile_summaries.values()
                for metric in summary["metrics"].values()
                for value in metric.values()
            ),
            "l0_l1_geometry_metrics_exactly_equal": l0_l1_equal,
            "spatial_annotations_used_for_diagnosis_only": True,
            "spatial_annotations_exposed_to_training": False,
            "loader_selection_from_audit_forbidden": True,
        },
        "elapsed_seconds": elapsed,
        "runtime": runtime_metadata(torch.device("cpu")),
    }
    save_json(result, args.output_dir / "loader_damage_audit.json")
    save_json(
        {
            "status": "pass",
            "phase": "complete",
            "profile_summaries": len(profile_summaries),
            "draws_per_profile": len(train_records) * draws_per_image,
            "official_test_images_or_masks_decoded": 0,
            "elapsed_seconds": elapsed,
        },
        args.output_dir / "sequence_status.json",
    )

    print("[CUB_LOADER_AUDIT_FINAL_RESULTS]", flush=True)
    for row in rows:
        print(
            "[CUB_LOADER_AUDIT_RESULT] "
            f"profile={row['profile']} "
            f"part_retention={row['visible_part_retention_mean']:.6f} "
            f"all_parts={row['all_visible_parts_retained_mean']:.6f} "
            f"mask_retention={row['foreground_mask_retention_mean']:.6f} "
            f"bbox_coverage={row['bounding_box_coverage_mean']:.6f} "
            f"near_empty={row['near_empty_crop_mean']:.6f} "
            f"background_only={row['background_only_crop_mean']:.6f}",
            flush=True,
        )
    print(
        f"[CUB_LOADER_AUDIT_COMPLETE] status=pass profiles=3 "
        f"draws_per_profile={len(train_records) * draws_per_image} "
        "official_test_images_or_masks_decoded=0 "
        f"elapsed_seconds={elapsed:.2f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "status": "failed",
                "phase": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "official_test_images_or_masks_decoded": 0,
            },
            args.output_dir / "sequence_status.json",
        )
        print(
            f"[CUB_LOADER_AUDIT_FAILED] {type(error).__name__}: {error}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
