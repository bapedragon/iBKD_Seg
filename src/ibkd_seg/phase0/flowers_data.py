"""압축 해제한 Oxford Flowers-102 이미지, 마스크, 라벨, split ID를 감사한다."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .checkpoints import sha256_file
from .masks import BLUE_BACKGROUND_RGB, recover_foreground_alpha


IMAGE_PATTERN = re.compile(r"image_(\d+)\.jpg$")
MASK_PATTERN = re.compile(r"segmim_(\d+)\.jpg$")


def _indexed_files(root: Path, glob_pattern: str, id_pattern: re.Pattern[str]) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in root.rglob(glob_pattern):
        match = id_pattern.search(path.name)
        if not match:
            continue
        image_id = int(match.group(1))
        if image_id in indexed:
            raise ValueError(f"duplicate ID {image_id}: {indexed[image_id]} and {path}")
        indexed[image_id] = path
    return indexed


def _only_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def _mat_ids(path: Path, key: str) -> set[int]:
    values = np.asarray(loadmat(path)[key]).reshape(-1)
    return {int(value) for value in values}


def _mask_profile(
    images: dict[int, Path],
    masks: dict[int, Path],
    sample_count: int,
) -> dict[str, Any]:
    paired_ids = sorted(set(images) & set(masks))
    if len(paired_ids) > sample_count:
        selected_indices = np.linspace(0, len(paired_ids) - 1, sample_count, dtype=int)
        selected_ids = [paired_ids[int(index)] for index in selected_indices]
    else:
        selected_ids = paired_ids
    colors: Counter[tuple[int, ...]] = Counter()
    modes: Counter[str] = Counter()
    foreground_fractions: list[float] = []
    transition_fractions: list[float] = []
    threshold_sensitivity_fractions: list[float] = []
    simple_key_disagreements: list[float] = []
    for image_id in selected_ids:
        with Image.open(images[image_id]) as image, Image.open(masks[image_id]) as mask:
            modes[mask.mode] += 1
            source = np.asarray(image.convert("RGB"), dtype=np.uint8)
            composite = np.asarray(mask.convert("RGB"), dtype=np.uint8)
            flattened = composite.reshape(-1, 3)
            unique, counts = np.unique(flattened, axis=0, return_counts=True)
            top_indices = np.argsort(counts)[-50:]
            colors.update(
                {
                    tuple(int(value) for value in unique[index]): int(counts[index])
                    for index in top_indices
                }
            )
            alpha = recover_foreground_alpha(source, composite)
            foreground = alpha >= 0.5
            blue_distance = np.linalg.norm(
                composite.astype(np.float32) - BLUE_BACKGROUND_RGB,
                axis=-1,
            )
            foreground_fractions.append(float(foreground.mean()))
            transition_fractions.append(float(((alpha > 0.05) & (alpha < 0.95)).mean()))
            threshold_sensitivity_fractions.append(
                float(((alpha >= 0.25) != (alpha >= 0.75)).mean())
            )
            simple_key_disagreements.append(float((foreground != (blue_distance > 30.0)).mean()))

    def summarize(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    return {
        "sample_count": len(selected_ids),
        "sample_ids": selected_ids,
        "modes": dict(modes),
        "conversion_rule": (
            "least-squares alpha from composite = alpha*original + "
            "(1-alpha)*RGB(0,0,255); foreground iff alpha >= 0.5"
        ),
        "foreground_fraction": summarize(foreground_fractions),
        "jpeg_transition_fraction_alpha_0p05_to_0p95": summarize(transition_fractions),
        "threshold_sensitivity_fraction_alpha_0p25_to_0p75": summarize(
            threshold_sensitivity_fractions
        ),
        "disagreement_with_blue_distance_gt_30": summarize(simple_key_disagreements),
        "top_rgb_colors": [
            {"rgb": list(color), "pixels": count}
            for color, count in colors.most_common(20)
        ],
    }


def _asset_records(data_root: Path, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for asset in assets:
        matches = list(data_root.rglob(asset["name"]))
        if len(matches) != 1:
            record = {
                "name": asset["name"],
                "status": "fail",
                "matches": [str(path.relative_to(data_root)) for path in matches],
                "expected_size_bytes": asset["size_bytes"],
            }
            if "sha256" in asset:
                record["expected_sha256"] = asset["sha256"]
            records.append(record)
            continue
        path = matches[0]
        size_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        expected_sha256 = asset.get("sha256")
        record = {
            "name": asset["name"],
            "status": (
                "pass"
                if size_bytes == asset["size_bytes"]
                and (expected_sha256 is None or actual_sha256 == expected_sha256)
                else "fail"
            ),
            "relative_path": str(path.relative_to(data_root)),
            "size_bytes": size_bytes,
            "expected_size_bytes": asset["size_bytes"],
            "sha256": actual_sha256,
        }
        if expected_sha256 is not None:
            record["expected_sha256"] = expected_sha256
        records.append(record)
    return records


def _mask_quality_summary(foreground_fractions: dict[int, float]) -> dict[str, Any]:
    if not foreground_fractions:
        return {
            "resolution": "native",
            "count": 0,
            "foreground_fraction_quantiles": None,
            "empty_ids": [],
            "near_empty_ids": [],
            "near_full_ids": [],
        }
    ids = sorted(foreground_fractions)
    values = np.asarray([foreground_fractions[image_id] for image_id in ids])
    return {
        "resolution": "native",
        "count": len(ids),
        "foreground_fraction_quantiles": {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "empty_ids": [
            image_id for image_id, value in foreground_fractions.items() if value == 0.0
        ],
        "near_empty_ids": [
            image_id for image_id, value in foreground_fractions.items() if value < 0.005
        ],
        "near_full_ids": [
            image_id for image_id, value in foreground_fractions.items() if value > 0.995
        ],
    }


def audit_dataset(data_root: Path, manifest_path: Path, mask_samples: int = 32) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["expected"]
    asset_records = _asset_records(data_root, manifest.get("assets", []))
    images = _indexed_files(data_root, "image_*.jpg", IMAGE_PATTERN)
    masks = _indexed_files(data_root, "segmim_*.jpg", MASK_PATTERN)
    setid_path = _only_file(data_root, "setid.mat")
    labels_path = _only_file(data_root, "imagelabels.mat")

    split_ids = {
        "train": _mat_ids(setid_path, "trnid"),
        "val": _mat_ids(setid_path, "valid"),
        "test": _mat_ids(setid_path, "tstid"),
    }
    labels = np.asarray(loadmat(labels_path)["labels"]).reshape(-1)
    all_split_ids = set().union(*split_ids.values())
    expected_ids = set(range(1, expected["images"] + 1))
    observed_classes = {int(value) for value in labels}
    intersections = {
        "train_val": sorted(split_ids["train"] & split_ids["val"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "val_test": sorted(split_ids["val"] & split_ids["test"]),
    }
    checks = {
        "image_count": len(images) == expected["images"],
        "mask_count": len(masks) == expected["masks"],
        "paired_ids": set(images) == set(masks),
        "canonical_image_ids": set(images) == expected_ids,
        "canonical_mask_ids": set(masks) == expected_ids,
        "train_count": len(split_ids["train"]) == expected["train_ids"],
        "val_count": len(split_ids["val"]) == expected["val_ids"],
        "test_count": len(split_ids["test"]) == expected["test_ids"],
        "split_disjoint": not any(intersections.values()),
        "split_covers_images": all_split_ids == set(images),
        "split_ids_are_valid": all_split_ids <= expected_ids,
        "label_count": len(labels) == expected["images"],
        "class_count": len(observed_classes) == expected["classes"],
        "canonical_class_ids": observed_classes == set(range(1, expected["classes"] + 1)),
        "asset_files": all(record["status"] == "pass" for record in asset_records),
    }

    size_mismatches: list[int] = []
    decode_failures: list[dict[str, Any]] = []
    foreground_fractions: dict[int, float] = {}
    for image_id in sorted(set(images) & set(masks)):
        try:
            with Image.open(images[image_id]) as image, Image.open(masks[image_id]) as mask:
                if image.size != mask.size:
                    size_mismatches.append(image_id)
                    continue
                source_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                mask_array = np.asarray(mask.convert("RGB"), dtype=np.uint8)
                foreground_fractions[image_id] = float(
                    (recover_foreground_alpha(source_array, mask_array) >= 0.5).mean()
                )
        except (OSError, ValueError) as error:
            decode_failures.append({"image_id": image_id, "error": str(error)})
    mask_quality = _mask_quality_summary(foreground_fractions)
    checks["all_image_mask_sizes_match"] = not size_mismatches
    checks["all_images_and_masks_decode"] = not decode_failures
    checks["all_masks_have_foreground"] = not mask_quality["empty_ids"]
    checks["all_masks_have_background"] = not mask_quality["near_full_ids"]

    return {
        "schema_version": 1,
        "audit": "flowers102_official_data_contract",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "counts": {
            "images": len(images),
            "masks": len(masks),
            "labels": len(labels),
            "classes": len(set(int(value) for value in labels)),
            "train": len(split_ids["train"]),
            "val": len(split_ids["val"]),
            "test": len(split_ids["test"]),
        },
        "checks": checks,
        "assets": asset_records,
        "split_intersections": intersections,
        "missing_mask_ids": sorted(set(images) - set(masks)),
        "missing_image_ids": sorted(set(masks) - set(images)),
        "size_mismatch_ids": size_mismatches,
        "decode_failures": decode_failures,
        "mask_quality": mask_quality,
        "mask_profile": _mask_profile(images, masks, mask_samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mask-samples", type=int, default=32)
    args = parser.parse_args(argv)
    report = audit_dataset(args.data_root, args.manifest, args.mask_samples)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
