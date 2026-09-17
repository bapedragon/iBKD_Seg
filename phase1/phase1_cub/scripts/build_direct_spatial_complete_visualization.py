#!/usr/bin/env python3
"""Build complete, audited visual evidence for the CUB direct-spatial report.

The script does not train or select any model.  It reloads the frozen seed-1
classification encoders and all five validation-selected part-probe heads, runs
inference on the eight image ids fixed before evaluation, and renders:

* part GT/prediction overlays for PCK@0.1 and normalized location error;
* a three-encoder-seed block-11 CKA heatmap.

Attention visualizations are built separately from the H200-produced attention
rollout PNGs by ``build_direct_spatial_attention_comparison.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ibkd_seg.phase1.cub_data import ARCHIVE_SHA256, read_records
from ibkd_seg.phase1.cub_direct_spatial import (
    GRID_SIZE,
    INPUT_SIZE,
    PART_COUNT,
    build_part_probe,
    load_spatial_annotations,
)
from ibkd_seg.phase1.data import IMAGENET_MEAN, IMAGENET_STD
from ibkd_seg.phase1.models import create_student, forward_student_spatial
from ibkd_seg.phase1.train_timing import file_sha256, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/reports/direct_spatial/"
    "resnet50_224_b128_guided_3seed_v2"
)
SEED1_REPORT_DIR = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/reports/direct_spatial/"
    "resnet50_224_b128_seed1_v2"
)
SEED1_ENCODER_REPORT_DIR = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/reports/frozen_probe/"
    "resnet50_224_b128_b64_guided_seed1_v4"
)
OUTPUT_DIR = REPORT_DIR / "qualitative_comparisons"
FIXED_IMAGE_ID_GROUPS = (
    (787, 2285, 3735, 5205),
    (6691, 8139, 9597, 11064),
)
METHODS = (
    ("LG", "lg"),
    ("ALG-w20", "alg_warmup20"),
    ("iBKD 0.25", "ibkd_lambda_0.25"),
    ("iBKD 0.5", "ibkd_lambda_0.5"),
)
PANEL_SIZE = 300
METHOD_PANEL_FOOTER = 42
GT_COLOR = (0, 230, 255, 255)
CORRECT_COLOR = (35, 210, 95, 215)
INCORRECT_COLOR = (255, 72, 88, 205)
BBOX_COLOR = (255, 190, 35, 255)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] | str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=font,
        fill=fill,
    )


def _records_for_fixed_ids(dataset_root: Path) -> list[Any]:
    wanted = {image_id for group in FIXED_IMAGE_ID_GROUPS for image_id in group}
    records = {record.image_id: record for record in read_records(dataset_root)}
    missing = wanted - records.keys()
    if missing:
        raise RuntimeError(f"fixed CUB image ids are missing: {sorted(missing)}")
    selected = [records[image_id] for image_id in sorted(wanted)]
    if any(record.is_train for record in selected):
        raise RuntimeError("all fixed qualitative image ids must belong to official test")
    for record in selected:
        if not (dataset_root / "images" / record.relative_path).is_file():
            raise FileNotFoundError(f"missing fixed CUB image: {record.relative_path}")
    return selected


def _load_image_batch(dataset_root: Path, records: Sequence[Any]) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for record in records:
        with Image.open(dataset_root / "images" / record.relative_path) as handle:
            image = handle.convert("RGB")
            image = TF.resize(
                image,
                [INPUT_SIZE, INPUT_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        tensors.append(TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD))
    return torch.stack(tensors)


def _validate_checkpoint(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"checkpoint byte count changed: {path}")
    if file_sha256(path) != entry["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint SHA-256 changed: {path}")


def _student_entries() -> dict[str, dict[str, Any]]:
    audit = _json(SEED1_ENCODER_REPORT_DIR / "checkpoint_audit.json")
    entries = {
        entry["variant"]: entry
        for entry in audit["entries"]
        if entry.get("kind") == "classification_encoder"
        and entry.get("batch_size") == 128
        and entry.get("encoder_seed") == 1
    }
    if set(entries) != {variant for _label, variant in METHODS}:
        raise RuntimeError("seed-1 batch-128 encoder audit is incomplete")
    return entries


def _probe_entries() -> dict[str, list[dict[str, Any]]]:
    audit = _json(SEED1_REPORT_DIR / "checkpoint_audit.json")
    grouped: dict[str, list[dict[str, Any]]] = {
        variant: [] for _label, variant in METHODS
    }
    for entry in audit["part_probes"]["entries"]:
        if entry.get("encoder_seed") == 1 and entry["variant"] in grouped:
            grouped[entry["variant"]].append(entry)
    for variant, entries in grouped.items():
        entries.sort(key=lambda entry: int(entry["probe_seed"]))
        if [int(entry["probe_seed"]) for entry in entries] != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"part-probe seed audit is incomplete: {variant}")
    return grouped


@torch.inference_mode()
def _predict_parts(
    *,
    student_root: Path,
    probe_root: Path,
    dataset_root: Path,
    records: Sequence[Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    images = _load_image_batch(dataset_root, records)
    student_entries = _student_entries()
    probe_entries = _probe_entries()
    predictions: dict[str, torch.Tensor] = {}
    sources: dict[str, Any] = {}
    for _label, variant in METHODS:
        student_entry = student_entries[variant]
        student_path = student_root / student_entry["path_under_ignored_raw_root"]
        _validate_checkpoint(student_path, student_entry)
        student_payload = torch.load(student_path, map_location="cpu", weights_only=True)
        student = create_student(num_classes=200, drop_path_rate=0.1)
        incompatible = student.load_state_dict(student_payload["student"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"student strict load failed: {variant}")
        if state_dict_sha256(student) != student_entry["model_state_sha256"]:
            raise RuntimeError(f"student state hash changed: {variant}")
        student.eval().requires_grad_(False)
        features, _ = forward_student_spatial(student, images)
        last = features[-1].cpu()
        if last.shape != (len(records), 192, GRID_SIZE, GRID_SIZE):
            raise RuntimeError(f"unexpected block-11 feature shape: {tuple(last.shape)}")

        one_variant: list[torch.Tensor] = []
        probe_sources: list[dict[str, Any]] = []
        for entry in probe_entries[variant]:
            probe_path = probe_root / entry["path_under_ignored_raw_root"]
            _validate_checkpoint(probe_path, entry)
            payload = torch.load(probe_path, map_location="cpu", weights_only=True)
            metadata = payload.get("metadata", {})
            if (
                metadata.get("variant") != variant
                or metadata.get("encoder_seed") != 1
                or metadata.get("probe_seed") != entry["probe_seed"]
                or metadata.get("selection_metric") != "validation_micro_PCK_at_0.1"
                or metadata.get("official_test_evaluations_at_checkpoint_write") != 0
            ):
                raise RuntimeError(f"part-probe metadata changed: {probe_path}")
            probe = build_part_probe(int(entry["probe_seed"]))
            incompatible = probe.load_state_dict(payload["probe"], strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"part-probe strict load failed: {probe_path}")
            if state_dict_sha256(probe) != entry["model_state_sha256"]:
                raise RuntimeError(f"part-probe state hash changed: {probe_path}")
            probe.eval().requires_grad_(False)
            logits = probe(last)
            indices = logits.flatten(2).argmax(dim=-1)
            rows = torch.div(indices, GRID_SIZE, rounding_mode="floor")
            columns = indices.remainder(GRID_SIZE)
            one_variant.append(torch.stack((columns, rows), dim=-1).cpu())
            probe_sources.append(
                {
                    "probe_seed": int(entry["probe_seed"]),
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                }
            )
        predictions[variant] = torch.stack(one_variant)
        sources[variant] = {
            "student_checkpoint_sha256": student_entry["checkpoint_sha256"],
            "student_model_state_sha256": student_entry["model_state_sha256"],
            "part_probes": probe_sources,
        }
        del student, student_payload, features, last
    return predictions, sources


def _geometry(
    *,
    dataset_root: Path,
    records: Sequence[Any],
    predictions: dict[str, torch.Tensor],
) -> tuple[tuple[str, ...], dict[int, dict[str, Any]]]:
    part_names, annotations = load_spatial_annotations(dataset_root)
    values: dict[int, dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        annotation = annotations[record.image_id]
        image_path = dataset_root / "images" / record.relative_path
        with Image.open(image_path) as handle:
            width, height = handle.size
        points = torch.tensor(annotation.points, dtype=torch.float32)
        official_visible = torch.tensor(annotation.visible, dtype=torch.bool)
        in_image = (
            points[:, 0].ge(0.0)
            & points[:, 0].le(float(width))
            & points[:, 1].ge(0.0)
            & points[:, 1].le(float(height))
        )
        visible = official_visible & in_image
        box = torch.tensor(annotation.bounding_box, dtype=torch.float32)
        normalizer = float(torch.maximum(box[2], box[3]).item())
        if not visible.any() or normalizer <= 0:
            raise RuntimeError(f"invalid fixed-image part geometry: {record.image_id}")
        row: dict[str, Any] = {
            "relative_path": record.relative_path,
            "image_size": [width, height],
            "bounding_box": list(annotation.bounding_box),
            "valid_part_ids": [
                index + 1 for index, value in enumerate(visible.tolist()) if value
            ],
            "gt_points_original_xy": [list(point) for point in annotation.points],
            "methods": {},
        }
        for _label, variant in METHODS:
            grid = predictions[variant][:, record_index].to(torch.float32)
            predicted_x = (grid[:, :, 0] + 0.5) * width / GRID_SIZE
            predicted_y = (grid[:, :, 1] + 0.5) * height / GRID_SIZE
            predicted = torch.stack((predicted_x, predicted_y), dim=-1)
            distances = torch.linalg.vector_norm(predicted - points.unsqueeze(0), dim=-1)
            normalized_error = distances / normalizer
            expanded_visible = visible.unsqueeze(0).expand_as(normalized_error)
            correct = normalized_error.le(0.1) & expanded_visible
            valid_count = int(expanded_visible.sum().item())
            row["methods"][variant] = {
                "probe_seeds": [1, 2, 3, 4, 5],
                "predicted_points_original_xy": predicted.tolist(),
                "correct_at_0.1": correct.tolist(),
                "five_probe_seed_micro_pck_at_0.1": float(
                    correct.sum().item() / valid_count
                ),
                "five_probe_seed_mean_normalized_location_error": float(
                    normalized_error[expanded_visible].mean().item()
                ),
                "evaluated_keypoints": valid_count,
            }
        values[record.image_id] = row
    return part_names, values


def _display_point(point: Sequence[float], image_size: Sequence[int]) -> tuple[float, float]:
    return (
        float(point[0]) * PANEL_SIZE / float(image_size[0]),
        float(point[1]) * PANEL_SIZE / float(image_size[1]),
    )


def _draw_gt_marker(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    part_id: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x, y = point
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GT_COLOR, outline=(0, 0, 0, 255), width=2)
    label = str(part_id)
    bounds = draw.textbbox((0, 0), label, font=font, stroke_width=2)
    draw.text(
        (x + 6, y - (bounds[3] - bounds[1]) / 2),
        label,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    color: tuple[int, int, int, int],
) -> None:
    x, y = point
    radius = 5
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=2)


def _part_panel(
    image: Image.Image,
    *,
    geometry: dict[str, Any],
    method: str | None,
) -> Image.Image:
    resized = image.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BILINEAR)
    if method is not None:
        resized = ImageEnhance.Brightness(resized).enhance(0.72)
    panel = Image.new("RGBA", (PANEL_SIZE, PANEL_SIZE + METHOD_PANEL_FOOTER), (248, 249, 251, 255))
    panel.alpha_composite(resized.convert("RGBA"), (0, 0))
    overlay = Image.new("RGBA", (PANEL_SIZE, PANEL_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    part_font = _font(14, bold=True)
    image_size = geometry["image_size"]
    valid_ids = set(geometry["valid_part_ids"])
    if method is None:
        x, y, width, height = geometry["bounding_box"]
        x0, y0 = _display_point((x, y), image_size)
        x1, y1 = _display_point((x + width, y + height), image_size)
        draw.rectangle((x0, y0, x1, y1), outline=BBOX_COLOR, width=3)
    for part_id, gt in enumerate(geometry["gt_points_original_xy"], start=1):
        if part_id not in valid_ids:
            continue
        gt_display = _display_point(gt, image_size)
        _draw_gt_marker(draw, gt_display, part_id, part_font)
        if method is None:
            continue
        method_data = geometry["methods"][method]
        for probe_index, predicted_set in enumerate(
            method_data["predicted_points_original_xy"]
        ):
            predicted = _display_point(predicted_set[part_id - 1], image_size)
            correct = method_data["correct_at_0.1"][probe_index][part_id - 1]
            color = CORRECT_COLOR if correct else INCORRECT_COLOR
            draw.line((*gt_display, *predicted), fill=(*color[:3], 55), width=1)
            _draw_cross(draw, predicted, color)
    panel.alpha_composite(overlay, (0, 0))
    footer = ImageDraw.Draw(panel)
    footer_font = _font(16, bold=method is not None)
    if method is None:
        footer_text = "GT parts (cyan) + bbox (yellow)"
    else:
        result = geometry["methods"][method]
        footer_text = (
            f"5-probe PCK {100 * result['five_probe_seed_micro_pck_at_0.1']:.1f}%"
            f"  |  norm. err {result['five_probe_seed_mean_normalized_location_error']:.3f}"
        )
    _center_text(
        footer,
        footer_text,
        (0, PANEL_SIZE, PANEL_SIZE, PANEL_SIZE + METHOD_PANEL_FOOTER),
        footer_font,
        (30, 35, 43, 255),
    )
    return panel.convert("RGB")


def _build_part_grid(
    *,
    dataset_root: Path,
    records_by_id: dict[int, Any],
    geometry_by_id: dict[int, dict[str, Any]],
    part_names: Sequence[str],
    image_ids: Sequence[int],
    output_path: Path,
) -> dict[str, Any]:
    columns = ("GT + bbox", *(label for label, _variant in METHODS))
    margin_left = 110
    header_height = 58
    row_gap = 8
    column_gap = 7
    legend_height = 104
    panel_height = PANEL_SIZE + METHOD_PANEL_FOOTER
    width = margin_left + len(columns) * PANEL_SIZE + (len(columns) - 1) * column_gap
    height = header_height + len(image_ids) * panel_height + (len(image_ids) - 1) * row_gap + legend_height
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    header_font = _font(22, bold=True)
    row_font = _font(18, bold=True)
    legend_font = _font(15)
    for column_index, label in enumerate(columns):
        left = margin_left + column_index * (PANEL_SIZE + column_gap)
        _center_text(draw, label, (left, 0, left + PANEL_SIZE, header_height), header_font, (25, 30, 38))

    output_rows: list[dict[str, Any]] = []
    for row_index, image_id in enumerate(image_ids):
        top = header_height + row_index * (panel_height + row_gap)
        record = records_by_id[image_id]
        with Image.open(dataset_root / "images" / record.relative_path) as handle:
            image = handle.convert("RGB")
        _center_text(draw, f"ID {image_id}", (0, top, margin_left - 6, top + panel_height), row_font, (25, 30, 38))
        panels = [_part_panel(image, geometry=geometry_by_id[image_id], method=None)]
        panels.extend(
            _part_panel(image, geometry=geometry_by_id[image_id], method=variant)
            for _label, variant in METHODS
        )
        for column_index, panel in enumerate(panels):
            left = margin_left + column_index * (PANEL_SIZE + column_gap)
            canvas.paste(panel, (left, top))
            draw.rectangle((left, top, left + PANEL_SIZE - 1, top + panel_height - 1), outline=(200, 205, 213), width=1)
        geometry = geometry_by_id[image_id]
        output_rows.append(
            {
                "image_id": image_id,
                "relative_path": geometry["relative_path"],
                "valid_part_ids": geometry["valid_part_ids"],
                "methods": {
                    variant: {
                        "five_probe_seed_micro_pck_at_0.1": geometry["methods"][variant][
                            "five_probe_seed_micro_pck_at_0.1"
                        ],
                        "five_probe_seed_mean_normalized_location_error": geometry[
                            "methods"
                        ][variant]["five_probe_seed_mean_normalized_location_error"],
                        "evaluated_keypoints": geometry["methods"][variant][
                            "evaluated_keypoints"
                        ],
                    }
                    for _label, variant in METHODS
                },
            }
        )

    legend_top = height - legend_height
    draw.line((margin_left, legend_top + 8, width, legend_top + 8), fill=(200, 205, 213), width=1)
    legend_1 = (
        "Cyan circle/number: GT part  |  green x: PCK@0.1 correct  |  red x: incorrect  |  "
        "each method overlays all five validation-selected probe seeds (coincident x marks overlap)."
    )
    legend_2 = (
        "Per-image footer averages every valid part across the five probe seeds. "
        "PCK threshold and normalized error use 0.1 x max(GT bbox width, height)."
    )
    draw.text((margin_left, legend_top + 19), legend_1, font=legend_font, fill=(50, 56, 66))
    draw.text((margin_left, legend_top + 43), legend_2, font=legend_font, fill=(50, 56, 66))
    part_legend = "  ".join(f"{index}: {name}" for index, name in enumerate(part_names, 1))
    midpoint = len(part_legend) // 2
    split = part_legend.rfind("  ", 0, midpoint)
    draw.text((margin_left, legend_top + 67), part_legend[:split], font=legend_font, fill=(50, 56, 66))
    draw.text((margin_left, legend_top + 86), part_legend[split + 2 :], font=legend_font, fill=(50, 56, 66))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(output_path),
        "width": width,
        "height": height,
        "image_ids": list(image_ids),
        "rows": output_rows,
    }


def _build_cka_heatmap(output_path: Path) -> dict[str, Any]:
    csv_path = REPORT_DIR / "spatial_cka_block11_per_encoder_seed.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        (row["variant"], int(row["encoder_seed"])): float(row["centered_linear_cka"])
        for row in rows
    }
    labels = {variant: label for label, variant in METHODS}
    margin_left = 190
    margin_top = 98
    cell_width = 190
    cell_height = 74
    columns = ("encoder seed 1", "encoder seed 2", "encoder seed 3", "mean +/- sample SD")
    width = margin_left + len(columns) * cell_width + 24
    height = margin_top + len(METHODS) * cell_height + 74
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25, bold=True)
    label_font = _font(20, bold=True)
    value_font = _font(20, bold=True)
    note_font = _font(16)
    draw.text((20, 16), "Spatial CKA block11 (higher is more teacher-like)", font=title_font, fill=(24, 29, 37))
    for column_index, label in enumerate(columns):
        left = margin_left + column_index * cell_width
        _center_text(draw, label, (left, 55, left + cell_width, margin_top), label_font, (35, 41, 50))

    table_rows: list[dict[str, Any]] = []
    for row_index, (_unused, variant) in enumerate(METHODS):
        top = margin_top + row_index * cell_height
        _center_text(draw, labels[variant], (8, top, margin_left - 8, top + cell_height), label_font, (35, 41, 50))
        values = [lookup[(variant, seed)] for seed in (1, 2, 3)]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        texts = [*(f"{value:.4f}" for value in values), f"{mean:.4f} +/- {sd:.4f}"]
        for column_index, (value, text_value) in enumerate(zip((*values, mean), texts, strict=True)):
            left = margin_left + column_index * cell_width
            # The observed block-11 values occupy roughly 0.20--0.45.  Use that
            # fixed range for a readable comparison without implying that the
            # colors themselves are a separate metric.
            intensity = max(0.0, min(1.0, (value - 0.20) / 0.25))
            color = (
                int(242 - 126 * intensity),
                int(247 - 91 * intensity),
                int(253 - 33 * intensity),
            )
            draw.rectangle((left, top, left + cell_width - 2, top + cell_height - 2), fill=color, outline=(190, 198, 208), width=1)
            _center_text(draw, text_value, (left, top, left + cell_width - 2, top + cell_height - 2), value_font, (18, 31, 48))
        table_rows.append({"variant": variant, "encoder_seed_values": values, "mean": mean, "sample_sd": sd})

    note_top = margin_top + len(METHODS) * cell_height + 19
    draw.text(
        (20, note_top),
        "Fixed 600-image validation split; student block11 vs. ResNet-50 layer3. Color scale: 0.20--0.45. CKA is not a per-image mask.",
        font=note_font,
        fill=(60, 67, 78),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path.relative_to(REPOSITORY_ROOT)),
        "sha256": file_sha256(output_path),
        "source_csv": str(csv_path.relative_to(REPOSITORY_ROOT)),
        "source_csv_sha256": file_sha256(csv_path),
        "rows": table_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--student-release-root", type=Path, required=True)
    parser.add_argument("--part-probe-release-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    student_root = args.student_release_root.resolve()
    probe_root = args.part_probe_release_root.resolve()
    records = _records_for_fixed_ids(dataset_root)
    records_by_id = {record.image_id: record for record in records}
    predictions, checkpoint_sources = _predict_parts(
        student_root=student_root,
        probe_root=probe_root,
        dataset_root=dataset_root,
        records=records,
    )
    part_names, geometry_by_id = _geometry(
        dataset_root=dataset_root,
        records=records,
        predictions=predictions,
    )

    part_outputs = []
    for group_index, image_ids in enumerate(FIXED_IMAGE_ID_GROUPS, start=1):
        part_outputs.append(
            _build_part_grid(
                dataset_root=dataset_root,
                records_by_id=records_by_id,
                geometry_by_id=geometry_by_id,
                part_names=part_names,
                image_ids=image_ids,
                output_path=OUTPUT_DIR / f"part_localization_seed1_group{group_index}.png",
            )
        )
    cka_output = _build_cka_heatmap(OUTPUT_DIR / "cka_block11_three_seed_heatmap.png")
    attention_manifest = OUTPUT_DIR / "manifest.json"
    manifest = {
        "schema_version": 1,
        "role": "complete_direct_spatial_visual_evidence",
        "selection_policy": (
            "all eight official-test image ids fixed before issue 737 evaluation; "
            "no post-hoc qualitative example selection"
        ),
        "fixed_image_ids": [image_id for group in FIXED_IMAGE_ID_GROUPS for image_id in group],
        "dataset": {
            "name": "CUB-200-2011",
            "official_archive_sha256": ARCHIVE_SHA256,
            "part_names": list(part_names),
        },
        "part_localization": {
            "encoder_seed": 1,
            "probe_seeds": [1, 2, 3, 4, 5],
            "probe_selection": "validation_micro_PCK_at_0.1",
            "display_aggregation": (
                "all five selected probe predictions are drawn; per-image footer pools "
                "all valid keypoints across the same five probe seeds"
            ),
            "supports_metrics": ["part_PCK_at_0.1", "mean_normalized_location_error"],
            "checkpoint_sources": checkpoint_sources,
            "outputs": part_outputs,
        },
        "spatial_cka": cka_output,
        "attention": {
            "supports_metrics": ["attention_patch_AP", "pointing", "foreground_attention_mass"],
            "manifest": str(attention_manifest.relative_to(REPOSITORY_ROOT)),
            "manifest_sha256": file_sha256(attention_manifest),
        },
        "metric_to_visual": {
            "Part PCK@0.1": "part_localization outputs (green/red prediction crosses)",
            "normalized location error": "part_localization outputs (GT-to-prediction distances and footer)",
            "CKA block11": "spatial_cka output",
            "Attention AP": "attention comparison outputs (red rollout over green GT)",
            "Pointing": "attention comparison outputs (rollout peak location vs. GT)",
            "FG mass": "attention comparison outputs (attention mass inside GT foreground)",
        },
        "not_training": True,
    }
    manifest_path = OUTPUT_DIR / "complete_visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    for output in part_outputs:
        print(f"wrote {output['path']}")
    print(f"wrote {cka_output['path']}")


if __name__ == "__main__":
    main()
