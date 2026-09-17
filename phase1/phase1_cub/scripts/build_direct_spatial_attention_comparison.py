#!/usr/bin/env python3
"""Build deterministic method-comparison grids from audited attention PNGs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/reports/direct_spatial/"
    "resnet50_224_b128_seed1_v2/attention_qualitative"
)
OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/reports/direct_spatial/"
    "resnet50_224_b128_guided_3seed_v2/qualitative_comparisons"
)

PANEL_SIZE = 224
IMAGE_ID_GROUPS = (
    (787, 2285, 3735, 5205),
    (6691, 8139, 9597, 11064),
)
METHODS = (
    ("LG", "lg"),
    ("ALG-w20", "alg_warmup20"),
    ("iBKD 0.25", "ibkd_lambda_0.25"),
    ("iBKD 0.5", "ibkd_lambda_0.5"),
)
COLUMNS = ("Original", "GT foreground", *(label for label, _ in METHODS))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _source_path(method: str, image_id: int) -> Path:
    return SOURCE_DIR / f"{method}_encoder_seed1_image{image_id}.png"


def _load_triptych(method: str, image_id: int) -> Image.Image:
    path = _source_path(method, image_id)
    image = Image.open(path).convert("RGB")
    expected = (PANEL_SIZE * 3, PANEL_SIZE)
    if image.size != expected:
        raise RuntimeError(f"unexpected source size for {path}: {image.size} != {expected}")
    return image


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
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


def build_grid(image_ids: tuple[int, ...], output_path: Path) -> dict[str, object]:
    margin_left = 125
    header_height = 58
    footer_height = 34
    gap = 5
    row_gap = 8
    width = margin_left + len(COLUMNS) * PANEL_SIZE + (len(COLUMNS) - 1) * gap
    height = (
        header_height
        + len(image_ids) * PANEL_SIZE
        + (len(image_ids) - 1) * row_gap
        + footer_height
    )
    canvas = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    header_font = _font(22, bold=True)
    row_font = _font(19, bold=True)
    footer_font = _font(16)

    for column_index, label in enumerate(COLUMNS):
        left = margin_left + column_index * (PANEL_SIZE + gap)
        _draw_centered(
            draw,
            label,
            (left, 0, left + PANEL_SIZE, header_height),
            header_font,
            (25, 30, 38),
        )

    source_files: list[dict[str, object]] = []
    for row_index, image_id in enumerate(image_ids):
        top = header_height + row_index * (PANEL_SIZE + row_gap)
        _draw_centered(
            draw,
            f"ID {image_id}",
            (0, top, margin_left - 8, top + PANEL_SIZE),
            row_font,
            (25, 30, 38),
        )

        reference = _load_triptych("lg", image_id)
        panels = [
            reference.crop((0, 0, PANEL_SIZE, PANEL_SIZE)),
            reference.crop((PANEL_SIZE, 0, PANEL_SIZE * 2, PANEL_SIZE)),
        ]
        for _, method in METHODS:
            source_path = _source_path(method, image_id)
            source = _load_triptych(method, image_id)
            panels.append(
                source.crop((PANEL_SIZE * 2, 0, PANEL_SIZE * 3, PANEL_SIZE))
            )
            source_files.append(
                {
                    "path": str(source_path.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(source_path),
                }
            )

        for column_index, panel in enumerate(panels):
            left = margin_left + column_index * (PANEL_SIZE + gap)
            canvas.paste(panel, (left, top))
            draw.rectangle(
                (left, top, left + PANEL_SIZE - 1, top + PANEL_SIZE - 1),
                outline=(205, 209, 216),
                width=1,
            )

    footer_top = height - footer_height
    _draw_centered(
        draw,
        "Green: GT foreground overlay    Red: CLS-to-patch attention rollout",
        (margin_left, footer_top, width, height),
        footer_font,
        (70, 76, 86),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return {
        "path": str(output_path.relative_to(REPOSITORY_ROOT)),
        "sha256": _sha256(output_path),
        "width": width,
        "height": height,
        "image_ids": list(image_ids),
        "source_files": source_files,
    }


def main() -> None:
    outputs = []
    for group_index, image_ids in enumerate(IMAGE_ID_GROUPS, start=1):
        outputs.append(
            build_grid(
                image_ids,
                OUTPUT_DIR / f"attention_method_comparison_seed1_group{group_index}.png",
            )
        )
    manifest = {
        "schema_version": 1,
        "role": "derived_qualitative_attention_comparison",
        "source_h200_issue": 737,
        "encoder_seed": 1,
        "selection_policy": "all eight image ids fixed before the direct-spatial evaluation; no post-hoc example selection",
        "columns": list(COLUMNS),
        "interpretation": {
            "green": "ground-truth foreground mask overlay",
            "red": "CLS-to-patch attention rollout overlay",
            "supports_metrics": ["attention_patch_AP", "pointing", "foreground_attention_mass"],
            "does_not_visualize": ["part_PCK", "normalized_location_error", "spatial_CKA"],
            "quantitative_result_source": "three encoder seeds; these grids show encoder seed 1 only",
        },
        "outputs": outputs,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    for output in outputs:
        print(f"wrote {output['path']}")


if __name__ == "__main__":
    main()
