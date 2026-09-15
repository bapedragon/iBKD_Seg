"""Shared validation and metric routines for CUB direct-spatial full runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .cub_data import DATASET_NAME, NUM_CLASSES
from .cub_direct_spatial import (
    GRID_SIZE,
    LinearCKAAccumulator,
    assert_probability,
    attention_gt_metrics,
    attention_rollout,
    feature_map_to_observations,
    load_mask_views,
    save_attention_triptych,
)
from .cub_probe_data import CubImageDataset, CubProbeRecord
from .models import create_student, forward_student_spatial
from .train_timing import file_sha256, state_dict_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_CONFIG_SHA256 = "55ac0598c11a4065f3b1416022e8fbb3de35b21cad94780ace2e2036d5430bc6"

SEED23_CONFIG_SHA256 = "bd71b02ebcca3240c7278819b4a34416a131914bd442887afcce6251a7e366d1"

EXPECTED_VALIDATION_SHA256 = "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"

EXPECTED_VARIANTS = (
    "lg",
    "alg_warmup20",
    "ibkd_lambda_0.25",
    "ibkd_lambda_0.5",
)

EXPECTED_METHODS: dict[str, tuple[str, float | None, int]] = {
    "lg": ("lg", None, 0),
    "alg_warmup20": ("alg", None, 20),
    "ibkd_lambda_0.25": ("ibkd", 0.25, 20),
    "ibkd_lambda_0.5": ("ibkd", 0.5, 20),
}

def log(message: str = "") -> None:
    print(message, flush=True)

def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload

def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path

def _validate_seed1_config(config: dict[str, Any], path: Path) -> None:
    scope = config.get("comparison_scope", {})
    dataset = config.get("dataset", {})
    encoder = config.get("encoder", {})
    part = config.get("part_localization_probe", {})
    cka = config.get("spatial_cka", {})
    attention = config.get("attention_gt_localization", {})
    smoke = config.get("smoke", {})
    execution = config.get("execution", {})
    gate = config.get("completion_gate", {})
    inputs = config.get("checkpoint_inputs", [])
    checks = {
        "config_hash": file_sha256(path) == EXPECTED_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_direct_spatial_smoke_v1",
        "locked_before_results": config.get("status")
        == "locked_before_direct_spatial_smoke_results_2026-09-09",
        "non_scientific": config.get("scientific_result") is False,
        "scope": scope.get("primary_student_batch_size") == 128
        and tuple(scope.get("variants", ())) == EXPECTED_VARIANTS
        and scope.get("eventual_encoder_seeds") == [1, 2, 3]
        and scope.get("smoke_encoder_seeds") == [1]
        and scope.get("batch64_included") is False
        and scope.get("method_or_lambda_selection_from_smoke") is False,
        "dataset": dataset.get("name") == DATASET_NAME
        and dataset.get("num_classes") == NUM_CLASSES
        and dataset.get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_SHA256,
        },
        "encoder": encoder.get("architecture") == "deit_tiny_patch16_224"
        and encoder.get("strict_load") is True
        and encoder.get("frozen") is True
        and encoder.get("eval_mode") is True
        and encoder.get("patch_grid") == [14, 14]
        and encoder.get("patch_channels") == 192,
        "part": part.get("role") == "primary_direct_spatial_metric"
        and part.get("feature", {}).get("student_block") == 11
        and part.get("head", {}).get("architecture") == "Conv2d(192,15,1,bias=True)"
        and part.get("head", {}).get("parameter_count") == 2895
        and part.get("target", {}).get("sigma_grid_pixels") == 1.0
        and part.get("loss") == "visible_part_masked_mean_squared_error"
        and part.get("learning_rates") == [0.01, 0.03, 0.1]
        and part.get("epochs") == 100
        and part.get("batch_size") == 64
        and part.get("probe_seeds") == [1, 2, 3, 4, 5]
        and part.get("selection", {}).get("split") == "validation",
        "cka": cka.get("split") == "fixed_validation_600"
        and cka.get("student_features")
        == "pre_final_norm_patch_tokens_blocks_0_through_11"
        and cka.get("teacher", {}).get("feature") == "layer3_output"
        and cka.get("observation_unit")
        == "aligned_image_and_14x14_spatial_position"
        and cka.get("metric") == "centered_linear_CKA"
        and cka.get("accumulator_dtype") == "float64"
        and cka.get("official_test_used") is False,
        "attention": attention.get("attention_rollout", {}).get("layers") == "all_12"
        and attention.get("attention_rollout", {}).get("head_fusion")
        == "arithmetic_mean"
        and attention.get("primary_metric")
        == "global_micro_patch_average_precision"
        and attention.get("no_threshold_tuning") is True,
        "smoke": smoke.get("official_test_accessed") is False
        and smoke.get("subset", {}).get("train_count") == 200
        and smoke.get("subset", {}).get("validation_count") == 200
        and smoke.get("part_probe")
        == {"probe_seeds": [1], "learning_rates": [0.01, 0.03, 0.1], "epochs": 2}
        and smoke.get("spatial_cka_validation_images") == 200
        and smoke.get("attention_validation_images") == 200
        and smoke.get("qualitative_examples_per_variant") == 4
        and smoke.get("all_metrics_non_scientific") is True,
        "inputs": len(inputs) == 4
        and tuple(item.get("variant") for item in inputs) == EXPECTED_VARIANTS
        and all(item.get("batch_size") == 128 and item.get("encoder_seed") == 1 for item in inputs),
        "execution": execution.get("requested_mig_slices") == 1
        and execution.get("feature_batch_size") == 16
        and execution.get("cka_batch_size") == 8
        and execution.get("attention_batch_size") == 16
        and execution.get("num_workers") == 4,
        "gate": gate
        == {
            "checkpoint_strict_loads": 4,
            "part_probe_lr_candidates": 12,
            "part_probe_validation_selections": 4,
            "spatial_cka_values": 48,
            "attention_metric_rows": 4,
            "qualitative_pngs": 16,
            "official_test_evaluations": 0,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid CUB direct-spatial smoke config: " + ", ".join(failures))

    provenance = config["protocol_provenance"]
    for source_name in ("base_v3", "seed1_encoder_release", "teacher_release"):
        source = provenance[source_name]
        source_path = _resolve_repository_path(
            source.get("path", source.get("manifest_path"))
        )
        digest_key = "sha256" if source_name == "base_v3" else "manifest_sha256"
        if not source_path.is_file() or file_sha256(source_path) != source[digest_key]:
            raise RuntimeError(f"direct-spatial provenance changed: {source_name}")

def _validate_seed23_config(config: dict[str, Any], path: Path) -> None:
    scope = config.get("comparison_scope", {})
    dataset = config.get("dataset", {})
    encoder = config.get("encoder", {})
    part = config.get("part_localization_probe", {})
    cka = config.get("spatial_cka", {})
    attention = config.get("attention_gt_localization", {})
    smoke = config.get("smoke", {})
    execution = config.get("execution", {})
    inputs = config.get("checkpoint_inputs", [])
    gate = config.get("completion_gate", {})
    checks = {
        "config_hash": file_sha256(path) == SEED23_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_r50_224_b128_seed2_3_direct_spatial_smoke_v2",
        "locked_before_results": config.get("status")
        == "locked_before_seed2_3_direct_spatial_smoke_results_2026-09-10",
        "non_scientific": config.get("scientific_result") is False,
        "scope": scope.get("primary_student_batch_size") == 128
        and tuple(scope.get("variants", ())) == EXPECTED_VARIANTS
        and scope.get("eventual_encoder_seeds") == [1, 2, 3]
        and scope.get("smoke_encoder_seeds") == [2, 3]
        and scope.get("batch64_included") is False
        and scope.get("method_lambda_or_protocol_selection_from_smoke") is False
        and scope.get("seed1_result_must_not_change_seed2_3_settings") is True,
        "dataset": dataset.get("name") == DATASET_NAME
        and dataset.get("num_classes") == NUM_CLASSES
        and dataset.get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_SHA256,
        }
        and dataset.get("annotations", {}).get("visibility_rule")
        == "official_visible_and_in_image_bounds"
        and dataset.get("annotations", {}).get("out_of_frame_visible_action")
        == "exclude_without_clipping",
        "encoder": encoder.get("architecture") == "deit_tiny_patch16_224"
        and encoder.get("checkpoint_purpose")
        == "phase1_cub_r50_224_seed_extension_full_student_v5"
        and encoder.get("checkpoint_protocol_config_sha256")
        == "f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9"
        and encoder.get("strict_load") is True
        and encoder.get("frozen") is True
        and encoder.get("eval_mode") is True
        and encoder.get("patch_grid") == [14, 14]
        and encoder.get("patch_channels") == 192,
        "part": part.get("role") == "primary_direct_spatial_metric"
        and part.get("feature", {}).get("student_block") == 11
        and part.get("head", {}).get("architecture") == "Conv2d(192,15,1,bias=True)"
        and part.get("head", {}).get("parameter_count") == 2895
        and part.get("target", {}).get("sigma_grid_pixels") == 1.0
        and part.get("target", {}).get("validity")
        == "official_visible_and_in_image_bounds"
        and part.get("target", {}).get("out_of_frame_visible_action")
        == "exclude_without_clipping"
        and part.get("loss") == "visible_part_masked_mean_squared_error"
        and part.get("learning_rates") == [0.01, 0.03, 0.1]
        and part.get("epochs") == 100
        and part.get("batch_size") == 64
        and part.get("probe_seeds") == [1, 2, 3, 4, 5]
        and part.get("selection", {}).get("split") == "validation",
        "cka": cka.get("split") == "fixed_validation_600"
        and cka.get("student_features")
        == "pre_final_norm_patch_tokens_blocks_0_through_11"
        and cka.get("metric") == "centered_linear_CKA"
        and cka.get("accumulator_dtype") == "float64"
        and cka.get("official_test_used") is False,
        "attention": attention.get("attention_rollout", {}).get("layers") == "all_12"
        and attention.get("attention_rollout", {}).get("head_fusion")
        == "arithmetic_mean"
        and attention.get("primary_metric")
        == "global_micro_patch_average_precision"
        and attention.get("smoke_split") == "fixed_validation_subset_only"
        and attention.get("no_threshold_tuning") is True,
        "smoke": smoke.get("official_test_accessed") is False
        and smoke.get("subset", {}).get("train_count") == 200
        and smoke.get("subset", {}).get("validation_count") == 200
        and smoke.get("part_probe")
        == {"probe_seeds": [1], "learning_rates": [0.01, 0.03, 0.1], "epochs": 2}
        and smoke.get("spatial_cka_validation_images") == 200
        and smoke.get("attention_validation_images") == 200
        and smoke.get("qualitative_examples_per_encoder") == 4
        and smoke.get("all_metrics_non_scientific") is True,
        "inputs": len(inputs) == 8
        and [(item.get("encoder_seed"), item.get("variant")) for item in inputs]
        == [(seed, variant) for seed in (2, 3) for variant in EXPECTED_VARIANTS]
        and all(item.get("batch_size") == 128 for item in inputs),
        "execution": execution.get("requested_mig_slices") == 1
        and execution.get("feature_batch_size") == 16
        and execution.get("cka_batch_size") == 8
        and execution.get("attention_batch_size") == 16
        and execution.get("num_workers") == 4,
        "gate": gate
        == {
            "checkpoint_strict_loads": 8,
            "part_probe_lr_candidates": 24,
            "part_probe_validation_selections": 8,
            "spatial_cka_values": 96,
            "attention_metric_rows": 8,
            "qualitative_pngs": 32,
            "official_test_evaluations": 0,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB seed-2/3 direct-spatial smoke config: "
            + ", ".join(failures)
        )

    provenance = config["protocol_provenance"]
    for source_name in (
        "direct_spatial_full_v2",
        "seed2_3_encoder_release",
        "teacher_release",
    ):
        source = provenance[source_name]
        source_path = _resolve_repository_path(
            source.get("path", source.get("manifest_path"))
        )
        digest_key = (
            "sha256" if source_name == "direct_spatial_full_v2" else "manifest_sha256"
        )
        if not source_path.is_file() or file_sha256(source_path) != source[digest_key]:
            raise RuntimeError(f"direct-spatial provenance changed: {source_name}")

def _validate_config(config: dict[str, Any], path: Path) -> None:
    digest = file_sha256(path)
    if digest == EXPECTED_CONFIG_SHA256:
        _validate_seed1_config(config, path)
    elif digest == SEED23_CONFIG_SHA256:
        _validate_seed23_config(config, path)
    else:
        raise RuntimeError(f"unsupported CUB direct-spatial smoke config hash: {digest}")

def _variant_metadata(variant: str) -> tuple[str, float | None, int]:
    try:
        return EXPECTED_METHODS[variant]
    except KeyError as error:
        raise RuntimeError(f"unexpected direct-spatial variant: {variant}") from error

def _load_student(
    root: Path,
    item: dict[str, Any],
    *,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = root / item["relative_path"]
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != item["bytes"]
        or file_sha256(checkpoint) != item["checkpoint_sha256"]
    ):
        raise RuntimeError(f"direct-spatial encoder byte contract failed: {item['variant']}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    method, fusion_ratio, warmup = _variant_metadata(item["variant"])
    encoder_contract = config["encoder"]
    expected = {
        "purpose": encoder_contract.get(
            "checkpoint_purpose",
            "phase1_cub_r50_224_batch_profile_full_student_v4",
        ),
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": method,
        "fusion_ratio_lambda": fusion_ratio,
        "batch_size": 128,
        "epochs": 300,
        "seed": int(item["encoder_seed"]),
        "guidance_controller_warmup_epochs": warmup,
        "validation_image_ids_sha256": EXPECTED_VALIDATION_SHA256,
        "protocol_config_sha256": encoder_contract.get(
            "checkpoint_protocol_config_sha256",
            "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633",
        ),
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    failures = [key for key, value in expected.items() if metadata.get(key) != value]
    if failures:
        raise RuntimeError(
            f"direct-spatial encoder metadata mismatch {item['variant']}: {','.join(failures)}"
        )
    model = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    state_hash = state_dict_sha256(model)
    floating_finite = all(
        bool(torch.isfinite(tensor).all())
        for tensor in payload["student"].values()
        if tensor.is_floating_point()
    )
    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
        or state_hash != item["model_state_sha256"]
        or state_hash != metadata.get("student_state_sha256")
        or not floating_finite
    ):
        raise RuntimeError(f"direct-spatial encoder state audit failed: {item['variant']}")
    model.to(device).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"direct-spatial encoder was not frozen: {item['variant']}")
    return model, {
        "variant": item["variant"],
        "encoder_seed": int(item["encoder_seed"]),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": item["checkpoint_sha256"],
        "model_state_sha256": state_hash,
        "strict_load": True,
        "all_floating_tensors_finite": True,
        "eval_mode": True,
        "trainable_parameters": 0,
    }

def _extract_last_features(
    model: torch.nn.Module,
    records: Sequence[CubProbeRecord],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        CubImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    values: list[torch.Tensor] = []
    for images, _image_ids in loader:
        features, _logits = forward_student_spatial(
            model, images.to(device, non_blocking=True)
        )
        last = features[-1]
        if last.shape[1:] != (192, GRID_SIZE, GRID_SIZE):
            raise RuntimeError(f"unexpected frozen student feature shape: {tuple(last.shape)}")
        values.append(last.detach().cpu())
    result = torch.cat(values)
    if result.shape != (len(records), 192, GRID_SIZE, GRID_SIZE):
        raise RuntimeError("frozen feature cache count or shape mismatch")
    return result

def _run_cka_analysis(
    *,
    variant: str,
    encoder_seed: int,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    records: Sequence[CubProbeRecord],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        CubImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    accumulators = [
        LinearCKAAccumulator(192, 1024, device=device, dtype=torch.float64)
        for _ in range(12)
    ]
    image_count = 0
    for images, _image_ids in loader:
        images = images.to(device, non_blocking=True)
        student_features, _ = forward_student_spatial(student, images)
        teacher_feature = teacher.forward_features(images)[1]
        if teacher_feature.shape[1:] != (1024, GRID_SIZE, GRID_SIZE):
            raise RuntimeError(f"unexpected teacher layer3 shape: {tuple(teacher_feature.shape)}")
        teacher_observations = feature_map_to_observations(teacher_feature)
        for accumulator, feature in zip(accumulators, student_features, strict=True):
            accumulator.update(feature_map_to_observations(feature), teacher_observations)
        image_count += images.shape[0]
    if image_count != len(records):
        raise RuntimeError("spatial CKA image count mismatch")
    rows = []
    for block, accumulator in enumerate(accumulators):
        value = accumulator.compute()
        assert_probability(value, name=f"{variant}/block{block}/linear_cka")
        rows.append(
            {
                "variant": variant,
                "encoder_seed": encoder_seed,
                "student_block": block,
                "teacher_feature": "resnet50_layer3",
                "images": image_count,
                "spatial_observations": image_count * GRID_SIZE * GRID_SIZE,
                "centered_linear_cka": value,
                "scientific_result": False,
            }
        )
    log(
        "[DIRECT_CKA_DONE] "
        f"variant={variant} encoder_seed={encoder_seed} values=12 "
        f"block11={rows[-1]['centered_linear_cka']:.6f}"
    )
    return rows

def _run_attention_analysis(
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
    rollout_values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    occupancies: list[torch.Tensor] = []
    offset = 0
    qualitative_ids = {
        record.image_id
        for record in sorted(records, key=lambda item: item.image_id)[:4]
    }
    for images, _image_ids in loader:
        batch_rollout = attention_rollout(student, images.to(device, non_blocking=True)).cpu()
        batch_records = records[offset : offset + len(batch_rollout)]
        for record, one_rollout in zip(batch_records, batch_rollout, strict=True):
            mask, occupancy = load_mask_views(record)
            masks.append(mask)
            occupancies.append(occupancy)
            if record.image_id in qualitative_ids:
                save_attention_triptych(
                    record=record,
                    rollout=one_rollout,
                    destination=(
                        output_dir
                        / "attention_gt"
                        / "qualitative"
                        / f"{variant}_encoder_seed{encoder_seed}_image{record.image_id}.png"
                    ),
                )
        rollout_values.append(batch_rollout)
        offset += len(batch_rollout)
    rollout = torch.cat(rollout_values)
    metrics = attention_gt_metrics(rollout, torch.stack(masks), torch.stack(occupancies))
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
        "qualitative_image_ids": sorted(qualitative_ids),
        "qualitative_png_count": len(qualitative_ids),
        "official_test_evaluations": 0,
        "scientific_result": False,
    }
    log(
        "[DIRECT_ATTENTION_DONE] "
        f"variant={variant} encoder_seed={encoder_seed} "
        f"patch_ap={metrics['global_micro_patch_average_precision']:.6f} "
        f"pointing={metrics['pointing_game_peak_inside_mask']:.6f}"
    )
    return result

def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def _save_cka_heatmap(rows: Sequence[dict[str, Any]], path: Path) -> None:
    lookup = {
        (int(row["encoder_seed"]), row["variant"], row["student_block"]): row[
            "centered_linear_cka"
        ]
        for row in rows
    }
    encoder_keys = sorted(
        {
            (int(row["encoder_seed"]), row["variant"])
            for row in rows
        },
        key=lambda value: (value[0], EXPECTED_VARIANTS.index(value[1])),
    )
    cell_width, cell_height = 54, 42
    margin_left, margin_top = 130, 34
    canvas = Image.new(
        "RGB",
        (margin_left + 12 * cell_width, margin_top + len(encoder_keys) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for block in range(12):
        draw.text((margin_left + block * cell_width + 18, 10), str(block), fill="black")
    labels = {
        "lg": "LG",
        "alg_warmup20": "ALG-w20",
        "ibkd_lambda_0.25": "iBKD-0.25",
        "ibkd_lambda_0.5": "iBKD-0.5",
    }
    show_encoder_seed = len({seed for seed, _variant in encoder_keys}) > 1
    for row_index, (encoder_seed, variant) in enumerate(encoder_keys):
        y = margin_top + row_index * cell_height
        label = (
            f"{labels[variant]} s{encoder_seed}"
            if show_encoder_seed
            else labels[variant]
        )
        draw.text((8, y + 14), label, fill="black")
        for block in range(12):
            value = float(lookup[(encoder_seed, variant, block)])
            color = (int(255 * value), 45, int(255 * (1.0 - value)))
            x = margin_left + block * cell_width
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), fill=color)
            draw.text((x + 7, y + 14), f"{value:.2f}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)

def file_digest_from_ids(records: Sequence[CubProbeRecord]) -> str:
    import hashlib

    payload = "\n".join(str(record.image_id) for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
