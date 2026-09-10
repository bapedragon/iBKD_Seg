#!/usr/bin/env python3
"""Smoke locked CUB direct-spatial diagnostics on audited guided encoders."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .cub_data import DATASET_NAME, NUM_CLASSES, resolve_dataset_root
from .cub_direct_spatial import (
    GRID_SIZE,
    LinearCKAAccumulator,
    assert_probability,
    attention_gt_metrics,
    attention_rollout,
    build_part_probe,
    evaluate_part_probe,
    feature_map_to_observations,
    load_mask_views,
    load_part_supervision,
    load_spatial_annotations,
    save_attention_triptych,
    select_lowest_image_id_per_class,
    train_part_candidate,
)
from .cub_probe_data import CubImageDataset, CubProbeRecord, load_train_validation_records
from .models import create_student, forward_student_spatial
from .run_cub_combined_smoke import _atomic_json_save, _atomic_torch_save, _runtime
from .run_cub_r50_teacher_full import load_scientific_teacher
from .train_timing import file_sha256, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/cub200_r50_224_b128_direct_spatial_smoke_v1.json"
)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
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
            "direct-spatial CLI changed locked runtime values: "
            f"{actual} != {expected}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUB direct-spatial H200 smoke requires CUDA")


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


@torch.no_grad()
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


def _part_smoke(
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
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
    smoke = config["smoke"]["part_probe"]
    candidates: list[dict[str, Any]] = []
    initial_hashes: set[str] = set()
    for learning_rate in smoke["learning_rates"]:
        result = train_part_candidate(
            train_features=train_features,
            train_supervision=train_supervision,
            validation_features=validation_features,
            validation_supervision=validation_supervision,
            learning_rate=float(learning_rate),
            epochs=int(smoke["epochs"]),
            seed=int(smoke["probe_seeds"][0]),
            batch_size=int(config["part_localization_probe"]["batch_size"]),
            device=device,
        )
        initial_hashes.add(result["initial_probe_state_sha256"])
        log(
            "[DIRECT_PART_CANDIDATE] "
            f"variant={variant} encoder_seed={encoder_seed} lr={learning_rate} "
            f"best_epoch={result['best_epoch']} "
            f"val_pck={result['best_validation']['micro_pck_at_0.1']:.6f}"
        )
        candidates.append(result)
    if len(initial_hashes) != 1:
        raise RuntimeError("part probe initial state changed across LR candidates")
    selected = min(
        candidates,
        key=lambda value: (
            -value["best_validation"]["micro_pck_at_0.1"],
            value["learning_rate"],
            value["best_epoch"],
        ),
    )
    probe = build_part_probe(int(smoke["probe_seeds"][0]))
    incompatible = probe.load_state_dict(selected["probe_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("selected part probe strict reload failed")
    probe.to(device).eval()
    reevaluated = evaluate_part_probe(
        probe,
        validation_features,
        validation_supervision,
        device=device,
        batch_size=int(config["part_localization_probe"]["batch_size"]),
    )
    if reevaluated != selected["best_validation"]:
        raise RuntimeError("selected part probe re-evaluation changed")
    legacy_seed1_smoke = config_sha256 == EXPECTED_CONFIG_SHA256
    checkpoint_name = (
        f"{variant}_seed1.pt"
        if legacy_seed1_smoke
        else f"{variant}_encoder_seed{encoder_seed}_probe_seed1.pt"
    )
    checkpoint = output_dir / "part_probe" / "checkpoints" / checkpoint_name
    _atomic_torch_save(
        {
            "probe": selected["probe_state"],
            "metadata": {
                "purpose": (
                    "non_scientific_cub_direct_spatial_smoke_part_probe_v1"
                    if legacy_seed1_smoke
                    else f"{config['protocol_id']}_selected_part_probe"
                ),
                "variant": variant,
                "encoder_seed": encoder_seed,
                "probe_seed": 1,
                "learning_rate": selected["learning_rate"],
                "selected_epoch": selected["best_epoch"],
                "official_test_evaluations": 0,
                "config_sha256": config_sha256,
            },
        },
        checkpoint,
    )
    candidate_rows = [
        {
            "variant": variant,
            "encoder_seed": encoder_seed,
            "probe_seed": value["seed"],
            "learning_rate": value["learning_rate"],
            "best_epoch": value["best_epoch"],
            "validation_micro_pck_at_0.1": value["best_validation"]["micro_pck_at_0.1"],
            "validation_mean_normalized_error": value["best_validation"][
                "mean_normalized_localization_error"
            ],
            "selected": value is selected,
            "scientific_result": False,
        }
        for value in candidates
    ]
    summary = {
        "variant": variant,
        "encoder_seed": encoder_seed,
        "probe_seed": 1,
        "candidate_count": len(candidates),
        "same_initial_probe_state_across_learning_rates": True,
        "initial_probe_state_sha256": next(iter(initial_hashes)),
        "selected_learning_rate": selected["learning_rate"],
        "selected_epoch": selected["best_epoch"],
        "validation": reevaluated,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "official_test_evaluations": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_result": False,
    }
    del train_features, validation_features, probe
    return candidate_rows, summary


@torch.no_grad()
def _cka_smoke(
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


@torch.no_grad()
def _attention_smoke(
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
    _validate_config(config, config_path)
    _validate_cli(args, config)
    config_sha256 = file_sha256(config_path)
    checkpoint_inputs = config["checkpoint_inputs"]
    encoder_seeds = sorted({int(item["encoder_seed"]) for item in checkpoint_inputs})

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

    splits, split_manifest, source = load_train_validation_records(
        args.data_dir.expanduser().resolve(), download=True
    )
    if (
        len(splits["train"]) != 5394
        or len(splits["validation"]) != 600
        or split_manifest.get("validation_image_ids_sha256") != EXPECTED_VALIDATION_SHA256
    ):
        raise RuntimeError("direct-spatial locked CUB split changed")
    train_records = select_lowest_image_id_per_class(splits["train"])
    validation_records = select_lowest_image_id_per_class(splits["validation"])
    dataset_root = resolve_dataset_root(args.data_dir.expanduser().resolve())
    part_names, annotations = load_spatial_annotations(dataset_root)
    if len(annotations) != 11788 or len(part_names) != 15:
        raise RuntimeError("official CUB spatial annotation inventory changed")
    visible_counts = [0] * 15
    for annotation in annotations.values():
        for index, visible in enumerate(annotation.visible):
            visible_counts[index] += int(visible)
    train_supervision = load_part_supervision(train_records, annotations)
    validation_supervision = load_part_supervision(validation_records, annotations)
    dataset_audit = {
        "status": "pass",
        "dataset": DATASET_NAME,
        "source": source,
        "split_manifest": split_manifest,
        "full_counts": {"train": 5394, "validation": 600, "official_test": 5794},
        "smoke_counts": {"train": 200, "validation": 200, "official_test": 0},
        "smoke_train_image_ids_sha256": file_digest_from_ids(train_records),
        "smoke_validation_image_ids_sha256": file_digest_from_ids(validation_records),
        "part_names": list(part_names),
        "part_annotation_images": len(annotations),
        "visible_counts_over_full_annotation_index": visible_counts,
        "global_annotation_index_parsed_for_integrity": True,
        "official_test_images_masks_or_metrics_accessed": False,
    }
    _atomic_json_save(dataset_audit, output_dir / "dataset_audit.json")

    teacher, _teacher_metadata, teacher_hash, teacher_state_hash = load_scientific_teacher(
        args.teacher_checkpoint.expanduser().resolve(), device=device
    )
    teacher_expected = config["protocol_provenance"]["teacher_release"]
    if (
        teacher_hash != teacher_expected["checkpoint_sha256"]
        or teacher_state_hash != teacher_expected["model_state_sha256"]
    ):
        raise RuntimeError("direct-spatial teacher identity changed")

    checkpoint_audits: list[dict[str, Any]] = []
    part_candidate_rows: list[dict[str, Any]] = []
    part_summaries: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    shared_part_initial_hashes: set[str] = set()

    for index, item in enumerate(config["checkpoint_inputs"], 1):
        variant = item["variant"]
        encoder_seed = int(item["encoder_seed"])
        _atomic_json_save(
            {
                "status": "running",
                "phase": "direct_spatial_diagnostics",
                "active_variant": variant,
                "active_encoder_seed": encoder_seed,
                "variants_complete": index - 1,
                "variants_expected": len(checkpoint_inputs),
                "official_test_accessed": False,
            },
            status_path,
        )
        student, audit = _load_student(
            args.student_release_dir.expanduser().resolve(),
            item,
            config=config,
            device=device,
        )
        checkpoint_audits.append(audit)
        candidate_rows, part_summary = _part_smoke(
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
        part_candidate_rows.extend(candidate_rows)
        part_summaries.append(part_summary)
        shared_part_initial_hashes.add(part_summary["initial_probe_state_sha256"])
        cka_rows.extend(
            _cka_smoke(
                variant=variant,
                encoder_seed=encoder_seed,
                student=student,
                teacher=teacher,
                records=validation_records,
                batch_size=args.cka_batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        )
        attention_rows.append(
            _attention_smoke(
                variant=variant,
                encoder_seed=encoder_seed,
                student=student,
                records=validation_records,
                batch_size=args.attention_batch_size,
                num_workers=args.num_workers,
                output_dir=output_dir,
                device=device,
            )
        )
        del student
        torch.cuda.empty_cache()

    if len(shared_part_initial_hashes) != 1:
        raise RuntimeError("matched part-probe initialization changed across encoders")
    qualitative_pngs = sorted((output_dir / "attention_gt/qualitative").glob("*.png"))
    gate = {
        "checkpoint_strict_loads": len(checkpoint_audits),
        "part_probe_lr_candidates": len(part_candidate_rows),
        "part_probe_validation_selections": len(part_summaries),
        "spatial_cka_values": len(cka_rows),
        "attention_metric_rows": len(attention_rows),
        "qualitative_pngs": len(qualitative_pngs),
        "official_test_evaluations": 0,
    }
    if gate != config["completion_gate"]:
        raise RuntimeError(f"direct-spatial completion gate failed: {gate}")

    _write_csv(part_candidate_rows, output_dir / "part_probe/candidates.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "matched_initialization_across_encoders": True,
            "initial_probe_state_sha256": next(iter(shared_part_initial_hashes)),
            "selections": part_summaries,
            "official_test_evaluations": 0,
            "scientific_result": False,
        },
        output_dir / "part_probe/results.json",
    )
    _write_csv(cka_rows, output_dir / "spatial_cka/results.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "rows": cka_rows,
            "official_test_used": False,
            "scientific_result": False,
        },
        output_dir / "spatial_cka/results.json",
    )
    _save_cka_heatmap(cka_rows, output_dir / "spatial_cka/layerwise_heatmap.png")
    _write_csv(attention_rows, output_dir / "attention_gt/results.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "rows": attention_rows,
            "official_test_evaluations": 0,
            "scientific_result": False,
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
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "scientific_result": False,
        "smoke_metrics_must_not_select_method_lambda_or_protocol": True,
        "student_batch_size": 128,
        "encoder_seeds": encoder_seeds,
        "variants": list(EXPECTED_VARIANTS),
        "runtime": {
            **_runtime(device),
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "completion_gate": gate,
        "official_test_accessed": False,
        "official_test_evaluations": 0,
        "output_files": {
            "dataset_audit": "dataset_audit.json",
            "checkpoint_audit": "checkpoint_audit.json",
            "part_probe": "part_probe/results.json",
            "spatial_cka": "spatial_cka/results.json",
            "spatial_cka_heatmap": "spatial_cka/layerwise_heatmap.png",
            "attention_gt": "attention_gt/results.json",
            "attention_qualitative_pngs": len(qualitative_pngs),
        },
    }
    if encoder_seeds == [1]:
        summary["encoder_seed"] = 1
    _atomic_json_save(summary, output_dir / "summary.json")
    _atomic_json_save(
        {
            "status": "complete",
            "phase": "complete",
            "completion_gate": gate,
            "finished_at_utc": summary["runtime"]["finished_at_utc"],
            "official_test_accessed": False,
            "failure": None,
        },
        status_path,
    )

    log("")
    log("[DIRECT_SPATIAL_SMOKE_RESULTS]")
    part_lookup = {
        (row["encoder_seed"], row["variant"]): row for row in part_summaries
    }
    cka_lookup = {
        (row["encoder_seed"], row["variant"], row["student_block"]): row
        for row in cka_rows
    }
    attention_lookup = {
        (row["encoder_seed"], row["variant"]): row for row in attention_rows
    }
    for item in checkpoint_inputs:
        variant = item["variant"]
        encoder_seed = int(item["encoder_seed"])
        log(
            f"[DIRECT_SPATIAL_RESULT] variant={variant} encoder_seed={encoder_seed} "
            f"part_val_pck={part_lookup[(encoder_seed, variant)]['validation']['micro_pck_at_0.1']:.6f} "
            "part_val_normalized_error="
            f"{part_lookup[(encoder_seed, variant)]['validation']['mean_normalized_localization_error']:.6f} "
            f"cka_block11={cka_lookup[(encoder_seed, variant, 11)]['centered_linear_cka']:.6f} "
            "attention_patch_ap="
            f"{attention_lookup[(encoder_seed, variant)]['global_micro_patch_average_precision']:.6f} "
            "attention_pointing="
            f"{attention_lookup[(encoder_seed, variant)]['pointing_game_peak_inside_mask']:.6f} "
            "foreground_attention_mass="
            f"{attention_lookup[(encoder_seed, variant)]['foreground_attention_mass_mean']:.6f}"
        )
    seed_marker = (
        ""
        if encoder_seeds == [1]
        else f"encoder_seeds={','.join(str(seed) for seed in encoder_seeds)} "
    )
    log(
        "[DIRECT_SPATIAL_SMOKE_DONE] status=pass "
        f"{seed_marker}"
        f"strict_loads={gate['checkpoint_strict_loads']} "
        f"part_candidates={gate['part_probe_lr_candidates']} "
        f"part_selections={gate['part_probe_validation_selections']} "
        f"cka_values={gate['spatial_cka_values']} "
        f"attention_rows={gate['attention_metric_rows']} "
        f"qualitative_pngs={gate['qualitative_pngs']} official_test=0 "
        f"elapsed_seconds={elapsed:.2f} "
        f"peak_allocated_bytes={summary['runtime']['peak_allocated_bytes']} "
        f"peak_reserved_bytes={summary['runtime']['peak_reserved_bytes']}"
    )
    return summary


def file_digest_from_ids(records: Sequence[CubProbeRecord]) -> str:
    import hashlib

    payload = "\n".join(str(record.image_id) for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_save(
            {
                "status": "failed",
                "phase": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
                "official_test_accessed": False,
            },
            output_dir / "sequence_status.json",
        )
        raise


if __name__ == "__main__":
    main()
