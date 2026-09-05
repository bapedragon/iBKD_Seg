#!/usr/bin/env python3
"""Run the batch-64 Phase 1 frozen-probe smoke without opening official test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import create_student
from .probe import (
    build_probe,
    confusion_from_tensors,
    evaluate_probe,
    module_state_sha256,
    probe_from_state,
    train_candidate,
)
from .probe_data import (
    PetImageDataset,
    PetRecord,
    ids_sha256,
    load_targets,
    load_train_validation_records,
)
from .train_timing import file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_phase1_v1.json"
)
DEFAULT_SMOKE_CONFIG = (
    REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_probe_smoke_b64_v1.json"
)
EXPECTED_VARIANTS = (
    "vanilla",
    "kd",
    "lg",
    "alg",
    "ibkd_lambda_0.25",
    "ibkd_lambda_0.5",
)


def log(message: str) -> None:
    print(message, flush=True)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _variant(method: str, fusion_ratio: float | None) -> str:
    if method != "ibkd":
        if fusion_ratio is not None:
            raise RuntimeError(f"non-iBKD method has lambda: {method}")
        return method
    if fusion_ratio == 0.25:
        return "ibkd_lambda_0.25"
    if fusion_ratio == 0.5:
        return "ibkd_lambda_0.5"
    raise RuntimeError(f"unexpected iBKD lambda: {fusion_ratio}")


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime(device: torch.device) -> dict[str, Any]:
    import timm
    import torchvision

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda": torch.version.cuda,
        "git_commit": _git_commit(),
    }


def _validate_smoke_config(
    protocol: dict[str, Any],
    smoke: dict[str, Any],
    protocol_path: Path,
) -> None:
    if smoke.get("scientific_result") is not False:
        raise RuntimeError("probe smoke must be non-scientific")
    if smoke.get("selection_from_smoke_metrics_forbidden") is not True:
        raise RuntimeError("smoke-metric selection must be forbidden")
    if smoke.get("official_test_accessed") is not False:
        raise RuntimeError("probe smoke must forbid official test access")
    if file_sha256(protocol_path) != smoke["source_protocol_config_sha256"]:
        raise RuntimeError("locked Phase 1 protocol config SHA-256 mismatch")
    if protocol["protocol_id"] != smoke["source_protocol"]:
        raise RuntimeError("probe smoke points to a different source protocol")
    source_probe = protocol["frozen_spatial_probe"]["probe"]
    smoke_probe = smoke["probe"]
    if smoke_probe["learning_rates"] != source_probe["learning_rates"]:
        raise RuntimeError("probe smoke must exercise the locked LR grid")
    if smoke_probe["batch_size"] != source_probe["batch_size"]:
        raise RuntimeError("probe smoke batch size differs from locked probe batch")
    if smoke_probe["probe_seeds"] != [source_probe["probe_seeds"][0]]:
        raise RuntimeError("probe smoke must use only the first locked probe seed")
    if tuple(smoke["classification_input"]["variants"]) != EXPECTED_VARIANTS:
        raise RuntimeError("probe smoke variant order differs from the locked matrix")
    if smoke["data"] != {
        "source": "official_trainval_only",
        "train_samples": 2940,
        "validation_samples": 740,
        "test_samples": 0,
        "random_augmentation": False,
    }:
        raise RuntimeError("probe smoke data contract changed")


def _validate_classification_input(
    classification_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    summary_path = classification_root / "classification_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            "batch-64 classification summary is missing at " f"{summary_path}"
        )
    suite = _load_json(summary_path)
    suite_contracts = suite.get("contracts", {})
    required = {
        "status": suite.get("status") == "complete",
        "scientific_result": suite.get("scientific_result") is True,
        "batch_size": suite.get("batch_size") == 64,
        "epochs": suite.get("epochs") == 300,
        "completed_tasks": suite.get("completed_tasks") == 19,
        "failed_tasks": suite.get("failed_tasks") == 0,
        "contracts": suite_contracts.get("all_passed") is True,
        "strict_reload": suite_contracts.get(
            "every_completed_checkpoint_strict_reloaded"
        )
        is True,
        "test_once": suite_contracts.get("every_completed_model_tested_once") is True,
        "test_not_selected": suite_contracts.get(
            "official_test_used_for_training_or_selection"
        )
        is False,
    }
    if not all(required.values()):
        failures = [name for name, passed in required.items() if not passed]
        raise RuntimeError(
            "batch-64 classification suite failed prerequisites: "
            + ", ".join(failures)
        )

    summary_paths = sorted((classification_root / "students").glob("*/summary.json"))
    if len(summary_paths) != 18:
        raise RuntimeError(
            f"expected 18 batch-64 student summaries, found {len(summary_paths)}"
        )
    all_entries: list[dict[str, Any]] = []
    observed_matrix: set[tuple[str, int]] = set()
    validation_hashes: set[str] = set()
    for path in summary_paths:
        row = _load_json(path)
        variant = _variant(row["method"], row.get("fusion_ratio_lambda"))
        seed = int(row["seed"])
        checks = {
            "status": row.get("status") == "complete",
            "scientific_result": row.get("scientific_result") is True,
            "batch_size": row.get("batch_size") == 64,
            "epochs": row.get("epochs") == 300,
            "official_test_evaluations": row.get("official_test_evaluations") == 1,
            "test_not_selected": row.get(
                "official_test_used_for_training_or_selection"
            )
            is False,
            "strict_reload": row.get("selected_checkpoint_strict_reloaded") is True,
        }
        if not all(checks.values()):
            failures = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"invalid classification summary {path}: {failures}")
        checkpoint_path = path.parent / "student_best_validation.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"classification checkpoint is missing: {checkpoint_path}")
        split_hash = row["split_manifest"]["validation_image_ids_sha256"]
        validation_hashes.add(split_hash)
        observed_matrix.add((variant, seed))
        all_entries.append(
            {
                "variant": variant,
                "method": row["method"],
                "fusion_ratio_lambda": row.get("fusion_ratio_lambda"),
                "encoder_seed": seed,
                "summary_path": path,
                "checkpoint_path": checkpoint_path,
                "summary": row,
                "validation_image_ids_sha256": split_hash,
            }
        )
    expected_matrix = {
        (variant, seed) for variant in EXPECTED_VARIANTS for seed in (1, 2, 3)
    }
    if observed_matrix != expected_matrix:
        raise RuntimeError("classification student matrix is incomplete or duplicated")
    if len(validation_hashes) != 1:
        raise RuntimeError("classification checkpoints used different validation splits")

    selected = [entry for entry in all_entries if entry["encoder_seed"] == 1]
    selected_by_variant = {entry["variant"]: entry for entry in selected}
    if set(selected_by_variant) != set(EXPECTED_VARIANTS):
        raise RuntimeError("could not resolve six encoder-seed-1 checkpoints")
    return (
        [selected_by_variant[variant] for variant in EXPECTED_VARIANTS],
        suite,
        next(iter(validation_hashes)),
    )


def _load_encoder(
    entry: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path: Path = entry["checkpoint_path"]
    row = entry["summary"]
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != row["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("student"), dict):
        raise RuntimeError(f"invalid student checkpoint payload: {checkpoint_path}")
    metadata = payload.get("metadata", {})
    metadata_checks = {
        "purpose": metadata.get("purpose") == "phase1_scientific_full_student",
        "dataset": metadata.get("dataset") == "Oxford-IIIT Pet",
        "architecture": metadata.get("architecture") == "deit_tiny_patch16_224",
        "method": metadata.get("method") == entry["method"],
        "lambda": metadata.get("fusion_ratio_lambda")
        == entry["fusion_ratio_lambda"],
        "batch": metadata.get("batch_size") == 64,
        "epochs": metadata.get("epochs") == 300,
        "seed": metadata.get("seed") == 1,
        "validation_split": metadata.get("validation_image_ids_sha256")
        == entry["validation_image_ids_sha256"],
        "test_before_checkpoint": metadata.get(
            "official_test_evaluations_at_checkpoint_write"
        )
        == 0,
    }
    if not all(metadata_checks.values()):
        failures = [name for name, passed in metadata_checks.items() if not passed]
        raise RuntimeError(f"checkpoint metadata mismatch {checkpoint_path}: {failures}")

    model = create_student(num_classes=37, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict encoder load returned incompatible keys")
    model_state_sha256 = module_state_sha256(model)
    if model_state_sha256 != row["student_state_sha256"]:
        raise RuntimeError(f"student state SHA-256 mismatch: {checkpoint_path}")
    if metadata.get("student_state_sha256") != model_state_sha256:
        raise RuntimeError(f"checkpoint metadata state hash mismatch: {checkpoint_path}")
    model.requires_grad_(False)
    model.eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("encoder freeze/eval contract failed")
    return model.to(device), {
        "checkpoint_sha256": checkpoint_sha256,
        "student_state_sha256": model_state_sha256,
        "strict_load": True,
        "eval_mode": True,
        "trainable_parameter_count": 0,
    }


def _load_cache(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("metadata") != expected:
        return None
    return payload


def _target_cache(
    records: Sequence[PetRecord],
    *,
    split: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
    cache_path: Path,
) -> tuple[dict[str, Any], bool]:
    probe_contract = protocol["frozen_spatial_probe"]
    input_size = int(probe_contract["image_input"]["size"])
    target_contract = probe_contract["probe"]["target"]
    grid_size = (
        int(target_contract["grid_height"]),
        int(target_contract["grid_width"]),
    )
    ignore_index = int(probe_contract["mask"]["ignore_index"])
    expected = {
        "kind": "phase1_pet_probe_targets_v1",
        "protocol_sha256": protocol_sha256,
        "split": split,
        "ids_sha256": ids_sha256(records),
        "count": len(records),
        "input_size": input_size,
        "grid_size": list(grid_size),
        "ignore_index": ignore_index,
    }
    cached = _load_cache(cache_path, expected)
    if cached is not None:
        input_targets = cached.get("input_targets")
        grid_targets = cached.get("grid_targets")
        if (
            isinstance(input_targets, torch.Tensor)
            and input_targets.shape == (len(records), input_size, input_size)
            and input_targets.dtype == torch.uint8
            and isinstance(grid_targets, torch.Tensor)
            and grid_targets.shape == (len(records), *grid_size)
            and grid_targets.dtype == torch.uint8
        ):
            log(f"[CACHE] targets {split}: hit ({len(records)} samples)")
            return cached, True

    log(f"[CACHE] targets {split}: building {len(records)} samples")
    input_targets = torch.empty((len(records), input_size, input_size), dtype=torch.uint8)
    grid_targets = torch.empty((len(records), *grid_size), dtype=torch.uint8)
    for index, record in enumerate(records):
        input_target, grid_target = load_targets(
            record,
            input_size=input_size,
            grid_size=grid_size,
            occupancy_threshold=float(target_contract["foreground_threshold"]),
            ignore_index=ignore_index,
        )
        input_targets[index] = input_target
        grid_targets[index] = grid_target
        if (index + 1) % 500 == 0 or index + 1 == len(records):
            log(f"[CACHE] targets {split}: {index + 1}/{len(records)}")
    payload = {
        "metadata": expected,
        "input_targets": input_targets,
        "grid_targets": grid_targets,
    }
    _atomic_torch_save(payload, cache_path)
    reloaded = _load_cache(cache_path, expected)
    if reloaded is None:
        raise RuntimeError(f"target cache safe reload failed: {cache_path}")
    return reloaded, True


@torch.inference_mode()
def _feature_cache(
    model: torch.nn.Module,
    entry: dict[str, Any],
    records: Sequence[PetRecord],
    *,
    split: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
    cache_path: Path,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], bool]:
    feature_contract = protocol["frozen_spatial_probe"]["encoder"]["feature"]
    feature_shape = (
        int(feature_contract["channels"]),
        int(feature_contract["height"]),
        int(feature_contract["width"]),
    )
    expected = {
        "kind": "phase1_pet_probe_frozen_features_v1",
        "protocol_sha256": protocol_sha256,
        "split": split,
        "ids_sha256": ids_sha256(records),
        "count": len(records),
        "variant": entry["variant"],
        "encoder_seed": entry["encoder_seed"],
        "checkpoint_sha256": entry["summary"]["checkpoint_sha256"],
        "student_state_sha256": entry["summary"]["student_state_sha256"],
        "block_index": int(feature_contract["block_index"]),
        "norm": bool(feature_contract["norm"]),
        "exclude_cls_token": bool(feature_contract["exclude_cls_token"]),
        "feature_shape": list(feature_shape),
        "feature_dtype": "float32",
        "amp": False,
    }
    cached = _load_cache(cache_path, expected)
    if cached is not None:
        features = cached.get("features")
        if (
            isinstance(features, torch.Tensor)
            and features.shape == (len(records), *feature_shape)
            and features.dtype == torch.float32
            and not features.requires_grad
        ):
            log(
                f"[CACHE] features {entry['variant']} {split}: "
                f"hit ({len(records)} samples)"
            )
            return cached, True

    log(
        f"[CACHE] features {entry['variant']} {split}: "
        f"building {len(records)} samples"
    )
    dataset = PetImageDataset(
        records,
        input_size=int(protocol["frozen_spatial_probe"]["image_input"]["size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=feature_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features = torch.empty((len(records), *feature_shape), dtype=torch.float32)
    observed_ids: list[str] = []
    offset = 0
    for batch_index, (images, batch_ids) in enumerate(loader, start=1):
        final_tokens, intermediates = model.forward_intermediates(
            images.to(device, non_blocking=True),
            indices=[int(feature_contract["block_index"])],
            norm=bool(feature_contract["norm"]),
            output_fmt=feature_contract["output_format"],
            intermediates_only=False,
        )
        del final_tokens
        if len(intermediates) != 1:
            raise RuntimeError("encoder did not return exactly one probe feature")
        batch_features = intermediates[0].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        if batch_features.shape[1:] != feature_shape:
            raise RuntimeError(
                f"unexpected frozen feature shape: {tuple(batch_features.shape)}"
            )
        if batch_features.requires_grad:
            raise RuntimeError("frozen feature unexpectedly requires gradients")
        end = offset + len(batch_features)
        features[offset:end] = batch_features
        offset = end
        observed_ids.extend(str(image_id) for image_id in batch_ids)
        if batch_index % 25 == 0 or offset == len(records):
            log(
                f"[CACHE] features {entry['variant']} {split}: "
                f"{offset}/{len(records)}"
            )
    expected_ids = [record.image_id for record in records]
    if offset != len(records) or observed_ids != expected_ids:
        raise RuntimeError("frozen feature cache sample order mismatch")
    payload = {"metadata": expected, "features": features}
    _atomic_torch_save(payload, cache_path)
    reloaded = _load_cache(cache_path, expected)
    if reloaded is None:
        raise RuntimeError(f"feature cache safe reload failed: {cache_path}")
    reloaded_features = reloaded.get("features")
    if not isinstance(reloaded_features, torch.Tensor) or not torch.equal(
        reloaded_features, features
    ):
        raise RuntimeError(f"feature cache round-trip mismatch: {cache_path}")
    return reloaded, True


def _baselines(
    train_targets: dict[str, Any],
    validation_targets: dict[str, Any],
    *,
    ignore_index: int,
) -> dict[str, Any]:
    train_grid = train_targets["grid_targets"]
    valid_counts = train_grid.ne(ignore_index).sum(dim=0)
    foreground_counts = train_grid.eq(1).sum(dim=0)
    train_mean_score = torch.zeros_like(valid_counts, dtype=torch.float32)
    usable = valid_counts.gt(0)
    train_mean_score[usable] = (
        foreground_counts[usable].float() / valid_counts[usable].float()
    )
    train_mean_grid = train_mean_score.ge(0.5)
    input_size = int(validation_targets["input_targets"].shape[-1])
    train_mean_input = F.interpolate(
        train_mean_score[None, None],
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].ge(0.5)
    validation_grid = validation_targets["grid_targets"]
    validation_input = validation_targets["input_targets"]
    report: dict[str, Any] = {}
    for name, grid_template, input_template in (
        (
            "all_background",
            torch.zeros_like(train_mean_grid),
            torch.zeros_like(train_mean_input),
        ),
        ("train_mean_mask", train_mean_grid, train_mean_input),
    ):
        report[name] = {
            "validation_grid": confusion_from_tensors(
                grid_template.expand_as(validation_grid),
                validation_grid,
                ignore_index=ignore_index,
            ).metrics(),
            "validation_input_224": confusion_from_tensors(
                input_template.expand_as(validation_input),
                validation_input,
                ignore_index=ignore_index,
            ).metrics(),
        }
    return report


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    names = (
        "foreground_iou",
        "background_iou",
        "mean_iou",
        "foreground_dice",
        "pixel_accuracy",
    )
    return all(math.isfinite(float(metrics[name])) for name in names)


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "encoder_seed",
        "selected_learning_rate",
        "selected_epoch",
        "validation_grid_mean_iou",
        "validation_input_224_mean_iou",
        "validation_input_224_foreground_iou",
        "validation_input_224_background_iou",
        "validation_input_224_foreground_dice",
        "validation_input_224_pixel_accuracy",
        "feature_cache_seconds",
        "probe_training_seconds",
        "peak_cuda_memory_bytes",
        "scientific_result",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            selection = result["selection"]
            grid = result["validation"]["grid_14x14"]
            input_metrics = result["validation"]["input_224"]
            writer.writerow(
                {
                    "variant": result["variant"],
                    "encoder_seed": result["encoder_seed"],
                    "selected_learning_rate": selection["learning_rate"],
                    "selected_epoch": selection["epoch"],
                    "validation_grid_mean_iou": grid["mean_iou"],
                    "validation_input_224_mean_iou": input_metrics["mean_iou"],
                    "validation_input_224_foreground_iou": input_metrics[
                        "foreground_iou"
                    ],
                    "validation_input_224_background_iou": input_metrics[
                        "background_iou"
                    ],
                    "validation_input_224_foreground_dice": input_metrics[
                        "foreground_dice"
                    ],
                    "validation_input_224_pixel_accuracy": input_metrics[
                        "pixel_accuracy"
                    ],
                    "feature_cache_seconds": result["timing"][
                        "feature_cache_seconds"
                    ],
                    "probe_training_seconds": result["timing"][
                        "probe_training_seconds"
                    ],
                    "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
                    "scientific_result": False,
                }
            )
    temporary.replace(path)


def _sequence_status(
    output_dir: Path,
    *,
    status: str,
    completed: int,
    active_variant: str | None,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "completed_encoders": completed,
            "expected_encoders": 6,
            "active_variant": active_variant,
            "scientific_result": False,
            "official_test_accessed": False,
            "failure": failure,
        },
        output_dir / "sequence_status.json",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    protocol = _load_json(args.protocol_config)
    smoke = _load_json(args.smoke_config)
    _validate_smoke_config(protocol, smoke, args.protocol_config)
    protocol_sha256 = file_sha256(args.protocol_config)
    smoke_config_sha256 = file_sha256(args.smoke_config)
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _sequence_status(
        args.output_dir,
        status="running",
        completed=0,
        active_variant=None,
    )

    log(
        "[PROBE_SMOKE_MODE] scientific_result=false official_test_accessed=false "
        "smoke_metrics_for_selection=forbidden"
    )
    entries, classification_suite, classification_split_hash = (
        _validate_classification_input(args.classification_root)
    )
    log(
        "[PROBE_SMOKE_INPUT] batch64_suite=pass student_runs=18 "
        "selected_encoder_seed=1 checkpoints=6"
    )

    records, split_manifest = load_train_validation_records(args.data_dir, download=True)
    if split_manifest["validation_image_ids_sha256"] != classification_split_hash:
        raise RuntimeError(
            "probe split differs from the classification validation split"
        )
    if {name: len(values) for name, values in records.items()} != {
        "train": 2940,
        "validation": 740,
    }:
        raise RuntimeError("probe train/validation counts changed")
    log(
        "[PROBE_SMOKE_DATA] official_trainval_only train=2940 validation=740 "
        "test=0 split_hash=" + classification_split_hash
    )

    target_started = time.monotonic()
    targets: dict[str, dict[str, Any]] = {}
    target_cache_reloaded: list[bool] = []
    for split in ("train", "validation"):
        targets[split], reloaded = _target_cache(
            records[split],
            split=split,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
        target_cache_reloaded.append(reloaded)
    target_seconds = time.monotonic() - target_started
    ignore_index = int(protocol["frozen_spatial_probe"]["mask"]["ignore_index"])
    allowed_targets = {0, 1, ignore_index}
    observed_targets = {
        int(value)
        for split in targets.values()
        for key in ("input_targets", "grid_targets")
        for value in torch.unique(split[key])
    }
    if not observed_targets.issubset(allowed_targets):
        raise RuntimeError(f"mapped target contract failed: {observed_targets}")
    baselines = _baselines(
        targets["train"],
        targets["validation"],
        ignore_index=ignore_index,
    )

    source_probe = protocol["frozen_spatial_probe"]["probe"]
    smoke_probe = smoke["probe"]
    learning_rates = [float(value) for value in smoke_probe["learning_rates"]]
    probe_seed = int(smoke_probe["probe_seeds"][0])
    smoke_epochs = int(smoke_probe["epochs"])
    log(
        "[PROBE_SMOKE_TASK_COUNT] encoders=6 lr_candidates=18 "
        f"epochs_per_candidate={smoke_epochs} probe_seed={probe_seed}"
    )

    results: list[dict[str, Any]] = []
    initial_probe_hashes: set[str] = set()
    batch_orders_by_epoch: dict[int, set[str]] = {
        epoch: set() for epoch in range(1, smoke_epochs + 1)
    }
    feature_cache_reloaded: list[bool] = []
    for entry in entries:
        variant = entry["variant"]
        _sequence_status(
            args.output_dir,
            status="running",
            completed=len(results),
            active_variant=variant,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, encoder_audit = _load_encoder(entry, device)

        feature_started = time.monotonic()
        features: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation"):
            features[split], reloaded = _feature_cache(
                model,
                entry,
                records[split],
                split=split,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                cache_path=args.cache_dir / "features" / variant / f"{split}.pt",
                device=device,
                feature_batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
            feature_cache_reloaded.append(reloaded)
        _synchronize(device)
        feature_seconds = time.monotonic() - feature_started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        candidate_started = time.monotonic()
        candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        for learning_rate in learning_rates:
            state, candidate = train_candidate(
                features["train"]["features"],
                targets["train"]["grid_targets"],
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                probe_config=source_probe,
                learning_rate=learning_rate,
                seed=probe_seed,
                device=device,
                epochs=smoke_epochs,
            )
            candidates.append((state, candidate))
            initial_probe_hashes.add(candidate["initial_probe_state_sha256"])
            for epoch, digest in enumerate(
                candidate["batch_order_sha256_by_epoch"], start=1
            ):
                batch_orders_by_epoch[epoch].add(digest)
        # max() retains the earlier list entry on an exact metric tie. LRs are
        # in ascending order, which implements lower-LR before earlier-epoch.
        selected_index = max(
            range(len(candidates)),
            key=lambda index: candidates[index][1][
                "best_validation_grid_mean_iou"
            ],
        )
        selected_state, selected = candidates[selected_index]
        probe = probe_from_state(source_probe, probe_seed, selected_state, device)
        grid_metrics = evaluate_probe(
            probe,
            features["validation"]["features"],
            targets["validation"]["grid_targets"],
            batch_size=int(source_probe["batch_size"]),
            device=device,
            ignore_index=ignore_index,
        )
        input_size = int(protocol["frozen_spatial_probe"]["image_input"]["size"])
        input_metrics = evaluate_probe(
            probe,
            features["validation"]["features"],
            targets["validation"]["input_targets"],
            batch_size=int(source_probe["batch_size"]),
            device=device,
            ignore_index=ignore_index,
            output_size=(input_size, input_size),
        )
        _synchronize(device)
        probe_seconds = time.monotonic() - candidate_started
        artifact_path = args.output_dir / "probes" / f"{variant}_seed1_smoke.pt"
        _atomic_torch_save(
            {
                "purpose": "phase1_probe_smoke_only",
                "scientific_result": False,
                "official_test_accessed": False,
                "protocol_sha256": protocol_sha256,
                "smoke_config_sha256": smoke_config_sha256,
                "variant": variant,
                "encoder_seed": 1,
                "encoder_checkpoint_sha256": encoder_audit["checkpoint_sha256"],
                "probe_seed": probe_seed,
                "selection": {
                    "split": "validation",
                    "learning_rate": selected["learning_rate"],
                    "epoch": selected["best_epoch"],
                    "validation_grid_mean_iou": selected[
                        "best_validation_grid_mean_iou"
                    ],
                },
                "model": selected_state,
            },
            artifact_path,
        )
        artifact_payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
        strict_probe = probe_from_state(
            source_probe,
            probe_seed,
            artifact_payload["model"],
            device,
        )
        del strict_probe
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        result = {
            "variant": variant,
            "method": entry["method"],
            "fusion_ratio_lambda": entry["fusion_ratio_lambda"],
            "encoder_seed": 1,
            "classification_checkpoint": {
                **encoder_audit,
                "path": str(entry["checkpoint_path"]),
            },
            "feature_contract": {
                "shape": [192, 14, 14],
                "dtype": "float32",
                "requires_grad": False,
                "block_index": 11,
                "norm": False,
                "exclude_cls_token": True,
            },
            "probe_seed": probe_seed,
            "selection": {
                "split": "validation",
                "learning_rate": selected["learning_rate"],
                "epoch": selected["best_epoch"],
                "validation_grid_mean_iou": selected[
                    "best_validation_grid_mean_iou"
                ],
                "smoke_plumbing_only": True,
            },
            "candidates": [candidate for _, candidate in candidates],
            "validation": {
                "grid_14x14": grid_metrics,
                "input_224": input_metrics,
            },
            "selected_probe_artifact": str(artifact_path),
            "selected_probe_strict_reloaded": True,
            "timing": {
                "feature_cache_seconds": feature_seconds,
                "probe_training_seconds": probe_seconds,
            },
            "peak_cuda_memory_bytes": peak_memory,
            "scientific_result": False,
        }
        results.append(result)
        _sequence_status(
            args.output_dir,
            status="running",
            completed=len(results),
            active_variant=None,
        )
        log(
            f"[PROBE_SMOKE_RESULT] variant={variant} encoder_seed=1 "
            f"lr={selected['learning_rate']:g} epoch={selected['best_epoch']} "
            f"val_grid_miou={float(grid_metrics['mean_iou']):.6f} "
            f"val_input_miou={float(input_metrics['mean_iou']):.6f} "
            f"feature_seconds={feature_seconds:.2f} probe_seconds={probe_seconds:.2f} "
            f"peak_cuda_memory_bytes={peak_memory} scientific_result=false"
        )
        del probe, features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    same_probe_initialization = len(initial_probe_hashes) == 1
    same_batch_order = all(len(values) == 1 for values in batch_orders_by_epoch.values())
    all_candidate_gradients = all(
        candidate["gradient_contract"]
        == {
            "cached_feature_gradient_tensor_count": 0,
            "probe_gradient_tensor_count": 2,
        }
        for result in results
        for candidate in result["candidates"]
    )
    all_finite = all(
        _finite_metrics(result["validation"][resolution])
        for result in results
        for resolution in ("grid_14x14", "input_224")
    )
    gates = {
        "classification_suite_complete_and_audited": True,
        "six_encoder_seed1_checkpoints_sha256_and_strict_load": len(results) == 6,
        "same_classification_and_probe_validation_split": True,
        "official_trainval_counts_2940_740": True,
        "official_test_accessed": False,
        "trimap_values_mapped_to_0_1_255": observed_targets.issubset(allowed_targets),
        "encoder_eval_and_zero_trainable_parameters": all(
            result["classification_checkpoint"]["eval_mode"]
            and result["classification_checkpoint"]["trainable_parameter_count"] == 0
            for result in results
        ),
        "features_float32_192x14x14_without_grad": all(
            result["feature_contract"]
            == {
                "shape": [192, 14, 14],
                "dtype": "float32",
                "requires_grad": False,
                "block_index": 11,
                "norm": False,
                "exclude_cls_token": True,
            }
            for result in results
        ),
        "target_cache_safe_reload": all(target_cache_reloaded),
        "feature_cache_safe_reload": all(feature_cache_reloaded),
        "same_probe_initial_state_across_encoders_and_lrs": same_probe_initialization,
        "same_batch_order_across_encoders_and_lrs": same_batch_order,
        "only_two_probe_parameter_gradients": all_candidate_gradients,
        "selected_probe_strict_reload": all(
            result["selected_probe_strict_reloaded"] for result in results
        ),
        "finite_validation_loss_and_metrics": all_finite,
        "selection_uses_validation_only": True,
        "smoke_metrics_for_scientific_selection_forbidden": True,
    }
    status = "pass" if all(gates.values()) else "fail"
    total_seconds = time.monotonic() - started
    feature_seconds_total = sum(
        result["timing"]["feature_cache_seconds"] for result in results
    )
    probe_seconds_total = sum(
        result["timing"]["probe_training_seconds"] for result in results
    )
    official_counts = protocol["dataset"]["official_split_counts"]
    full_feature_multiplier = (
        3
        * int(official_counts["total"])
        / int(official_counts["trainval"])
    )
    estimate = {
        "basis": "rough_linear_extrapolation_from_full_train_validation_smoke",
        "batch64_full_target_cache_seconds": (
            target_seconds
            * int(official_counts["total"])
            / int(official_counts["trainval"])
        ),
        "batch64_full_feature_cache_seconds": (
            feature_seconds_total * full_feature_multiplier
        ),
        "batch64_full_probe_candidate_training_seconds": probe_seconds_total * 750,
        "multiplier_explanation": {
            "target_cache": "7,349 full samples / 3,680 smoke trainval samples",
            "feature_cache": (
                "3 encoder seeds / 1 smoke encoder seed x "
                "7,349 full samples / 3,680 smoke trainval samples"
            ),
            "probe_training": "3 encoder seeds x 5 probe seeds x (100/2) epochs",
        },
        "excluded_from_extrapolation": [
            "selected-probe official-test evaluation",
            "qualitative panel export",
            "queue time and storage variation",
        ],
        "not_a_runtime_guarantee": True,
    }
    summary = {
        "status": status,
        "smoke_id": smoke["smoke_id"],
        "purpose": smoke["purpose"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "protocol": {
            "id": protocol["protocol_id"],
            "config_path": str(args.protocol_config),
            "config_sha256": protocol_sha256,
            "smoke_config_path": str(args.smoke_config),
            "smoke_config_sha256": smoke_config_sha256,
        },
        "classification_input": {
            "root": str(args.classification_root),
            "summary_sha256": file_sha256(
                args.classification_root / "classification_summary.json"
            ),
            "suite_status": classification_suite["status"],
            "batch_size": 64,
            "student_suite_runs": 18,
            "used_encoder_seed": 1,
            "used_checkpoints": 6,
        },
        "data": {
            "root": str(args.data_dir),
            "source": "official_trainval_only",
            "counts": {"train": 2940, "validation": 740, "test": 0},
            "validation_image_ids_sha256": classification_split_hash,
            "target_values_observed": sorted(observed_targets),
        },
        "smoke_matrix": {
            "variants": list(EXPECTED_VARIANTS),
            "encoder_seeds": [1],
            "probe_seeds": [probe_seed],
            "learning_rates": learning_rates,
            "epochs": smoke_epochs,
            "probe_batch_size": int(source_probe["batch_size"]),
            "lr_candidates": 18,
        },
        "baselines_validation_only": baselines,
        "results": results,
        "contracts": gates,
        "timing": {
            "target_cache_seconds": target_seconds,
            "feature_cache_seconds_total": feature_seconds_total,
            "probe_training_seconds_total": probe_seconds_total,
            "suite_seconds": total_seconds,
            "rough_full_batch64_extrapolation": estimate,
        },
        "runtime": _runtime(device),
    }
    _atomic_json_save(summary, args.output_dir / "probe_smoke_summary.json")
    _write_csv(results, args.output_dir / "probe_smoke_summary.csv")
    _sequence_status(
        args.output_dir,
        status=status,
        completed=len(results),
        active_variant=None,
    )
    contract_text = " ".join(
        f"{name}={str(value).lower()}" for name, value in gates.items()
    )
    log("[PROBE_SMOKE_CONTRACT] " + contract_text)
    log(
        f"[PROBE_SMOKE_DONE] status={status} completed={len(results)}/6 "
        f"seconds={total_seconds:.2f} official_test_accessed=false "
        f"scientific_result=false output={args.output_dir}"
    )
    if status != "pass":
        raise RuntimeError("one or more frozen-probe smoke gates failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--smoke-config", type=Path, default=DEFAULT_SMOKE_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.feature_batch_size <= 0 or args.num_workers < 0:
        parser.error("feature batch size must be positive and workers non-negative")
    return args


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _sequence_status(
            args.output_dir,
            status="failed",
            completed=0,
            active_variant=None,
            failure=f"{type(error).__name__}: {error}",
        )
        log(f"[PROBE_SMOKE_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
