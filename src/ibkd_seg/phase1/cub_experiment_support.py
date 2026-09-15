"""Shared CUB experiment I/O, cache, and reporting support."""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .cub_probe_data import (
    CubImageDataset,
    CubProbeRecord,
    ids_sha256,
    load_targets,
)

EXPECTED_VARIANTS = (
    "vanilla",
    "kd",
    "lg",
    "alg_warmup20",
    "ibkd_lambda_0.25",
    "ibkd_lambda_0.5",
)

VARIANT_ARGUMENTS: dict[str, tuple[str, float | None, int | None]] = {
    "vanilla": ("vanilla", None, None),
    "kd": ("kd", None, None),
    "lg": ("lg", None, 0),
    "alg_warmup20": ("alg", None, 20),
    "ibkd_lambda_0.25": ("ibkd", 0.25, 20),
    "ibkd_lambda_0.5": ("ibkd", 0.5, 20),
}

def log(message: str = "") -> None:
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
    }

def _load_cache(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("metadata") != expected:
        return None
    return payload

def _target_cache(
    records: Sequence[CubProbeRecord],
    *,
    split: str,
    config: dict[str, Any],
    config_sha256: str,
    cache_path: Path,
) -> tuple[dict[str, Any], bool]:
    input_size = int(config["frozen_probe"]["image_input"]["size"])
    target_config = config["frozen_probe"]["probe"]["target"]
    grid_size = (
        int(target_config["grid_height"]),
        int(target_config["grid_width"]),
    )
    threshold = float(target_config["foreground_occupancy_threshold"])
    expected = {
        "kind": "phase1_cub_probe_targets_v1",
        "config_sha256": config_sha256,
        "split": split,
        "ids_sha256": ids_sha256(records),
        "count": len(records),
        "input_size": input_size,
        "grid_size": list(grid_size),
        "foreground_rule": "grayscale_value_greater_than_0",
        "foreground_occupancy_threshold": threshold,
    }
    cached = _load_cache(cache_path, expected)
    if cached is not None:
        log(f"[CUB_CACHE] targets {split}: hit ({len(records)} samples)")
        return cached, True

    log(f"[CUB_CACHE] targets {split}: building {len(records)} samples")
    input_targets = torch.empty(
        (len(records), input_size, input_size), dtype=torch.uint8
    )
    grid_targets = torch.empty((len(records), *grid_size), dtype=torch.uint8)
    for index, record in enumerate(records):
        input_target, grid_target = load_targets(
            record,
            input_size=input_size,
            grid_size=grid_size,
            occupancy_threshold=threshold,
        )
        input_targets[index] = input_target
        grid_targets[index] = grid_target
        if (index + 1) % 500 == 0 or index + 1 == len(records):
            log(f"[CUB_CACHE] targets {split}: {index + 1}/{len(records)}")
    payload = {
        "metadata": expected,
        "input_targets": input_targets,
        "grid_targets": grid_targets,
    }
    _atomic_torch_save(payload, cache_path)
    reloaded = _load_cache(cache_path, expected)
    if reloaded is None:
        raise RuntimeError(f"CUB target cache safe reload failed: {cache_path}")
    if not torch.equal(reloaded["input_targets"], input_targets) or not torch.equal(
        reloaded["grid_targets"], grid_targets
    ):
        raise RuntimeError(f"CUB target cache round-trip mismatch: {cache_path}")
    return reloaded, True

def _feature_cache(
    model: torch.nn.Module,
    records: Sequence[CubProbeRecord],
    *,
    split: str,
    variant: str,
    encoder_seed: int,
    checkpoint_sha256: str,
    state_sha256: str,
    config: dict[str, Any],
    config_sha256: str,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], bool]:
    contract = config["frozen_probe"]["encoder"]["feature"]
    feature_shape = (
        int(contract["channels"]),
        int(contract["height"]),
        int(contract["width"]),
    )
    expected = {
        "kind": "phase1_cub_probe_frozen_features_v1",
        "config_sha256": config_sha256,
        "split": split,
        "ids_sha256": ids_sha256(records),
        "count": len(records),
        "variant": variant,
        "encoder_seed": encoder_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "student_state_sha256": state_sha256,
        "block_index": int(contract["block_index"]),
        "norm": bool(contract["norm"]),
        "exclude_cls_token": bool(contract["exclude_cls_token"]),
        "feature_shape": list(feature_shape),
        "feature_dtype": "float32",
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
                f"[CUB_CACHE] features {variant} {split}: hit ({len(records)} samples)"
            )
            return cached, True

    dataset = CubImageDataset(
        records,
        input_size=int(config["frozen_probe"]["image_input"]["size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    features = torch.empty((len(records), *feature_shape), dtype=torch.float32)
    expected_ids = [str(record.image_id) for record in records]
    observed_ids: list[str] = []
    offset = 0
    log(f"[CUB_CACHE] features {variant} {split}: building {len(records)} samples")
    with torch.inference_mode():
        for batch_index, (images, batch_ids) in enumerate(loader, start=1):
            final_tokens, intermediates = model.forward_intermediates(
                images.to(device, non_blocking=True),
                indices=[int(contract["block_index"])],
                norm=bool(contract["norm"]),
                output_fmt=str(contract["output_format"]),
                intermediates_only=False,
            )
            del final_tokens
            if len(intermediates) != 1:
                raise RuntimeError("CUB encoder did not return one probe feature")
            batch_features = intermediates[0].detach().to(
                device="cpu", dtype=torch.float32
            )
            if tuple(batch_features.shape[1:]) != feature_shape:
                raise RuntimeError(
                    f"unexpected CUB frozen feature shape: {batch_features.shape}"
                )
            end = offset + len(batch_features)
            features[offset:end] = batch_features
            offset = end
            observed_ids.extend(str(image_id) for image_id in batch_ids)
            if batch_index % 25 == 0 or offset == len(records):
                log(
                    f"[CUB_CACHE] features {variant} {split}: "
                    f"{offset}/{len(records)}"
                )
    if offset != len(records) or observed_ids != expected_ids:
        raise RuntimeError(f"CUB feature sample order mismatch: {variant}/{split}")
    payload = {"metadata": expected, "features": features}
    _atomic_torch_save(payload, cache_path)
    reloaded = _load_cache(cache_path, expected)
    if reloaded is None or not torch.equal(reloaded["features"], features):
        raise RuntimeError(f"CUB feature cache round-trip failed: {cache_path}")
    return reloaded, True

def _finite_metrics(metrics: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(metrics[name]))
        for name in (
            "foreground_iou",
            "background_iou",
            "mean_iou",
            "foreground_dice",
            "pixel_accuracy",
        )
    )
