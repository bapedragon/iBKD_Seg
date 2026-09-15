"""Shared Oxford-IIIT Pet frozen-probe runtime support."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .models import create_student
from .probe import module_state_sha256
from .probe_data import PetImageDataset, PetRecord, ids_sha256, load_targets
from .train_timing import file_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT / "phase1/phase1_pet/configs/oxford_iiit_pet_phase1_v1.json"
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

def _validate_classification_input(
    classification_root: Path,
    *,
    expected_batch_size: int = 64,
    encoder_seeds: Sequence[int] = (1,),
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if expected_batch_size not in (64, 128):
        raise ValueError("expected_batch_size must be 64 or 128")
    summary_path = classification_root / "classification_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"batch-{expected_batch_size} classification summary is missing at "
            f"{summary_path}"
        )
    suite = _load_json(summary_path)
    suite_contracts = suite.get("contracts", {})
    required = {
        "status": suite.get("status") == "complete",
        "scientific_result": suite.get("scientific_result") is True,
        "batch_size": suite.get("batch_size") == expected_batch_size,
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
            f"batch-{expected_batch_size} classification suite failed prerequisites: "
            + ", ".join(failures)
        )

    summary_paths = sorted((classification_root / "students").glob("*/summary.json"))
    if len(summary_paths) != 18:
        raise RuntimeError(
            f"expected 18 batch-{expected_batch_size} student summaries, "
            f"found {len(summary_paths)}"
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
            "batch_size": row.get("batch_size") == expected_batch_size,
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

    requested_seeds = tuple(int(seed) for seed in encoder_seeds)
    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("encoder_seeds must be non-empty and unique")
    if not set(requested_seeds).issubset({1, 2, 3}):
        raise ValueError("Phase 1 encoder seeds must be selected from 1, 2, 3")
    selected_by_key = {
        (entry["variant"], entry["encoder_seed"]): entry
        for entry in all_entries
        if entry["encoder_seed"] in requested_seeds
    }
    expected_selected = {
        (variant, seed) for variant in EXPECTED_VARIANTS for seed in requested_seeds
    }
    if set(selected_by_key) != expected_selected:
        raise RuntimeError("could not resolve the requested classification checkpoints")
    return (
        [
            selected_by_key[(variant, seed)]
            for variant in EXPECTED_VARIANTS
            for seed in requested_seeds
        ],
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
        "batch": metadata.get("batch_size") == row["batch_size"],
        "epochs": metadata.get("epochs") == 300,
        "seed": metadata.get("seed") == entry["encoder_seed"],
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

def _finite_metrics(metrics: dict[str, Any]) -> bool:
    names = (
        "foreground_iou",
        "background_iou",
        "mean_iou",
        "foreground_dice",
        "pixel_accuracy",
    )
    return all(math.isfinite(float(metrics[name])) for name in names)
