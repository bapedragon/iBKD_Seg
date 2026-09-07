#!/usr/bin/env python3
"""Run CUB batch-128 classification -> frozen-probe smoke end to end.

The smoke trains one two-epoch ResNet-56 teacher and six two-epoch DeiT-Tiny
students, then strictly reloads and freezes each student before exercising the
three probe learning-rate candidates for two epochs.  The sole ALG condition
uses the predeclared 20-epoch controller decision warm-up.  Official test images
and masks are never opened by this entry point.
"""

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
from torch.utils.data import DataLoader

from .cub_data import (
    ARCHIVE_MD5,
    DATASET_NAME,
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
    NUM_CLASSES,
    OFFICIAL_TEST_COUNT,
    OFFICIAL_TRAIN_COUNT,
)
from .cub_probe_data import (
    SEGMENTATION_ARCHIVE_MD5,
    CubImageDataset,
    CubProbeRecord,
    ids_sha256,
    load_targets,
    load_train_validation_records,
)
from .models import create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .train_timing import file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/cub200_b128_combined_smoke_v2.json"
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUB H200 smoke requires CUDA")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _validate_config(config: dict[str, Any]) -> None:
    checks = {
        "smoke_id": config.get("smoke_id")
        == "cub200_phase1_b128_classification_to_frozen_probe_smoke_v2",
        "non_scientific": config.get("scientific_result") is False,
        "selection_forbidden": config.get(
            "selection_from_smoke_metrics_forbidden"
        )
        is True,
        "test_forbidden": config.get("official_test_accessed") is False,
        "full_protocol_unlocked": config.get("full_protocol_status") == "not_locked",
        "dataset": config.get("dataset", {}).get("name") == DATASET_NAME,
        "classes": config.get("dataset", {}).get("num_classes") == NUM_CLASSES,
        "official_counts": config.get("dataset", {}).get("official_counts")
        == {
            "train": OFFICIAL_TRAIN_COUNT,
            "test": OFFICIAL_TEST_COUNT,
            "total": OFFICIAL_TRAIN_COUNT + OFFICIAL_TEST_COUNT,
        },
        "derived_counts": config.get("dataset", {}).get("split", {}).get("counts")
        == {
            "train": DERIVED_TRAIN_COUNT,
            "validation": DERIVED_VALIDATION_COUNT,
            "test": 0,
        },
        "archive_md5s": config.get("dataset", {}).get("image_archive", {}).get(
            "md5"
        )
        == ARCHIVE_MD5
        and config.get("dataset", {}).get("segmentation_archive", {}).get("md5")
        == SEGMENTATION_ARCHIVE_MD5,
        "split": config.get("dataset", {}).get("split", {}).get(
            "validation_per_class"
        )
        == 3
        and config.get("dataset", {}).get("split", {}).get("split_seed") == 2027,
        "mask_threshold": config.get("dataset", {}).get("mask", {}).get(
            "foreground_rule"
        )
        == "grayscale_value_greater_than_0",
        "classification_epochs": config.get("classification", {}).get(
            "actual_epochs"
        )
        == 2,
        "batch128": config.get("classification", {}).get("student", {}).get(
            "batch_size"
        )
        == 128,
        "teacher": config.get("classification", {}).get("teacher", {}).get(
            "architecture"
        )
        == "cifar_style_resnet56_6n_plus_2_n9",
        "variants": tuple(config.get("classification", {}).get("variants", ()))
        == EXPECTED_VARIANTS,
        "alg_warmup20_only": config.get("classification", {}).get(
            "controller", {}
        ).get("alg_warmup_epochs")
        == 20
        and config.get("classification", {}).get("controller", {}).get(
            "canonical_alg_warmup0_included"
        )
        is False,
        "ibkd_warmup": config.get("classification", {}).get(
            "controller", {}
        ).get("ibkd_warmup_epochs")
        == 20,
        "probe_lrs": config.get("frozen_probe", {}).get("probe", {}).get(
            "learning_rates"
        )
        == [0.01, 0.03, 0.1],
        "probe_epochs": config.get("frozen_probe", {}).get("probe", {}).get(
            "epochs"
        )
        == 2,
        "probe_seed": config.get("frozen_probe", {}).get("probe", {}).get(
            "probe_seeds"
        )
        == [1],
        "task_count": config.get("task_count")
        == {
            "teacher": 1,
            "classification_students": 6,
            "probe_lr_candidates": 18,
            "selected_smoke_probes": 6,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid CUB combined smoke config: " + ", ".join(failures))


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    classification_complete: int,
    probe_candidates_complete: int,
    selected_probes_complete: int,
    active_variant: str | None = None,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "classification_complete": classification_complete,
            "classification_expected": 6,
            "probe_candidates_complete": probe_candidates_complete,
            "probe_candidates_expected": 18,
            "selected_probes_complete": selected_probes_complete,
            "selected_probes_expected": 6,
            "active_variant": active_variant,
            "scientific_result": False,
            "official_test_accessed": False,
            "failure": failure,
        },
        output_dir / "sequence_status.json",
    )


def _run_command(command: list[str], *, label: str) -> None:
    log(f"[CUB_SMOKE_TASK_START] {label} command={' '.join(command)}")
    subprocess.run(command, check=True)
    log(f"[CUB_SMOKE_TASK_DONE] {label}")


def _complete_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete smoke task summary: {path}")
    if payload.get("scientific_result") is not False:
        raise RuntimeError(f"smoke task marked scientific: {path}")
    if payload.get("official_test_accessed") is not False:
        raise RuntimeError(f"smoke task accessed official test: {path}")
    return payload


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


def _load_encoder(
    checkpoint_path: Path,
    summary: dict[str, Any],
    *,
    variant: str,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    method, _, controller_warmup_epochs = VARIANT_ARGUMENTS[variant]
    alg_controller_warmup_epochs = (
        int(controller_warmup_epochs) if method == "alg" else 0
    )
    expected = {
        "purpose": "phase1_cub_combined_smoke_student",
        "scientific_result": False,
        "official_test_accessed": False,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": method,
        "batch_size": 128,
        "seed": 1,
        "actual_epochs": 2,
        "planned_epochs": 300,
        "controller_warmup_epochs": alg_controller_warmup_epochs,
        "guidance_controller_warmup_epochs": controller_warmup_epochs,
        "validation_image_ids_sha256": validation_hash,
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"CUB smoke student checkpoint mismatch {variant}:{key}: "
                f"expected={value!r} got={metadata.get(key)!r}"
            )
    if summary.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise RuntimeError(f"CUB smoke checkpoint SHA mismatch: {variant}")
    student = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = student.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict encoder load failed: {variant}")
    state_hash = state_dict_sha256(student)
    if state_hash != metadata.get("student_state_sha256"):
        raise RuntimeError(f"CUB smoke student state hash mismatch: {variant}")
    student.to(device).eval()
    student.requires_grad_(False)
    audit = {
        "strict_load": True,
        "eval_mode": not student.training,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in student.parameters()
            if parameter.requires_grad
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "student_state_sha256": state_hash,
    }
    if not (
        audit["eval_mode"]
        and audit["trainable_parameter_count"] == 0
        and audit["strict_load"]
    ):
        raise RuntimeError(f"CUB frozen encoder contract failed: {variant}")
    return student, audit


def _feature_cache(
    model: torch.nn.Module,
    records: Sequence[CubProbeRecord],
    *,
    split: str,
    variant: str,
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
        "encoder_seed": 1,
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


def _write_classification_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "controller_warmup_epochs",
        "validation_macro_top1_epoch2",
        "validation_overall_top1_epoch2",
        "avg_epoch_seconds",
        "peak_cuda_memory_bytes",
        "checkpoint_sha256",
        "scientific_result",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            summary = row["summary"]
            final_epoch = summary["epochs"][-1]
            writer.writerow(
                {
                    "variant": row["variant"],
                    "controller_warmup_epochs": row[
                        "controller_warmup_epochs"
                    ],
                    "validation_macro_top1_epoch2": final_epoch["validation"][
                        "macro_top1"
                    ],
                    "validation_overall_top1_epoch2": final_epoch["validation"][
                        "overall_top1"
                    ],
                    "avg_epoch_seconds": summary["avg_epoch_seconds"],
                    "peak_cuda_memory_bytes": max(
                        int(epoch.get("peak_cuda_memory_bytes") or 0)
                        for epoch in summary["epochs"]
                    ),
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "scientific_result": False,
                }
            )
    temporary.replace(path)


def _write_probe_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "selected_learning_rate",
        "selected_epoch",
        "validation_grid_mean_iou",
        "validation_input_224_mean_iou",
        "validation_input_224_foreground_iou",
        "validation_input_224_background_iou",
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
        for row in rows:
            input_metrics = row["validation"]["input_224"]
            writer.writerow(
                {
                    "variant": row["variant"],
                    "selected_learning_rate": row["selection"]["learning_rate"],
                    "selected_epoch": row["selection"]["epoch"],
                    "validation_grid_mean_iou": row["validation"]["grid_14x14"][
                        "mean_iou"
                    ],
                    "validation_input_224_mean_iou": input_metrics["mean_iou"],
                    "validation_input_224_foreground_iou": input_metrics[
                        "foreground_iou"
                    ],
                    "validation_input_224_background_iou": input_metrics[
                        "background_iou"
                    ],
                    "feature_cache_seconds": row["timing"]["feature_cache_seconds"],
                    "probe_training_seconds": row["timing"][
                        "probe_training_seconds"
                    ],
                    "peak_cuda_memory_bytes": row["peak_cuda_memory_bytes"],
                    "scientific_result": False,
                }
            )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.num_workers < 0 or args.feature_batch_size <= 0:
        raise ValueError("invalid CUB smoke loader settings")
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUB combined smoke is an H200 CUDA check")
    config = _load_json(args.config)
    _validate_config(config)
    config_sha256 = file_sha256(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        args.output_dir,
        status="running",
        phase="dataset_and_mask_setup",
        classification_complete=0,
        probe_candidates_complete=0,
        selected_probes_complete=0,
    )

    log("=" * 96)
    log("CUB-200-2011 PHASE 1 — BATCH 128 CLASSIFICATION TO FROZEN PROBE SMOKE")
    log("=" * 96)
    log(
        "[CUB_COMBINED_SMOKE_POLICY] scientific_result=false "
        "selection_from_smoke_metrics=forbidden official_test_accessed=false "
        "full_protocol_status=not_locked"
    )

    records, split_manifest, source = load_train_validation_records(
        args.data_dir,
        download=True,
    )
    counts = {split: len(values) for split, values in records.items()}
    if counts != {
        "train": DERIVED_TRAIN_COUNT,
        "validation": DERIVED_VALIDATION_COUNT,
    }:
        raise RuntimeError(f"unexpected CUB smoke record counts: {counts}")
    validation_hash = split_manifest["validation_image_ids_sha256"]
    _atomic_json_save(
        {
            "status": "pass",
            "source": source,
            "split_manifest": split_manifest,
            "counts": {**counts, "test": 0},
            "mask_pairs_present": DERIVED_TRAIN_COUNT + DERIVED_VALIDATION_COUNT,
            "official_test_images_or_masks_opened": False,
        },
        args.output_dir / "data_smoke_audit.json",
    )
    log(
        "[CUB_SMOKE_DATA] source=official_train train=5394 validation=600 "
        f"test=0 val_per_class=3 split_seed=2027 split_sha256={validation_hash}"
    )

    classification_root = args.output_dir / "classification"
    teacher_root = classification_root / "teacher"
    teacher_name = "cub_teacher_resnet56_32_b128_smoke_2ep_seed1"
    teacher_dir = teacher_root / teacher_name
    teacher_checkpoint = teacher_dir / "timing_teacher_latest.pt"
    teacher_command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_timing",
        "--timing-run",
        "--dataset",
        "cub",
        "--kind",
        "teacher",
        "--batch-size",
        "128",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(teacher_root),
        "--run-name",
        teacher_name,
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--seed",
        "1",
    ]
    _run_command(teacher_command, label="teacher_resnet56_seed1")
    teacher_summary = _complete_summary(teacher_dir / "summary.json")
    if (
        teacher_summary.get("dataset") != DATASET_NAME
        or teacher_summary.get("split_manifest", {}).get(
            "validation_image_ids_sha256"
        )
        != validation_hash
        or not teacher_checkpoint.is_file()
    ):
        raise RuntimeError("CUB smoke teacher contract failed")

    student_root = classification_root / "students"
    classification_rows: list[dict[str, Any]] = []
    for variant in EXPECTED_VARIANTS:
        _write_status(
            args.output_dir,
            status="running",
            phase="classification_students",
            classification_complete=len(classification_rows),
            probe_candidates_complete=0,
            selected_probes_complete=0,
            active_variant=variant,
        )
        method, fusion_ratio, controller_warmup_epochs = VARIANT_ARGUMENTS[variant]
        run_name = f"cub_{variant}_deit_tiny_b128_smoke_2ep_seed1"
        run_dir = student_root / run_name
        command = [
            sys.executable,
            "-m",
            "ibkd_seg.phase1.train_timing",
            "--timing-run",
            "--dataset",
            "cub",
            "--kind",
            "student",
            "--method",
            method,
            "--batch-size",
            "128",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(student_root),
            "--run-name",
            run_name,
            "--num-workers",
            str(args.num_workers),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--seed",
            "1",
            "--save-student-checkpoint",
        ]
        if method != "vanilla":
            command.extend(["--teacher-checkpoint", str(teacher_checkpoint)])
        if fusion_ratio is not None:
            command.extend(["--fusion-ratio", str(fusion_ratio)])
        if method == "alg":
            assert controller_warmup_epochs is not None
            command.extend(
                [
                    "--alg-controller-warmup-epochs",
                    str(controller_warmup_epochs),
                ]
            )
        _run_command(command, label=f"classification_{variant}")
        summary_path = run_dir / "summary.json"
        summary = _complete_summary(summary_path)
        checkpoint_path = run_dir / "timing_student_latest.pt"
        expected_values = {
            "dataset": DATASET_NAME,
            "method": method,
            "batch_size": 128,
            "seed": 1,
            "fusion_ratio_lambda": fusion_ratio,
            "actual_epochs": 2,
            "alg_controller_warmup_epochs": (
                controller_warmup_epochs if method == "alg" else 0
            ),
            "guidance_controller_warmup_epochs": controller_warmup_epochs,
        }
        for key, value in expected_values.items():
            if summary.get(key) != value:
                raise RuntimeError(
                    f"CUB smoke classification mismatch {variant}:{key}"
                )
        if (
            summary["split_manifest"]["validation_image_ids_sha256"]
            != validation_hash
            or not checkpoint_path.is_file()
            or summary.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        ):
            raise RuntimeError(f"CUB smoke student artifact failed: {variant}")
        classification_rows.append(
            {
                "variant": variant,
                "method": method,
                "fusion_ratio_lambda": fusion_ratio,
                "controller_warmup_epochs": controller_warmup_epochs,
                "summary_path": str(summary_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "summary": summary,
            }
        )

    initial_hashes = {
        row["summary"]["initial_student_state_sha256"]
        for row in classification_rows
    }
    guided_teacher_hashes = {
        row["summary"]["teacher_checkpoint_sha256"]
        for row in classification_rows
        if row["method"] != "vanilla"
    }
    if len(initial_hashes) != 1 or guided_teacher_hashes != {
        teacher_summary["checkpoint_sha256"]
    }:
        raise RuntimeError("CUB smoke paired initialization/shared teacher failed")
    _write_classification_csv(
        classification_rows,
        args.output_dir / "classification_smoke_results.csv",
    )

    target_started = time.monotonic()
    targets: dict[str, dict[str, Any]] = {}
    target_reload_checks: list[bool] = []
    for split in ("train", "validation"):
        targets[split], reloaded = _target_cache(
            records[split],
            split=split,
            config=config,
            config_sha256=config_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
        target_reload_checks.append(reloaded)
    target_seconds = time.monotonic() - target_started
    target_values = {
        int(value)
        for payload in targets.values()
        for key in ("input_targets", "grid_targets")
        for value in torch.unique(payload[key])
    }
    if target_values != {0, 1}:
        raise RuntimeError(f"CUB binary target values changed: {target_values}")

    probe_config = config["frozen_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    probe_seed = int(probe_config["probe_seeds"][0])
    probe_epochs = int(probe_config["epochs"])
    probe_rows: list[dict[str, Any]] = []
    feature_reload_checks: list[bool] = []
    initial_probe_hashes: set[str] = set()
    batch_orders_by_epoch = {
        epoch: set() for epoch in range(1, probe_epochs + 1)
    }
    log(
        "[CUB_PROBE_SMOKE_TASK_COUNT] encoders=6 lr_candidates=18 "
        "epochs_per_candidate=2 probe_seed=1"
    )
    for classification in classification_rows:
        variant = classification["variant"]
        _write_status(
            args.output_dir,
            status="running",
            phase="frozen_probe",
            classification_complete=6,
            probe_candidates_complete=len(probe_rows) * len(learning_rates),
            selected_probes_complete=len(probe_rows),
            active_variant=variant,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        checkpoint_path = Path(classification["checkpoint_path"])
        model, encoder_audit = _load_encoder(
            checkpoint_path,
            classification["summary"],
            variant=variant,
            validation_hash=validation_hash,
            device=device,
        )
        feature_started = time.monotonic()
        features: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation"):
            features[split], reloaded = _feature_cache(
                model,
                records[split],
                split=split,
                variant=variant,
                checkpoint_sha256=encoder_audit["checkpoint_sha256"],
                state_sha256=encoder_audit["student_state_sha256"],
                config=config,
                config_sha256=config_sha256,
                cache_path=args.cache_dir
                / "features"
                / variant
                / f"{split}.pt",
                device=device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
            feature_reload_checks.append(reloaded)
        _synchronize(device)
        feature_seconds = time.monotonic() - feature_started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        probe_started = time.monotonic()
        candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        for learning_rate in learning_rates:
            state, candidate = train_candidate(
                features["train"]["features"],
                targets["train"]["grid_targets"],
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                probe_config=probe_config,
                learning_rate=learning_rate,
                seed=probe_seed,
                device=device,
                epochs=probe_epochs,
            )
            candidates.append((state, candidate))
            initial_probe_hashes.add(candidate["initial_probe_state_sha256"])
            for epoch, digest in enumerate(
                candidate["batch_order_sha256_by_epoch"], start=1
            ):
                batch_orders_by_epoch[epoch].add(digest)
            log(
                f"[CUB_PROBE_CANDIDATE] variant={variant} "
                f"completed={len(candidates)}/3 lr={learning_rate:g} "
                f"best_epoch={candidate['best_epoch']} "
                "validation_grid_miou="
                f"{candidate['best_validation_grid_mean_iou']:.6f} "
                "scientific_result=false"
            )
        selected_index = max(
            range(len(candidates)),
            key=lambda index: candidates[index][1][
                "best_validation_grid_mean_iou"
            ],
        )
        selected_state, selected = candidates[selected_index]
        probe = probe_from_state(probe_config, probe_seed, selected_state, device)
        validation_metrics, _ = evaluate_probe_both_resolutions(
            probe,
            features["validation"]["features"],
            targets["validation"]["grid_targets"],
            targets["validation"]["input_targets"],
            batch_size=int(probe_config["batch_size"]),
            device=device,
            input_size=int(config["frozen_probe"]["image_input"]["size"]),
            ignore_index=int(probe_config["loss"]["ignore_index"]),
        )
        _synchronize(device)
        probe_seconds = time.monotonic() - probe_started
        probe_path = args.output_dir / "probes" / f"{variant}_seed1_smoke.pt"
        _atomic_torch_save(
            {
                "purpose": "phase1_cub_frozen_probe_smoke_only",
                "scientific_result": False,
                "official_test_accessed": False,
                "config_sha256": config_sha256,
                "variant": variant,
                "encoder_seed": 1,
                "encoder_checkpoint_sha256": encoder_audit[
                    "checkpoint_sha256"
                ],
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
                "model": selected_state,
            },
            probe_path,
        )
        saved_probe = torch.load(probe_path, map_location="cpu", weights_only=True)
        strict_probe = probe_from_state(
            probe_config,
            probe_seed,
            saved_probe["model"],
            device,
        )
        del strict_probe, probe
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        result = {
            "variant": variant,
            "method": classification["method"],
            "fusion_ratio_lambda": classification["fusion_ratio_lambda"],
            "encoder_seed": 1,
            "encoder_audit": encoder_audit,
            "probe_seed": probe_seed,
            "candidates": [candidate for _, candidate in candidates],
            "selection": {
                "split": "validation",
                "learning_rate": selected["learning_rate"],
                "epoch": selected["best_epoch"],
                "validation_grid_mean_iou": selected[
                    "best_validation_grid_mean_iou"
                ],
                "smoke_plumbing_only": True,
            },
            "validation": validation_metrics,
            "probe_artifact": str(probe_path.resolve()),
            "probe_artifact_sha256": file_sha256(probe_path),
            "selected_probe_strict_reloaded": True,
            "timing": {
                "feature_cache_seconds": feature_seconds,
                "probe_training_seconds": probe_seconds,
            },
            "peak_cuda_memory_bytes": peak_memory,
            "scientific_result": False,
        }
        if not all(
            _finite_metrics(metrics) for metrics in validation_metrics.values()
        ):
            raise RuntimeError(f"non-finite CUB probe metrics: {variant}")
        probe_rows.append(result)
        del features, candidates
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_probe_csv(probe_rows, args.output_dir / "probe_smoke_results.csv")
    classification_estimate_seconds = sum(
        float(row["summary"]["avg_epoch_seconds"]) * 300 * 3
        for row in classification_rows
    )
    teacher_estimate_seconds = float(teacher_summary["avg_epoch_seconds"]) * 300
    probe_estimate_seconds = sum(
        float(row["timing"]["probe_training_seconds"])
        * (int(probe_config["planned_epochs"]) / probe_epochs)
        * 15
        for row in probe_rows
    )
    contracts = {
        "config_validated": True,
        "non_scientific": True,
        "official_test_not_accessed": True,
        "full_protocol_still_unlocked": True,
        "official_train_derived_counts_5394_600": counts
        == {"train": 5394, "validation": 600},
        "classification_and_probe_split_match": all(
            row["summary"]["split_manifest"]["validation_image_ids_sha256"]
            == validation_hash
            for row in classification_rows
        ),
        "teacher_completed": teacher_summary["status"] == "complete",
        "six_classification_students_completed": len(classification_rows) == 6,
        "same_initial_student_state_across_variants": len(initial_hashes) == 1,
        "one_teacher_shared_by_guided_students": len(guided_teacher_hashes) == 1,
        "binary_masks_only": target_values == {0, 1},
        "target_cache_safe_reload": all(target_reload_checks),
        "encoder_strict_loaded_frozen_eval": all(
            row["encoder_audit"]["strict_load"]
            and row["encoder_audit"]["eval_mode"]
            and row["encoder_audit"]["trainable_parameter_count"] == 0
            for row in probe_rows
        ),
        "feature_cache_safe_reload": all(feature_reload_checks),
        "eighteen_probe_lr_candidates_completed": len(probe_rows) * 3 == 18,
        "same_probe_initial_state_across_all_candidates": len(
            initial_probe_hashes
        )
        == 1,
        "same_probe_batch_order_across_all_candidates": all(
            len(values) == 1 for values in batch_orders_by_epoch.values()
        ),
        "six_selected_probes_strict_reloaded": len(probe_rows) == 6
        and all(row["selected_probe_strict_reloaded"] for row in probe_rows),
        "finite_validation_metrics": all(
            all(_finite_metrics(metrics) for metrics in row["validation"].values())
            for row in probe_rows
        ),
    }
    status = "pass" if all(contracts.values()) else "fail"
    elapsed_seconds = time.monotonic() - started
    summary = {
        "status": status,
        "completed_at_utc": _utc_now(),
        "smoke_id": config["smoke_id"],
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "full_protocol_status": "not_locked",
        "config": {
            "path": str(args.config),
            "sha256": config_sha256,
        },
        "data": {
            "source": source,
            "counts": {**counts, "test": 0},
            "validation_image_ids_sha256": validation_hash,
            "target_preparation_seconds": target_seconds,
        },
        "teacher": teacher_summary,
        "classification": classification_rows,
        "frozen_probe": probe_rows,
        "contracts": {"all_passed": status == "pass", **contracts},
        "timing": {
            "smoke_suite_seconds": elapsed_seconds,
            "rough_full_estimate": {
                "teacher_1x300ep_seconds": teacher_estimate_seconds,
                "classification_6variants_x3seeds_x300ep_seconds": (
                    classification_estimate_seconds
                ),
                "probe_6variants_x3encoders_x5probe_seeds_x3lr_x100ep_seconds": (
                    probe_estimate_seconds
                ),
                "warning": "linear_extrapolation_for_job_partitioning_only",
            },
        },
        "runtime": _runtime(device),
    }
    summary_path = args.output_dir / "combined_smoke_summary.json"
    _atomic_json_save(summary, summary_path)
    _write_status(
        args.output_dir,
        status=status,
        phase="complete",
        classification_complete=len(classification_rows),
        probe_candidates_complete=len(probe_rows) * 3,
        selected_probes_complete=len(probe_rows),
    )

    log("[CUB_SMOKE_FINAL_CLASSIFICATION_RESULTS]")
    for row in classification_rows:
        final_epoch = row["summary"]["epochs"][-1]
        log(
            f"[CUB_CLASSIFICATION_SMOKE_RESULT] variant={row['variant']} "
            f"epoch2_val_macro_top1={final_epoch['validation']['macro_top1']:.4f} "
            f"epoch2_val_overall_top1={final_epoch['validation']['overall_top1']:.4f} "
            f"avg_epoch_seconds={row['summary']['avg_epoch_seconds']:.3f} "
            "scientific_result=false"
        )
        controller = row["summary"].get("controller")
        if controller is not None:
            log(
                f"[CUB_CONTROLLER_SMOKE] variant={row['variant']} "
                f"kind={controller['kind']} warmup={controller['warmup_epochs']} "
                f"active={str(controller['active']).lower()} "
                f"stop_epoch={controller['stop_epoch']} "
                f"beta_history={controller['beta_history']}"
            )
    log("[CUB_SMOKE_FINAL_PROBE_RESULTS]")
    for row in probe_rows:
        log(
            f"[CUB_PROBE_SMOKE_RESULT] variant={row['variant']} "
            f"selected_lr={row['selection']['learning_rate']:g} "
            f"selected_epoch={row['selection']['epoch']} "
            "validation_grid_miou="
            f"{row['validation']['grid_14x14']['mean_iou']:.6f} "
            "validation_input_miou="
            f"{row['validation']['input_224']['mean_iou']:.6f} "
            "scientific_result=false"
        )
    log(
        "[CUB_ROUGH_FULL_ESTIMATE] teacher="
        f"{format_duration(teacher_estimate_seconds)} classification="
        f"{format_duration(classification_estimate_seconds)} probe="
        f"{format_duration(probe_estimate_seconds)} "
        "linear_extrapolation_only=true"
    )
    log(
        f"[CUB_COMBINED_SMOKE_DONE] status={status} teacher=1/1 "
        f"classification={len(classification_rows)}/6 "
        f"probe_candidates={len(probe_rows) * 3}/18 "
        f"selected_probes={len(probe_rows)}/6 tasks=25/25 scientific_result=false "
        f"official_test_accessed=false seconds={elapsed_seconds:.2f} "
        f"summary={summary_path.resolve()}"
    )
    if status != "pass":
        raise RuntimeError("CUB combined smoke failed one or more contracts")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            previous = _load_json(args.output_dir / "sequence_status.json")
        except Exception:
            previous = {}
        _write_status(
            args.output_dir,
            status="failed",
            phase="failed",
            classification_complete=int(
                previous.get("classification_complete", 0)
            ),
            probe_candidates_complete=int(
                previous.get("probe_candidates_complete", 0)
            ),
            selected_probes_complete=int(
                previous.get("selected_probes_complete", 0)
            ),
            active_variant=previous.get("active_variant"),
            failure=f"{type(error).__name__}: {error}",
        )
        log(f"[CUB_COMBINED_SMOKE_FAILED] {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
