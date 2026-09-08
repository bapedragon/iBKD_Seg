#!/usr/bin/env python3
"""Train the locked CUB Phase 1 ResNet-50/224 scratch teacher.

This entry point is intentionally separate from the legacy ResNet-56/32
trainer.  It accepts no hyperparameter overrides: every scientific setting is
read from, and checked against, the locked v3 protocol before data or CUDA
training is started.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cub_data import (
    DATASET_NAME,
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
    NUM_CLASSES,
    OFFICIAL_TEST_COUNT,
    build_resnet50_official_test_loader,
    build_resnet50_train_validation_loaders,
)
from .data import save_json
from .models import ResNet50CUB
from .train_timing import (
    atomic_torch_save,
    file_sha256,
    format_duration,
    runtime_metadata,
    seed_everything,
    state_dict_sha256,
    synchronize,
    teacher_parameter_groups,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/cub200_r50_224_b128_full_v3.json"
)
EXPECTED_CONFIG_SHA256 = (
    "e3faff49101a8cffc5d0836f2cf299177547cea5243715ce51cc288b743626dc"
)
EXPECTED_VALIDATION_HASH = (
    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
)
CHECKPOINT_PURPOSE = "phase1_cub_r50_224_scientific_teacher_v3"


def log(message: str = "") -> None:
    print(message, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-teacher", action="store_true", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _validate_config(config: dict[str, Any], *, config_path: Path) -> None:
    actual_config_hash = file_sha256(config_path)
    classification = config.get("classification", {})
    teacher = classification.get("teacher", {})
    split = config.get("dataset", {}).get("split", {})
    test_policy = classification.get("test_policy", {})
    expected_teacher = {
        "architecture": "torchvision_resnet50",
        "initialization": "scratch",
        "external_pretraining": False,
        "backbone_initialization": "torchvision_default_random",
        "classification_head_initialization": {
            "weight": "normal",
            "weight_std": 0.01,
            "bias": 0.0,
        },
        "input_size": 224,
        "seed": 1,
        "epochs": 200,
        "train_batch_size": 128,
        "eval_batch_size": 200,
        "optimizer": {
            "name": "sgd",
            "learning_rate": 0.05,
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 0.0001,
            "weight_decay_exclusions": ["bias", "normalization_parameters"],
        },
        "scheduler": {
            "name": "linear_warmup_then_cosine",
            "warmup_epochs": 5,
            "warmup_learning_rate": 0.0,
            "minimum_learning_rate": 0.0,
        },
        "train_transform": {
            "random_resized_crop": 224,
            "interpolation": "bicubic",
            "horizontal_flip_probability": 0.5,
            "normalization": "imagenet",
        },
        "evaluation_transform": {
            "resize_shorter_side": 256,
            "center_crop": 224,
            "interpolation": "bicubic",
            "normalization": "imagenet",
        },
        "feature_stages": ["layer2", "layer3", "layer4"],
        "feature_channels": [512, 1024, 2048],
        "feature_spatial_sizes": [28, 14, 7],
        "selection_metric": "validation_macro_top1",
        "selection_tie_break": "earlier_epoch",
        "one_fixed_checkpoint_shared_by_all_guided_students": True,
    }
    checks = {
        "config_sha256": actual_config_hash == EXPECTED_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id")
        == "cub200_phase1_resnet50_224_scratch_b128_full_v3",
        "locked_before_results": str(config.get("status", "")).startswith(
            "locked_before_v3_smoke_and_full_results"
        ),
        "scientific_result": config.get("scientific_result") is True,
        "unconditional_matrix": config.get("matrix_commitment")
        == {
            "unconditional": True,
            "variants": [
                "vanilla",
                "kd",
                "lg",
                "alg_warmup20",
                "ibkd_lambda_0.25",
                "ibkd_lambda_0.5",
            ],
            "encoder_seeds": [1, 2, 3],
            "configuration_changes_after_smoke": False,
            "method_or_lambda_selection_from_smoke_or_test": False,
        },
        "dataset": config.get("dataset", {}).get("name") == DATASET_NAME
        and config.get("dataset", {}).get("num_classes") == NUM_CLASSES,
        "split": split.get("expected_counts")
        == {
            "train": DERIVED_TRAIN_COUNT,
            "validation": DERIVED_VALIDATION_COUNT,
            "test": OFFICIAL_TEST_COUNT,
        }
        and split.get("validation_per_class") == 3
        and split.get("split_seed") == 2027
        and split.get("validation_image_ids_sha256")
        == EXPECTED_VALIDATION_HASH,
        "classification_inputs": config.get("dataset", {}).get(
            "classification_encoder_may_use"
        )
        == ["rgb_image", "species_label"]
        and "segmentation_mask"
        in config.get("dataset", {}).get(
            "classification_encoder_must_not_use", []
        ),
        "teacher": teacher == expected_teacher,
        "test_policy": test_policy.get("selection_uses_official_test") is False
        and test_policy.get("evaluation")
        == "once_per_validation_selected_checkpoint"
        and test_policy.get("official_test_reporting_enabled_before_full_matrix")
        is True
        and test_policy.get("no_method_or_lambda_selection_from_test") is True,
        "execution": config.get("execution", {}).get("precision") == "float32"
        and config.get("execution", {}).get("classification_num_workers") == 4
        and config.get("execution", {}).get("classification_eval_batch_size")
        == 200,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid locked CUB ResNet-50 teacher v3 config: "
            + ", ".join(failures)
        )


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct_by_class = torch.zeros(NUM_CLASSES, dtype=torch.long)
    total_by_class = torch.zeros(NUM_CLASSES, dtype=torch.long)
    total_correct = 0
    total_top5 = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        matches = logits.argmax(dim=1).eq(targets)
        top5_matches = logits.topk(5, dim=1).indices.eq(targets[:, None]).any(dim=1)
        total_correct += int(matches.sum())
        total_top5 += int(top5_matches.sum())
        total += targets.numel()
        cpu_targets = targets.detach().cpu()
        total_by_class += torch.bincount(cpu_targets, minlength=NUM_CLASSES)
        correct_by_class += torch.bincount(
            cpu_targets[matches.detach().cpu()], minlength=NUM_CLASSES
        )
    if total != len(loader.dataset):
        raise RuntimeError(
            f"evaluation count mismatch: evaluated={total} dataset={len(loader.dataset)}"
        )
    if bool((total_by_class == 0).any()):
        raise RuntimeError("evaluation split is missing at least one CUB class")
    macro = (correct_by_class.float() / total_by_class.float()).mean().item()
    return {
        "overall_top1": 100.0 * total_correct / total,
        "macro_top1": 100.0 * macro,
        "top5": 100.0 * total_top5 / total,
        "samples": total,
    }


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    rows = [
        {
            "epoch": row["epoch"],
            "learning_rate": row["learning_rate"],
            "train_loss": row["train_loss"],
            "train_top1": row["train_top1"],
            "validation_overall_top1": row["validation"]["overall_top1"],
            "validation_macro_top1": row["validation"]["macro_top1"],
            "validation_top5": row["validation"]["top5"],
            "seconds_including_validation": row[
                "seconds_including_validation"
            ],
            "peak_cuda_memory_bytes": row["peak_cuda_memory_bytes"],
            "peak_cuda_memory_reserved_bytes": row[
                "peak_cuda_memory_reserved_bytes"
            ],
        }
        for row in history
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_progress(
    path: Path,
    *,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_validation: dict[str, float],
    started_at: str,
) -> None:
    save_json(
        {
            "status": "training",
            "protocol_id": "cub200_phase1_resnet50_224_scratch_b128_full_v3",
            "scientific_result": True,
            "teacher": "torchvision_resnet50_224_scratch",
            "started_at_utc": started_at,
            "updated_at_utc": _utc_now(),
            "completed_epochs": len(history),
            "planned_epochs": 200,
            "best_validation_epoch": best_epoch,
            "best_validation": best_validation,
            "latest_epoch": history[-1],
            "official_test_evaluations": 0,
        },
        path,
    )


def load_scientific_teacher(
    checkpoint_path: Path,
    *,
    device: torch.device,
    validation_hash: str = EXPECTED_VALIDATION_HASH,
    config_sha256: str = EXPECTED_CONFIG_SHA256,
) -> tuple[ResNet50CUB, dict[str, Any], str, str]:
    """Strictly load and freeze the single teacher shared by v3 students."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("CUB v3 teacher checkpoint payload is malformed")
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": CHECKPOINT_PURPOSE,
        "scientific_result": True,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "torchvision_resnet50",
        "initialization": "scratch",
        "external_pretraining": False,
        "input_size": 224,
        "epochs": 200,
        "seed": 1,
        "batch_size": 128,
        "selection_metric": "validation_macro_top1",
        "selection_tie_break": "earlier_epoch",
        "validation_image_ids_sha256": validation_hash,
        "full_config_sha256": config_sha256,
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    failures = [
        key for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    ]
    if failures:
        raise RuntimeError(
            "CUB v3 teacher checkpoint contract mismatch: " + ", ".join(failures)
        )
    model = ResNet50CUB(num_classes=NUM_CLASSES)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("CUB v3 teacher strict state load failed")
    state_hash = state_dict_sha256(model)
    if state_hash != metadata.get("model_state_sha256"):
        raise RuntimeError("CUB v3 teacher model-state SHA-256 mismatch")
    model.to(device).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CUB v3 shared teacher was not completely frozen")
    return model, metadata, file_sha256(checkpoint_path), state_hash


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_workers != 4:
        raise ValueError("locked CUB v3 teacher requires num-workers=4")
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
    _validate_config(config, config_path=config_path)

    import timm

    if timm.__version__ != "1.0.27":
        raise RuntimeError(f"expected timm==1.0.27, found {timm.__version__}")

    if not torch.cuda.is_available():
        raise RuntimeError("locked CUB v3 teacher full run requires CUDA")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    seed_everything(1)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "training_status.json"
    checkpoint_path = output_dir / "teacher_best_validation.pt"
    started_at = _utc_now()
    started = time.perf_counter()

    model = ResNet50CUB(num_classes=NUM_CLASSES).to(device)
    initial_state_hash = state_dict_sha256(model)
    train_loader, validation_loader, split_manifest = (
        build_resnet50_train_validation_loaders(
            args.data_dir,
            train_batch_size=128,
            eval_batch_size=200,
            num_workers=args.num_workers,
            seed=1,
            device=device,
        )
    )
    if (
        len(train_loader.dataset) != DERIVED_TRAIN_COUNT
        or len(validation_loader.dataset) != DERIVED_VALIDATION_COUNT
        or split_manifest.get("validation_image_ids_sha256")
        != EXPECTED_VALIDATION_HASH
        or train_loader.batch_size != 128
        or not train_loader.drop_last
        or validation_loader.batch_size != 200
        or validation_loader.drop_last
    ):
        raise RuntimeError("constructed CUB ResNet-50 loaders violate locked v3 contract")

    save_json(split_manifest, output_dir / "validation_split.json")
    save_json(
        {
            "protocol_config": str(config_path),
            "protocol_config_sha256": EXPECTED_CONFIG_SHA256,
            "protocol": config,
        },
        output_dir / "protocol_snapshot.json",
    )

    optimizer = torch.optim.SGD(
        teacher_parameter_groups(model, weight_decay=1e-4),
        lr=0.05,
        momentum=0.9,
        nesterov=False,
    )
    from timm.scheduler import CosineLRScheduler

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=200,
        lr_min=0.0,
        warmup_t=5,
        warmup_lr_init=0.0,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro = float("-inf")
    best_validation: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    max_allocated = 0
    max_reserved = 0
    effective_train_samples = len(train_loader) * int(train_loader.batch_size)
    log(
        "[CUB_R50_TEACHER_FULL_START] scientific_result=true "
        "architecture=torchvision_resnet50 initialization=scratch "
        "pretrained=false input=224 batch=128 epochs=200 seed=1 "
        f"train={len(train_loader.dataset)} validation={len(validation_loader.dataset)} "
        f"effective_train_samples_per_epoch={effective_train_samples} "
        f"config_sha256={EXPECTED_CONFIG_SHA256}"
    )

    for epoch in range(1, 201):
        synchronize(device)
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite teacher loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            batch_size = targets.numel()
            total += batch_size
            total_loss += float(loss.detach()) * batch_size
            total_correct += int(logits.argmax(dim=1).eq(targets).sum())
        if total != effective_train_samples:
            raise RuntimeError(
                f"training sample count changed: expected={effective_train_samples} got={total}"
            )
        validation = _evaluate(model, validation_loader, device)
        synchronize(device)
        elapsed = time.perf_counter() - epoch_started
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        max_allocated = max(max_allocated, peak_allocated)
        max_reserved = max(max_reserved, peak_reserved)
        row = {
            "epoch": epoch,
            "learning_rate": epoch_lr,
            "train_loss": total_loss / total,
            "train_top1": 100.0 * total_correct / total,
            "validation": validation,
            "seconds_including_validation": elapsed,
            "peak_cuda_memory_bytes": peak_allocated,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
        }
        history.append(row)
        # Strict improvement preserves the earlier epoch on an exact tie.
        if validation["macro_top1"] > best_macro:
            best_macro = validation["macro_top1"]
            best_epoch = epoch
            best_validation = dict(validation)
            best_state = _clone_state_dict(model)
        assert best_validation is not None
        _write_progress(
            status_path,
            history=history,
            best_epoch=best_epoch,
            best_validation=best_validation,
            started_at=started_at,
        )
        log(
            f"[CUB_R50_TEACHER_EPOCH] {epoch}/200 lr={epoch_lr:.8g} "
            f"loss={total_loss / total:.6f} train_top1={100.0 * total_correct / total:.4f} "
            f"val_top1={validation['overall_top1']:.4f} "
            f"val_macro_top1={validation['macro_top1']:.4f} "
            f"val_top5={validation['top5']:.4f} best_epoch={best_epoch} "
            f"time={elapsed:.2f}s peak_allocated={peak_allocated} "
            f"peak_reserved={peak_reserved}"
        )
        scheduler.step(epoch)

    if best_state is None or best_validation is None:
        raise RuntimeError("teacher training produced no validation selection")
    model.load_state_dict(best_state, strict=True)
    selected_state_hash = state_dict_sha256(model)
    checkpoint_metadata = {
        "purpose": CHECKPOINT_PURPOSE,
        "scientific_result": True,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "torchvision_resnet50",
        "initialization": "scratch",
        "external_pretraining": False,
        "torchvision_weights_argument": None,
        "input_size": 224,
        "epochs": 200,
        "seed": 1,
        "batch_size": 128,
        "selection_metric": "validation_macro_top1",
        "selection_tie_break": "earlier_epoch",
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "validation_image_ids_sha256": EXPECTED_VALIDATION_HASH,
        "full_config_sha256": EXPECTED_CONFIG_SHA256,
        "model_state_sha256": selected_state_hash,
        "one_fixed_checkpoint_shared_by_all_guided_students": True,
        "official_test_policy": "once_after_validation_selection_and_strict_reload",
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    atomic_torch_save(
        {"model": best_state, "metadata": checkpoint_metadata}, checkpoint_path
    )

    del train_loader, validation_loader, optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    selected_model, reloaded_metadata, checkpoint_hash, reloaded_state_hash = (
        load_scientific_teacher(checkpoint_path, device=device)
    )
    if reloaded_state_hash != selected_state_hash:
        raise RuntimeError("strictly reloaded teacher differs from selected state")

    # The untouched official test loader is created only after selection is final.
    test_loader = build_resnet50_official_test_loader(
        args.data_dir,
        eval_batch_size=200,
        num_workers=args.num_workers,
        device=device,
    )
    if len(test_loader.dataset) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("official CUB test count changed")
    test_started = time.perf_counter()
    official_test = _evaluate(selected_model, test_loader, device)
    synchronize(device)
    official_test_seconds = time.perf_counter() - test_started

    _write_history_csv(history, output_dir / "teacher_history.csv")
    total_seconds = time.perf_counter() - started
    summary = {
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "scientific_result": True,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "kind": "teacher",
        "architecture": "torchvision_resnet50",
        "initialization": "scratch",
        "external_pretraining": False,
        "input_size": 224,
        "epochs": 200,
        "seed": 1,
        "train_batch_size": 128,
        "eval_batch_size": 200,
        "train_drop_last": True,
        "train_samples": DERIVED_TRAIN_COUNT,
        "effective_train_samples_per_epoch": effective_train_samples,
        "validation_samples": DERIVED_VALIDATION_COUNT,
        "official_test_samples": OFFICIAL_TEST_COUNT,
        "initial_state_sha256": initial_state_hash,
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "official_test": official_test,
        "official_test_evaluations": 1,
        "official_test_used_for_training_or_selection": False,
        "selected_checkpoint_strict_reloaded": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": selected_state_hash,
        "checkpoint_metadata": reloaded_metadata,
        "full_config": str(config_path),
        "full_config_sha256": EXPECTED_CONFIG_SHA256,
        "split_manifest": split_manifest,
        "training_and_test_seconds": total_seconds,
        "official_test_seconds": official_test_seconds,
        "peak_cuda_memory_bytes": max_allocated,
        "peak_cuda_memory_reserved_bytes": max_reserved,
        "history_file": str(output_dir / "teacher_history.csv"),
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "runtime": runtime_metadata(device),
    }
    save_json(summary, output_dir / "teacher_summary.json")
    save_json(summary, status_path)
    log(
        "[CUB_R50_TEACHER_RESULT] "
        f"selected_epoch={best_epoch} "
        f"validation_top1={best_validation['overall_top1']:.4f} "
        f"validation_macro_top1={best_validation['macro_top1']:.4f} "
        f"official_test_top1={official_test['overall_top1']:.4f} "
        f"official_test_macro_top1={official_test['macro_top1']:.4f} "
        f"official_test_top5={official_test['top5']:.4f}"
    )
    log(
        "[CUB_R50_TEACHER_ARTIFACT] "
        f"checkpoint={checkpoint_path} checkpoint_sha256={checkpoint_hash} "
        f"model_state_sha256={selected_state_hash}"
    )
    log(
        "[CUB_R50_TEACHER_FULL_DONE] status=pass epochs=200 "
        "official_test_evaluations=1 strict_reload=true "
        f"elapsed={format_duration(total_seconds)} "
        f"peak_allocated={max_allocated} peak_reserved={max_reserved}"
    )
    return summary


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "scientific_result": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at_utc": _utc_now(),
        }
        if torch.cuda.is_available():
            failure["peak_cuda_memory_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )
            failure["peak_cuda_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved()
            )
        save_json(failure, output_dir / "failure.json")
        log(
            "[CUB_R50_TEACHER_FULL_FAILED] "
            f"error={type(error).__name__}:{error}"
        )
        raise


if __name__ == "__main__":
    main()
