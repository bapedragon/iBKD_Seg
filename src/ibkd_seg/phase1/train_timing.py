#!/usr/bin/env python3
"""Run one full-data, two-epoch Phase 1 timing task.

This entry point cannot run a scientific/full experiment.  Official test is
closed by default; the locked CUB v3 smoke may explicitly exercise it after an
unconditional full-matrix commitment.  Every checkpoint and metric remains
timing-only by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .controllers import GuidanceController
from .cub_data import (
    DATASET_NAME as CUB_DATASET_NAME,
    NUM_CLASSES as CUB_NUM_CLASSES,
    build_official_test_loader as build_cub_official_test_loader,
    build_resnet50_official_test_loader,
    build_resnet50_train_validation_loaders,
    build_train_validation_loaders as build_cub_train_validation_loaders,
)
from .data import (
    NUM_CLASSES as PET_NUM_CLASSES,
    build_train_validation_loaders as build_pet_train_validation_loaders,
    save_json,
)
from .models import (
    IBKD,
    LocalityGuidance,
    RESNET50_TEACHER_CHANNELS,
    ResNet50CUB,
    ResNet56,
    create_student,
    forward_student_spatial,
    teacher_view,
)


ACTUAL_EPOCHS = 2
PLANNED_EPOCHS = 300
RESNET50_TEACHER_PLANNED_EPOCHS = 200
METHODS = ("vanilla", "kd", "lg", "alg", "ibkd")
TEACHER_ARCHITECTURES = ("resnet56_32", "resnet50_224_scratch")
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.9


def _dataset_key(args: argparse.Namespace) -> str:
    value = str(getattr(args, "dataset", "pet"))
    if value not in {"pet", "cub"}:
        raise ValueError(f"unsupported Phase 1 timing dataset: {value!r}")
    return value


def _dataset_contract(
    args: argparse.Namespace,
) -> tuple[str, int, Any]:
    if _dataset_key(args) == "cub":
        return CUB_DATASET_NAME, CUB_NUM_CLASSES, build_cub_train_validation_loaders
    return "Oxford-IIIT Pet", PET_NUM_CLASSES, build_pet_train_validation_loaders


def _is_resnet50_teacher(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "teacher_architecture", "resnet56_32"))
        == "resnet50_224_scratch"
    )


def _teacher_architecture_name(args: argparse.Namespace) -> str:
    if _is_resnet50_teacher(args):
        return "torchvision_resnet50_224_scratch"
    return "cifar_style_resnet56_6n_plus_2_n9"


def _teacher_image_size(args: argparse.Namespace) -> int:
    return 224 if _is_resnet50_teacher(args) else 32


def _teacher_channels(args: argparse.Namespace) -> tuple[int, int, int]:
    return RESNET50_TEACHER_CHANNELS if _is_resnet50_teacher(args) else (16, 32, 64)


def _teacher_planned_epochs(args: argparse.Namespace) -> int:
    return RESNET50_TEACHER_PLANNED_EPOCHS if _is_resnet50_teacher(args) else PLANNED_EPOCHS


def _teacher_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if _is_resnet50_teacher(args):
        return ResNet50CUB(num_classes=num_classes)
    return ResNet56(num_classes=num_classes)


def _teacher_inputs(args: argparse.Namespace, images: torch.Tensor) -> torch.Tensor:
    return teacher_view(images, image_size=_teacher_image_size(args))


def log(message: str = "") -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    import timm
    import torchvision

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda": torch.version.cuda,
        "git_commit": git_commit(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-run", action="store_true", required=True)
    parser.add_argument("--dataset", choices=("pet", "cub"), default="pet")
    parser.add_argument("--kind", choices=("teacher", "student"), required=True)
    parser.add_argument(
        "--teacher-architecture",
        choices=TEACHER_ARCHITECTURES,
        default="resnet56_32",
    )
    parser.add_argument(
        "--access-official-test",
        action="store_true",
        help=(
            "Timing-only v3 escape hatch. Evaluate the latest 2-epoch smoke "
            "checkpoint on CUB official test after validation."
        ),
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fusion-ratio", type=float)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--scientific-cub-r50-teacher",
        action="store_true",
        help=(
            "Load the audited Phase 1 CUB ResNet-50/224 v3 teacher instead of "
            "a two-epoch timing-only teacher. CUB student timing only."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--alg-controller-warmup-epochs",
        type=int,
        default=0,
        help=(
            "ALG controller-only stop-decision delay. Canonical ALG uses 0; "
            "20 is used only by an explicitly labeled ALG-w20 protocol."
        ),
    )
    parser.add_argument(
        "--save-student-checkpoint",
        action="store_true",
        help="Persist the non-scientific timing student for frozen-probe smoke.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    dataset_key = _dataset_key(args)
    teacher_architecture = str(
        getattr(args, "teacher_architecture", "resnet56_32")
    )
    access_official_test = bool(getattr(args, "access_official_test", False))
    scientific_cub_teacher = bool(
        getattr(args, "scientific_cub_r50_teacher", False)
    )
    if teacher_architecture == "resnet50_224_scratch" and dataset_key != "cub":
        raise ValueError("ResNet-50/224 timing teacher is CUB-only")
    if access_official_test and not (
        dataset_key == "cub" and teacher_architecture == "resnet50_224_scratch"
    ):
        raise ValueError(
            "Official-test timing access is reserved for the locked CUB v3 smoke"
        )
    if args.batch_size not in {64, 128}:
        raise ValueError("Timing matrix batch size must be 64 or 128")
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid eval batch size or worker count")
    if args.seed != 1:
        raise ValueError("Phase 1 timing is fixed to seed 1")
    if args.kind == "teacher":
        if scientific_cub_teacher:
            raise ValueError("Teacher timing cannot consume a scientific teacher")
        if args.alg_controller_warmup_epochs != 0:
            raise ValueError("Teacher timing does not accept ALG controller warm-up")
        if args.save_student_checkpoint:
            raise ValueError("Teacher timing cannot save a student checkpoint")
        if args.method is not None or args.fusion_ratio is not None:
            raise ValueError("Teacher timing does not accept method/lambda")
        if args.batch_size != 128:
            raise ValueError("Teacher timing batch is fixed to 128")
    else:
        if args.method is None:
            raise ValueError("Student timing requires --method")
        if args.method == "ibkd":
            if args.fusion_ratio not in {0.25, 0.5}:
                raise ValueError("iBKD timing lambda must be 0.25 or 0.5")
        elif args.fusion_ratio is not None:
            raise ValueError("Only iBKD accepts --fusion-ratio")
        if args.method != "vanilla" and args.teacher_checkpoint is None:
            raise ValueError("Guided student timing requires --teacher-checkpoint")
        if scientific_cub_teacher and not (
            dataset_key == "cub"
            and teacher_architecture == "resnet50_224_scratch"
            and args.teacher_checkpoint is not None
        ):
            raise ValueError(
                "Scientific teacher reuse is restricted to CUB ResNet-50/224 students"
            )
        if args.method == "alg":
            if args.alg_controller_warmup_epochs not in {0, 20}:
                raise ValueError(
                    "ALG controller warm-up must be canonical 0 or diagnostic 20"
                )
        elif args.alg_controller_warmup_epochs != 0:
            raise ValueError("Only ALG accepts --alg-controller-warmup-epochs")
        if (
            args.save_student_checkpoint
            and dataset_key == "pet"
            and not (
                args.method == "alg" and args.alg_controller_warmup_epochs == 20
            )
        ):
            raise ValueError(
                "Timing student checkpoint export is reserved for ALG warm-up-20 smoke"
            )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    teacher: bool,
    planned_epochs: int = PLANNED_EPOCHS,
    teacher_warmup_epochs: int = 0,
) -> Any:
    from timm.scheduler import CosineLRScheduler

    if teacher:
        return CosineLRScheduler(
            optimizer,
            t_initial=planned_epochs,
            lr_min=0.0,
            warmup_t=teacher_warmup_epochs,
            warmup_lr_init=(0.0 if teacher_warmup_epochs else 0.0),
        )
    return CosineLRScheduler(
        optimizer,
        t_initial=PLANNED_EPOCHS,
        lr_min=5e-6,
        warmup_t=20,
        warmup_lr_init=5e-7,
    )


def teacher_parameter_groups(
    model: nn.Module, *, weight_decay: float = 5e-4
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    teacher: bool,
    teacher_image_size: int = 32,
    num_classes: int = PET_NUM_CLASSES,
) -> dict[str, float]:
    model.eval()
    correct_by_class = torch.zeros(num_classes, dtype=torch.long)
    total_by_class = torch.zeros(num_classes, dtype=torch.long)
    total_correct = 0
    total_top5 = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(
            teacher_view(images, image_size=teacher_image_size)
            if teacher
            else images
        )
        predictions = logits.argmax(dim=1)
        matches = predictions.eq(targets)
        total_correct += int(matches.sum())
        top5_matches = logits.topk(5, dim=1).indices.eq(targets[:, None]).any(dim=1)
        total_top5 += int(top5_matches.sum())
        total += targets.numel()
        total_by_class += torch.bincount(targets.cpu(), minlength=num_classes)
        correct_by_class += torch.bincount(
            targets[matches].cpu(), minlength=num_classes
        )
    if bool((total_by_class == 0).any()):
        raise RuntimeError("Validation split is missing at least one breed")
    macro = (correct_by_class.float() / total_by_class.float()).mean().item()
    return {
        "overall_top1": 100.0 * total_correct / max(1, total),
        "macro_top1": 100.0 * macro,
        "top5": 100.0 * total_top5 / max(1, total),
    }


def run_teacher(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dataset_name, num_classes, loader_builder = _dataset_contract(args)
    seed_everything(args.seed)
    model = _teacher_model(args, num_classes).to(device)
    initial_hash = state_dict_sha256(model)
    if _is_resnet50_teacher(args):
        loader_builder = build_resnet50_train_validation_loaders
    train_loader, validation_loader, manifest = loader_builder(
        args.data_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    teacher_lr = 0.05 if _is_resnet50_teacher(args) else 0.1
    teacher_weight_decay = 1e-4 if _is_resnet50_teacher(args) else 5e-4
    teacher_nesterov = not _is_resnet50_teacher(args)
    teacher_warmup_epochs = 5 if _is_resnet50_teacher(args) else 0
    planned_epochs = _teacher_planned_epochs(args)
    optimizer = torch.optim.SGD(
        teacher_parameter_groups(model, weight_decay=teacher_weight_decay),
        lr=teacher_lr,
        momentum=0.9,
        nesterov=teacher_nesterov,
    )
    scheduler = create_scheduler(
        optimizer,
        teacher=True,
        planned_epochs=planned_epochs,
        teacher_warmup_epochs=teacher_warmup_epochs,
    )
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, run_dir / "validation_split.json")
    epoch_rows: list[dict[str, Any]] = []

    log("[TIMING_ONLY] teacher accuracy/checkpoint is not valid for scientific runs")
    for epoch in range(1, ACTUAL_EPOCHS + 1):
        synchronize(device)
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(_teacher_inputs(args, images))
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            batch = targets.numel()
            total += batch
            total_loss += float(loss.detach()) * batch
            correct += int(logits.argmax(dim=1).eq(targets).sum())
        validation = evaluate(
            model,
            validation_loader,
            device,
            teacher=True,
            teacher_image_size=_teacher_image_size(args),
            num_classes=num_classes,
        )
        synchronize(device)
        elapsed = time.perf_counter() - start
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / total,
            "train_top1": 100.0 * correct / total,
            "validation": validation,
            "seconds_including_validation": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
        }
        epoch_rows.append(row)
        log(
            f"[TEACHER_EPOCH] {epoch}/{ACTUAL_EPOCHS} "
            f"time={elapsed:.2f}s peak_cuda_bytes={peak_memory} "
            f"peak_cuda_reserved_bytes={peak_reserved} "
            f"val_macro={validation['macro_top1']:.2f}"
        )
        scheduler.step(epoch)

    official_test: dict[str, float] | None = None
    official_test_seconds = 0.0
    if bool(getattr(args, "access_official_test", False)):
        test_loader = build_resnet50_official_test_loader(
            args.data_dir,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        synchronize(device)
        test_started = time.perf_counter()
        official_test = evaluate(
            model,
            test_loader,
            device,
            teacher=True,
            teacher_image_size=_teacher_image_size(args),
            num_classes=num_classes,
        )
        synchronize(device)
        official_test_seconds = time.perf_counter() - test_started
        log(
            "[TEACHER_OFFICIAL_TEST_SMOKE] "
            f"top1={official_test['overall_top1']:.4f} "
            f"macro_top1={official_test['macro_top1']:.4f} "
            "scientific_result=false"
        )

    checkpoint_path = run_dir / "timing_teacher_latest.pt"
    checkpoint = {
        "model": model.state_dict(),
        "metadata": {
            "purpose": "phase1_timing_only_not_scientific",
            "dataset": dataset_name,
            "num_classes": num_classes,
            "architecture": _teacher_architecture_name(args),
            "input_size": _teacher_image_size(args),
            "actual_epochs": ACTUAL_EPOCHS,
            "planned_epochs": planned_epochs,
            "seed": args.seed,
            "official_test_accessed": bool(
                getattr(args, "access_official_test", False)
            ),
            "validation_image_ids_sha256": manifest["validation_image_ids_sha256"],
        },
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    average = sum(row["seconds_including_validation"] for row in epoch_rows) / len(
        epoch_rows
    )
    return {
        "status": "complete",
        "purpose": "runtime_feasibility_only",
        "scientific_result": False,
        "official_test_accessed": bool(
            getattr(args, "access_official_test", False)
        ),
        "dataset": dataset_name,
        "kind": "teacher",
        "batch_size": args.batch_size,
        "actual_epochs": ACTUAL_EPOCHS,
        "planned_epochs": planned_epochs,
        "avg_epoch_seconds": average,
        "estimated_planned_seconds": average * planned_epochs,
        "official_test": official_test,
        "official_test_seconds": official_test_seconds,
        "teacher_contract": {
            "architecture": _teacher_architecture_name(args),
            "input_size": _teacher_image_size(args),
            "feature_channels": list(_teacher_channels(args)),
            "feature_spatial_sizes": (
                [28, 14, 7] if _is_resnet50_teacher(args) else [32, 16, 8]
            ),
            "optimizer": {
                "name": "sgd",
                "learning_rate": teacher_lr,
                "momentum": 0.9,
                "nesterov": teacher_nesterov,
                "weight_decay": teacher_weight_decay,
            },
            "warmup_epochs": teacher_warmup_epochs,
        },
        "initial_state_sha256": initial_hash,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "split_manifest": manifest,
        "epochs": epoch_rows,
        "runtime": runtime_metadata(device),
    }


def load_timing_teacher(
    checkpoint_path: Path,
    *,
    validation_hash: str,
    device: torch.device,
    teacher_architecture: str = "resnet56_32",
    official_test_accessed: bool = False,
    dataset_name: str = "Oxford-IIIT Pet",
    num_classes: int = PET_NUM_CLASSES,
) -> tuple[nn.Module, dict[str, Any], str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": "phase1_timing_only_not_scientific",
        "dataset": dataset_name,
        "num_classes": num_classes,
        "actual_epochs": ACTUAL_EPOCHS,
        "architecture": (
            "torchvision_resnet50_224_scratch"
            if teacher_architecture == "resnet50_224_scratch"
            else "cifar_style_resnet56_6n_plus_2_n9"
        ),
        "official_test_accessed": official_test_accessed,
        "validation_image_ids_sha256": validation_hash,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Timing teacher contract mismatch for {key}: "
                f"expected={value!r} got={metadata.get(key)!r}"
            )
    teacher = (
        ResNet50CUB(num_classes=num_classes)
        if teacher_architecture == "resnet50_224_scratch"
        else ResNet56(num_classes=num_classes)
    )
    teacher.load_state_dict(payload["model"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, metadata, file_sha256(checkpoint_path)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    temperature = KD_TEMPERATURE
    return F.kl_div(
        F.log_softmax(student_logits.float() / temperature, dim=1),
        F.softmax(teacher_logits.float() / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)


def run_student(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    assert args.method is not None
    dataset_name, num_classes, loader_builder = _dataset_contract(args)
    seed_everything(args.seed)
    student = create_student(num_classes=num_classes, drop_path_rate=0.1).to(device)
    initial_hash = state_dict_sha256(student)
    train_loader, validation_loader, manifest = loader_builder(
        args.data_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, run_dir / "validation_split.json")

    teacher: nn.Module | None = None
    teacher_metadata: dict[str, Any] | None = None
    teacher_hash: str | None = None
    teacher_model_state_hash: str | None = None
    guidance: nn.Module | None = None
    controller: GuidanceController | None = None
    if args.method != "vanilla":
        assert args.teacher_checkpoint is not None
        if bool(getattr(args, "scientific_cub_r50_teacher", False)):
            from .run_cub_r50_teacher_full import load_scientific_teacher

            (
                teacher,
                teacher_metadata,
                teacher_hash,
                teacher_model_state_hash,
            ) = load_scientific_teacher(
                args.teacher_checkpoint,
                device=device,
                validation_hash=manifest["validation_image_ids_sha256"],
            )
        else:
            teacher, teacher_metadata, teacher_hash = load_timing_teacher(
                args.teacher_checkpoint,
                validation_hash=manifest["validation_image_ids_sha256"],
                device=device,
                teacher_architecture=str(
                    getattr(args, "teacher_architecture", "resnet56_32")
                ),
                official_test_accessed=bool(
                    getattr(args, "access_official_test", False)
                ),
                dataset_name=dataset_name,
                num_classes=num_classes,
            )
    if args.method in {"lg", "alg"}:
        guidance = LocalityGuidance(
            teacher_channels=_teacher_channels(args)
        ).to(device)
        controller = GuidanceController(
            kind=args.method,
            warmup_epochs=(
                args.alg_controller_warmup_epochs if args.method == "alg" else 0
            ),
        )
    elif args.method == "ibkd":
        guidance = IBKD(teacher_channels=_teacher_channels(args)).to(device)
        controller = GuidanceController(kind="ibkd", warmup_epochs=20)

    parameters = list(student.parameters())
    if guidance is not None:
        parameters.extend(guidance.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
    )
    scheduler = create_scheduler(optimizer, teacher=False)
    epoch_rows: list[dict[str, Any]] = []
    log(
        "[TIMING_ONLY] student validation accuracy is diagnostic only; "
        "it must not select batch, lambda, method, or checkpoint"
    )
    log(
        f"[STUDENT_CONTRACT] method={args.method} batch={args.batch_size} "
        f"lambda={args.fusion_ratio} initial_sha256={initial_hash} fp32=True "
        f"alg_controller_warmup={args.alg_controller_warmup_epochs}"
    )

    for epoch in range(1, ACTUAL_EPOCHS + 1):
        beta = 0.0 if controller is None else controller.beta_for_epoch(epoch)
        synchronize(device)
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        student.train()
        if guidance is not None:
            guidance.train()
        total = 0
        correct = 0
        totals = {"loss": 0.0, "ce": 0.0, "guidance": 0.0, "align": 0.0, "fuse": 0.0}
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            teacher_features: list[torch.Tensor] | None = None
            teacher_logits: torch.Tensor | None = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_inputs = _teacher_inputs(args, images)
                    if args.method == "kd":
                        teacher_logits = teacher(teacher_inputs)
                    elif beta > 0.0:
                        teacher_features = list(teacher.forward_features(teacher_inputs))

            align = images.new_zeros(())
            fuse = images.new_zeros(())
            feature_loss = images.new_zeros(())
            if args.method in {"lg", "alg", "ibkd"}:
                student_features, logits = forward_student_spatial(student, images)
            else:
                logits = student(images)
                student_features = []
            ce = F.cross_entropy(logits, targets)
            if args.method == "vanilla":
                loss = ce
            elif args.method == "kd":
                assert teacher_logits is not None
                feature_loss = kd_loss(logits, teacher_logits)
                loss = (1.0 - KD_ALPHA) * ce + KD_ALPHA * feature_loss
            elif args.method in {"lg", "alg"}:
                assert isinstance(guidance, LocalityGuidance)
                assert teacher_features is not None
                feature_loss = guidance(student_features, teacher_features)
                loss = ce + beta * feature_loss
            else:
                assert isinstance(guidance, IBKD)
                assert teacher_features is not None
                assert args.fusion_ratio is not None
                align, fuse = guidance(student_features, teacher_features)
                feature_loss = args.fusion_ratio * fuse + (1.0 - args.fusion_ratio) * align
                loss = ce + beta * feature_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Encountered non-finite training loss")
            loss.backward()
            optimizer.step()

            batch = targets.numel()
            total += batch
            correct += int(logits.argmax(dim=1).eq(targets).sum())
            totals["loss"] += float(loss.detach()) * batch
            totals["ce"] += float(ce.detach()) * batch
            totals["guidance"] += float(feature_loss.detach()) * batch
            totals["align"] += float(align.detach()) * batch
            totals["fuse"] += float(fuse.detach()) * batch

        average_guidance = totals["guidance"] / total
        if controller is not None:
            controller.observe(epoch, average_guidance, beta_used=beta)
        validation = evaluate(
            student,
            validation_loader,
            device,
            teacher=False,
            num_classes=num_classes,
        )
        synchronize(device)
        elapsed = time.perf_counter() - start
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "beta": beta,
            "train_loss": totals["loss"] / total,
            "train_ce": totals["ce"] / total,
            "train_guidance": average_guidance,
            "train_alignment": totals["align"] / total,
            "train_fusion": totals["fuse"] / total,
            "train_top1": 100.0 * correct / total,
            "validation": validation,
            "seconds_including_validation": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
        }
        epoch_rows.append(row)
        log(
            f"[STUDENT_EPOCH] method={args.method} batch={args.batch_size} "
            f"lambda={args.fusion_ratio} epoch={epoch}/{ACTUAL_EPOCHS} "
            f"time={elapsed:.2f}s peak_cuda_bytes={peak_memory} "
            f"peak_cuda_reserved_bytes={peak_reserved} "
            f"val_macro={validation['macro_top1']:.2f}"
        )
        scheduler.step(epoch)

    official_test: dict[str, float] | None = None
    official_test_seconds = 0.0
    if bool(getattr(args, "access_official_test", False)):
        test_loader = build_cub_official_test_loader(
            args.data_dir,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        synchronize(device)
        test_started = time.perf_counter()
        official_test = evaluate(
            student,
            test_loader,
            device,
            teacher=False,
            num_classes=num_classes,
        )
        synchronize(device)
        official_test_seconds = time.perf_counter() - test_started
        log(
            f"[STUDENT_OFFICIAL_TEST_SMOKE] method={args.method} "
            f"lambda={args.fusion_ratio} top1={official_test['overall_top1']:.4f} "
            f"macro_top1={official_test['macro_top1']:.4f} "
            "scientific_result=false"
        )

    average = sum(row["seconds_including_validation"] for row in epoch_rows) / len(
        epoch_rows
    )
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    student_state_hash: str | None = None
    if args.save_student_checkpoint:
        checkpoint_path = run_dir / "timing_student_latest.pt"
        student_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in student.state_dict().items()
        }
        student_state_hash = state_dict_sha256(student)
        atomic_torch_save(
            {
                "student": student_state,
                "metadata": {
                    "purpose": (
                        "phase1_cub_r50_224_guided_smoke_student_v3"
                        if _is_resnet50_teacher(args)
                        else "phase1_cub_combined_smoke_student"
                        if _dataset_key(args) == "cub"
                        else "phase1_alg_warmup20_combined_smoke_student"
                    ),
                    "scientific_result": False,
                    "official_test_accessed": bool(
                        getattr(args, "access_official_test", False)
                    ),
                    "dataset": dataset_name,
                    "num_classes": num_classes,
                    "architecture": "deit_tiny_patch16_224",
                    "method": args.method,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "actual_epochs": ACTUAL_EPOCHS,
                    "planned_epochs": PLANNED_EPOCHS,
                    "teacher_architecture": str(
                        getattr(args, "teacher_architecture", "resnet56_32")
                    ),
                    "teacher_checkpoint_kind": (
                        "cub_r50_v3_scientific"
                        if bool(
                            getattr(args, "scientific_cub_r50_teacher", False)
                        )
                        else "timing_smoke"
                    ),
                    "teacher_checkpoint_sha256": teacher_hash,
                    "teacher_model_state_sha256": teacher_model_state_hash,
                    "controller_warmup_epochs": args.alg_controller_warmup_epochs,
                    "guidance_controller_warmup_epochs": (
                        None if controller is None else controller.warmup_epochs
                    ),
                    "validation_image_ids_sha256": manifest[
                        "validation_image_ids_sha256"
                    ],
                    "student_state_sha256": student_state_hash,
                    "official_test_evaluations_at_checkpoint_write": int(
                        bool(getattr(args, "access_official_test", False))
                    ),
                },
            },
            checkpoint_path,
        )
        checkpoint_sha256 = file_sha256(checkpoint_path)
        log(
            "[TIMING_STUDENT_CHECKPOINT] "
            f"path={checkpoint_path.resolve()} sha256={checkpoint_sha256} "
            "scientific_result=false official_test_accessed="
            f"{str(bool(getattr(args, 'access_official_test', False))).lower()}"
        )

    return {
        "status": "complete",
        "purpose": "runtime_and_memory_feasibility_only",
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": bool(
            getattr(args, "access_official_test", False)
        ),
        "dataset": dataset_name,
        "kind": "student",
        "method": args.method,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "fusion_ratio_lambda": args.fusion_ratio,
        "actual_epochs": ACTUAL_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
        "avg_epoch_seconds": average,
        "estimated_planned_seconds": average * PLANNED_EPOCHS,
        "official_test": official_test,
        "official_test_seconds": official_test_seconds,
        "initial_student_state_sha256": initial_hash,
        "teacher_checkpoint_sha256": teacher_hash,
        "teacher_model_state_sha256": teacher_model_state_hash,
        "teacher_checkpoint_kind": (
            "cub_r50_v3_scientific"
            if bool(getattr(args, "scientific_cub_r50_teacher", False))
            else "timing_smoke"
            if teacher is not None
            else None
        ),
        "teacher_metadata": teacher_metadata,
        "controller": None if controller is None else controller.state_dict(),
        "alg_controller_warmup_epochs": args.alg_controller_warmup_epochs,
        "guidance_controller_warmup_epochs": (
            None if controller is None else controller.warmup_epochs
        ),
        "checkpoint": (
            None if checkpoint_path is None else str(checkpoint_path.resolve())
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "student_state_sha256": student_state_hash,
        "optimizer_contract": "shared_single_group_adamw_all_trainable_parameters_wd_0.05",
        "split_manifest": manifest,
        "epochs": epoch_rows,
        "runtime": runtime_metadata(device),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    try:
        validate_args(args)
        import timm

        if timm.__version__ != "1.0.27":
            raise RuntimeError(f"Expected timm==1.0.27, found {timm.__version__}")
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(
            f"[PHASE1_TIMING_START] dataset={_dataset_key(args)} "
            f"kind={args.kind} method={args.method} "
            f"teacher_architecture={args.teacher_architecture} "
            f"batch={args.batch_size} device={device} actual_epochs=2 "
            f"planned_epochs={_teacher_planned_epochs(args) if args.kind == 'teacher' else PLANNED_EPOCHS}"
        )
        payload = (
            run_teacher(args, device)
            if args.kind == "teacher"
            else run_student(args, device)
        )
        save_json(payload, run_dir / "summary.json")
        log(
            f"[PHASE1_TIMING_DONE] dataset={_dataset_key(args)} "
            f"kind={args.kind} method={args.method} "
            f"batch={args.batch_size} avg_epoch={payload['avg_epoch_seconds']:.2f}s "
            "estimated_planned="
            f"{format_duration(payload['estimated_planned_seconds'])}"
        )
    except Exception as error:
        message = str(error)
        failure_kind = (
            "cuda_oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in message.lower()
            else "runtime_error"
        )
        peak_memory = None
        peak_reserved = None
        if torch.cuda.is_available():
            peak_memory = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        save_json(
            {
                "status": "failed",
                "purpose": "runtime_memory_oom_and_job_partitioning_only",
                "scientific_result": False,
                "official_test_accessed": bool(
                    getattr(args, "access_official_test", False)
                ),
                "kind": args.kind,
                "method": args.method,
                "batch_size": args.batch_size,
                "fusion_ratio_lambda": args.fusion_ratio,
                "failure_kind": failure_kind,
                "error_type": type(error).__name__,
                "error": message,
                "peak_cuda_memory_bytes": peak_memory,
                "peak_cuda_memory_reserved_bytes": peak_reserved,
            },
            run_dir / "failure.json",
        )
        log(
            f"[PHASE1_TIMING_FAILED] dataset={_dataset_key(args)} "
            f"kind={args.kind} method={args.method} "
            f"batch={args.batch_size} failure_kind={failure_kind} "
            f"error={type(error).__name__}:{message}"
        )
        raise


if __name__ == "__main__":
    main()
