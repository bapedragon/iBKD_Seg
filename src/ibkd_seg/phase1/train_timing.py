#!/usr/bin/env python3
"""Run one full-data, two-epoch Phase 1 timing task.

This entry point cannot run a scientific/full experiment and never opens the
Oxford-IIIT Pet official test split.  Its checkpoints and accuracies are marked
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controllers import GuidanceController
from .data import NUM_CLASSES, build_train_validation_loaders, save_json
from .models import (
    IBKD,
    LocalityGuidance,
    ResNet56,
    create_student,
    forward_student_spatial,
    teacher_view,
)


ACTUAL_EPOCHS = 2
PLANNED_EPOCHS = 300
METHODS = ("vanilla", "kd", "lg", "alg", "ibkd")
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.9


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
    parser.add_argument("--kind", choices=("teacher", "student"), required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fusion-ratio", type=float)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size not in {64, 128}:
        raise ValueError("Timing matrix batch size must be 64 or 128")
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid eval batch size or worker count")
    if args.seed != 1:
        raise ValueError("Phase 1 timing is fixed to seed 1")
    if args.kind == "teacher":
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


def create_scheduler(optimizer: torch.optim.Optimizer, *, teacher: bool) -> Any:
    from timm.scheduler import CosineLRScheduler

    if teacher:
        return CosineLRScheduler(optimizer, t_initial=PLANNED_EPOCHS, lr_min=0.0)
    return CosineLRScheduler(
        optimizer,
        t_initial=PLANNED_EPOCHS,
        lr_min=5e-6,
        warmup_t=20,
        warmup_lr_init=5e-7,
    )


def teacher_parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
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
        {"params": decay, "weight_decay": 5e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    teacher: bool,
) -> dict[str, float]:
    model.eval()
    correct_by_class = torch.zeros(NUM_CLASSES, dtype=torch.long)
    total_by_class = torch.zeros(NUM_CLASSES, dtype=torch.long)
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(teacher_view(images) if teacher else images)
        predictions = logits.argmax(dim=1)
        matches = predictions.eq(targets)
        total_correct += int(matches.sum())
        total += targets.numel()
        total_by_class += torch.bincount(targets.cpu(), minlength=NUM_CLASSES)
        correct_by_class += torch.bincount(
            targets[matches].cpu(), minlength=NUM_CLASSES
        )
    if bool((total_by_class == 0).any()):
        raise RuntimeError("Validation split is missing at least one breed")
    macro = (correct_by_class.float() / total_by_class.float()).mean().item()
    return {
        "overall_top1": 100.0 * total_correct / max(1, total),
        "macro_top1": 100.0 * macro,
    }


def run_teacher(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed_everything(args.seed)
    model = ResNet56(num_classes=NUM_CLASSES).to(device)
    initial_hash = state_dict_sha256(model)
    train_loader, validation_loader, manifest = build_train_validation_loaders(
        args.data_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )
    optimizer = torch.optim.SGD(
        teacher_parameter_groups(model),
        lr=0.1,
        momentum=0.9,
        nesterov=True,
    )
    scheduler = create_scheduler(optimizer, teacher=True)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, run_dir / "validation_split.json")
    epoch_rows: list[dict[str, Any]] = []

    log("[TIMING_ONLY] teacher accuracy/checkpoint is not valid for scientific runs")
    for epoch in range(1, ACTUAL_EPOCHS + 1):
        synchronize(device)
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(teacher_view(images))
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            batch = targets.numel()
            total += batch
            total_loss += float(loss.detach()) * batch
            correct += int(logits.argmax(dim=1).eq(targets).sum())
        validation = evaluate(model, validation_loader, device, teacher=True)
        synchronize(device)
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / total,
            "train_top1": 100.0 * correct / total,
            "validation": validation,
            "seconds_including_validation": elapsed,
        }
        epoch_rows.append(row)
        log(
            f"[TEACHER_EPOCH] {epoch}/{ACTUAL_EPOCHS} "
            f"time={elapsed:.2f}s val_macro={validation['macro_top1']:.2f}"
        )
        scheduler.step(epoch)

    checkpoint_path = run_dir / "timing_teacher_latest.pt"
    checkpoint = {
        "model": model.state_dict(),
        "metadata": {
            "purpose": "phase1_timing_only_not_scientific",
            "dataset": "Oxford-IIIT Pet",
            "num_classes": NUM_CLASSES,
            "architecture": "cifar_style_resnet56_6n_plus_2_n9",
            "actual_epochs": ACTUAL_EPOCHS,
            "planned_epochs": PLANNED_EPOCHS,
            "seed": args.seed,
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
        "official_test_accessed": False,
        "kind": "teacher",
        "batch_size": args.batch_size,
        "actual_epochs": ACTUAL_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
        "avg_epoch_seconds": average,
        "estimated_planned_seconds": average * PLANNED_EPOCHS,
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
) -> tuple[ResNet56, dict[str, Any], str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": "phase1_timing_only_not_scientific",
        "dataset": "Oxford-IIIT Pet",
        "num_classes": NUM_CLASSES,
        "actual_epochs": ACTUAL_EPOCHS,
        "validation_image_ids_sha256": validation_hash,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Timing teacher contract mismatch for {key}: "
                f"expected={value!r} got={metadata.get(key)!r}"
            )
    teacher = ResNet56(num_classes=NUM_CLASSES)
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
    seed_everything(args.seed)
    student = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1).to(device)
    initial_hash = state_dict_sha256(student)
    train_loader, validation_loader, manifest = build_train_validation_loaders(
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

    teacher: ResNet56 | None = None
    teacher_metadata: dict[str, Any] | None = None
    teacher_hash: str | None = None
    guidance: nn.Module | None = None
    controller: GuidanceController | None = None
    if args.method != "vanilla":
        assert args.teacher_checkpoint is not None
        teacher, teacher_metadata, teacher_hash = load_timing_teacher(
            args.teacher_checkpoint,
            validation_hash=manifest["validation_image_ids_sha256"],
            device=device,
        )
    if args.method in {"lg", "alg"}:
        guidance = LocalityGuidance().to(device)
        controller = GuidanceController(kind=args.method, warmup_epochs=0)
    elif args.method == "ibkd":
        guidance = IBKD().to(device)
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
        f"lambda={args.fusion_ratio} initial_sha256={initial_hash} fp32=True"
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
                    teacher_inputs = teacher_view(images)
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
        validation = evaluate(student, validation_loader, device, teacher=False)
        synchronize(device)
        elapsed = time.perf_counter() - start
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
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
        }
        epoch_rows.append(row)
        log(
            f"[STUDENT_EPOCH] method={args.method} batch={args.batch_size} "
            f"lambda={args.fusion_ratio} epoch={epoch}/{ACTUAL_EPOCHS} "
            f"time={elapsed:.2f}s peak_cuda_bytes={peak_memory} "
            f"val_macro={validation['macro_top1']:.2f}"
        )
        scheduler.step(epoch)

    average = sum(row["seconds_including_validation"] for row in epoch_rows) / len(
        epoch_rows
    )
    return {
        "status": "complete",
        "purpose": "runtime_and_memory_feasibility_only",
        "scientific_result": False,
        "selection_from_smoke_metrics_forbidden": True,
        "official_test_accessed": False,
        "kind": "student",
        "method": args.method,
        "batch_size": args.batch_size,
        "fusion_ratio_lambda": args.fusion_ratio,
        "actual_epochs": ACTUAL_EPOCHS,
        "planned_epochs": PLANNED_EPOCHS,
        "avg_epoch_seconds": average,
        "estimated_planned_seconds": average * PLANNED_EPOCHS,
        "initial_student_state_sha256": initial_hash,
        "teacher_checkpoint_sha256": teacher_hash,
        "teacher_metadata": teacher_metadata,
        "controller": None if controller is None else controller.state_dict(),
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
            f"[PHASE1_TIMING_START] kind={args.kind} method={args.method} "
            f"batch={args.batch_size} device={device} actual_epochs=2 planned_epochs=300"
        )
        payload = (
            run_teacher(args, device)
            if args.kind == "teacher"
            else run_student(args, device)
        )
        save_json(payload, run_dir / "summary.json")
        log(
            f"[PHASE1_TIMING_DONE] kind={args.kind} method={args.method} "
            f"batch={args.batch_size} avg_epoch={payload['avg_epoch_seconds']:.2f}s "
            f"estimated_300={format_duration(payload['estimated_planned_seconds'])}"
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
        if torch.cuda.is_available():
            peak_memory = int(torch.cuda.max_memory_allocated())
        save_json(
            {
                "status": "failed",
                "purpose": "runtime_memory_oom_and_job_partitioning_only",
                "scientific_result": False,
                "official_test_accessed": False,
                "kind": args.kind,
                "method": args.method,
                "batch_size": args.batch_size,
                "fusion_ratio_lambda": args.fusion_ratio,
                "failure_kind": failure_kind,
                "error_type": type(error).__name__,
                "error": message,
                "peak_cuda_memory_bytes": peak_memory,
            },
            run_dir / "failure.json",
        )
        log(
            f"[PHASE1_TIMING_FAILED] kind={args.kind} method={args.method} "
            f"batch={args.batch_size} failure_kind={failure_kind} "
            f"error={type(error).__name__}:{message}"
        )
        raise


if __name__ == "__main__":
    main()
