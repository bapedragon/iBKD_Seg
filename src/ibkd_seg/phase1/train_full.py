#!/usr/bin/env python3
"""Train one validation-selected Phase 1 teacher or student for 300 epochs."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controllers import GuidanceController
from .cub_data import (
    DATASET_NAME as CUB_DATASET_NAME,
    NUM_CLASSES as CUB_NUM_CLASSES,
    build_official_test_loader as build_cub_official_test_loader,
    build_train_validation_loaders as build_cub_train_validation_loaders,
)
from .data import (
    NUM_CLASSES as PET_NUM_CLASSES,
    build_official_test_loader as build_pet_official_test_loader,
    build_train_validation_loaders as build_pet_train_validation_loaders,
    save_json,
)
from .models import (
    IBKD,
    LocalityGuidance,
    RESNET50_TEACHER_CHANNELS,
    ResNet56,
    create_student,
    forward_student_spatial,
    teacher_view,
)
from .train_timing import (
    KD_ALPHA,
    PLANNED_EPOCHS,
    atomic_torch_save,
    create_scheduler,
    evaluate,
    file_sha256,
    format_duration,
    kd_loss,
    log,
    runtime_metadata,
    seed_everything,
    state_dict_sha256,
    synchronize,
    teacher_parameter_groups,
)


METHODS = ("vanilla", "kd", "lg", "alg", "ibkd")
TEACHER_ARCHITECTURES = ("resnet56_32", "resnet50_224_scratch")
CUB_R50_BATCH_PROFILE_CONFIG_SHA256 = (
    "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633"
)
ALG_WARMUP20_DIAGNOSTIC_ID = (
    "oxford_iiit_pet_alg_controller_warmup20_posthoc_v1"
)


def _dataset_key(args: argparse.Namespace) -> str:
    value = str(getattr(args, "dataset", "pet"))
    if value not in {"pet", "cub"}:
        raise ValueError(f"unsupported Phase 1 full dataset: {value!r}")
    return value


def _dataset_contract(args: argparse.Namespace) -> tuple[str, int, Any, Any]:
    if _dataset_key(args) == "cub":
        return (
            CUB_DATASET_NAME,
            CUB_NUM_CLASSES,
            build_cub_train_validation_loaders,
            build_cub_official_test_loader,
        )
    return (
        "Oxford-IIIT Pet",
        PET_NUM_CLASSES,
        build_pet_train_validation_loaders,
        build_pet_official_test_loader,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true", required=True)
    parser.add_argument("--dataset", choices=("pet", "cub"), default="pet")
    parser.add_argument("--kind", choices=("teacher", "student"), required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fusion-ratio", type=float)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--teacher-architecture",
        choices=TEACHER_ARCHITECTURES,
        default="resnet56_32",
    )
    parser.add_argument(
        "--scientific-cub-r50-teacher",
        action="store_true",
        help="Strictly reuse the audited issue-722 CUB ResNet-50/224 teacher.",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        help="Locked batch-profile protocol snapshot bound to this student run.",
    )
    parser.add_argument(
        "--batch-profile-role",
        choices=("locked_v3_partial_cell", "batch64_sensitivity"),
    )
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--alg-controller-warmup-epochs",
        type=int,
        default=0,
        help="Canonical ALG uses 0; the labeled post-hoc diagnostic uses 20.",
    )
    parser.add_argument(
        "--posthoc-diagnostic-id",
        type=str,
        help="Required diagnostic identity when ALG controller warm-up is 20.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    dataset_key = _dataset_key(args)
    teacher_architecture = str(getattr(args, "teacher_architecture", "resnet56_32"))
    scientific_cub_r50 = bool(getattr(args, "scientific_cub_r50_teacher", False))
    protocol_config = getattr(args, "protocol_config", None)
    batch_profile_role = getattr(args, "batch_profile_role", None)
    if args.batch_size not in {64, 128}:
        raise ValueError("Phase 1 full-run batch must be 64 or 128")
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid evaluation batch size or worker count")
    if args.kind == "teacher":
        if scientific_cub_r50 or protocol_config is not None:
            raise ValueError("Legacy teacher training cannot use the CUB R50 student flags")
        if teacher_architecture != "resnet56_32":
            raise ValueError("Use run_cub_r50_teacher_full for a ResNet-50 teacher")
        if batch_profile_role is not None:
            raise ValueError("Teacher does not accept a batch-profile role")
        if args.alg_controller_warmup_epochs != 0:
            raise ValueError("Teacher does not accept ALG controller warm-up")
        if args.posthoc_diagnostic_id is not None:
            raise ValueError("Teacher does not accept a post-hoc diagnostic id")
        if args.seed != 1 or args.batch_size != 128:
            raise ValueError("Teacher is fixed to seed 1 and batch 128")
        if args.method is not None or args.fusion_ratio is not None:
            raise ValueError("Teacher does not accept method or lambda")
        return
    if args.seed not in {1, 2, 3}:
        raise ValueError("Student seed must be 1, 2, or 3")
    if teacher_architecture == "resnet50_224_scratch":
        if not scientific_cub_r50 or dataset_key != "cub":
            raise ValueError(
                "ResNet-50/224 full students require --scientific-cub-r50-teacher on CUB"
            )
        if protocol_config is None or batch_profile_role is None:
            raise ValueError(
                "CUB ResNet-50/224 full students require protocol config and profile role"
            )
        profile = json.loads(protocol_config.read_text(encoding="utf-8"))
        if file_sha256(protocol_config) != CUB_R50_BATCH_PROFILE_CONFIG_SHA256:
            raise ValueError("CUB ResNet-50/224 batch-profile protocol SHA-256 changed")
        if profile.get("protocol_id") != (
            "cub200_phase1_r50_224_guided_b128_b64_seed1_full_v4"
        ):
            raise ValueError("Unexpected CUB ResNet-50/224 batch-profile protocol")
        expected_role = (
            "locked_v3_partial_cell" if args.batch_size == 128 else "batch64_sensitivity"
        )
        if batch_profile_role != expected_role:
            raise ValueError("Batch size and batch-profile role disagree")
        if args.seed != 1:
            raise ValueError("The v4 guided batch-profile run is fixed to encoder seed 1")
        allowed = profile.get("result_scope", {}).get("variants", [])
        variant = (
            f"ibkd_lambda_{args.fusion_ratio}"
            if args.method == "ibkd"
            else "alg_warmup20"
            if args.method == "alg"
            else args.method
        )
        if variant not in allowed:
            raise ValueError(f"Method is outside the locked v4 profile: {variant}")
    elif scientific_cub_r50 or protocol_config is not None or batch_profile_role:
        raise ValueError("CUB R50 flags require --teacher-architecture resnet50_224_scratch")
    if args.method is None:
        raise ValueError("Student run requires --method")
    if args.method == "ibkd":
        if args.fusion_ratio not in {0.25, 0.5}:
            raise ValueError("iBKD lambda must be 0.25 or 0.5")
    elif args.fusion_ratio is not None:
        raise ValueError("Only iBKD accepts --fusion-ratio")
    if args.method != "vanilla" and args.teacher_checkpoint is None:
        raise ValueError("Guided student run requires --teacher-checkpoint")
    if args.method == "alg":
        if args.alg_controller_warmup_epochs not in {0, 20}:
            raise ValueError(
                "ALG controller warm-up must be canonical 0 or diagnostic 20"
            )
        if dataset_key == "cub":
            if args.alg_controller_warmup_epochs != 20:
                raise ValueError("CUB confirmatory ALG is fixed to warm-up 20")
            if args.batch_size != 128 and not scientific_cub_r50:
                raise ValueError("CUB ALG warm-up-20 is fixed to batch 128")
            if args.posthoc_diagnostic_id is not None:
                raise ValueError("CUB ALG-w20 is prespecified, not a Pet diagnostic")
        elif args.alg_controller_warmup_epochs == 20:
            if args.batch_size != 128:
                raise ValueError("ALG warm-up-20 diagnostic is fixed to batch 128")
            if args.posthoc_diagnostic_id != ALG_WARMUP20_DIAGNOSTIC_ID:
                raise ValueError(
                    "ALG warm-up-20 requires the fixed post-hoc diagnostic id"
                )
        elif args.posthoc_diagnostic_id is not None:
            raise ValueError("Canonical ALG cannot carry a post-hoc diagnostic id")
    elif args.alg_controller_warmup_epochs != 0:
        raise ValueError("Only ALG accepts --alg-controller-warmup-epochs")
    elif args.posthoc_diagnostic_id is not None:
        raise ValueError("Only diagnostic ALG accepts --posthoc-diagnostic-id")


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def write_epoch_status(
    path: Path,
    *,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_macro_top1: float,
) -> None:
    save_json(
        {
            "status": "training",
            "dataset": _dataset_contract(args)[0],
            "kind": args.kind,
            "method": args.method,
            "batch_size": args.batch_size,
            "fusion_ratio_lambda": args.fusion_ratio,
            "seed": args.seed,
            "posthoc_diagnostic_id": args.posthoc_diagnostic_id,
            "alg_controller_warmup_epochs": args.alg_controller_warmup_epochs,
            "completed_epochs": len(history),
            "planned_epochs": PLANNED_EPOCHS,
            "best_validation_epoch": best_epoch,
            "best_validation_macro_top1": best_macro_top1,
            "latest_epoch": history[-1],
        },
        path,
    )


def load_full_teacher(
    checkpoint_path: Path,
    *,
    validation_hash: str,
    device: torch.device,
    dataset_name: str = "Oxford-IIIT Pet",
    num_classes: int = PET_NUM_CLASSES,
) -> tuple[ResNet56, dict[str, Any], str, str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": "phase1_scientific_full_teacher",
        "dataset": dataset_name,
        "num_classes": num_classes,
        "architecture": "cifar_style_resnet56_6n_plus_2_n9",
        "epochs": PLANNED_EPOCHS,
        "seed": 1,
        "selection_metric": "validation_macro_top1",
        "validation_image_ids_sha256": validation_hash,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Full teacher contract mismatch for {key}: "
                f"expected={value!r} got={metadata.get(key)!r}"
            )
    teacher = ResNet56(num_classes=num_classes)
    teacher.load_state_dict(payload["model"], strict=True)
    actual_state_hash = state_dict_sha256(teacher)
    if actual_state_hash != metadata.get("model_state_sha256"):
        raise RuntimeError("Full teacher model-state SHA-256 mismatch")
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, metadata, file_sha256(checkpoint_path), actual_state_hash


def run_teacher(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dataset_name, num_classes, loader_builder, test_loader_builder = _dataset_contract(
        args
    )
    seed_everything(args.seed)
    model = ResNet56(num_classes=num_classes).to(device)
    initial_hash = state_dict_sha256(model)
    train_loader, validation_loader, manifest = loader_builder(
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
    status_path = run_dir / "training_status.json"
    checkpoint_path = run_dir / "teacher_best_validation.pt"
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro = float("-inf")
    best_validation: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    start_all = time.perf_counter()

    for epoch in range(1, PLANNED_EPOCHS + 1):
        synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        total = 0
        correct = 0
        total_loss = 0.0
        epoch_lr = float(optimizer.param_groups[0]["lr"])
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
        validation = evaluate(
            model,
            validation_loader,
            device,
            teacher=True,
            num_classes=num_classes,
        )
        synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "train_loss": total_loss / total,
            "train_top1": 100.0 * correct / total,
            "validation": validation,
            "seconds_including_validation": elapsed,
        }
        history.append(row)
        if validation["macro_top1"] > best_macro:
            best_macro = validation["macro_top1"]
            best_epoch = epoch
            best_validation = dict(validation)
            best_state = clone_state_dict(model)
        write_epoch_status(
            status_path,
            args=args,
            history=history,
            best_epoch=best_epoch,
            best_macro_top1=best_macro,
        )
        log(
            f"[TEACHER_EPOCH] {epoch}/{PLANNED_EPOCHS} lr={epoch_lr:.8g} "
            f"loss={total_loss / total:.4f} val_macro={validation['macro_top1']:.2f} "
            f"best_epoch={best_epoch} time={elapsed:.2f}s"
        )
        scheduler.step(epoch)

    if best_state is None or best_validation is None:
        raise RuntimeError("Teacher training produced no selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    selected_state_hash = state_dict_sha256(model)
    metadata = {
        "purpose": "phase1_scientific_full_teacher",
        "dataset": dataset_name,
        "num_classes": num_classes,
        "architecture": "cifar_style_resnet56_6n_plus_2_n9",
        "epochs": PLANNED_EPOCHS,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "selection_metric": "validation_macro_top1",
        "selection_tie_break": "earlier_epoch",
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "validation_image_ids_sha256": manifest["validation_image_ids_sha256"],
        "model_state_sha256": selected_state_hash,
        "official_test_policy": "once_after_validation_selection",
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    atomic_torch_save({"model": best_state, "metadata": metadata}, checkpoint_path)

    # Reload the serialized selection into a fresh model before the one test pass.
    del train_loader, validation_loader
    model.to("cpu")
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selected_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResNet56(num_classes=num_classes)
    model.load_state_dict(selected_payload["model"], strict=True)
    if state_dict_sha256(model) != selected_state_hash:
        raise RuntimeError("Reloaded teacher state does not match selected state")
    model.to(device)

    # Official test is instantiated only after validation selection and strict load.
    test_loader = test_loader_builder(
        args.data_dir,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        teacher=True,
        num_classes=num_classes,
    )
    summary = {
        "status": "complete",
        "scientific_result": True,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "kind": "teacher",
        "epochs": PLANNED_EPOCHS,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "initial_state_sha256": initial_hash,
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "official_test": test_metrics,
        "official_test_evaluations": 1,
        "official_test_used_for_training_or_selection": False,
        "selected_checkpoint_strict_reloaded": True,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_state_sha256": selected_state_hash,
        "split_manifest": manifest,
        "training_seconds": time.perf_counter() - start_all,
        "history": history,
        "runtime": runtime_metadata(device),
    }
    save_json({**summary, "status": "complete"}, status_path)
    return summary


def run_student(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    assert args.method is not None
    dataset_name, num_classes, loader_builder, test_loader_builder = _dataset_contract(
        args
    )
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
    status_path = run_dir / "training_status.json"
    checkpoint_path = run_dir / "student_best_validation.pt"

    scientific_cub_r50 = bool(args.scientific_cub_r50_teacher)
    protocol_config_sha256 = (
        file_sha256(args.protocol_config) if args.protocol_config is not None else None
    )
    teacher: nn.Module | None = None
    teacher_metadata: dict[str, Any] | None = None
    teacher_checkpoint_hash: str | None = None
    teacher_state_hash: str | None = None
    guidance: nn.Module | None = None
    controller: GuidanceController | None = None
    if args.method != "vanilla":
        assert args.teacher_checkpoint is not None
        if scientific_cub_r50:
            from .run_cub_r50_teacher_full import load_scientific_teacher

            (
                teacher,
                teacher_metadata,
                teacher_checkpoint_hash,
                teacher_state_hash,
            ) = load_scientific_teacher(
                args.teacher_checkpoint,
                validation_hash=manifest["validation_image_ids_sha256"],
                device=device,
            )
        else:
            (
                teacher,
                teacher_metadata,
                teacher_checkpoint_hash,
                teacher_state_hash,
            ) = load_full_teacher(
                args.teacher_checkpoint,
                validation_hash=manifest["validation_image_ids_sha256"],
                device=device,
                dataset_name=dataset_name,
                num_classes=num_classes,
            )
    if args.method in {"lg", "alg"}:
        guidance = LocalityGuidance(
            teacher_channels=(
                RESNET50_TEACHER_CHANNELS if scientific_cub_r50 else (16, 32, 64)
            )
        ).to(device)
        controller = GuidanceController(
            kind=args.method,
            warmup_epochs=(
                args.alg_controller_warmup_epochs if args.method == "alg" else 0
            ),
        )
    elif args.method == "ibkd":
        guidance = IBKD(
            teacher_channels=(
                RESNET50_TEACHER_CHANNELS if scientific_cub_r50 else (16, 32, 64)
            )
        ).to(device)
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
    # Guidance-module initialization must not shift student DropPath randomness.
    # The DataLoader owns a separate seeded generator, so this also preserves the
    # matched stochastic student path for a given seed across all six variants.
    seed_everything(args.seed)
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro = float("-inf")
    best_validation: dict[str, float] | None = None
    best_student_state: dict[str, torch.Tensor] | None = None
    best_guidance_state: dict[str, torch.Tensor] | None = None
    best_controller_state: dict[str, Any] | None = None
    start_all = time.perf_counter()
    log(
        f"[STUDENT_START] method={args.method} batch={args.batch_size} "
        f"lambda={args.fusion_ratio} seed={args.seed} initial_sha256={initial_hash} "
        f"alg_controller_warmup={args.alg_controller_warmup_epochs} "
        f"posthoc_diagnostic_id={args.posthoc_diagnostic_id}"
    )

    for epoch in range(1, PLANNED_EPOCHS + 1):
        beta = 0.0 if controller is None else controller.beta_for_epoch(epoch)
        synchronize(device)
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        student.train()
        if guidance is not None:
            guidance.train()
        total = 0
        correct = 0
        totals = {"loss": 0.0, "ce": 0.0, "guidance": 0.0, "align": 0.0, "fuse": 0.0}
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            teacher_features: list[torch.Tensor] | None = None
            teacher_logits: torch.Tensor | None = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_inputs = teacher_view(
                        images, image_size=224 if scientific_cub_r50 else 32
                    )
                    if args.method == "kd":
                        teacher_logits = teacher(teacher_inputs)
                    elif beta > 0.0:
                        teacher_features = list(teacher.forward_features(teacher_inputs))

            align = images.new_zeros(())
            fuse = images.new_zeros(())
            feature_loss = images.new_zeros(())
            if args.method in {"lg", "alg", "ibkd"} and beta > 0.0:
                student_features, logits = forward_student_spatial(student, images)
            else:
                student_features = []
                logits = student(images)
            ce = F.cross_entropy(logits, targets)
            if args.method == "vanilla":
                loss = ce
            elif args.method == "kd":
                assert teacher_logits is not None
                feature_loss = kd_loss(logits, teacher_logits)
                loss = (1.0 - KD_ALPHA) * ce + KD_ALPHA * feature_loss
            elif args.method in {"lg", "alg"}:
                if beta > 0.0:
                    assert isinstance(guidance, LocalityGuidance)
                    assert teacher_features is not None
                    feature_loss = guidance(student_features, teacher_features)
                loss = ce + beta * feature_loss
            else:
                if beta > 0.0:
                    assert isinstance(guidance, IBKD)
                    assert teacher_features is not None
                    assert args.fusion_ratio is not None
                    align, fuse = guidance(student_features, teacher_features)
                    feature_loss = (
                        args.fusion_ratio * fuse
                        + (1.0 - args.fusion_ratio) * align
                    )
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
        elapsed = time.perf_counter() - epoch_start
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        )
        row = {
            "epoch": epoch,
            "lr": epoch_lr,
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
        history.append(row)
        if validation["macro_top1"] > best_macro:
            best_macro = validation["macro_top1"]
            best_epoch = epoch
            best_validation = dict(validation)
            best_student_state = clone_state_dict(student)
            best_guidance_state = (
                None if guidance is None else clone_state_dict(guidance)
            )
            best_controller_state = (
                None if controller is None else copy.deepcopy(controller.state_dict())
            )
        write_epoch_status(
            status_path,
            args=args,
            history=history,
            best_epoch=best_epoch,
            best_macro_top1=best_macro,
        )
        log(
            f"[STUDENT_EPOCH] method={args.method} batch={args.batch_size} "
            f"lambda={args.fusion_ratio} seed={args.seed} "
            f"epoch={epoch}/{PLANNED_EPOCHS} lr={epoch_lr:.8g} beta={beta:g} "
            f"loss={totals['loss'] / total:.4f} val_macro={validation['macro_top1']:.2f} "
            f"best_epoch={best_epoch} time={elapsed:.2f}s"
        )
        scheduler.step(epoch)

    if best_student_state is None or best_validation is None:
        raise RuntimeError("Student training produced no selected checkpoint")
    student.load_state_dict(best_student_state, strict=True)
    selected_student_hash = state_dict_sha256(student)
    is_alg_warmup20_diagnostic = (
        _dataset_key(args) == "pet"
        and args.posthoc_diagnostic_id == ALG_WARMUP20_DIAGNOSTIC_ID
    )
    metadata = {
        "purpose": (
            "phase1_cub_r50_224_batch_profile_full_student_v4"
            if scientific_cub_r50
            else "phase1_posthoc_alg_warmup20_full_student"
            if is_alg_warmup20_diagnostic
            else "phase1_scientific_full_student"
        ),
        "posthoc_diagnostic": is_alg_warmup20_diagnostic,
        "posthoc_diagnostic_id": args.posthoc_diagnostic_id,
        "canonical_phase1_result_replaced": False,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "architecture": "deit_tiny_patch16_224",
        "method": args.method,
        "fusion_ratio_lambda": args.fusion_ratio,
        "batch_size": args.batch_size,
        "epochs": PLANNED_EPOCHS,
        "seed": args.seed,
        "selection_metric": "validation_macro_top1",
        "selection_tie_break": "earlier_epoch",
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "validation_image_ids_sha256": manifest["validation_image_ids_sha256"],
        "student_state_sha256": selected_student_hash,
        "initial_student_state_sha256": initial_hash,
        "teacher_model_state_sha256": teacher_state_hash,
        "teacher_checkpoint_sha256": teacher_checkpoint_hash,
        "teacher_architecture": args.teacher_architecture,
        "protocol_config_sha256": protocol_config_sha256,
        "batch_profile_role": args.batch_profile_role,
        "eligible_locked_v3_matrix_cell": scientific_cub_r50 and args.batch_size == 128,
        "final_confirmatory_matrix_complete": False if scientific_cub_r50 else None,
        "controller_warmup_epochs": args.alg_controller_warmup_epochs,
        "guidance_controller_warmup_epochs": (
            None if controller is None else controller.warmup_epochs
        ),
        "official_test_policy": "once_after_validation_selection",
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    atomic_torch_save(
        {
            "student": best_student_state,
            "guidance": best_guidance_state,
            "controller": best_controller_state,
            "metadata": metadata,
        },
        checkpoint_path,
    )

    # Reload the serialized selection into a fresh student before the one test pass.
    del train_loader, validation_loader
    student.to("cpu")
    if guidance is not None:
        guidance.to("cpu")
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selected_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student = create_student(num_classes=num_classes, drop_path_rate=0.1)
    student.load_state_dict(selected_payload["student"], strict=True)
    if state_dict_sha256(student) != selected_student_hash:
        raise RuntimeError("Reloaded student state does not match selected state")
    student.to(device)

    # Official test is instantiated only after validation selection and strict load.
    test_loader = test_loader_builder(
        args.data_dir,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    test_metrics = evaluate(
        student,
        test_loader,
        device,
        teacher=False,
        num_classes=num_classes,
    )
    summary = {
        "status": "complete",
        "scientific_result": True,
        "confirmatory_main_result": (
            args.batch_size == 128
            if scientific_cub_r50
            else not is_alg_warmup20_diagnostic
        ),
        "posthoc_diagnostic": is_alg_warmup20_diagnostic,
        "posthoc_diagnostic_id": args.posthoc_diagnostic_id,
        "canonical_phase1_result_replaced": False,
        "kind": "student",
        "dataset": dataset_name,
        "num_classes": num_classes,
        "method": args.method,
        "fusion_ratio_lambda": args.fusion_ratio,
        "batch_size": args.batch_size,
        "epochs": PLANNED_EPOCHS,
        "seed": args.seed,
        "initial_student_state_sha256": initial_hash,
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "official_test": test_metrics,
        "official_test_evaluations": 1,
        "official_test_used_for_training_or_selection": False,
        "selected_checkpoint_strict_reloaded": True,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "student_state_sha256": selected_student_hash,
        "teacher_checkpoint_sha256": teacher_checkpoint_hash,
        "teacher_model_state_sha256": teacher_state_hash,
        "teacher_metadata": teacher_metadata,
        "teacher_architecture": args.teacher_architecture,
        "protocol_config_sha256": protocol_config_sha256,
        "batch_profile_role": args.batch_profile_role,
        "eligible_locked_v3_matrix_cell": scientific_cub_r50 and args.batch_size == 128,
        "final_confirmatory_matrix_complete": False if scientific_cub_r50 else None,
        "controller_final": None if controller is None else controller.state_dict(),
        "alg_controller_warmup_epochs": args.alg_controller_warmup_epochs,
        "guidance_controller_warmup_epochs": (
            None if controller is None else controller.warmup_epochs
        ),
        "optimizer_contract": "shared_single_group_adamw_all_trainable_parameters_wd_0.05",
        "split_manifest": manifest,
        "training_seconds": time.perf_counter() - start_all,
        "history": history,
        "runtime": runtime_metadata(device),
    }
    save_json({**summary, "status": "complete"}, status_path)
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    try:
        validate_args(args)
        import timm

        if timm.__version__ != "1.0.27":
            raise RuntimeError(f"Expected timm==1.0.27, found {timm.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 1 full run requires CUDA")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        log(
            f"[PHASE1_FULL_START] dataset={_dataset_contract(args)[0]} "
            f"kind={args.kind} method={args.method} "
            f"batch={args.batch_size} lambda={args.fusion_ratio} seed={args.seed} "
            f"teacher_architecture={args.teacher_architecture} "
            f"alg_controller_warmup={args.alg_controller_warmup_epochs} "
            f"posthoc_diagnostic_id={args.posthoc_diagnostic_id}"
        )
        payload = (
            run_teacher(args, device)
            if args.kind == "teacher"
            else run_student(args, device)
        )
        save_json(payload, run_dir / "summary.json")
        log(
            f"[PHASE1_FULL_DONE] dataset={_dataset_contract(args)[0]} "
            f"kind={args.kind} method={args.method} "
            f"batch={args.batch_size} lambda={args.fusion_ratio} seed={args.seed} "
            f"selected_epoch={payload['selected_epoch']} "
            f"test_macro={payload['official_test']['macro_top1']:.3f} "
            f"alg_controller_warmup={args.alg_controller_warmup_epochs} "
            f"elapsed={format_duration(payload['training_seconds'])}"
        )
    except Exception as error:
        message = str(error)
        failure_kind = (
            "cuda_oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            or "out of memory" in message.lower()
            else "runtime_error"
        )
        save_json(
            {
                "status": "failed",
                "scientific_result": False,
                "dataset": _dataset_contract(args)[0],
                "kind": args.kind,
                "method": args.method,
                "fusion_ratio_lambda": args.fusion_ratio,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "posthoc_diagnostic_id": args.posthoc_diagnostic_id,
                "alg_controller_warmup_epochs": args.alg_controller_warmup_epochs,
                "teacher_architecture": args.teacher_architecture,
                "batch_profile_role": args.batch_profile_role,
                "failure_kind": failure_kind,
                "error_type": type(error).__name__,
                "error": message,
                "official_test_evaluations": 0,
            },
            run_dir / "failure.json",
        )
        log(
            f"[PHASE1_FULL_FAILED] dataset={_dataset_contract(args)[0]} "
            f"kind={args.kind} method={args.method} "
            f"batch={args.batch_size} seed={args.seed} "
            f"failure={failure_kind}:{type(error).__name__}:{message}"
        )
        raise


if __name__ == "__main__":
    main()
