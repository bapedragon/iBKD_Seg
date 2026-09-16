"""Real-data CUB direct-segmentation smoke for Vanilla/LG/ALG/iBKD.

This entrypoint verifies data, optimization, guidance, checkpoint, and metric
connections.  Its three-step diagnostic scores are never scientific results.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from ibkd_seg.cityscapes.data import json_hash, save_json, sha256
from ibkd_seg.cityscapes.evaluation import autocast
from ibkd_seg.cityscapes.models import Segmenter, build_guidance
from ibkd_seg.cityscapes.runtime import (
    REPO,
    controller_for,
    seed_all,
    state_hash,
)

from .data import prepare_fixed_batches


CONFIG = REPO / "phase1/phase1_cub_Seg/configs/direct_segmentation_smoke_v1.json"
METHODS = ("vanilla", "lg", "alg", "ibkd")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id", "scientific_result", "data", "task", "teacher", "student",
        "initialization", "input_size", "crop_size", "num_classes", "decoder_channels",
        "drop_path_rate", "gradient_checkpointing", "batch_size",
        "validation_batch_size", "steps", "train_samples", "validation_samples",
        "data_seed", "teacher_seed", "student_seed", "methods", "precision",
        "optimizer", "learning_rate", "weight_decay", "gradient_clip_norm",
        "attention_query_chunk", "guidance_beta", "alg_window", "alg_threshold",
        "alg_warmup_epochs", "ibkd_warmup_epochs", "ibkd_fusion_ratio",
        "primary_metric", "reported_metrics", "full_training_authorized",
    }
    if set(config) != required:
        raise ValueError(
            f"CUB segmentation smoke config keys differ: missing={required - set(config)}, "
            f"extra={set(config) - required}"
        )
    if config["scientific_result"] is not False or config["full_training_authorized"] is not False:
        raise ValueError("Smoke must remain explicitly non-scientific and bounded")
    if config["methods"] != list(METHODS) or config["num_classes"] != 2:
        raise ValueError("Smoke requires Vanilla/LG/ALG/iBKD and two output classes")
    if config["crop_size"] != [config["input_size"], config["input_size"]]:
        raise ValueError("crop_size and square input_size must agree")
    for key in (
        "input_size", "decoder_channels", "batch_size", "validation_batch_size",
        "steps", "train_samples", "validation_samples", "attention_query_chunk",
        "alg_window",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid positive integer: {key}")
    if config["input_size"] % 16 or config["decoder_channels"] % 8:
        raise ValueError("input_size must divide into ViT patches; decoder width into GroupNorm")
    if config["train_samples"] != config["steps"] * config["batch_size"]:
        raise ValueError("train_samples must equal steps * batch_size")
    for key in ("data_seed", "teacher_seed", "student_seed", "alg_warmup_epochs", "ibkd_warmup_epochs"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"invalid nonnegative integer: {key}")
    for key in ("learning_rate", "weight_decay", "gradient_clip_norm", "guidance_beta"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"invalid positive scalar: {key}")
    if not 0 <= config["drop_path_rate"] < 1 or not 0 < config["ibkd_fusion_ratio"] <= 1:
        raise ValueError("invalid drop_path_rate or ibkd_fusion_ratio")
    if config["precision"] not in {"fp32", "bf16"} or config["optimizer"] != "adamw":
        raise ValueError("unsupported precision or optimizer")
    if config["primary_metric"] != "two_class_miou":
        raise ValueError("CUB binary segmentation smoke is centered on two-class mIoU")
    return config


def source_hash() -> str:
    paths = [
        Path(__file__),
        Path(__file__).with_name("data.py"),
        REPO / "src/ibkd_seg/cityscapes/models.py",
        REPO / "src/ibkd_seg/phase1/models.py",
        REPO / "src/ibkd_seg/phase1/controllers.py",
        REPO / "src/ibkd_seg/phase1/cub_data.py",
        REPO / "src/ibkd_seg/phase1/cub_probe_data.py",
    ]
    return json_hash({str(path.relative_to(REPO)): sha256(path) for path in paths})


def check_device(device: torch.device, config: dict) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This smoke profile requires CUDA BF16 support")
    elif config["precision"] != "fp32":
        raise RuntimeError("CPU verification requires the explicit FP32 small profile")


def optimizer_for(model, guidance, config):
    modules = [model] + ([] if guidance is None else [guidance])
    decay, no_decay = [], []
    for module in modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            target = (
                no_decay
                if parameter.ndim <= 1
                or name.endswith("bias")
                or "pos_embed" in name
                or "cls_token" in name
                else decay
            )
            target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
    )


def _has_nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(parameter.grad.detach().abs().max() > 0)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def binary_metrics(matrix: torch.Tensor) -> dict:
    if tuple(matrix.shape) != (2, 2) or int(matrix.sum()) <= 0:
        raise ValueError("binary confusion matrix must be nonempty and 2x2")
    values = matrix.double().cpu()
    intersection = values.diag()
    union = values.sum(0) + values.sum(1) - intersection
    iou = intersection / union.clamp_min(1)
    foreground_denom = 2 * intersection[1] + values[0, 1] + values[1, 0]
    return {
        "two_class_miou": float(iou.mean()),
        "background_iou": float(iou[0]),
        "foreground_iou": float(iou[1]),
        "foreground_dice": float(2 * intersection[1] / foreground_denom.clamp_min(1)),
        "pixel_accuracy": float(intersection.sum() / values.sum()),
        "valid_pixels": int(values.sum()),
        "confusion_matrix_truth_rows_prediction_columns": matrix.tolist(),
    }


@torch.inference_mode()
def evaluate(model, batches, device, config):
    model.eval()
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    for images, targets in batches:
        images = images.to(device)
        with autocast(device, config["precision"]):
            logits = model(images)
        logits = F.interpolate(
            logits.float(),
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits.argmax(1).cpu()
        matrix += torch.bincount(
            2 * targets.reshape(-1) + prediction.reshape(-1),
            minlength=4,
        ).reshape(2, 2)
    return binary_metrics(matrix)


def _environment(device, config):
    import timm
    import torchvision

    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": config["precision"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
    }


def run_teacher(config, data_dir, output, device):
    train_batches, validation_batches, identity, _ = prepare_fixed_batches(data_dir, config)
    seed_all(config["teacher_seed"])
    model = Segmenter("teacher", config).to(device)
    initial_hash = state_hash(model)
    optimizer = optimizer_for(model, None, config)
    losses = []
    seed_all(config["teacher_seed"] + 10_000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, (images, targets) in enumerate(train_batches, 1):
        images, targets = images.to(device), targets.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, config["precision"]):
            logits = model(images)
            logits = F.interpolate(
                logits.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite teacher segmentation loss")
        loss.backward()
        if not _has_nonzero_gradient(model.encoder) or not _has_nonzero_gradient(model.decoder):
            raise RuntimeError("teacher encoder/decoder gradient path is missing")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        losses.append({"step": index, "ce": float(loss.detach()), "grad_norm": float(norm)})
        print(f"[CUB_SEG_TEACHER_STEP] step={index} ce={losses[-1]['ce']:.6g}", flush=True)
    final_hash = state_hash(model)
    if final_hash == initial_hash:
        raise RuntimeError("teacher optimizer did not change model state")
    validation = evaluate(model, validation_batches, device, config)
    checkpoint = output / "teacher.pt"
    torch.save(
        {
            "artifact": "cub_direct_segmentation_smoke_teacher",
            "scientific_result": False,
            "config_sha256": json_hash(config),
            "source_sha256": source_hash(),
            "data_identity_sha256": json_hash(identity),
            "model": model.state_dict(),
            "model_state_sha256": final_hash,
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded = Segmenter("teacher", config)
    reloaded.load_state_dict(saved["model"], strict=True)
    if state_hash(reloaded) != final_hash:
        raise RuntimeError("strict teacher checkpoint reload changed state")
    summary = {
        "status": "passed",
        "scientific_result": False,
        "kind": "teacher",
        "steps": len(losses),
        "losses": losses,
        "diagnostic_validation": validation,
        "initial_model_state_sha256": initial_hash,
        "final_model_state_sha256": final_hash,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "strict_reload": "passed",
        "data_identity": identity,
        "environment": _environment(device, config),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "score_use": "three-step pipeline diagnostic only; not a trained teacher result",
    }
    save_json(output / "summary.json", summary)
    return summary


def _load_teacher(checkpoint, config, identity, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "artifact": "cub_direct_segmentation_smoke_teacher",
        "scientific_result": False,
        "config_sha256": json_hash(config),
        "source_sha256": source_hash(),
        "data_identity_sha256": json_hash(identity),
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError("teacher checkpoint contract differs from this smoke run")
    teacher = Segmenter("teacher", config).to(device)
    teacher.load_state_dict(saved["model"], strict=True)
    teacher.requires_grad_(False).eval()
    if state_hash(teacher) != saved["model_state_sha256"]:
        raise RuntimeError("loaded teacher state hash mismatch")
    return teacher, saved["model_state_sha256"]


def run_method(method, config, data_dir, teacher_checkpoint, output, device):
    train_batches, validation_batches, identity, _ = prepare_fixed_batches(data_dir, config)
    seed_all(config["student_seed"])
    model = Segmenter("student", config).to(device)
    initial_hash = state_hash(model)
    guidance = None if method == "vanilla" else build_guidance(method, config).to(device)
    teacher = teacher_hash = None
    if method != "vanilla":
        teacher, teacher_hash = _load_teacher(
            teacher_checkpoint, config, identity, device
        )
    optimizer = optimizer_for(model, guidance, config)
    controller = controller_for(method, config)
    beta = 0.0 if controller is None else controller.beta_for_epoch(1)
    seed_all(config["student_seed"] + 10_000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses, seconds = [], []
    modules = {"encoder": model.encoder, "decoder": model.decoder}
    if guidance is not None:
        modules["guidance"] = guidance
    parameters = [
        parameter
        for module in modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    for index, (images, targets) in enumerate(train_batches, 1):
        images, targets = images.to(device), targets.to(device)
        model.train()
        if guidance is not None:
            guidance.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with autocast(device, config["precision"]):
            logits, student_features = model(images, return_features=True)
            logits = F.interpolate(
                logits.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            ce = F.cross_entropy(logits, targets)
            guided = ce.new_zeros(())
            if guidance is not None:
                with torch.no_grad():
                    teacher_features = teacher.features(images)[1:]
                if method == "ibkd":
                    align, fuse = guidance(student_features, teacher_features)
                    ratio = config["ibkd_fusion_ratio"]
                    guided = (1 - ratio) * align + ratio * fuse
                else:
                    guided = guidance(student_features, teacher_features)
            loss = ce + beta * guided
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {method} segmentation/guidance loss")
        loss.backward()
        for name, module in modules.items():
            if not _has_nonzero_gradient(module):
                raise RuntimeError(f"missing nonzero gradient path: {name}")
        if teacher is not None and any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("frozen teacher received gradients")
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config["gradient_clip_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds.append(time.perf_counter() - started)
        losses.append(
            {
                "step": index,
                "total": float(loss.detach()),
                "ce": float(ce.detach()),
                "guidance": float(guided.detach()),
                "beta": beta,
                "grad_norm": float(norm),
            }
        )
        print(
            f"[CUB_DIRECT_SEGMENTATION_STEP] method={method} step={index} "
            f"loss={losses[-1]['total']:.6g} seconds={seconds[-1]:.3f}",
            flush=True,
        )
    final_hash = state_hash(model)
    if final_hash == initial_hash:
        raise RuntimeError("student optimizer did not change model state")
    if controller is not None:
        controller.observe(
            1,
            sum(row["guidance"] for row in losses) / len(losses),
            beta_used=beta,
        )
    if teacher is not None and state_hash(teacher) != teacher_hash:
        raise RuntimeError("frozen teacher state changed")
    validation = evaluate(model, validation_batches, device, config)

    checkpoint = output / "smoke_checkpoint.pt"
    torch.save(
        {
            "artifact": "cub_direct_segmentation_method_smoke",
            "scientific_result": False,
            "method": method,
            "config_sha256": json_hash(config),
            "source_sha256": source_hash(),
            "data_identity_sha256": json_hash(identity),
            "model": model.state_dict(),
            "guidance": None if guidance is None else guidance.state_dict(),
            "optimizer": optimizer.state_dict(),
            "controller": None if controller is None else controller.state_dict(),
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded_model = Segmenter("student", config)
    reloaded_model.load_state_dict(saved["model"], strict=True)
    if state_hash(reloaded_model) != final_hash:
        raise RuntimeError("strict student checkpoint reload changed state")
    if guidance is not None:
        reloaded_guidance = build_guidance(method, config)
        reloaded_guidance.load_state_dict(saved["guidance"], strict=True)

    result = {
        "status": "passed",
        "scientific_result": False,
        "method": method,
        "steps": len(losses),
        "losses": losses,
        "seconds_per_step": seconds,
        "diagnostic_validation": validation,
        "primary_diagnostic_metric": "two_class_miou",
        "student_initial_state_sha256": initial_hash,
        "student_final_state_sha256": final_hash,
        "teacher_state_sha256": teacher_hash,
        "teacher_frozen_verified": teacher is not None,
        "strict_reload": "passed",
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "controller": None if controller is None else controller.state_dict(),
        "data_identity": identity,
        "environment": _environment(device, config),
        "student_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "guidance_parameters": (
            0 if guidance is None else sum(parameter.numel() for parameter in guidance.parameters())
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "score_use": "fixed-subset three-step smoke diagnostic; no ranking or checkpoint selection",
    }
    save_json(output / "summary.json", result)
    del model, guidance, teacher, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _small_cpu_config(config):
    reduced = dict(config)
    reduced.update(
        protocol_id=config["protocol_id"] + "_cpu_small",
        input_size=64,
        crop_size=[64, 64],
        decoder_channels=8,
        batch_size=1,
        validation_batch_size=1,
        steps=1,
        train_samples=1,
        validation_samples=1,
        attention_query_chunk=16,
        precision="fp32",
    )
    return reduced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-small", action="store_true")
    parser.add_argument("--teacher", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--teacher-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.teacher and args.method:
        parser.error("--teacher and --method are mutually exclusive")
    config = load_config(args.config)
    if args.cpu_small:
        if args.device != "cpu":
            parser.error("--cpu-small requires --device cpu")
        config = _small_cpu_config(config)
    elif args.device != "cuda":
        parser.error("CPU requires --cpu-small; H200 geometry must not be silently reduced")
    device = torch.device(args.device)
    check_device(device, config)
    torch.set_num_threads(2 if device.type == "cpu" else 4)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)

    if args.teacher:
        run_teacher(config, args.data_dir, output, device)
        return
    if args.method:
        if args.method != "vanilla" and args.teacher_checkpoint is None:
            parser.error("guided methods require --teacher-checkpoint")
        run_method(
            args.method,
            config,
            args.data_dir,
            args.teacher_checkpoint,
            output,
            device,
        )
        return

    _, _, identity, audit = prepare_fixed_batches(args.data_dir, config)
    save_json(output / "dataset_audit.json", audit)
    report = {
        "status": "running",
        "scientific_result": False,
        "config": config,
        "config_sha256": json_hash(config),
        "source_sha256": source_hash(),
        "data_identity": identity,
        "teacher": None,
        "runs": [],
        "limitations": [
            "Teacher and students run only three optimizer steps from scratch",
            "Only six train and eight validation images are used after the full pair inventory check",
            "Diagnostic scores cannot rank methods or establish segmentation performance",
            "No official test image is used for model input, selection, or metrics",
        ],
    }
    save_json(output / "smoke_summary.json", report)
    common = [
        sys.executable,
        "-m",
        "ibkd_seg.cub_segmentation.smoke",
        "--device",
        args.device,
        "--data-dir",
        str(args.data_dir.resolve()),
        "--config",
        str(args.config.resolve()),
    ]
    if args.cpu_small:
        common.append("--cpu-small")
    try:
        teacher_output = output / "teacher"
        print("[CUB_DIRECT_SEGMENTATION_TEACHER_START]", flush=True)
        subprocess.run(
            common + ["--output-dir", str(teacher_output), "--teacher"],
            check=True,
        )
        report["teacher"] = json.loads((teacher_output / "summary.json").read_text())
        teacher_checkpoint = teacher_output / "teacher.pt"
        save_json(output / "smoke_summary.json", report)
        for method in METHODS:
            print(f"[CUB_DIRECT_SEGMENTATION_METHOD_START] method={method}", flush=True)
            command = common + ["--output-dir", str(output / method), "--method", method]
            if method != "vanilla":
                command += ["--teacher-checkpoint", str(teacher_checkpoint)]
            subprocess.run(command, check=True)
            row = json.loads((output / method / "summary.json").read_text())
            report["runs"].append(row)
            save_json(output / "smoke_summary.json", report)
            metrics = row["diagnostic_validation"]
            print(
                f"[CUB_DIRECT_SEGMENTATION_METHOD_DONE] method={method} "
                f"diagnostic_miou={metrics['two_class_miou']:.6f} "
                f"foreground_iou={metrics['foreground_iou']:.6f}",
                flush=True,
            )
        if len({row["student_initial_state_sha256"] for row in report["runs"]}) != 1:
            raise RuntimeError("student initialization differs across methods")
        guided = [row for row in report["runs"] if row["method"] != "vanilla"]
        if {row["teacher_state_sha256"] for row in guided} != {
            report["teacher"]["final_model_state_sha256"]
        }:
            raise RuntimeError("guided methods did not share the trained teacher")
        if len({json_hash(row["data_identity"]) for row in report["runs"]}) != 1:
            raise RuntimeError("method input images/masks/transforms differ")
        if any(row["data_identity"] != identity for row in report["runs"]):
            raise RuntimeError("child data identity differs from parent audit")
        report["status"] = "passed"
        save_json(output / "smoke_summary.json", report)
        final_results = {
            "teacher": {
                "losses": report["teacher"]["losses"],
                "diagnostic_validation": report["teacher"]["diagnostic_validation"],
            },
            "methods": {
                row["method"]: {
                    "losses": row["losses"],
                    "diagnostic_validation": row["diagnostic_validation"],
                    "controller": row["controller"],
                    "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                }
                for row in report["runs"]
            },
        }
        print(
            "[CUB_DIRECT_SEGMENTATION_SMOKE_DONE] "
            f"status=passed methods=4/4 scientific_result=false summary={output / 'smoke_summary.json'}",
            flush=True,
        )
        print(
            "[CUB_DIRECT_SEGMENTATION_SMOKE_FINAL_RESULTS] "
            + json.dumps(final_results, sort_keys=True),
            flush=True,
        )
    except Exception as error:
        report.update(status="failed", error=repr(error))
        save_json(output / "smoke_summary.json", report)
        print(
            f"[CUB_DIRECT_SEGMENTATION_SMOKE_DONE] status=failed error={error!r}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
