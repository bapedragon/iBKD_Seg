"""Resumable exploratory full training for CUB binary segmentation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ibkd_seg.cityscapes.data import json_hash, save_json, sha256
from ibkd_seg.cityscapes.evaluation import autocast
from ibkd_seg.cityscapes.models import Segmenter, build_guidance
from ibkd_seg.cityscapes.runtime import (
    REPO,
    controller_for,
    restore_controller,
    seed_all,
    state_hash,
)
from ibkd_seg.phase1.cub_data import (
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
    OFFICIAL_TEST_COUNT,
)
from ibkd_seg.phase1.cub_probe_data import (
    ids_sha256,
    load_official_test_records,
    load_train_validation_records,
)

from .data import load_pair
from .smoke import METHODS, binary_metrics


CONFIG = REPO / "phase1/phase1_cub_Seg/configs/direct_segmentation_full_v1.json"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id", "status", "scientific_result", "task", "data", "split",
        "input_size", "crop_size", "num_classes", "decoder_channels",
        "drop_path_rate", "gradient_checkpointing", "teacher", "student",
        "batch_size", "evaluation_batch_size", "num_workers", "precision",
        "gradient_clip_norm", "train_transform", "evaluation_transform", "methods",
        "guidance_beta", "alg_window", "alg_threshold", "alg_warmup_epochs",
        "ibkd_warmup_epochs", "ibkd_fusion_ratio", "attention_query_chunk",
        "selection_metric", "selection_tie_break", "reported_metrics",
        "validation_every_epochs", "checkpoint_every_epochs", "resume", "test_policy",
        "full_training_authorized",
    }
    if set(config) != required:
        raise ValueError(
            f"full config keys differ: missing={required - set(config)}, "
            f"extra={set(config) - required}"
        )
    if config["scientific_result"] is not False or config["full_training_authorized"] is not True:
        raise ValueError("v1 must remain an explicitly authorized exploratory full run")
    if config["status"] != "locked_after_real_data_smoke_before_full_results":
        raise ValueError("full protocol must be locked before results")
    if config["methods"] != list(METHODS) or config["num_classes"] != 2:
        raise ValueError("full matrix requires Vanilla/LG/ALG/iBKD and two classes")
    if config["split"] != {
        "train": 5394,
        "validation": 600,
        "test": 5794,
        "source": "official train split with fixed 3-per-class validation holdout; official test used only after validation selection",
        "seed": 2027,
    }:
        raise ValueError("CUB split contract changed")
    if config["crop_size"] != [config["input_size"], config["input_size"]]:
        raise ValueError("crop_size must match square input_size")
    for key in (
        "input_size", "decoder_channels", "batch_size", "evaluation_batch_size",
        "validation_every_epochs", "checkpoint_every_epochs", "attention_query_chunk",
        "alg_window",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid positive integer: {key}")
    if config["input_size"] % 16 or config["decoder_channels"] % 8:
        raise ValueError("invalid ViT patch or GroupNorm geometry")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    for role in ("teacher", "student"):
        block = config[role]
        if block["seed"] < 0 or block["epochs"] <= 0 or block["warmup_epochs"] < 0:
            raise ValueError(f"invalid {role} seed/epoch schedule")
        if block["warmup_epochs"] >= block["epochs"]:
            raise ValueError(f"{role} warmup must be shorter than training")
        for key in ("learning_rate", "weight_decay"):
            if not math.isfinite(block[key]) or block[key] <= 0:
                raise ValueError(f"invalid {role}.{key}")
        if not 0 <= block["minimum_learning_rate"] < block["learning_rate"]:
            raise ValueError(f"invalid {role} minimum learning rate")
    if config["teacher"]["optimizer"] != "sgd" or config["student"]["optimizer"] != "adamw":
        raise ValueError("optimizer contract changed")
    if not 0 <= config["teacher"]["momentum"] < 1:
        raise ValueError("invalid teacher momentum")
    if config["precision"] not in {"bf16", "fp32"}:
        raise ValueError("unsupported precision")
    if config["selection_metric"] != "two_class_miou" or config["selection_tie_break"] != "earlier_epoch":
        raise ValueError("selection contract changed")
    if config["reported_metrics"] != [
        "two_class_miou",
        "foreground_iou",
        "background_iou",
        "foreground_dice",
        "pixel_accuracy",
    ]:
        raise ValueError("reported metric contract changed")
    if config["validation_every_epochs"] != 1 or config["checkpoint_every_epochs"] != 1:
        raise ValueError("v1 validates and checkpoints every epoch")
    if config["resume"] != "strict_epoch_boundary":
        raise ValueError("resume contract changed")
    return config


def source_hash() -> str:
    paths = [
        Path(__file__),
        Path(__file__).with_name("data.py"),
        Path(__file__).with_name("smoke.py"),
        REPO / "src/ibkd_seg/cityscapes/models.py",
        REPO / "src/ibkd_seg/phase1/models.py",
        REPO / "src/ibkd_seg/phase1/controllers.py",
        REPO / "src/ibkd_seg/phase1/cub_data.py",
        REPO / "src/ibkd_seg/phase1/cub_probe_data.py",
    ]
    return json_hash({str(path.relative_to(REPO)): sha256(path) for path in paths})


class CubSegmentationDataset(Dataset):
    def __init__(self, records, config, *, training: bool, seed: int):
        self.records = list(records)
        self.config = config
        self.training = training
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        flip = False
        if self.training:
            digest = hashlib.sha256(
                f"{self.seed}:{self.epoch}:{record.image_id}".encode()
            ).digest()
            flip = bool(digest[0] & 1)
        image, target = load_pair(
            record,
            input_size=self.config["input_size"],
            horizontal_flip=flip,
        )
        return image, target, record.image_id


def load_train_validation(data_dir: Path):
    records, manifest, source = load_train_validation_records(data_dir, download=True)
    counts = {key: len(value) for key, value in records.items()}
    if counts != {"train": DERIVED_TRAIN_COUNT, "validation": DERIVED_VALIDATION_COUNT}:
        raise RuntimeError(f"CUB train/validation count changed: {counts}")
    if {row.image_id for row in records["train"]} & {
        row.image_id for row in records["validation"]
    }:
        raise RuntimeError("CUB train/validation overlap")
    identity = {
        "dataset": "CUB-200-2011 official RGB and segmentation masks",
        "split_manifest_sha256": json_hash(manifest),
        "train_samples": len(records["train"]),
        "validation_samples": len(records["validation"]),
        "train_image_ids_sha256": ids_sha256(records["train"]),
        "validation_image_ids_sha256": ids_sha256(records["validation"]),
        "mask_mapping": "official grayscale > 0 is bird=1; else background=0",
        "official_test_used_for_selection": False,
    }
    return records, identity, {"manifest": manifest, "source": source}


def make_train_loader(dataset, config, epoch):
    dataset.epoch = epoch
    generator = torch.Generator().manual_seed(config["split"]["seed"] * 1_000_003 + epoch)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=False,
        generator=generator,
    )


def make_eval_loader(records, config):
    dataset = CubSegmentationDataset(
        records,
        config,
        training=False,
        seed=config["split"]["seed"],
    )
    return DataLoader(
        dataset,
        batch_size=config["evaluation_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=False,
    )


@torch.inference_mode()
def evaluate(model, loader, device, config):
    model.eval()
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
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


def check_device(device, config):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The full profile requires a CUDA device")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The full profile requires CUDA BF16 support")


def _parameter_groups(model, guidance, weight_decay):
    decay, no_decay = [], []
    modules = [model] + ([] if guidance is None else [guidance])
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
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def optimizer_for(kind, model, guidance, config):
    block = config[kind]
    groups = _parameter_groups(model, guidance, block["weight_decay"])
    if kind == "teacher":
        return torch.optim.SGD(
            groups,
            lr=block["learning_rate"],
            momentum=block["momentum"],
        )
    return torch.optim.AdamW(groups, lr=block["learning_rate"])


def learning_rate(step, steps_per_epoch, role_config):
    total = role_config["epochs"] * steps_per_epoch
    warmup = role_config["warmup_epochs"] * steps_per_epoch
    maximum = role_config["learning_rate"]
    minimum = role_config["minimum_learning_rate"]
    if step < warmup:
        return maximum * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup - 1))
    return minimum + (maximum - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def _has_gradient(module):
    return any(
        parameter.grad is not None and bool(parameter.grad.detach().abs().max() > 0)
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def atomic_torch_save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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


def load_teacher(checkpoint, config, data_identity, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("artifact") != "cub_direct_segmentation_validation_selected"
        or payload["identity"]["role"] != "teacher"
        or payload["identity"]["config_sha256"] != json_hash(config)
        or payload["identity"]["source_sha256"] != source_hash()
        or payload["identity"]["data_identity_sha256"] != json_hash(data_identity)
    ):
        raise RuntimeError("teacher checkpoint contract differs from this full run")
    summary_path = checkpoint.parent / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("teacher completion summary is missing")
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete" or summary["best_checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("teacher checkpoint is not the completed validation selection")
    teacher = Segmenter("teacher", config).to(device)
    teacher.load_state_dict(payload["model"], strict=True)
    teacher.requires_grad_(False).eval()
    teacher_hash = state_hash(teacher)
    if teacher_hash != payload["model_state_sha256"]:
        raise RuntimeError("teacher model state hash mismatch")
    return teacher, teacher_hash


def run_training(
    *,
    role,
    method,
    config,
    data_dir,
    output,
    device,
    teacher_checkpoint=None,
    resume=False,
):
    records, data_identity, _ = load_train_validation(data_dir)
    guided = role == "student" and method != "vanilla"
    if role not in {"teacher", "student"}:
        raise ValueError(role)
    if role == "teacher" and (method is not None or teacher_checkpoint is not None):
        raise ValueError("teacher cannot have a student method/checkpoint")
    if role == "student" and method not in METHODS:
        raise ValueError("unknown student method")
    if guided != (teacher_checkpoint is not None):
        raise ValueError("only guided students require the selected teacher checkpoint")

    output.mkdir(parents=True, exist_ok=True)
    completed = output / "summary.json"
    if completed.is_file():
        summary = json.loads(completed.read_text())
        if (
            summary.get("status") == "complete"
            and summary.get("config_sha256") == json_hash(config)
            and summary.get("source_sha256") == source_hash()
            and summary.get("data_identity_sha256") == json_hash(data_identity)
        ):
            best = output / "best.pt"
            latest = output / "latest.pt"
            if (
                not best.is_file()
                or sha256(best) != summary.get("best_checkpoint_sha256")
                or not latest.is_file()
                or sha256(latest) != summary.get("latest_checkpoint_sha256")
            ):
                raise RuntimeError("completed run checkpoint is missing or changed")
            print(
                f"[CUB_SEG_FULL_ALREADY_COMPLETE] role={role} method={method}",
                flush=True,
            )
            return summary
        raise RuntimeError("existing completion summary differs from this run")

    seed = config[role]["seed"]
    seed_all(seed)
    model = Segmenter(role, config).to(device)
    initial_hash = state_hash(model)
    guidance = None
    teacher = None
    teacher_hash = None
    if role == "student" and method != "vanilla":
        guidance = build_guidance(method, config).to(device)
        teacher, teacher_hash = load_teacher(
            teacher_checkpoint,
            config,
            data_identity,
            device,
        )
    controller = controller_for(method, config) if role == "student" else None
    optimizer = optimizer_for(role, model, guidance, config)
    identity = {
        "protocol_id": config["protocol_id"],
        "config_sha256": json_hash(config),
        "source_sha256": source_hash(),
        "data_identity_sha256": json_hash(data_identity),
        "role": role,
        "method": method,
        "seed": seed,
        "initial_model_state_sha256": initial_hash,
        "teacher_checkpoint_sha256": (
            sha256(teacher_checkpoint) if teacher_checkpoint is not None else None
        ),
        "teacher_model_state_sha256": teacher_hash,
    }
    identity_path = output / "identity.json"
    if identity_path.is_file() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("existing output identity differs from current run")
    save_json(output / "identity.json", identity)
    save_json(output / "config.json", config)
    save_json(output / "environment.json", _environment(device, config))

    train_dataset = CubSegmentationDataset(
        records["train"], config, training=True, seed=config["split"]["seed"]
    )
    validation_loader = make_eval_loader(records["validation"], config)
    epochs = config[role]["epochs"]
    first_epoch = 1
    best_score = -1.0
    best_epoch = None
    best_validation = None
    history = []
    prior_seconds = 0.0
    seed_all(seed + 10_000)
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    if latest_path.is_file():
        if not resume:
            raise FileExistsError(f"checkpoint exists; pass --resume: {latest_path}")
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise RuntimeError("resume identity differs from current data/config/source")
        model.load_state_dict(saved["model"], strict=True)
        if guidance is not None:
            guidance.load_state_dict(saved["guidance"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_controller(controller, saved["controller"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first_epoch = saved["epoch"] + 1
        best_score = saved["best_score"]
        best_epoch = saved["best_epoch"]
        best_validation = saved["best_validation"]
        history = saved["history"]
        prior_seconds = saved["elapsed_seconds"]
        if best_epoch is not None and (
            not best_path.is_file()
            or sha256(best_path) != saved["best_checkpoint_sha256"]
        ):
            raise RuntimeError("selected best checkpoint is missing or changed")
    elif resume and any(output.iterdir()):
        allowed = {"config.json", "environment.json", "identity.json"}
        if {path.name for path in output.iterdir()} - allowed:
            raise RuntimeError("nonempty output has no resumable latest.pt")

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first_epoch, epochs + 1):
        loader = make_train_loader(train_dataset, config, epoch)
        model.train()
        if guidance is not None:
            guidance.train()
        beta = 0.0 if controller is None else controller.beta_for_epoch(epoch)
        sums = {"total": 0.0, "ce": 0.0, "guidance": 0.0}
        seen = 0
        last_lr = None
        for batch_index, (images, targets, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            step = (epoch - 1) * len(loader) + batch_index
            last_lr = learning_rate(step, len(loader), config[role])
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, config["precision"]):
                logits, features = model(images, return_features=True)
                logits = F.interpolate(
                    logits.float(),
                    size=targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                ce = F.cross_entropy(logits, targets)
                feature_loss = ce.new_zeros(())
                if guidance is not None and beta > 0:
                    with torch.no_grad():
                        teacher_features = teacher.features(images)[1:]
                    if method == "ibkd":
                        align, fuse = guidance(features, teacher_features)
                        ratio = config["ibkd_fusion_ratio"]
                        feature_loss = (1 - ratio) * align + ratio * fuse
                    else:
                        feature_loss = guidance(features, teacher_features)
                loss = ce + beta * feature_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: role={role} method={method} epoch={epoch}")
            loss.backward()
            if not _has_gradient(model.encoder) or not _has_gradient(model.decoder):
                raise RuntimeError("encoder/decoder gradient path is missing")
            if guidance is not None and beta > 0 and not _has_gradient(guidance):
                raise RuntimeError("guidance gradient path is missing")
            if teacher is not None and any(parameter.grad is not None for parameter in teacher.parameters()):
                raise RuntimeError("frozen teacher received gradients")
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(
                parameters,
                config["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            batch = len(images)
            seen += batch
            for key, value in (
                ("total", loss),
                ("ce", ce),
                ("guidance", feature_loss),
            ):
                sums[key] += float(value.detach()) * batch
        losses = {key: value / seen for key, value in sums.items()}
        if controller is not None:
            controller.observe(epoch, losses["guidance"], beta_used=beta)
        validation = evaluate(model, validation_loader, device, config)
        if validation[config["selection_metric"]] > best_score:
            best_score = validation[config["selection_metric"]]
            best_epoch = epoch
            best_validation = validation
            selected_hash = state_hash(model)
            atomic_torch_save(
                best_path,
                {
                    "artifact": "cub_direct_segmentation_validation_selected",
                    "scientific_result": False,
                    "identity": identity,
                    "config": config,
                    "epoch": epoch,
                    "validation": validation,
                    "model_state_sha256": selected_hash,
                    "model": model.state_dict(),
                },
            )
        elapsed = prior_seconds + time.perf_counter() - started
        row = {
            "epoch": epoch,
            "losses": losses,
            "beta": beta,
            "learning_rate": last_lr,
            "validation": validation,
            "best_epoch": best_epoch,
            "best_two_class_miou": best_score,
            "guidance_stop_epoch": None if controller is None else controller.stop_epoch,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        best_hash = sha256(best_path)
        atomic_torch_save(
            latest_path,
            {
                "artifact": "cub_direct_segmentation_latest",
                "identity": identity,
                "epoch": epoch,
                "model": model.state_dict(),
                "guidance": None if guidance is None else guidance.state_dict(),
                "optimizer": optimizer.state_dict(),
                "controller": None if controller is None else controller.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_validation": best_validation,
                "best_checkpoint_sha256": best_hash,
                "history": history,
                "elapsed_seconds": elapsed,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
        )
        save_json(output / "history.json", history)
        print(
            json.dumps(
                {
                    "event": "cub_seg_full_epoch",
                    "role": role,
                    "method": method,
                    **row,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if best_epoch is None or best_validation is None or not best_path.is_file():
        raise RuntimeError("training completed without a validation-selected checkpoint")
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"], strict=True)
    if state_hash(model) != selected["model_state_sha256"]:
        raise RuntimeError("strict selected checkpoint reload changed model state")
    test_records, test_source = load_official_test_records(data_dir, download=True)
    if len(test_records) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("official CUB test count changed")
    test_metrics = evaluate(model, make_eval_loader(test_records, config), device, config)
    summary = {
        "status": "complete",
        "scientific_result": False,
        "role": role,
        "method": method,
        "config_sha256": json_hash(config),
        "source_sha256": source_hash(),
        "data_identity_sha256": json_hash(data_identity),
        "identity": identity,
        "epochs": epochs,
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "official_test": test_metrics,
        "official_test_samples": len(test_records),
        "official_test_image_ids_sha256": ids_sha256(test_records),
        "official_test_source": test_source,
        "last_epoch_losses": history[-1]["losses"],
        "guidance_stop_epoch": None if controller is None else controller.stop_epoch,
        "best_checkpoint_sha256": sha256(best_path),
        "best_checkpoint_bytes": best_path.stat().st_size,
        "latest_checkpoint_sha256": sha256(latest_path),
        "strict_selected_reload": "passed",
        "teacher_frozen_verified": teacher is not None,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "elapsed_seconds": prior_seconds + time.perf_counter() - started,
        "score_use": "exploratory full run; selected only by validation two-class mIoU",
    }
    save_json(output / "summary.json", summary)
    print(
        "[CUB_SEG_FULL_RUN_RESULT] "
        + json.dumps(
            {
                "role": role,
                "method": method,
                "selected_epoch": best_epoch,
                "validation": best_validation,
                "test": test_metrics,
                "last_epoch_losses": history[-1]["losses"],
                "guidance_stop_epoch": summary["guidance_stop_epoch"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del model, guidance, teacher, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def final_results(report):
    def view(summary):
        return {
            "selected_epoch": summary["selected_epoch"],
            "validation": summary["selected_validation"],
            "test": summary["official_test"],
            "last_epoch_losses": summary["last_epoch_losses"],
            "guidance_stop_epoch": summary["guidance_stop_epoch"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
        }

    return {
        "protocol_id": report["config"]["protocol_id"],
        "scientific_result": False,
        "teacher": view(report["teacher"]),
        "methods": {row["method"]: view(row) for row in report["runs"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--role", choices=("teacher", "student"), help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--teacher-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    check_device(device, config)
    torch.set_num_threads(4)
    output = args.output_dir.resolve()

    if args.role:
        if args.role == "student" and args.method is None:
            parser.error("student role requires --method")
        run_training(
            role=args.role,
            method=args.method,
            config=config,
            data_dir=args.data_dir,
            output=output,
            device=device,
            teacher_checkpoint=args.teacher_checkpoint,
            resume=args.resume,
        )
        return
    if args.method or args.teacher_checkpoint:
        parser.error("--method/--teacher-checkpoint require an internal --role")

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("existing output uses a different full config")
    save_json(config_path, config)
    records, data_identity, audit = load_train_validation(args.data_dir)
    del records
    save_json(
        output / "dataset_audit.json",
        {
            "status": "passed",
            "identity": data_identity,
            **audit,
            "official_test_accessed_during_initial_audit": False,
            "note": "Official test is loaded only inside each completed validation-selected run.",
        },
    )
    report_path = output / "full_summary.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("config") != config or report.get("data_identity") != data_identity:
            raise RuntimeError("existing full summary differs from current contract")
    else:
        report = {
            "status": "running",
            "scientific_result": False,
            "config": config,
            "config_sha256": json_hash(config),
            "source_sha256": source_hash(),
            "data_identity": data_identity,
            "teacher": None,
            "runs": [],
        }
        save_json(report_path, report)

    common = [
        sys.executable,
        "-u",
        "-m",
        "ibkd_seg.cub_segmentation.full",
        "--device",
        args.device,
        "--data-dir",
        str(args.data_dir.resolve()),
        "--config",
        str(args.config.resolve()),
    ]
    if args.resume:
        common.append("--resume")
    try:
        teacher_output = output / "teacher"
        subprocess.run(
            common
            + [
                "--output-dir",
                str(teacher_output),
                "--role",
                "teacher",
            ],
            check=True,
        )
        report["teacher"] = json.loads((teacher_output / "summary.json").read_text())
        save_json(report_path, report)
        teacher_checkpoint = teacher_output / "best.pt"
        completed_methods = {row["method"] for row in report["runs"]}
        for method in METHODS:
            command = common + [
                "--output-dir",
                str(output / method),
                "--role",
                "student",
                "--method",
                method,
            ]
            if method != "vanilla":
                command += ["--teacher-checkpoint", str(teacher_checkpoint)]
            subprocess.run(command, check=True)
            row = json.loads((output / method / "summary.json").read_text())
            if method in completed_methods:
                report["runs"] = [item for item in report["runs"] if item["method"] != method]
            report["runs"].append(row)
            save_json(report_path, report)
        order = {method: index for index, method in enumerate(METHODS)}
        report["runs"].sort(key=lambda row: order[row["method"]])
        if len({row["identity"]["initial_model_state_sha256"] for row in report["runs"]}) != 1:
            raise RuntimeError("student initial model state differs across methods")
        guided = [row for row in report["runs"] if row["method"] != "vanilla"]
        selected = torch.load(teacher_checkpoint, map_location="cpu", weights_only=True)
        if {row["identity"]["teacher_model_state_sha256"] for row in guided} != {
            selected["model_state_sha256"]
        }:
            raise RuntimeError("guided methods did not share one selected teacher")
        if any(row["data_identity_sha256"] != json_hash(data_identity) for row in report["runs"]):
            raise RuntimeError("method dataset identities differ")
        report["status"] = "complete"
        report["final_results"] = final_results(report)
        save_json(report_path, report)
        print(
            "[CUB_DIRECT_SEGMENTATION_FULL_DONE] "
            f"status=complete methods=4/4 scientific_result=false summary={report_path}",
            flush=True,
        )
        print(
            "[CUB_DIRECT_SEGMENTATION_FULL_FINAL_RESULTS] "
            + json.dumps(report["final_results"], sort_keys=True),
            flush=True,
        )
    except Exception as error:
        report.update(status="failed", error=repr(error))
        save_json(report_path, report)
        print(f"[CUB_DIRECT_SEGMENTATION_FULL_FAILED] error={error!r}", flush=True)
        raise


if __name__ == "__main__":
    main()
