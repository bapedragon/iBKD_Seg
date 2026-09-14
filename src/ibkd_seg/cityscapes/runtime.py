"""Reproducible teacher/student training, epoch-boundary resume, and evaluation."""

from __future__ import annotations

import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ibkd_seg.phase1.controllers import GuidanceController

from .data import Cityscapes, IGNORE, json_hash, save_json, sha256, verify_manifest
from .evaluation import autocast, evaluate
from .models import Segmenter, build_guidance

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "phase4/phase4_cityscapes/configs/scratch_pixel_accuracy_v2.json"
METHODS = ("vanilla", "lg", "alg", "ibkd")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict):
    expected = set(json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    required = expected - {"selection_metric"}
    if not required.issubset(config) or set(config) - expected:
        raise ValueError(f"Config keys differ: missing={required - set(config)}, extra={set(config) - expected}")
    if config.get("selection_metric", "miou") not in {"pixel_accuracy", "miou"}:
        raise ValueError("Selection metric must be pixel_accuracy or miou")
    if config["initialization"] != "scratch":
        raise ValueError("v1 implements scratch initialization only")
    if config["status"] not in {"pilot_not_paper_result", "synthetic_smoke_not_scientific"}:
        raise ValueError("This runner is an explicitly labeled pilot")
    if not isinstance(config["protocol_id"], str) or not config["protocol_id"]:
        raise ValueError("protocol_id is required")
    for key in ("crop_size", "eval_stride"):
        if len(config[key]) != 2 or any(type(v) is not int or v <= 0 for v in config[key]):
            raise ValueError(f"Invalid {key}")
    if any(v % 32 for v in config["crop_size"]):
        raise ValueError("Crop dimensions must be multiples of 32")
    if any(s > c for s, c in zip(config["eval_stride"], config["crop_size"])):
        raise ValueError("Evaluation stride cannot exceed crop")
    for key in ("epochs", "batch_size", "decoder_channels", "crop_attempts", "attention_query_chunk", "validation_every", "alg_window"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("num_workers", "warmup_epochs", "alg_warmup_epochs", "ibkd_warmup_epochs", "teacher_seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"Invalid {key}")
    if config["decoder_channels"] % 8 or config["warmup_epochs"] >= config["epochs"]:
        raise ValueError("Decoder width must be divisible by 8, warmup shorter than training")
    for key in ("learning_rate", "weight_decay", "poly_power", "gradient_clip_norm", "guidance_beta"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("max_category_ratio", "warmup_start_factor", "ibkd_fusion_ratio"):
        if not 0 < config[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if not 0 <= config["drop_path_rate"] < 1 or not math.isfinite(config["alg_threshold"]):
        raise ValueError("Invalid drop path / threshold")
    scales = config["scale_range"]
    if len(scales) != 2 or not 0 < scales[0] <= scales[1] or not all(math.isfinite(x) for x in scales):
        raise ValueError("Invalid scale range")
    if type(config["gradient_checkpointing"]) is not bool or config["precision"] not in {"fp32", "bf16"}:
        raise ValueError("Invalid execution settings")
    seeds = config["student_seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Invalid student seeds")
    if config["methods"] != list(METHODS):
        raise ValueError("The pilot matrix must include Vanilla/LG/ALG/iBKD")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(module):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hash():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [Path(__file__).parents[1] / "phase1" / name for name in ("models.py", "controllers.py")]
    return json_hash({str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)})


def atomic_save(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def restore_controller(controller, state):
    if controller is None:
        if state is not None:
            raise ValueError("Unexpected controller state")
        return
    if state["kind"] != controller.kind:
        raise ValueError("Controller type mismatch")
    for attribute, key in (
        ("active", "active"), ("stop_epoch", "stop_epoch"),
        ("losses", "loss_history"), ("derivatives", "derivative_history"),
        ("smoothed_derivatives", "smoothed_derivative_history"), ("beta_history", "beta_history"),
    ):
        setattr(controller, attribute, state[key])


def controller_for(method, config):
    if method not in {"lg", "alg", "ibkd"}:
        return None
    return GuidanceController(
        kind=method, beta=config["guidance_beta"], window=config["alg_window"],
        threshold=config["alg_threshold"],
        warmup_epochs=config["ibkd_warmup_epochs"] if method == "ibkd" else config["alg_warmup_epochs"] if method == "alg" else 0,
    )


def lr_at(step, steps_per_epoch, config):
    warmup = config["warmup_epochs"] * steps_per_epoch
    total = config["epochs"] * steps_per_epoch
    if step < warmup:
        factor = config["warmup_start_factor"] + (1 - config["warmup_start_factor"]) * step / max(1, warmup)
    else:
        factor = (1 - (step - warmup) / max(1, total - warmup)) ** config["poly_power"]
    return config["learning_rate"] * factor


def optimizer_for(model, guidance, config):
    decay, no_decay = [], []
    modules = [model] + ([] if guidance is None else [guidance])
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                target = no_decay if parameter.ndim <= 1 or name.endswith("bias") or "pos_embed" in name or "cls_token" in name else decay
                target.append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config["learning_rate"])


def validate_manifest_contract(manifest, config):
    synthetic = config["status"] == "synthetic_smoke_not_scientific"
    if manifest.get("synthetic") is not synthetic or set(manifest["splits"]) != {"train", "val"}:
        raise ValueError("Synthetic/real data or split contract mismatch")
    if not synthetic:
        if [len(manifest["splits"][s]) for s in ("train", "val")] != [2975, 500]:
            raise ValueError("Official fine train/val counts are required")


def check_device(device, config):
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This runner supports CPU smoke or CUDA training")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this machine; use CPU only for smoke")
    if config["precision"] == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 profile requires a compatible CUDA GPU")


def train(*, config, data_root, manifest, output, kind, method, seed, device,
          teacher_checkpoint=None, resume=False, stop_after_epoch=None):
    """Save latest at epoch boundaries; stop_after_epoch is used for resume verification."""
    validate_config(config)
    validate_manifest_contract(manifest, config)
    check_device(device, config)
    if kind not in {"teacher", "student"}:
        raise ValueError(kind)
    if kind == "teacher":
        if method is not None or teacher_checkpoint is not None or seed != config["teacher_seed"]:
            raise ValueError("Teacher uses its fixed seed and no method/teacher checkpoint")
    elif method not in METHODS or seed not in config["student_seeds"]:
        raise ValueError("Student method/seed not in protocol")
    guided = kind == "student" and method != "vanilla"
    if guided != (teacher_checkpoint is not None):
        raise ValueError("Only guided students require a teacher checkpoint")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output is not empty; choose another directory or --resume: {output}")
    if resume and not (output / "latest.pt").is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {output / 'latest.pt'}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = json_hash(manifest)
    code_hash = source_hash()
    seed_all(seed)
    model = Segmenter(kind, config).to(device)
    initial_hash = state_hash(model)
    teacher, teacher_hash = None, None
    if guided:
        teacher_hash = sha256(teacher_checkpoint)
        payload = torch.load(teacher_checkpoint, map_location="cpu", weights_only=True)
        teacher_identity = payload["identity"]
        if (teacher_identity["kind"] != "teacher" or teacher_identity["config_sha256"] != json_hash(config)
                or teacher_identity["manifest_sha256"] != manifest_hash or teacher_identity["source_sha256"] != code_hash):
            raise ValueError("Teacher checkpoint has a different dataset/config/source contract")
        if payload.get("artifact") != "validation_selected_segmenter":
            raise ValueError("Use the teacher's validation-selected best.pt")
        teacher_summary_path = teacher_checkpoint.parent / "summary.json"
        if not teacher_summary_path.is_file():
            raise ValueError("Teacher completion summary is missing")
        teacher_summary = json.loads(teacher_summary_path.read_text())
        if teacher_summary["status"] != "complete" or teacher_summary["best_checkpoint_sha256"] != teacher_hash:
            raise ValueError("Teacher run is incomplete or checkpoint hash differs from summary")
        teacher = Segmenter("teacher", config).to(device)
        teacher.load_state_dict(payload["model"], strict=True)
        teacher.requires_grad_(False).eval()
        del payload
    guidance = build_guidance(method, config) if guided else None
    if guidance is not None:
        guidance = guidance.to(device)
    controller = controller_for(method, config)
    optimizer = optimizer_for(model, guidance, config)
    identity = {
        "protocol_id": config["protocol_id"], "config_sha256": json_hash(config),
        "manifest_sha256": manifest_hash, "source_sha256": code_hash,
        "kind": kind, "method": method, "seed": seed,
        "teacher_checkpoint_sha256": teacher_hash, "initial_model_sha256": initial_hash,
        "synthetic": manifest["synthetic"],
        "selection_metric": config.get("selection_metric", "miou"),
    }
    train_data = Cityscapes(data_root, manifest["splits"]["train"], config, training=True, seed=seed)
    val_data = Cityscapes(data_root, manifest["splits"]["val"], config, training=False, seed=seed)
    validation_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    first_epoch, best, best_epoch, history, prior_seconds = 1, -1.0, None, [], 0.0
    selection_metric = identity["selection_metric"]
    best_validation = None
    # Align dropout RNG after method-specific initialization; data has an independent RNG.
    seed_all(seed + 10000)
    if resume:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume identity differs: dataset/config/source/method/teacher/initialization changed")
        model.load_state_dict(saved["model"], strict=True)
        if guidance is not None:
            guidance.load_state_dict(saved["guidance"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_controller(controller, saved["controller"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first_epoch = saved["epoch"] + 1
        best, best_epoch, history = saved["best_score"], saved["best_epoch"], saved["history"]
        best_validation = saved["best_validation"]
        prior_seconds = saved["elapsed_seconds"]
        if best_epoch is not None:
            if not (output / "best.pt").is_file() or sha256(output / "best.pt") != saved["best_checkpoint_sha256"]:
                raise ValueError("Selected checkpoint is missing or changed since latest.pt")
        del saved
    save_json(output / "config.json", config)
    save_json(output / "identity.json", identity)
    save_json(output / "environment.json", {
        "torch": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "guidance_parameters": 0 if guidance is None else sum(p.numel() for p in guidance.parameters()),
        "precision": config["precision"],
    })
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_epoch = first_epoch - 1
    for epoch in range(first_epoch, config["epochs"] + 1):
        train_data.epoch = epoch
        generator = torch.Generator().manual_seed(seed * 1000003 + epoch)
        loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                            num_workers=config["num_workers"], generator=generator,
                            pin_memory=device.type == "cuda", drop_last=False)
        model.train()
        if guidance is not None:
            guidance.train()
        beta = 0.0 if controller is None else controller.beta_for_epoch(epoch)
        sums = {"loss": 0.0, "ce": 0.0, "guidance": 0.0}
        seen, skipped = 0, 0
        for batch_index, (images, targets, _) in enumerate(loader):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if not (targets != IGNORE).any():
                skipped += len(images)
                continue
            lr = lr_at((epoch - 1) * len(loader) + batch_index, len(loader), config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, config["precision"]):
                logits, student_features = model(images, return_features=True)
                logits = F.interpolate(logits.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False)
                ce = F.cross_entropy(logits, targets, ignore_index=IGNORE)
                feature_loss = ce.new_zeros(())
                if beta > 0:
                    with torch.no_grad():
                        teacher_features = teacher.features(images)[1:]
                    if method == "ibkd":
                        align, fuse = guidance(student_features, teacher_features)
                        ratio = config["ibkd_fusion_ratio"]
                        feature_loss = (1 - ratio) * align + ratio * fuse
                    else:
                        feature_loss = guidance(student_features, teacher_features)
                loss = ce + beta * feature_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            parameters = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            batch = len(images)
            seen += batch
            for name, value in (("loss", loss), ("ce", ce), ("guidance", feature_loss)):
                sums[name] += float(value.detach()) * batch
            if batch_index == 0 or (batch_index + 1) % 100 == 0:
                print(json.dumps({"event": "train", "kind": kind, "method": method, "seed": seed,
                                  "epoch": epoch, "batch": batch_index + 1, "batches": len(loader),
                                  "loss": float(loss.detach()), "lr": lr, "beta": beta}), flush=True)
        if not seen:
            raise RuntimeError("All training crops were void; no updates were made")
        losses = {key: value / seen for key, value in sums.items()}
        if controller is not None:
            controller.observe(epoch, losses["guidance"], beta_used=beta)
        validation = None
        if epoch % config["validation_every"] == 0 or epoch == config["epochs"]:
            validation = evaluate(model, validation_loader, config, device)
            if not manifest["synthetic"] and validation["evaluated_classes"] != 19:
                raise RuntimeError("Full Cityscapes validation must evaluate all 19 classes")
            if validation[selection_metric] > best:
                best, best_epoch = validation[selection_metric], epoch
                best_validation = validation
                atomic_save(output / "best.pt", {
                    "artifact": "validation_selected_segmenter", "identity": identity,
                    "config": config, "model": model.state_dict(), "epoch": epoch,
                    "validation": validation,
                })
        last_epoch = epoch
        record = {"epoch": epoch, **losses, "beta": beta, "skipped_void_samples": skipped,
                  "guidance_stop_epoch": None if controller is None else controller.stop_epoch,
                  "validation": validation, "elapsed_seconds": prior_seconds + time.perf_counter() - started}
        history.append(record)
        print(json.dumps({"event": "epoch_complete", "kind": kind, "method": method, "seed": seed,
                          **{k: v for k, v in record.items() if k != "validation"},
                          "selection_metric": selection_metric,
                          "val_pixel_accuracy": None if validation is None else validation["pixel_accuracy"],
                          "val_miou": None if validation is None else validation["miou"]}), flush=True)
        atomic_save(output / "latest.pt", {
            "identity": identity, "model": model.state_dict(),
            "guidance": None if guidance is None else guidance.state_dict(), "optimizer": optimizer.state_dict(),
            "controller": None if controller is None else controller.state_dict(),
            "epoch": epoch, "best_score": best, "best_epoch": best_epoch,
            "best_validation": best_validation, "history": history,
            "elapsed_seconds": record["elapsed_seconds"],
            "best_checkpoint_sha256": None if best_epoch is None else sha256(output / "best.pt"),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })
        save_json(output / "history.json", history)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
    summary = {
        "status": "complete" if last_epoch == config["epochs"] else "paused_at_epoch_boundary",
        "scientific_result": False, "identity": identity,
        "selection_metric": selection_metric, "selected_epoch": best_epoch,
        "selected_val_pixel_accuracy": None if best_validation is None else best_validation["pixel_accuracy"],
        "selected_val_miou": None if best_validation is None else best_validation["miou"],
        "best_checkpoint_sha256": None if best_epoch is None else sha256(output / "best.pt"),
        "best_checkpoint_bytes": None if best_epoch is None else (output / "best.pt").stat().st_size,
        "guidance_stop_epoch": None if controller is None else controller.stop_epoch,
        "elapsed_seconds": prior_seconds + time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "metric_note": f"Selected by val {selection_metric}; all metrics from that checkpoint, not independent test",
    }
    save_json(output / "summary.json", summary)
    del optimizer, model, guidance, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def evaluate_checkpoint(checkpoint_path, data_root, manifest, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("artifact") != "validation_selected_segmenter":
        raise ValueError("Evaluation expects best.pt")
    config = payload["config"]
    validate_config(config)
    validate_manifest_contract(manifest, config)
    check_device(device, config)
    if payload["identity"]["manifest_sha256"] != json_hash(manifest) or payload["identity"]["source_sha256"] != source_hash():
        raise ValueError("Evaluation data/source differs from training")
    model = Segmenter(payload["identity"]["kind"], config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    dataset = Cityscapes(data_root, manifest["splits"]["val"], config, training=False, seed=0)
    loader = DataLoader(dataset, batch_size=1, num_workers=config["num_workers"])
    return {"checkpoint_sha256": sha256(checkpoint_path), "identity": payload["identity"],
            "validation": evaluate(model, loader, config, device)}
