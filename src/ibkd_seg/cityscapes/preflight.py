"""Measure configured full-shape train steps on synthetic tensors, without checkpoints."""

from __future__ import annotations

import gc
import time

import torch
from torch.nn import functional as F

from .data import json_hash, save_json
from .evaluation import autocast
from .models import Segmenter, build_guidance
from .runtime import METHODS, check_device, optimizer_for, seed_all, source_hash, validate_config


def measure(kind, method, config, device, steps):
    seed_all(1)
    model = Segmenter(kind, config).to(device).train()
    guidance = build_guidance(method, config) if kind == "student" and method != "vanilla" else None
    teacher = None
    if guidance is not None:
        guidance = guidance.to(device).train()
        teacher = Segmenter("teacher", config).to(device).eval().requires_grad_(False)
    optimizer = optimizer_for(model, guidance, config)
    height, width = config["crop_size"]
    images = torch.randn(config["batch_size"], 3, height, width, device=device)
    target = torch.randint(0, 19, (config["batch_size"], height, width), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    elapsed, last_loss = [], None
    for _ in range(steps):
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, config["precision"]):
            logits, features = model(images, return_features=True)
            logits = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, target)
            if guidance is not None:
                with torch.no_grad():
                    teacher_features = teacher.features(images)[1:]
                if method == "ibkd":
                    align, fuse = guidance(features, teacher_features)
                    ratio = config["ibkd_fusion_ratio"]
                    guided = (1 - ratio) * align + ratio * fuse
                else:
                    guided = guidance(features, teacher_features)
                loss = loss + config["guidance_beta"] * guided
        if not torch.isfinite(loss):
            raise RuntimeError("Preflight loss is not finite")
        loss.backward()
        params = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, config["gradient_clip_norm"], error_if_nonfinite=True)
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Teacher received gradients")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - begin)
        last_loss = float(loss.detach())
    return {
        "kind": kind, "method": method, "steps": steps, "last_loss": last_loss,
        "seconds_per_step_after_first": sum(elapsed[1:]) / (steps - 1),
        "step_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "guidance_parameters": 0 if guidance is None else sum(p.numel() for p in guidance.parameters()),
    }


def run_preflight(config, device, output, steps=3):
    validate_config(config)
    check_device(device, config)
    if steps < 2:
        raise ValueError("Use at least two steps to include initialized AdamW state")
    if output.exists():
        raise FileExistsError(f"Do not overwrite an existing preflight report: {output}")
    report = {"status": "running", "synthetic": True, "scientific_result": False,
              "config_sha256": json_hash(config), "source_sha256": source_hash(),
              "crop_size": config["crop_size"], "batch_size": config["batch_size"],
              "precision": config["precision"], "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "note": "Random teacher/inputs; model steps only, no data loading or full validation timing", "runs": []}
    save_json(output, report)
    for kind, method in [("teacher", None)] + [("student", method) for method in METHODS]:
        try:
            result = measure(kind, method, config, device, steps)
        except Exception as error:
            report.update(status="failed", failed_kind=kind, failed_method=method, error=str(error))
            save_json(output, report)
            raise
        report["runs"].append(result)
        save_json(output, report)
        print(f"preflight {kind}/{method}: {result}", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "passed"
    save_json(output, report)
    return report
