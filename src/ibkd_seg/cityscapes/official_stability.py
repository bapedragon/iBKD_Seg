"""Bounded stability diagnostics for the crop512 L/16 protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .official_api import bootstrap


def _ratio(value, reference):
    if reference == 0:
        return None
    return abs(value) / abs(reference)


def series_diagnostics(rows, key, *, reference_steps, tail_steps):
    values = [float(row[key]) for row in rows]
    if not values:
        return None
    reference = statistics.median(values[:reference_steps])
    tail = statistics.median(values[-tail_steps:])
    peak = max(abs(value) for value in values)
    return {
        "reference_median": reference,
        "tail_median": tail,
        "peak_absolute": peak,
        "peak_ratio": _ratio(peak, reference),
        "tail_ratio": _ratio(tail, reference),
    }


def stability_decision(rows, config, *, expected_steps, runtime_error=None):
    diagnostics = {
        key: series_diagnostics(
            rows, key,
            reference_steps=config["reference_steps"],
            tail_steps=config["tail_steps"],
        )
        for key in ("loss", "ce", "guidance", "grad_norm_unclipped")
    }
    reasons = []
    if runtime_error is not None:
        reasons.append(f"runtime_error:{runtime_error}")
    if len(rows) != expected_steps:
        reasons.append(f"incomplete_steps:{len(rows)}/{expected_steps}")
    for key in ("loss", "guidance", "grad_norm_unclipped"):
        item = diagnostics[key]
        if item is None or (key == "guidance" and item["reference_median"] == 0 and item["peak_absolute"] == 0):
            continue
        if not all(math.isfinite(float(row[key])) for row in rows):
            reasons.append(f"{key}:nonfinite")
            continue
        if item["peak_ratio"] is None or item["peak_ratio"] > config["max_peak_ratio"]:
            reasons.append(f"{key}:peak_ratio={item['peak_ratio']}")
        if item["tail_ratio"] is None or item["tail_ratio"] > config["max_tail_ratio"]:
            reasons.append(f"{key}:tail_ratio={item['tail_ratio']}")
    return {"stable": not reasons, "reasons": reasons, "diagnostics": diagnostics}


def online_divergence_reason(rows, config):
    """Return a locked hard-failure reason without waiting for all 500 steps."""
    if len(rows) < 2:
        return None
    for key in ("loss", "guidance", "grad_norm_unclipped"):
        reference = abs(float(rows[0][key]))
        current = abs(float(rows[-1][key]))
        if reference == 0:
            continue
        ratio = current / reference
        if not math.isfinite(ratio) or ratio > config["max_peak_ratio"]:
            return f"{key}:step={rows[-1]['step']}:ratio={ratio}"
    return None


def validate_config(config):
    beta_grid_2000 = "cityscapes_segmenter_l16_crop512_beta_grid2000_v6"
    locked = {
        "train_samples": 2975,
        "val_samples": 500 if config.get("protocol_id") == beta_grid_2000 else 20,
        "batch_size": 8,
        "image_size": 1024,
        "crop_size": 512,
        "window_size": 512,
        "window_stride": 512,
        "decoder_layers": 1,
        "total_steps": 80000,
        "precision": "fp32",
        "gradient_clipping": False,
        "automatic_hyperparameter_changes": False,
        "test_used": False,
    }
    mismatches = {key: (config.get(key), value) for key, value in locked.items()
                  if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Stability protocol mismatch: {mismatches}")
    profiles = {
        "cityscapes_segmenter_l16_crop512_stability_v1": {
            "methods": ["vanilla", "lg", "alg", "ibkd"],
            "stability_steps": 500,
            "guidance_beta": 2.5,
            "guidance_beta_by_method": None,
        },
        "cityscapes_segmenter_l16_crop512_beta_screen100_v2": {
            "methods": ["lg", "ibkd"],
            "stability_steps": 100,
            "guidance_beta": 0.05,
            "guidance_beta_by_method": {"lg": 0.05, "ibkd": 0.5},
        },
        "cityscapes_segmenter_l16_crop512_beta_confirm500_v3": {
            "methods": ["lg", "alg", "ibkd"],
            "stability_steps": 500,
            "guidance_beta": 0.05,
            "guidance_beta_by_method": {"lg": 0.05, "alg": 0.05, "ibkd": 0.5},
        },
        "cityscapes_segmenter_l16_crop512_repro25_v4": {
            "methods": ["lg", "alg"],
            "stability_steps": 25,
            "guidance_beta": 0.05,
            "guidance_beta_by_method": {"lg": 0.05, "alg": 0.05},
            "strict_determinism": True,
        },
        "cityscapes_segmenter_l16_crop512_beta_grid500_v5": {
            "methods": ["lg", "alg", "ibkd"],
            "stability_steps": 500,
            "guidance_beta": 0.05,
            "guidance_beta_by_method": None,
            "guidance_beta_candidates_by_method": {
                "lg": [0.02, 0.05, 0.1, 0.2],
                "alg": [0.02, 0.05, 0.1, 0.2],
                "ibkd": [0.1, 0.25, 0.5, 1.0],
            },
        },
        beta_grid_2000: {
            "methods": ["lg", "alg", "ibkd"],
            "stability_steps": 2000,
            "guidance_beta": 0.05,
            "guidance_beta_by_method": None,
            "guidance_beta_candidates_by_method": {
                "lg": [0.02, 0.05, 0.1, 0.2],
                "alg": [0.02, 0.05, 0.1, 0.2],
                "ibkd": [0.1, 0.25, 0.5, 1.0],
            },
        },
    }
    profile = profiles.get(config.get("protocol_id"))
    if profile is None:
        raise ValueError(f"Unknown stability protocol: {config.get('protocol_id')!r}")
    profile_mismatches = {key: (config.get(key), value) for key, value in profile.items()
                          if config.get(key) != value}
    if profile_mismatches:
        raise ValueError(f"Stability profile mismatch: {profile_mismatches}")
    if not 0 < config["reference_steps"] <= config["tail_steps"] <= config["stability_steps"]:
        raise ValueError("Invalid stability windows")
    if min(config["max_peak_ratio"], config["max_tail_ratio"]) <= 1:
        raise ValueError("Stability ratios must exceed one")


def guidance_beta_for(method, config, override=None):
    candidates = config.get("guidance_beta_candidates_by_method")
    if candidates is not None:
        if method not in candidates:
            raise ValueError(f"No locked guidance beta candidates for {method}")
        if override is None:
            raise ValueError(f"The beta-grid protocol requires an explicit beta for {method}")
        beta = float(override)
        if not any(math.isclose(beta, float(value), rel_tol=0.0, abs_tol=1e-12)
                   for value in candidates[method]):
            raise ValueError(f"Beta {beta} is outside the locked grid for {method}")
        return beta
    if override is not None:
        raise ValueError("A beta override is allowed only for the locked beta-grid protocol")
    mapping = config.get("guidance_beta_by_method")
    if mapping is not None:
        if method not in mapping:
            raise ValueError(f"No locked guidance beta for {method}")
        return float(mapping[method])
    return float(config["guidance_beta"])


def beta_run_id(method, beta):
    value = format(float(beta), "g").replace("-", "m").replace(".", "p")
    return f"{method}_beta_{value}"


def terminal_result(runs, config, consistency_errors):
    methods = {}
    for row in runs:
        method = row["method"]
        run_id = row.get("run_id", method)
        if "decision" not in row:
            methods[run_id] = row
            continue
        scores = row.get("diagnostic_validation")
        methods[run_id] = {
            "method": method,
            "status": row["status"],
            "completed_steps": row["completed_steps"],
            "expected_steps": row["expected_steps"],
            "first_step": row.get("first_step"),
            "final_step": row.get("final_step"),
            "stability_diagnostics": row["decision"]["diagnostics"],
            "stability_reasons": row["decision"]["reasons"],
            "diagnostic_pixel_accuracy": None if scores is None else scores["pixel_accuracy"],
            "diagnostic_miou": None if scores is None else scores["miou"],
            "diagnostic_validation": scores,
            "validation_samples": row["validation_samples"],
            "guidance_beta": row["effective_guidance_beta"],
            "guidance_active_steps": row.get("guidance_active_steps"),
            "guidance_stop_epoch": row.get("guidance_stop_epoch"),
            "decoder_layers": row["decoder_layers"],
            "schedule_total_steps": row["schedule_total_steps"],
            "train_seconds": row["train_seconds"],
            "invocation_seconds": row["invocation_seconds"],
            "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
            "teacher_frozen_verified": row["teacher_frozen_verified"],
            "parameters_finite": row["parameters_finite"],
            "optimizer_state_finite": row["optimizer_state_finite"],
            "runtime_error": row["runtime_error"],
        }
    stable_count = sum(row.get("status") == "stable" for row in runs)
    beta_candidates = config.get("guidance_beta_candidates_by_method")
    if beta_candidates is None:
        beta_by_method = {
            method: guidance_beta_for(method, config)
            for method in config["methods"] if method != "vanilla"
        }
    else:
        beta_by_method = None
    stable_candidates = None
    unstable_candidates = None
    if beta_candidates is not None:
        stable_candidates = {
            method: [
                row.get("effective_guidance_beta")
                for row in runs
                if row.get("method") == method and row.get("status") == "stable"
            ]
            for method in config["methods"]
        }
        unstable_candidates = {
            method: [
                row.get("effective_guidance_beta")
                for row in runs
                if row.get("method") == method and row.get("status") != "stable"
            ]
            for method in config["methods"]
        }
    return {
        "status": "completed",
        "protocol_id": config["protocol_id"],
        "crop_size": config["crop_size"],
        "batch_size": config["batch_size"],
        "schedule_total_steps": config["total_steps"],
        "guidance_beta_by_method": beta_by_method,
        "guidance_beta_candidates_by_method": beta_candidates,
        "guidance_beta_by_run": {
            row.get("run_id", row["method"]): row.get("effective_guidance_beta")
            for row in runs
        },
        "stable_beta_candidates_by_method": stable_candidates,
        "unstable_beta_candidates_by_method": unstable_candidates,
        "all_methods_stable": stable_count == len(runs) and not consistency_errors,
        "stable_methods": stable_count,
        "total_methods": len(runs),
        "consistency_errors": consistency_errors,
        "scientific_result": False,
        "full_training_authorized": False,
        "methods": methods,
    }


def _module_norm(module):
    import torch
    if module is None:
        return None
    total = torch.zeros((), device=next(module.parameters()).device)
    for parameter in module.parameters():
        total += parameter.detach().float().square().sum()
    return float(total.sqrt())


def run_method(args, config, output):
    import torch
    import torchvision
    import timm
    from torch.nn import functional as F
    from segm.model.utils import inference
    import segm.utils.torch as ptu
    from . import official_api as api
    from .data import json_hash, save_json, sha256
    from .evaluation import confusion_update, metrics
    from .full_data import FullDataset, batch_hash, train_loader
    from .runtime import controller_for, seed_all, source_hash, state_hash

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The stability diagnostic requires exactly one CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ptu.device = device
    torch.set_num_threads(4)

    manifest = json.loads(args.manifest.read_text())
    labels = json.loads(args.labels_report.read_text())
    datasets = {split: FullDataset(args.data_dir, manifest, config, split) for split in ("train", "val")}
    strict_determinism = bool(config.get("strict_determinism", False))
    if strict_determinism and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8", ":16:8"
    }:
        raise RuntimeError(
            "Strict determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    seed_all(config["seed"], strict_determinism=strict_determinism)
    model = api.student(args.cache_root, image_size=config["crop_size"],
                        decoder_layers=config["decoder_layers"]).to(device).train()
    initial_student = state_hash(model)
    seed_all(config["seed"] + 1000, strict_determinism=strict_determinism)
    effective_config = dict(config)
    effective_config["guidance_beta"] = guidance_beta_for(args.method, config, args.beta)
    guide = api.guidance(args.method, effective_config)
    teacher = None
    teacher_hash = None
    if guide is not None:
        guide = guide.to(device).train()
        teacher = api.teacher(args.cache_root).to(device)
        teacher_hash = state_hash(teacher)
    capture = api.FeatureCapture(model)
    optimizer, scheduler = api.optimizer_scheduler(
        model, guide, args.cache_root, total_steps=config["total_steps"])
    if scheduler.iter_max != config["total_steps"] or not optimizer.defaults["nesterov"]:
        raise RuntimeError("Unexpected optimizer or scheduler")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    controller = controller_for(args.method, effective_config)
    seed_all(config["seed"] + 2000, strict_determinism=strict_determinism)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    input_checks = {}
    stream = hashlib.sha256()
    steps_per_epoch = math.ceil(len(datasets["train"]) / config["batch_size"])
    milestones = {1, 2, 3, 4, 5, 10, 25, 50, 100, 200, 300, 400, config["stability_steps"]}
    for boundary in range(steps_per_epoch, config["stability_steps"] + 1, steps_per_epoch):
        milestones.update((boundary, boundary + 1))
    runtime_error = None
    began_run = time.perf_counter()
    log_path = output / "steps.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("w") as log:
            epoch = 1
            while len(rows) < config["stability_steps"]:
                beta = controller.beta_for_epoch(epoch) if controller else 0.0
                epoch_guidance_sum = 0.0
                epoch_samples = 0
                loader = train_loader(datasets["train"], epoch, 0, device)
                for image, target, ids in loader:
                    step_number = len(rows) + 1
                    digest = batch_hash(image, target, ids)
                    stream.update(f"{step_number}:{digest}".encode())
                    if step_number in milestones:
                        input_checks[str(step_number)] = {"ids": list(ids), "sha256": digest}
                    image = image.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    started = time.perf_counter()
                    if guide is None:
                        logits = model(image)
                        features = None
                    else:
                        logits, features = capture.forward(model, image)
                    ce = F.cross_entropy(logits, target, ignore_index=255)
                    guided = ce.new_zeros(())
                    if guide is not None:
                        with torch.no_grad():
                            teacher_features = teacher.extract_feat(api.teacher_input(image))[1:]
                        if args.method == "ibkd":
                            alignment, fusion = guide(features, teacher_features)
                            ratio = config["ibkd_fusion_ratio"]
                            guided = (1 - ratio) * alignment + ratio * fusion
                        else:
                            guided = guide(features, teacher_features)
                    loss = ce + beta * guided
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"nonfinite_loss step={step_number} ce={float(ce.detach())} guidance={float(guided.detach())}")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
                    if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                        raise RuntimeError("Frozen teacher received gradients")
                    lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step_update(scheduler.last_epoch + 1)
                    torch.cuda.synchronize()
                    row = {
                        "step": step_number,
                        "epoch": epoch,
                        "loss": float(loss.detach()),
                        "ce": float(ce.detach()),
                        "guidance": float(guided.detach()),
                        "beta": beta,
                        "weighted_guidance_to_ce_ratio": (
                            float((beta * guided / ce).detach()) if float(ce.detach()) != 0 else None
                        ),
                        "lr": lr,
                        "grad_norm_unclipped": float(norm),
                        "seconds": time.perf_counter() - started,
                    }
                    if step_number in milestones:
                        row["guidance_parameter_norm"] = _module_norm(guide)
                    rows.append(row)
                    epoch_guidance_sum += row["guidance"] * len(ids)
                    epoch_samples += len(ids)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    if step_number <= 5 or step_number % 25 == 0:
                        log.flush()
                        print(
                            f"[L16_STABILITY_STEP] method={args.method} step={step_number}/{config['stability_steps']} "
                            f"loss={row['loss']:.6g} ce={row['ce']:.6g} guidance={row['guidance']:.6g} "
                            f"weighted_guidance_to_ce={row['weighted_guidance_to_ce_ratio']:.6g} "
                            f"grad_norm={row['grad_norm_unclipped']:.6g} seconds={row['seconds']:.2f}",
                            flush=True,
                        )
                    divergence = online_divergence_reason(rows, config)
                    if divergence is not None:
                        raise FloatingPointError(f"online_divergence:{divergence}")
                    if len(rows) == config["stability_steps"]:
                        break
                if len(rows) < config["stability_steps"]:
                    if controller:
                        controller.observe(epoch, epoch_guidance_sum / epoch_samples, beta_used=beta)
                    epoch += 1
    except BaseException as error:
        runtime_error = repr(error)

    decision = stability_decision(
        rows, config, expected_steps=config["stability_steps"], runtime_error=runtime_error)
    parameters_finite = all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
    optimizer_state_finite = all(
        bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    if not parameters_finite or not optimizer_state_finite:
        decision["stable"] = False
        decision["reasons"].append("nonfinite_parameter_or_optimizer_state")
    scores = None
    if len(rows) == config["stability_steps"] and all(math.isfinite(row["loss"]) for row in rows):
        model.eval()
        confusion = torch.zeros(19, 19, dtype=torch.int64)
        expected_pixels = 0
        with torch.no_grad():
            for index in range(len(datasets["val"])):
                ims, metas, target, _ = datasets["val"][index]
                prediction = inference(model, ims, metas, tuple(target.shape), config["window_size"],
                                       config["window_stride"], batch_size=1).argmax(0).cpu()
                confusion_update(confusion, prediction, target)
                expected_pixels += int((target != 255).sum())
        scores = metrics(confusion)
        if scores["valid_pixels"] != expected_pixels:
            raise RuntimeError("Validation pixel accounting mismatch")

    teacher_frozen = teacher is None or state_hash(teacher) == teacher_hash
    if not teacher_frozen:
        decision["stable"] = False
        decision["reasons"].append("teacher_state_changed")
    positive_beta_seen = False
    guidance_stop_epoch = None
    for row in rows:
        if row["beta"] > 0:
            positive_beta_seen = True
        elif positive_beta_seen:
            guidance_stop_epoch = row["epoch"]
            break
    result = {
        "status": "stable" if decision["stable"] else "unstable",
        "method": args.method,
        "run_id": args.run_id or args.method,
        "scientific_result": False,
        "full_training_authorized": False,
        "completed_steps": len(rows),
        "expected_steps": config["stability_steps"],
        "effective_guidance_beta": 0.0 if guide is None else effective_config["guidance_beta"],
        "guidance_active_steps": sum(row["beta"] > 0 for row in rows),
        "guidance_stop_epoch": guidance_stop_epoch,
        "first_step": rows[0] if rows else None,
        "final_step": rows[-1] if rows else None,
        "decision": decision,
        "diagnostic_validation": scores,
        "validation_samples": config["val_samples"] if scores is not None else 0,
        "input_checks": input_checks,
        "input_stream_sha256": stream.hexdigest(),
        "student_initial_state_sha256": initial_student,
        "student_final_state_sha256": state_hash(model),
        "guidance_final_state_sha256": None if guide is None else state_hash(guide),
        "teacher_state_sha256": teacher_hash,
        "teacher_frozen_verified": teacher_frozen,
        "parameters_finite": parameters_finite,
        "optimizer_state_finite": optimizer_state_finite,
        "decoder_layers": len(model.decoder.blocks),
        "schedule_total_steps": scheduler.iter_max,
        "train_seconds": sum(row["seconds"] for row in rows),
        "invocation_seconds": time.perf_counter() - began_run,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "runtime_error": runtime_error,
        "config": config,
        "identity": {
            "manifest_sha256": sha256(args.manifest),
            "labels_report_sha256": sha256(args.labels_report),
            "labels_identity_sha256": json_hash(labels),
            "source_sha256": source_hash(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "strict_determinism_requested": strict_determinism,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }
    save_json(output / "summary.json", result)
    print(
        f"[L16_STABILITY_METHOD_DONE] run={result['run_id']} method={args.method} "
        f"beta={result['effective_guidance_beta']} status={result['status']} "
        f"steps={len(rows)}/{config['stability_steps']} reasons={json.dumps(decision['reasons'])}",
        flush=True,
    )
    return 0


def run_suite(args, config, output, config_path):
    from .data import json_hash, save_json, verify_manifest
    from .full_data import prepare_labels
    from .official_assets import verify

    output.mkdir(parents=True, exist_ok=True)
    provenance = verify(args.cache_root)
    manifest_data = json.loads(args.manifest.read_text())
    verify_manifest(args.data_dir, manifest_data)
    prepare_labels(
        args.data_dir,
        args.manifest,
        {"train": config["train_samples"], "val": config["val_samples"]},
        output,
    )
    save_json(output / "config.json", config)
    save_json(output / "provenance.json", provenance)
    candidates = config.get("guidance_beta_candidates_by_method")
    if candidates is None:
        plans = [(method, None, method) for method in config["methods"]]
    else:
        plans = [
            (method, float(beta), beta_run_id(method, beta))
            for method in config["methods"]
            for beta in candidates[method]
        ]
    runs = []
    for method, beta, run_id in plans:
        beta_text = "locked" if beta is None else format(beta, "g")
        print(f"[L16_STABILITY_START] run={run_id} method={method} beta={beta_text}", flush=True)
        command = [
            sys.executable, "-u", "-m", "ibkd_seg.cityscapes.official_stability",
            "--cache-root", str(args.cache_root), "--data-dir", str(args.data_dir),
            "--manifest", str(args.manifest), "--labels-report", str(output / "labels.json"),
            "--output-dir", str(output / run_id), "--config", str(config_path),
            "--device", args.device, "--method", method, "--run-id", run_id,
        ]
        if beta is not None:
            command.extend(("--beta", str(beta)))
        completed = subprocess.run(command, check=False)
        summary_path = output / run_id / "summary.json"
        if summary_path.exists():
            row = json.loads(summary_path.read_text())
        else:
            row = {"status": "infrastructure_error", "method": method,
                   "run_id": run_id, "effective_guidance_beta": beta,
                   "returncode": completed.returncode}
        runs.append(row)
        save_json(output / "stability_summary.json", {"status": "running", "runs": runs})

    comparable = [row for row in runs if "student_initial_state_sha256" in row]
    consistency_errors = []
    if len({row["student_initial_state_sha256"] for row in comparable}) > 1:
        consistency_errors.append("student_initial_state_mismatch")
    step_one = [row.get("input_checks", {}).get("1") for row in comparable]
    if step_one and (any(value is None for value in step_one)
                     or len({json_hash(value) for value in step_one}) > 1):
        consistency_errors.append("first_batch_mismatch")
    terminal = terminal_result(runs, config, consistency_errors)
    final = {
        "status": "completed",
        "all_methods_stable": terminal["all_methods_stable"],
        "stable_methods": terminal["stable_methods"],
        "total_methods": len(plans),
        "consistency_errors": consistency_errors,
        "scientific_result": False,
        "full_training_authorized": False,
        "config_sha256": json_hash(config),
        "terminal_result": terminal,
        "runs": runs,
    }
    save_json(output / "stability_summary.json", final)
    if candidates is not None:
        save_json(output / "beta_grid_summary.json", final)
    marker = ("[CITYSCAPES_L16_BETA_GRID_DONE] " if candidates is not None
              else "[CITYSCAPES_L16_STABILITY_DONE] ")
    print(marker
          + json.dumps(terminal, sort_keys=True, allow_nan=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache-root", "data-dir", "manifest", "output-dir", "config"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--labels-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("vanilla", "lg", "alg", "ibkd"), help=argparse.SUPPRESS)
    parser.add_argument("--beta", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    for key in ("cache_root", "data_dir", "manifest", "output_dir", "config", "labels_report"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap(args.cache_root)
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.method:
        if args.labels_report is None:
            parser.error("--method requires --labels-report")
        return run_method(args, config, args.output_dir)
    return run_suite(args, config, args.output_dir, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
