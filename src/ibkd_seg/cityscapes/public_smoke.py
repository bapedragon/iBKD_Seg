"""Synthetic H200 smoke entrypoint and shared DeepLabV3 -> Segmenter checks.

Not a Cityscapes training entrypoint. No data or model downloads, no full-run
option, no ranking or accuracy claims. Every method runs in its own process.
The separate real_smoke entrypoint supplies real batches to measure().
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import json_hash, save_json, sha256
from .evaluation import autocast, confusion_update, metrics, sliding_logits
from .public_models import DeepLabV3Teacher, SegmenterStudent, public_guidance
from .runtime import (REPO, check_device, controller_for, restore_controller,
                      seed_all, source_hash, state_hash)

CONFIG = REPO / "phase4/phase4_cityscapes/configs/deeplabv3_segmenter_smoke_v1.json"


def synthetic_batch(config, device):
    generator = torch.Generator().manual_seed(config["seed"] + 100)
    height, width = config["crop_size"]
    rgb = torch.rand(config["batch_size"], 3, height, width, generator=generator)
    target = torch.arange(width).remainder(19)[None, None].expand(config["batch_size"], height, width).clone()
    target[:, :2] = 255
    return rgb.to(device), target.to(device)


def teacher_input(rgb):
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (rgb - mean) / std


def verify_controller_paths(config):
    result = {}
    for method in ("lg", "alg", "ibkd"):
        controller = controller_for(method, config)
        for epoch in range(1, 56):
            beta = controller.beta_for_epoch(epoch)
            controller.observe(epoch, 1.0, beta_used=beta)
        expected = {"lg": None, "alg": 2, "ibkd": 20}[method]
        if controller.stop_epoch != expected:
            raise RuntimeError(f"Controller diagnostic failed: {method}")
        restored = controller_for(method, config)
        restore_controller(restored, controller.state_dict())
        if restored.state_dict() != controller.state_dict():
            raise RuntimeError("Controller restore changed state")
        result[method] = {"synthetic_constant_loss_stop_epoch": controller.stop_epoch,
                          "state_restore": "passed"}
    return result


def measure(method, config, device, output, *, train_batches=None, eval_batches=None, data_identity=None):
    import timm
    import torchvision

    synthetic = data_identity is None
    if synthetic:
        if train_batches is not None or eval_batches is not None:
            raise ValueError("Real batches require data provenance")
    elif train_batches is None or len(train_batches) != config["steps"] or not eval_batches:
        raise ValueError("Real smoke requires one batch per step and validation samples")
    marker = "CITYSCAPES_ARCH_SMOKE" if synthetic else "CITYSCAPES_REAL_SMOKE"
    check_device(device, config)
    torch.set_num_threads(4 if device.type == "cuda" else 2)
    seed_all(config["seed"])
    model = SegmenterStudent(config).to(device).train()
    initial = state_hash(model)
    guidance = public_guidance(method, config)
    teacher = None
    teacher_hash = None
    if guidance is not None:
        guidance = guidance.to(device).train()
        # All guided methods see exactly the same frozen random teacher.
        seed_all(config["seed"] + 1000)
        teacher = DeepLabV3Teacher().to(device).eval().requires_grad_(False)
        teacher_hash = state_hash(teacher)
    modules = {"encoder": model.encoder, "decoder": model.decoder}
    if guidance is not None:
        modules["guidance"] = guidance
    params = [p for module in modules.values() for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    controller = controller_for(method, config)
    beta = 0.0 if controller is None else controller.beta_for_epoch(1)
    rgb, target = (synthetic_batch(config, device) if synthetic else
                   tuple(value.to(device) for value in train_batches[0]))
    if teacher is not None:
        with torch.no_grad(), autocast(device, config["precision"]):
            teacher_logits, teacher_features = teacher(teacher_input(rgb), return_features=True)
        if teacher_logits.shape[:2] != (config["batch_size"], 19) or not torch.isfinite(teacher_logits).all():
            raise RuntimeError("DeepLabV3 19-class ASPP forward failed")
        teacher_shapes = [list(value.shape) for value in teacher_features[1:]]
        del teacher_logits, teacher_features
    else:
        teacher_shapes = None
    seed_all(config["seed"] + 2000)  # Matched dropout RNG as well as initialization/input.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, config["precision"]):
            logits, features = model((rgb - 0.5) / 0.5, return_features=True)
            if logits.shape != (config["batch_size"], 19, *config["crop_size"]):
                raise RuntimeError("Segmenter output geometry is wrong")
            ce = F.cross_entropy(logits.float(), target, ignore_index=255)
            guided = ce.new_zeros(())
            if guidance is not None:
                with torch.no_grad():
                    teacher_features = teacher.features(teacher_input(rgb))[1:]
                if method == "ibkd":
                    align, fuse = guidance(features, teacher_features)
                    ratio = config["ibkd_fusion_ratio"]
                    guided = (1 - ratio) * align + ratio * fuse
                else:
                    guided = guidance(features, teacher_features)
            loss = ce + beta * guided
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite segmentation/guidance loss")
        loss.backward()
        for name, module in modules.items():
            if not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in module.parameters()):
                raise RuntimeError(f"Missing/nonzero gradient path: {name}")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, config["gradient_clip_norm"], error_if_nonfinite=True)
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Frozen teacher received gradients")
        optimizer.step()
        return {"total": float(loss.detach()), "ce": float(ce.detach()),
                "guidance": float(guided.detach()), "beta": beta,
                "grad_norm": float(grad_norm),
                "student_shapes": [list(value.shape) for value in features]}

    elapsed, losses = [], []
    checkpoint_path = output / "resume_checkpoint.pt"
    for index in range(config["steps"]):
        if not synthetic:
            rgb, target = (value.to(device) for value in train_batches[index])
        if index == config["steps"] - 1:
            torch.save({"model": model.state_dict(),
                        "guidance": None if guidance is None else guidance.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "controller": None if controller is None else controller.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "completed_steps": index, "config": config,
                        "config_sha256": json_hash(config), "source_sha256": source_hash(),
                        "method": method, "teacher_state_sha256": teacher_hash,
                        "data_identity": data_identity,
                        "synthetic": synthetic, "scientific_result": False}, checkpoint_path)
        start = time.perf_counter()
        losses.append(step())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - start)
        print(f"[{marker}_STEP] method={method} step={index+1} loss={losses[-1]['total']:.6g} seconds={elapsed[-1]:.3f}", flush=True)
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    if state_hash(model) == initial:
        raise RuntimeError("Optimizer did not update the student")
    if controller is not None:
        controller.observe(1, sum(row["guidance"] for row in losses) / len(losses), beta_used=beta)
    expected_controller = None if controller is None else copy.deepcopy(controller.state_dict())
    expected_states = {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                       for name, module in modules.items()}
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["config_sha256"] != json_hash(config) or saved["source_sha256"] != source_hash():
        raise RuntimeError("Resume provenance mismatch")
    if saved["data_identity"] != data_identity:
        raise RuntimeError("Resume dataset/input provenance mismatch")
    model.load_state_dict(saved["model"], strict=True)
    if guidance is not None:
        guidance.load_state_dict(saved["guidance"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    restore_controller(controller, saved["controller"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    del saved
    resumed_loss = step()
    if controller is not None:
        controller.observe(1, (sum(row["guidance"] for row in losses[:-1]) + resumed_loss["guidance"]) / len(losses), beta_used=beta)
    tolerance = dict(rtol=2e-5, atol=2e-6) if device.type == "cuda" else dict(rtol=0, atol=0)
    for name, module in modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), expected_states[name][key], **tolerance,
                                       msg=f"Resume changed {name}.{key}")
    if controller is not None:
        # CUDA reductions can differ slightly; ensure the same discrete decision.
        if (controller.active, controller.stop_epoch) != (expected_controller["active"], expected_controller["stop_epoch"]):
            raise RuntimeError("Resume changed controller decision")
    del expected_states
    if teacher is not None and state_hash(teacher) != teacher_hash:
        raise RuntimeError("Frozen teacher parameters or BatchNorm buffers changed")
    model.eval()
    if synthetic:
        eval_h, eval_w = config["native_eval_size"]
        eval_rgb = torch.rand(1, 3, eval_h, eval_w, generator=torch.Generator().manual_seed(777))
        eval_target = torch.arange(eval_w).remainder(19)[None, None].expand(1, eval_h, eval_w).clone()
        eval_target[:, :2] = 255
        eval_batches = [(eval_rgb, eval_target)]
    start = time.perf_counter()
    matrix = torch.zeros(19, 19, dtype=torch.int64)
    expected_pixels = 0
    for eval_rgb, eval_target in eval_batches:
        prediction = sliding_logits(model, (eval_rgb - 0.5) / 0.5, config, device).argmax(1).cpu()
        confusion_update(matrix, prediction, eval_target)
        expected_pixels += int((eval_target != 255).sum())
    result_metrics = metrics(matrix)
    if result_metrics["valid_pixels"] != expected_pixels:
        raise RuntimeError("Void exclusion or native evaluation geometry failed")
    eval_seconds = time.perf_counter() - start
    result = {
        "status": "passed", "synthetic": synthetic, "scientific_result": False,
        "method": method, "config_sha256": json_hash(config), "source_sha256": source_hash(),
        "student_initial_state_sha256": initial, "teacher_state_sha256": teacher_hash,
        "teacher_frozen_verified": teacher is not None,
        "crop_size": config["crop_size"], "batch_size": config["batch_size"],
        "precision": config["precision"], "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
        "python": platform.python_version(), "cuda": torch.version.cuda,
        "student_parameters": sum(p.numel() for p in model.parameters()),
        "guidance_parameters": 0 if guidance is None else sum(p.numel() for p in guidance.parameters()),
        "teacher_feature_shapes": teacher_shapes, "losses": losses,
        "step_seconds": elapsed, "seconds_per_step_after_first": sum(elapsed[1:]) / (len(elapsed)-1),
        "train_peak_allocated_bytes": peak_allocated, "train_peak_reserved_bytes": peak_reserved,
        "native_synthetic_eval_size": config["native_eval_size"],
        "native_synthetic_eval_seconds": eval_seconds,
        "synthetic_pixel_accuracy": result_metrics["pixel_accuracy"], "synthetic_miou": result_metrics["miou"],
        "synthetic_metrics": result_metrics,
        "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size,
                       "sha256": sha256(checkpoint_path)},
        "strict_reload": "passed", "resumed_update_matches": "passed",
        "resume_tolerance": tolerance,
        "controller_diagnostics": verify_controller_paths(config),
    }
    if not synthetic:
        for key in ("native_synthetic_eval_size", "native_synthetic_eval_seconds",
                    "synthetic_pixel_accuracy", "synthetic_miou", "synthetic_metrics"):
            result.pop(key)
        result.update(data_identity=data_identity, validation_samples=len(eval_batches),
                      eval_sizes_hw=[list(rgb.shape[-2:]) for rgb, _ in eval_batches],
                      diagnostic_eval_seconds=eval_seconds,
                      diagnostic_pixel_accuracy=result_metrics["pixel_accuracy"],
                      diagnostic_miou=result_metrics["miou"], diagnostic_metrics=result_metrics,
                      pretrained_weights_used=False, full_validation=False,
                      score_use="smoke diagnostic only; no checkpoint selection or method ranking")
    save_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-small", action="store_true", help="Explicit reduced-resolution CPU verification")
    parser.add_argument("--method", choices=("vanilla", "lg", "alg", "ibkd"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.cpu_small:
        if args.device != "cpu":
            parser.error("--cpu-small requires --device cpu")
        config.update(protocol_id=config["protocol_id"] + "_cpu_small",
                      crop_size=[32, 64], eval_stride=[32, 48], native_eval_size=[48, 96],
                      attention_query_chunk=7, precision="fp32")
    elif args.device != "cuda":
        parser.error("CPU runs require --cpu-small; full-shape BF16 is a CUDA check")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    if args.method:
        try:
            measure(args.method, config, torch.device(args.device), output)
        except Exception as error:
            save_json(output / "failure.json", {"status": "failed", "method": args.method,
                                                "error": repr(error), "scientific_result": False})
            raise
        return
    report = {"status": "running", "synthetic": True, "scientific_result": False,
              "config": config, "runs": [], "limitations": [
                  "Cityscapes data and pretrained weights are not used",
                  "No real validation score or full-training time estimate",
                  "Student is Seg-S/16, not the author's Cityscapes Seg-L/16 result",
                  "Torchvision DeepLabV3 is not an MMSegmentation checkpoint-compatible implementation",
                  "SGD steps/AMP/clipping/batch are smoke settings, not a locked full-training recipe"]}
    save_json(output / "smoke_summary.json", report)
    for method in config["methods"]:
        command = [sys.executable, "-m", "ibkd_seg.cityscapes.public_smoke", "--device", args.device,
                   "--output-dir", str(output / method), "--method", method]
        if args.cpu_small:
            command.append("--cpu-small")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            report.update(status="failed", failed_method=method, returncode=error.returncode)
            save_json(output / "smoke_summary.json", report)
            raise
        report["runs"].append(json.loads((output / method / "summary.json").read_text()))
        save_json(output / "smoke_summary.json", report)
    if len({row["student_initial_state_sha256"] for row in report["runs"]}) != 1:
        raise RuntimeError("Student/decoder initialization differs across methods")
    if len({row["teacher_state_sha256"] for row in report["runs"] if row["method"] != "vanilla"}) != 1:
        raise RuntimeError("Guided methods used different teachers")
    report["status"] = "passed"
    save_json(output / "smoke_summary.json", report)
    print(f"[CITYSCAPES_ARCH_SMOKE_DONE] status=passed methods=4/4 scientific_result=false summary={output / 'smoke_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
