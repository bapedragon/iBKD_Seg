"""Three updates of the original pretrained L/16; never a full-training runner."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from .official_api import bootstrap


def measure(args, config, output):
    import torch
    import torchvision
    import timm
    from torch.nn import functional as F
    from segm.model.utils import inference
    import segm.utils.torch as ptu
    from . import official_api as api
    from .official_data import batches
    from .data import json_hash, save_json, sha256
    from .evaluation import confusion_update, metrics
    from .runtime import seed_all, state_hash, source_hash, controller_for, restore_controller
    from .public_smoke import verify_controller_paths

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one CUDA GPU")
        # FP32 recipe; do not silently enable TF32/AMP or reduce shapes.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    ptu.device = device
    torch.set_num_threads(4)
    data, identity = batches(args.data_dir, args.manifest, config, cpu_small=args.cpu_small)
    save_json(output / "data_identity.json", identity)
    seed_all(config["seed"])
    model = api.student(args.cache_root).to(device).train()
    initial = state_hash(model)
    seed_all(config["seed"] + 1000)
    guide = api.guidance(args.method, config)
    teacher = None
    teacher_hash = None
    if guide is not None:
        guide = guide.to(device).train()
        teacher = api.teacher(args.cache_root).to(device)
        teacher_hash = state_hash(teacher)
    capture = api.FeatureCapture(model)
    optimizer, scheduler = api.optimizer_scheduler(model, guide, args.cache_root)
    modules = {"encoder": model.encoder, "decoder": model.decoder}
    if guide is not None:
        modules["guidance"] = guide
    params = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
    controller = controller_for(args.method, config)
    beta = controller.beta_for_epoch(1) if controller else 0.0
    teacher_diagnostic = None
    if teacher is not None:
        image, target = (x.to(device) for x in data["train"][0])
        with torch.no_grad():
            features = teacher.extract_feat(api.teacher_input(image))
            logits = teacher.decode_head(features)
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, target, ignore_index=255)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite pretrained teacher output")
            teacher_diagnostic = {"ce": float(loss), "feature_shapes": [list(x.shape) for x in features[1:]]}
        del image, target, features, logits, loss
    seed_all(config["seed"] + 2000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step(batch):
        image, target = (x.to(device) for x in batch)
        optimizer.zero_grad(set_to_none=True)
        logits, features = capture.forward(model, image)
        ce = F.cross_entropy(logits, target, ignore_index=255)
        guided = ce.new_zeros(())
        if guide is not None:
            with torch.no_grad():
                t_features = teacher.extract_feat(api.teacher_input(image))[1:]
            if args.method == "ibkd":
                alignment, fusion = guide(features, t_features)
                ratio = config["ibkd_fusion_ratio"]
                guided = (1 - ratio) * alignment + ratio * fusion
            else:
                guided = guide(features, t_features)
        loss = ce + beta * guided
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        for name, module in modules.items():
            if not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in module.parameters()):
                raise RuntimeError(f"No nonzero gradients in {name}")
        # Infinity threshold measures/checks finite gradients without clipping.
        norm = torch.nn.utils.clip_grad_norm_(params, float("inf"), error_if_nonfinite=True)
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Frozen teacher received gradients")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step_update(scheduler.last_epoch + 1)
        return {"loss": float(loss.detach()), "ce": float(ce.detach()), "guidance": float(guided.detach()),
                "beta": beta, "lr": lr, "grad_norm_unclipped": float(norm),
                "feature_shapes": [list(x.shape) for x in features]}

    rows = []
    checkpoint_path = output / "resume_checkpoint.pt"
    source = source_hash()
    for index, batch in enumerate(data["train"]):
        if index == config["steps"] - 1:
            torch.save({"model": model.state_dict(), "guide": None if guide is None else guide.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "controller": None if controller is None else controller.state_dict(),
                        "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "config_sha256": json_hash(config), "source_sha256": source,
                        "input_sha256": identity["input_sha256"], "method": args.method}, checkpoint_path)
        start = time.perf_counter()
        row = step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        print(f"[OFFICIAL_L16_STEP] method={args.method} step={index+1} loss={row['loss']:.6g} "
              f"ce={row['ce']:.6g} guidance={row['guidance']:.6g} seconds={row['seconds']:.2f}", flush=True)
    if state_hash(model) == initial:
        raise RuntimeError("Student did not update")
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    if controller:
        controller.observe(1, sum(r["guidance"] for r in rows) / len(rows), beta_used=beta)
    expected_controller = copy.deepcopy(controller.state_dict()) if controller else None
    expected = {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                for name, module in modules.items()}
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (saved["config_sha256"] != json_hash(config) or saved["source_sha256"] != source_hash()
            or saved["input_sha256"] != identity["input_sha256"] or saved["method"] != args.method):
        raise RuntimeError("Resume provenance mismatch")
    model.load_state_dict(saved["model"], strict=True)
    if guide is not None:
        guide.load_state_dict(saved["guide"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_controller(controller, saved["controller"])
    torch.set_rng_state(saved["rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    del saved
    replay = step(data["train"][-1])
    tolerance = dict(rtol=2e-5, atol=2e-6) if device.type == "cuda" else dict(rtol=0, atol=0)
    for name, module in modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value.detach().cpu(), expected[name][key], **tolerance,
                                       msg=f"Resume changed {name}.{key}")
    if scheduler.state_dict() != expected_scheduler:
        raise RuntimeError("Resume changed LR schedule")
    if controller:
        controller.observe(1, (sum(r["guidance"] for r in rows[:-1]) + replay["guidance"]) / len(rows), beta_used=beta)
        if (controller.active, controller.stop_epoch) != (expected_controller["active"], expected_controller["stop_epoch"]):
            raise RuntimeError("Resume changed guidance decision")
    del expected
    if teacher is not None and state_hash(teacher) != teacher_hash:
        raise RuntimeError("Teacher parameters/BN buffers changed")
    model.eval()
    confusion = torch.zeros(19, 19, dtype=torch.int64)
    expected_pixels = 0
    with torch.no_grad():
        for ims, metas, target in data["val"]:
            prediction = inference(model, ims, metas, tuple(target.shape), config["window_size"],
                                   config["window_stride"], batch_size=1).argmax(0).cpu()
            confusion_update(confusion, prediction, target)
            expected_pixels += int((target != 255).sum())
    scores = metrics(confusion)
    if scores["valid_pixels"] != expected_pixels:
        raise RuntimeError("Evaluation did not exclude exactly the void pixels")
    checkpoint_record = {"bytes": checkpoint_path.stat().st_size, "sha256": sha256(checkpoint_path),
                         "retained": False, "strict_reload": "passed", "replayed_update": "passed"}
    result = dict(status="passed", method=args.method, scientific_result=False,
                  pretrained_weights_used=True, model_zoo_result_reproduced=False,
                  full_validation=False, validation_samples=2, config=config, data_identity=identity,
                  source_sha256=source, student_initial_state_sha256=initial,
                  teacher_state_sha256=teacher_hash, teacher_frozen_verified=teacher is not None,
                  teacher_diagnostic=teacher_diagnostic, student_parameters=sum(p.numel() for p in model.parameters()),
                  losses=rows, checkpoint=checkpoint_record, resume_tolerance=tolerance,
                  controller_diagnostics=verify_controller_paths(config),
                  diagnostic_pixel_accuracy=scores["pixel_accuracy"], diagnostic_miou=scores["miou"],
                  diagnostic_metrics=scores, train_peak_allocated_bytes=peak,
                  environment={"torch": str(torch.__version__), "torchvision": torchvision.__version__,
                               "timm": timm.__version__, "python": platform.python_version(),
                               "cuda": torch.version.cuda, "device": str(device),
                               "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None})
    # Only this smoke's newly created temporary checkpoint is removed after verification.
    checkpoint_path.unlink()
    save_json(output / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache-root", "data-dir", "manifest", "output-dir"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-small", action="store_true")
    parser.add_argument("--method", choices=("vanilla", "lg", "alg", "ibkd"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.device == "cpu") != args.cpu_small:
        parser.error("CPU requires --cpu-small; CUDA runs may not shrink the recipe")
    for key in ("cache_root", "data_dir", "manifest", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    bootstrap(args.cache_root)
    from .runtime import REPO
    from .data import save_json, verify_manifest, json_hash
    from .official_assets import verify
    from .official_api import recipe
    config = json.loads((REPO / "phase4/phase4_cityscapes/configs/official_l16_smoke_v2.json").read_text())
    if args.cpu_small:
        config.update(protocol_id=config["protocol_id"] + "_cpu_small", batch_size=2,
                      crop_size=32, window_size=32, window_stride=24, attention_query_chunk=7)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", scientific_result=False, config=config, runs=[])
    try:
        # Includes every source and full public weight checksum; no random fallback.
        provenance = verify(args.cache_root)
        save_json(output / "provenance.json", provenance)
        save_json(output / "config.json", config)
        net, upstream = recipe(args.cache_root)
        save_json(output / "upstream_recipe.json", {"net": net, "dataset": upstream})
        if args.method:
            measure(args, config, output)
            return
        verify_manifest(args.data_dir, json.loads(args.manifest.read_text()))
        for method in config["methods"]:
            print(f"[OFFICIAL_L16_START] method={method} pretrained=true", flush=True)
            command = [sys.executable, "-m", "ibkd_seg.cityscapes.official_smoke",
                       "--cache-root", str(args.cache_root), "--data-dir", str(args.data_dir),
                       "--manifest", str(args.manifest), "--output-dir", str(output / method),
                       "--device", args.device, "--method", method]
            if args.cpu_small:
                command.append("--cpu-small")
            subprocess.run(command, check=True)
            row = json.loads((output / method / "summary.json").read_text())
            report["runs"].append(row)
            save_json(output / "smoke_summary.json", report)
            print(f"[OFFICIAL_L16_METHOD_DONE] method={method} status=passed "
                  f"diagnostic_pixel_accuracy={row['diagnostic_pixel_accuracy']:.6f} "
                  f"diagnostic_miou={row['diagnostic_miou']:.6f} "
                  f"peak_cuda_bytes={row['train_peak_allocated_bytes']}", flush=True)
        for field in ("student_initial_state_sha256",):
            if len({r[field] for r in report["runs"]}) != 1:
                raise RuntimeError(f"Methods differ: {field}")
        if len({r["teacher_state_sha256"] for r in report["runs"] if r["method"] != "vanilla"}) != 1:
            raise RuntimeError("Methods used different teachers")
        if len({json_hash(r["data_identity"]) for r in report["runs"]}) != 1:
            raise RuntimeError("Methods used different inputs/labels/augmentations")
        report["status"] = "passed"
        save_json(output / "smoke_summary.json", report)
        print("[CITYSCAPES_OFFICIAL_L16_SMOKE_DONE] status=passed methods=4/4 scientific_result=false", flush=True)
    except Exception as error:
        report.update(status="failed", error=repr(error), failed_method=args.method or locals().get("method"))
        save_json(output / "failure.json", report)
        print(f"[CITYSCAPES_OFFICIAL_L16_SMOKE_DONE] status=failed error={error!r}", flush=True)
        raise


if __name__ == "__main__":
    main()
