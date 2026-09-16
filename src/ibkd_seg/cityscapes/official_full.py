"""Full, resumable single-seed Cityscapes training with pinned upstream L/16."""
from __future__ import annotations

import argparse
import json
import math
import platform
import signal
import time
from pathlib import Path

from .official_api import bootstrap


def validation_due(epoch, config):
    return (epoch - 1) % config["validation_every"] == 0 or epoch == config["epochs"]


def improves(scores, best):
    return best is None or scores["pixel_accuracy"] > best["metrics"]["pixel_accuracy"]


def initial_progress():
    return dict(epoch=1, next_batch=0, global_step=0, phase="train", beta=None,
                sums=dict(loss=0.0, ce=0.0, guidance=0.0), samples=0,
                best=None, history=[], first_batch_hashes={}, train_seconds=0.0,
                validation_seconds=0.0)


def run(args, config, output, stop):
    import random
    import numpy as np
    import cv2
    import PIL
    import torch
    import torchvision
    import timm
    import mmcv
    import mmseg
    from torch.nn import functional as F
    from segm.model.utils import inference
    import segm.utils.torch as ptu
    from . import official_api as api
    from .official_assets import verify
    from .full_data import prepare_labels, FullDataset, train_loader, batch_hash
    from .full_checkpoint import save_checkpoint, load_checkpoint, save_best, tree_hash
    from .data import save_json, json_hash
    from .evaluation import confusion_update, metrics
    from .runtime import seed_all, state_hash, source_hash, controller_for, restore_controller

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    ptu.device = device
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    provenance = verify(args.cache_root)
    save_json(output / "provenance.json", provenance)
    net, upstream = api.recipe(args.cache_root)
    save_json(output / "upstream_recipe.json", dict(net=net, dataset=upstream))
    # The optimizer factory uses the original config; never let local metadata
    # pretend to override its actual learning-rate schedule.
    for key in ("learning_rate", "epochs", "batch_size", "crop_size"):
        if not args.cpu_small and config[key] != upstream[key]:
            raise ValueError(f"Upstream recipe mismatch: {key}")
    manifest, data_hash = prepare_labels(args.data_dir, args.manifest,
                                         {s: config[s + "_samples"] for s in ("train", "val")}, output)
    datasets = {s: FullDataset(args.data_dir, manifest, config, s) for s in ("train", "val")}
    save_json(output / "train_pipeline.json", datasets["train"].base.config.data.train.pipeline)
    seed_all(config["seed"])
    model = api.student(args.cache_root).to(device)
    initial_hash = state_hash(model)
    seed_all(config["seed"] + 1000)
    guide = api.guidance(args.method, config)
    teacher, teacher_hash = None, None
    if guide is not None:
        guide = guide.to(device)
        teacher = api.teacher(args.cache_root).to(device)
        teacher_hash = state_hash(teacher)
    capture = api.FeatureCapture(model)
    optimizer, scheduler = api.optimizer_scheduler(model, guide, args.cache_root)
    if scheduler.iter_max != config["total_steps"] or not optimizer.defaults["nesterov"]:
        raise ValueError("Unexpected upstream optimizer/scheduler")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    controller = controller_for(args.method, config)
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "torchvision": str(torchvision.__version__), "timm": timm.__version__,
                   "mmcv": mmcv.__version__, "mmseg": mmseg.__version__,
                   "numpy": np.__version__, "pillow": PIL.__version__, "opencv": cv2.__version__,
                   "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "device": device.type, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
    signature = dict(config_sha256=json_hash(config), source_sha256=source_hash(),
                     data_sha256=data_hash, upstream_sha256=json_hash(provenance),
                     initial_student_sha256=initial_hash, teacher_sha256=teacher_hash,
                     method=args.method, environment=environment)
    seed_all(config["seed"] + 2000)
    progress = initial_progress()
    resume_info = None
    if args.resume:
        saved, resume_info = load_checkpoint(args.resume, signature, output)
        model.load_state_dict(saved["model"], strict=True)
        if guide is not None:
            guide.load_state_dict(saved["guide"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_controller(controller, saved["controller"])
        progress = saved["progress"]
        random.setstate(saved["python_rng"])
        rng = saved["numpy_rng"]
        np.random.set_state((rng[0], np.array(rng[1], dtype=np.uint32), *rng[2:]))
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
        print(f"[L16_RESUMED] method={args.method} step={progress['global_step']} generation={resume_info['generation']}", flush=True)
    save_json(output / "identity.json", signature)
    save_json(output / "environment.json", environment)
    save_json(output / "config.json", config)
    started_step = progress["global_step"]
    steps_per_epoch = math.ceil(len(datasets["train"]) / config["batch_size"])
    expected_steps = steps_per_epoch * config["epochs"]
    if not args.cpu_small and expected_steps != config["total_steps"]:
        raise ValueError("Full data/epoch count does not match upstream schedule")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def controller_state():
        return None if controller is None else controller.state_dict()

    def checkpoint():
        # Check the values actually being persisted, including momentum.
        tensors = parameters + [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise RuntimeError("Nonfinite parameter/optimizer state; retaining previous checkpoint")
        rng = np.random.get_state()
        payload = dict(signature=signature, model=model.state_dict(),
                       guide=None if guide is None else guide.state_dict(),
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       controller=controller_state(), progress=progress,
                       python_rng=random.getstate(), numpy_rng=(rng[0], rng[1].tolist(), *rng[2:]),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        save_checkpoint(output, payload)
        save_json(output / "history.json", progress["history"])
        save_json(output / "progress.json", progress)

    def pause_reason():
        if stop["signal"]:
            return stop["signal"]
        if args.max_hours and time.monotonic() - args.started >= args.max_hours * 3600:
            return "max_hours"
        if args.max_steps and progress["global_step"] - started_step >= args.max_steps:
            return "max_steps"
        return None

    def step(image, target):
        model.train()
        if guide is not None:
            guide.train()
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if not torch.isfinite(image).all() or not (target != 255).any():
            raise RuntimeError("Invalid input/all-void batch; no automatic resampling")
        optimizer.zero_grad(set_to_none=True)
        active = progress["beta"] > 0
        if active:
            logits, features = capture.forward(model, image)
        else:
            logits = model(image)
        ce = F.cross_entropy(logits, target, ignore_index=255)
        guided = ce.new_zeros(())
        if active:
            with torch.no_grad():
                t_features = teacher.extract_feat(api.teacher_input(image))[1:]
            if args.method == "ibkd":
                alignment, fusion = guide(features, t_features)
                ratio = config["ibkd_fusion_ratio"]
                guided = (1 - ratio) * alignment + ratio * fusion
            else:
                guided = guide(features, t_features)
        loss = ce + progress["beta"] * guided
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss: ce={float(ce.detach())} guidance={float(guided.detach())}")
        loss.backward()
        # Measure only. No finite gradient is clipped or silently skipped.
        norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
        if progress["next_batch"] == 0:
            modules = [model.encoder, model.decoder] + ([guide] if active else [])
            if any(not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in m.parameters()) for m in modules):
                raise RuntimeError("Missing encoder/decoder/active-guidance gradients")
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Frozen teacher received gradients")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step_update(scheduler.last_epoch + 1)
        return dict(loss=float(loss.detach()), ce=float(ce.detach()), guidance=float(guided.detach()),
                    lr=lr, grad_norm_unclipped=float(norm))

    def evaluate():
        model.eval()
        confusion = torch.zeros(19, 19, dtype=torch.int64)
        expected_pixels = 0
        with torch.no_grad():
            for index in range(len(datasets["val"])):
                if pause_reason():
                    return None  # Incomplete validation is never used for selection.
                ims, metas, target, _ = datasets["val"][index]
                prediction = inference(model, ims, metas, tuple(target.shape), config["window_size"],
                                       config["window_stride"], batch_size=1).argmax(0).cpu()
                confusion_update(confusion, prediction, target)
                expected_pixels += int((target != 255).sum())
        result = metrics(confusion)
        if result["valid_pixels"] != expected_pixels or (not args.cpu_small and result["evaluated_classes"] != 19):
            raise RuntimeError("Full validation pixel/class count mismatch")
        return result

    # Fresh runs have a finite step-zero recovery point. Resume creates a new
    # generation immediately, preserving portable best-student references.
    checkpoint()
    pause = None
    log_path = output / f"steps_{time.time_ns()}.jsonl"
    try:
        with log_path.open("w") as log:
            while progress["phase"] != "complete":
                epoch = progress["epoch"]
                if progress["phase"] == "train" and progress["next_batch"] == steps_per_epoch:
                    if progress["samples"] != len(datasets["train"]):
                        raise RuntimeError("Epoch omitted or duplicated samples")
                    if controller:
                        controller.observe(epoch, progress["sums"]["guidance"] / progress["samples"], beta_used=progress["beta"])
                    if teacher is not None and state_hash(teacher) != teacher_hash:
                        raise RuntimeError("Teacher weights or BN statistics changed")
                    progress["phase"] = "evaluate"
                    checkpoint()  # Resume must not observe the epoch twice.
                pause = pause_reason()
                if pause:
                    break
                if progress["phase"] == "train":
                    if progress["beta"] is None:
                        progress["beta"] = controller.beta_for_epoch(epoch) if controller else 0.0
                    loader = train_loader(datasets["train"], epoch, progress["next_batch"], device)
                    for image, target, ids in loader:
                        if progress["next_batch"] == 0:
                            progress["first_batch_hashes"][str(epoch)] = batch_hash(image, target, ids)
                        began = time.perf_counter()
                        row = step(image, target)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        seconds = time.perf_counter() - began
                        progress["train_seconds"] += seconds
                        progress["global_step"] += 1
                        progress["next_batch"] += 1
                        progress["samples"] += len(ids)
                        for key in progress["sums"]:
                            progress["sums"][key] += row[key] * len(ids)
                        log.write(json.dumps(dict(epoch=epoch, step=progress["global_step"], beta=progress["beta"],
                                                  seconds=seconds, ids=list(ids), **row), allow_nan=False) + "\n")
                        if progress["global_step"] % config["checkpoint_every_steps"] == 0:
                            log.flush()
                            checkpoint()
                        if progress["global_step"] <= 3:
                            print(f"[L16_STEP] method={args.method} step={progress['global_step']} loss={row['loss']:.6g} seconds={seconds:.2f}", flush=True)
                        if pause_reason():
                            break
                    del loader
                    continue
                began = time.perf_counter()
                scores = evaluate() if validation_due(epoch, config) else None
                progress["validation_seconds"] += time.perf_counter() - began
                if validation_due(epoch, config) and scores is None:
                    pause = pause_reason()
                    break
                if scores is not None and improves(scores, progress["best"]):
                    progress["best"] = save_best(output, model, signature, epoch, scores)
                row = dict(epoch=epoch, beta=progress["beta"],
                           **{k: v / progress["samples"] for k, v in progress["sums"].items()},
                           validation=scores, guidance_stop_epoch=None if controller is None else controller.stop_epoch)
                progress["history"].append(row)
                val_text = "" if scores is None else f" pixel_acc={scores['pixel_accuracy']:.6f} miou={scores['miou']:.6f}"
                print(f"[L16_EPOCH] method={args.method} epoch={epoch}/{config['epochs']} step={progress['global_step']} "
                      f"loss={row['loss']:.6g} guidance={row['guidance']:.6g} beta={row['beta']}{val_text}", flush=True)
                progress.update(epoch=epoch + 1, next_batch=0, beta=None,
                                sums=dict(loss=0.0, ce=0.0, guidance=0.0), samples=0,
                                phase="complete" if epoch == config["epochs"] else "train")
                checkpoint()
            checkpoint()
    except BaseException as error:
        # Do not overwrite a finite recovery point with the failing update.
        save_json(output / "failure.json", dict(status="failed", error=repr(error),
                  method=args.method, epoch=progress["epoch"], attempted_step=progress["global_step"] + 1,
                  recovery="resume.json", automatic_hyperparameter_changes=False))
        print(f"[CITYSCAPES_L16_FULL_DONE] status=failed method={args.method} recovery={output / 'resume.json'}", flush=True)
        raise
    if progress["phase"] == "complete" and progress["global_step"] != expected_steps:
        raise RuntimeError("Completed run has wrong update count")
    best = progress["best"]
    result = dict(status="complete" if progress["phase"] == "complete" else "paused", pause_reason=pause,
                  method=args.method, seed=config["seed"], run_kind=config["run_kind"],
                  global_step=progress["global_step"], expected_steps=expected_steps,
                  completed_epochs=len(progress["history"]), selected=best,
                  selection_metric="pixel_accuracy", secondary_metric="miou_at_same_checkpoint",
                  full_validation=not args.cpu_small and best is not None,
                  model_zoo_result_reproduced=False, test_used=False,
                  controller=controller_state(), resume=resume_info,
                  first_batch_hashes=progress["first_batch_hashes"],
                  train_seconds=progress["train_seconds"], validation_seconds=progress["validation_seconds"],
                  invocation_seconds=time.monotonic() - args.started,
                  final_student_sha256=state_hash(model),
                  final_guide_sha256=None if guide is None else state_hash(guide),
                  final_optimizer_sha256=tree_hash(optimizer.state_dict()),
                  final_scheduler=scheduler.state_dict(), teacher_frozen_verified=teacher is not None,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
    save_json(output / "summary.json", result)
    print(f"[CITYSCAPES_L16_FULL_DONE] status={result['status']} method={args.method} "
          f"epochs={result['completed_epochs']}/{config['epochs']} step={progress['global_step']} summary={output / 'summary.json'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache-root", "data-dir", "manifest", "output-dir"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("vanilla", "lg", "alg", "ibkd"))
    parser.add_argument("--resume", type=Path, help="Path to the saved bundle's resume.json")
    parser.add_argument("--max-hours", type=float, default=0, help="Soft pause limit including runner setup; 0 disables")
    parser.add_argument("--max-steps", type=int, default=0, help="Pause after this invocation's updates; 0 disables")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-small", action="store_true", help="Verification only: reduced images/data, full L/16")
    args = parser.parse_args()
    if (args.device == "cpu") != args.cpu_small:
        parser.error("CPU requires --cpu-small; CUDA may not reduce the recipe")
    if not math.isfinite(args.max_hours) or min(args.max_hours, args.max_steps) < 0:
        parser.error("Pause limits must be finite and nonnegative")
    for key in ("cache_root", "data_dir", "manifest", "output_dir", "resume"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if args.resume != args.output_dir / "resume.json":
            parser.error("Nonempty output requires resuming that directory's resume.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.started = time.monotonic()
    stop = {"signal": None}
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, frame: stop.update(signal=signal.Signals(number).name))
    bootstrap(args.cache_root)
    from .runtime import REPO
    config = json.loads((REPO / "phase4/phase4_cityscapes/configs/official_l16_full_v1.json").read_text())
    if args.cpu_small:
        config.update(protocol_id=config["protocol_id"] + "_cpu_small", run_kind="cpu_verification_only",
                      epochs=2, train_samples=4, val_samples=2, batch_size=2, image_size=64,
                      crop_size=32, window_size=32, window_stride=24, attention_query_chunk=7,
                      validation_every=1, data_workers=0)
    run(args, config, args.output_dir, stop)


if __name__ == "__main__":
    main()
