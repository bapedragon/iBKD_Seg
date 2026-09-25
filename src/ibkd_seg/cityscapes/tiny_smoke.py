"""Calibrate Tiny guidance and exercise the locked 25-update Cityscapes paths.

The source/teacher/data recipe is inherited from the L/16 track. This is a
bounded connection diagnostic, not a beta sweep or a scientific result.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "phase4/Cityscapes_Segmenter-Ti16/configs/smoke25_v1.json"
FSKD_CONFIG = CONFIG.with_name("smoke25_fskd_v2.json")


def fixed_fskd_calibration(rows, protocol):
    """Observe the fixed baseline without deriving or tuning a beta."""
    return {"ce_median": statistics.median(row["ce"] for row in rows),
            "guidance_median": statistics.median(row["guidance"] for row in rows),
            "raw_components_median": {name: statistics.median(row["components"][name] for row in rows)
                                      for name in rows[0]["components"]},
            "beta_candidates": [], "pilot_beta": None, "optimizer_updates": 0,
            "validation_used": False, "status": "fixed_fskd_recipe_no_beta_search",
            "loss_coefficients": protocol["coefficients"]}


def beta_candidates(rows, target_ratio=0.03, multipliers=(1, 2, 4, 8)):
    """Loss-scale heuristic only; all inputs must come from pre-update batches."""
    ce = [float(row["ce"]) for row in rows]
    guide = [float(row["guidance"]) for row in rows]
    if not ce or not all(math.isfinite(x) and x > 0 for x in ce):
        raise ValueError("Invalid segmentation CE calibration")
    if not all(math.isfinite(x) and x >= 0 for x in guide):
        raise ValueError("Invalid guidance calibration")
    c, g = statistics.median(ce), statistics.median(guide)
    if g <= 1e-10:
        raise ValueError("Guidance is too small to derive a meaningful beta grid")
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError("Invalid initial guidance/CE target")
    values = [target_ratio * c / g * multiple for multiple in multipliers]
    if len(values) != 4 or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Expected four finite positive beta candidates")
    measured = []
    for beta in values:
        ratios = [beta * raw / task for raw, task in zip(guide, ce, strict=True)]
        measured.append({"beta": beta, "weighted_guidance_to_seg_loss_min": min(ratios),
                         "weighted_guidance_to_seg_loss_median": statistics.median(ratios),
                         "weighted_guidance_to_seg_loss_max": max(ratios)})
    return {"seg_loss_type": "pixelwise_ce_ignore255_mean", "ce_median": c,
            "guidance_median": g, "beta_candidates": values,
            "pilot_beta": values[0], "measured_ratios": measured,
            "optimizer_updates": 0, "validation_used": False,
            "status": "proposed_not_stability_or_performance_selected"}


def rng_state():
    import numpy as np
    import torch
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    import numpy as np
    import torch
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def assert_state_close(actual, expected, *, rtol, atol):
    import torch
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=rtol, atol=atol)
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("State dictionary keys changed")
        for key in actual:
            assert_state_close(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError("State sequence length changed")
        for left, right in zip(actual, expected, strict=True):
            assert_state_close(left, right, rtol=rtol, atol=atol)
    elif actual != expected:
        raise AssertionError(f"State metadata changed: {actual!r} != {expected!r}")


def load_config(path):
    config = json.loads(path.read_text())
    locked = next((p for p in (CONFIG, FSKD_CONFIG) if p.name == path.name), None)
    if locked is None or config != json.loads(locked.read_text()):
        raise ValueError("Use the locked Tiny smoke config; protocol edits need a new revision")
    return config


def measure(args, config, plan, output):
    import torch
    from torch.nn import functional as F
    import segm.utils.torch as ptu
    from segm.model.utils import inference
    from . import official_api as api
    from .data import json_hash, save_json, sha256
    from .evaluation import confusion_update, metrics
    from .full_data import FullDataset, train_loader, batch_hash
    from .ibkd_deterministic import apply_deterministic_candidate
    from .public_smoke import verify_controller_paths
    from .runtime import seed_all, state_hash, source_hash, controller_for, restore_controller

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Tiny H200 smoke requires exactly one visible CUDA GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    device = torch.device("cuda")
    ptu.device = device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(config["ibkd_cpu_threads"] if plan["method"] == "ibkd" else 4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = json.loads(args.manifest.read_text())
    datasets = {split: FullDataset(args.data_dir, manifest, config, split) for split in ("train", "val")}

    seed_all(config["seed"])
    model = api.student(args.cache_root, backbone=config["student_backbone"],
                        image_size=config["crop_size"], decoder_layers=config["decoder_layers"],
                        recompute=config["gradient_checkpointing"]).to(device).train()
    initial_student = state_hash(model)
    initial_student_state = cpu_state(model)
    seed_all(config["seed"] + 1000)
    is_fskd = plan["method"] == "fskd"
    soft_rank_execution = None
    if is_fskd:
        from .tiny_fskd import CLSAttentionCapture, TinyFSKD, weighted_components, verify_soft_rank
        guide = TinyFSKD(crop=config["crop_size"])
        soft_rank_execution = verify_soft_rank(device)
    else:
        guide = api.guidance(plan["method"], config)
    candidate_contract = None
    if plan["method"] == "ibkd":
        candidate_contract = apply_deterministic_candidate(guide)
        if not candidate_contract["applied"]:
            raise RuntimeError("Missing deterministic iBKD operators")
    teacher = None
    teacher_hash = None
    initial_guide = None
    if guide is not None:
        guide = guide.to(device).train()
        initial_guide = state_hash(guide)
        initial_guide_state = cpu_state(guide)
        teacher = api.teacher(args.cache_root).to(device)
        teacher_hash = state_hash(teacher)
    capture = api.FeatureCapture(model)
    attention_capture = CLSAttentionCapture(model) if is_fskd else None
    teacher_diagnostic = None

    def losses(batch):
        nonlocal teacher_diagnostic
        images, target = (value.to(device) for value in batch[:2])
        if not torch.isfinite(images).all() or not (target != 255).any():
            raise RuntimeError("Invalid input/all-void batch")
        if guide is None:
            logits = model(images)
            features = None
        elif is_fskd:
            logits, features, cls_attention = attention_capture.forward(capture, model, images)
        else:
            logits, features = capture.forward(model, images)
        ce = F.cross_entropy(logits, target, ignore_index=255)
        alignment, fusion, guided = ce.new_zeros(()), ce.new_zeros(()), ce.new_zeros(())
        if guide is not None:
            with torch.no_grad():
                teacher_features = teacher.extract_feat(api.teacher_input(images))
                teacher_logits = teacher.decode_head(teacher_features) if is_fskd or teacher_diagnostic is None else None
                if teacher_diagnostic is None:
                    diagnostic_logits = F.interpolate(teacher_logits, target.shape[-2:], mode="bilinear", align_corners=False)
                    teacher_ce = F.cross_entropy(diagnostic_logits, target, ignore_index=255)
                    if not torch.isfinite(teacher_ce):
                        raise RuntimeError("Nonfinite pretrained teacher output")
                    teacher_diagnostic = {"ce": float(teacher_ce),
                                          "feature_shapes": [list(x.shape) for x in teacher_features[1:]],
                                          "all_backbone_feature_shapes": [list(x.shape) for x in teacher_features]}
            if is_fskd:
                components = guide(features, cls_attention, teacher_features, logits, teacher_logits, target)
                weighted = weighted_components(components)
                if not all(bool(torch.isfinite(x)) for x in components.values()):
                    raise FloatingPointError("Nonfinite FSKD loss component")
                if torch.is_grad_enabled() and not components["attention"].requires_grad:
                    raise RuntimeError("FSKD attention was detached from the student")
                guided = sum(weighted.values())
            elif plan["method"] == "ibkd":
                alignment, fusion = guide(features, teacher_features[1:])
                guided = (1 - plan["lambda"]) * alignment + plan["lambda"] * fusion
            else:
                guided = guide(features, teacher_features[1:])
        if not all(bool(torch.isfinite(value)) for value in (ce, guided, alignment, fusion)):
            raise FloatingPointError("Nonfinite segmentation/guidance loss")
        values = {"ce": float(ce.detach()), "guidance": float(guided.detach()),
                  "alignment": float(alignment.detach()), "fusion": float(fusion.detach())}
        if is_fskd:
            values.update(components={key: float(value.detach()) for key, value in components.items()},
                          weighted_components={key: float(value.detach()) for key, value in weighted.items()})
        return ce, guided, values

    calibration_rows, calibration_hashes = [], []
    seed_all(config["seed"] + 2000)
    initial_rng = rng_state()
    with torch.no_grad():
        for index, batch in enumerate(itertools.islice(train_loader(datasets["train"], 1, 0, device),
                                                       config["calibration_batches"]), 1):
            calibration_hashes.append(batch_hash(*batch))
            _, _, row = losses(batch)
            calibration_rows.append(row)
            if index == 1 or index % 5 == 0:
                print(f"[TI16_CALIBRATION] run={plan['id']} batch={index}/{config['calibration_batches']} "
                      f"seg_loss={row['ce']:.6g} guidance={row['guidance']:.6g} optimizer_updates=0", flush=True)
    if len(calibration_rows) != config["calibration_batches"]:
        raise RuntimeError("Incomplete calibration batches")
    calibration = (fixed_fskd_calibration(calibration_rows, config["fskd"]) if is_fskd else
                   beta_candidates(calibration_rows, config["beta_initial_ce_ratio"], config["beta_multipliers"])
                   if guide is not None else {"ce_median": statistics.median(r["ce"] for r in calibration_rows),
                                              "beta_candidates": [], "pilot_beta": 0.0, "optimizer_updates": 0,
                                              "validation_used": False, "status": "vanilla_ce_only"})
    model.load_state_dict(initial_student_state, strict=True)
    del initial_student_state
    if guide is not None:
        guide.load_state_dict(initial_guide_state, strict=True)
        del initial_guide_state
    restore_rng(initial_rng)
    if state_hash(model) != initial_student or (guide is not None and state_hash(guide) != initial_guide):
        raise RuntimeError("Calibration changed the initial training state")
    save_json(output / "calibration.json", {**calibration, "batches": calibration_rows,
                                           "input_hashes": calibration_hashes, "state_restored": True})
    print("[TI16_BETA_PROPOSAL] " + json.dumps({"run": plan["id"], **calibration}, allow_nan=False), flush=True)
    effective = dict(config, guidance_beta=calibration["pilot_beta"] if not is_fskd else 1.0)
    controller = controller_for(plan["method"], effective)
    beta = 1.0 if is_fskd else controller.beta_for_epoch(1) if controller else 0.0
    optimizer, scheduler = api.optimizer_scheduler(model, guide, args.cache_root, total_steps=config["total_steps"])
    if scheduler.iter_max != 80000 or not optimizer.defaults["nesterov"]:
        raise RuntimeError("Tiny smoke must preserve the 80k Nesterov SGD schedule")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    modules = {"student": model, **({"guidance": guide} if guide is not None else {})}
    checked_gradients = {"encoder": model.encoder, "decoder": model.decoder,
                         **({"guidance": guide} if guide is not None else {})}
    if is_fskd:
        checked_gradients.update({f"fskd_alignment_{i}": adapter for i, adapter in enumerate(guide.align)})

    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        ce, guided, values = losses(batch)
        loss = ce + beta * guided
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite total loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
        for name, module in checked_gradients.items():
            if not any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in module.parameters()):
                raise RuntimeError(f"No nonzero gradients in {name}")
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Frozen teacher received gradients")
        values.update(loss=float(loss.detach()), beta=None if is_fskd else beta,
                      guidance_multiplier=beta, weighted_guidance=beta * values["guidance"],
                      weighted_guidance_to_seg_loss=beta * values["guidance"] / values["ce"],
                      grad_norm_unclipped=float(norm), lr=optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step_update(scheduler.last_epoch + 1)
        return values

    rows, training_hashes = [], []
    teacher_initial_hash = teacher_hash
    checkpoint_path = output / "resume_probe.pt"
    config_hash = json_hash(config)
    source = source_hash()
    if config.get("fskd"):
        source = json_hash({"common": source, "shared_loss_primitives": sha256(Path(__file__).parent / "b0/losses.py")})
    torch.cuda.reset_peak_memory_stats()
    for index, batch in enumerate(itertools.islice(train_loader(datasets["train"], 1, 0, device), config["steps"]), 1):
        digest = batch_hash(*batch)
        training_hashes.append(digest)
        if digest != calibration_hashes[index - 1]:
            raise RuntimeError("Calibration/training input or augmentation differs")
        if index == config["steps"]:
            torch.save({"modules": {name: cpu_state(module) for name, module in modules.items()},
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "controller": None if controller is None else controller.state_dict(),
                        "rng": rng_state(), "completed_steps": index - 1, "next_input_sha256": digest,
                        "config_sha256": config_hash, "source_sha256": source, "run_id": plan["id"]}, checkpoint_path)
        torch.cuda.synchronize()
        start = time.perf_counter()
        row = update(batch)
        torch.cuda.synchronize()
        row.update(step=index, seconds=time.perf_counter() - start)
        rows.append(row)
        save_json(output / "training_progress.json", {"run_id": plan["id"], "completed_steps": index, "last": row})
        if index == 1 or index % 5 == 0:
            print(f"[TI16_SMOKE_STEP] run={plan['id']} step={index}/{config['steps']} "
                  f"loss={row['loss']:.6g} seg_loss={row['ce']:.6g} guidance={row['guidance']:.6g} "
                  f"guidance_multiplier={beta:.8g} ratio={row['weighted_guidance_to_seg_loss']:.6g} seconds={row['seconds']:.2f}", flush=True)
            if is_fskd:
                print("[TI16_FSKD_COMPONENTS] " + json.dumps({"step": index, "raw": row["components"],
                      "weighted": row["weighted_components"]}, allow_nan=False), flush=True)
    if len(rows) != config["steps"]:
        raise RuntimeError("Incomplete smoke training")
    if state_hash(model) == initial_student:
        raise RuntimeError("Student parameters did not update")
    peak = torch.cuda.max_memory_allocated()
    expected_modules = {name: cpu_state(module) for name, module in modules.items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    # This checkpoint was created by this process from verified local tensors.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (saved["config_sha256"] != config_hash or saved["source_sha256"] != source or
            saved["next_input_sha256"] != training_hashes[-1] or saved["run_id"] != plan["id"] or
            saved["completed_steps"] != config["steps"] - 1):
        raise RuntimeError("Resume checkpoint identity differs")
    for name, module in modules.items():
        module.load_state_dict(saved["modules"][name], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    restore_controller(controller, saved["controller"])
    restore_rng(saved["rng"])
    del saved
    replay_batch = next(iter(train_loader(datasets["train"], 1, config["steps"] - 1, device)))
    if batch_hash(*replay_batch) != training_hashes[-1]:
        raise RuntimeError("Resume changed input order or augmentation")
    replay = update(replay_batch)
    tolerance = {"rtol": config["resume_rtol"], "atol": config["resume_atol"]}
    for name, module in modules.items():
        assert_state_close(module.state_dict(), expected_modules[name], **tolerance)
    assert_state_close(optimizer.state_dict(), expected_optimizer, **tolerance)
    assert_state_close(scheduler.state_dict(), expected_scheduler, **tolerance)
    for key in ("loss", "ce", "guidance", "grad_norm_unclipped"):
        if not math.isclose(replay[key], rows[-1][key], rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"]):
            raise RuntimeError(f"Replayed {key} differs")
    if is_fskd:
        for name in replay["components"]:
            if not math.isclose(replay["components"][name], rows[-1]["components"][name],
                                rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"]):
                raise RuntimeError(f"Replayed FSKD component differs: {name}")
    # Evaluate the uninterrupted endpoint, keeping approximate replay only as a diagnostic.
    for name, module in modules.items():
        module.load_state_dict(expected_modules[name], strict=True)
    del expected_modules, expected_optimizer
    if teacher is not None and state_hash(teacher) != teacher_initial_hash:
        raise RuntimeError("Teacher parameters/BN state changed")
    model.eval()
    matrix = torch.zeros(19, 19, dtype=torch.int64)
    expected_pixels, val_ids = 0, []
    with torch.no_grad():
        for index in range(config["diagnostic_validation_samples"]):
            ims, metas, target, sample_id = datasets["val"][index]
            prediction = inference(model, ims, metas, tuple(target.shape), config["window_size"],
                                   config["window_stride"], batch_size=1).argmax(0).cpu()
            confusion_update(matrix, prediction, target)
            expected_pixels += int((target != 255).sum())
            val_ids.append(sample_id)
    scores = metrics(matrix)
    if scores["valid_pixels"] != expected_pixels:
        raise RuntimeError("Metric void-pixel accounting differs")
    checkpoint = {"bytes": checkpoint_path.stat().st_size, "sha256": sha256(checkpoint_path),
                  "strict_reload": "passed", "replayed_update": "passed", "retained": False,
                  "scope": "single_update_resume_probe_not_full_training_resume", "tolerance": tolerance}
    result = {"status": "passed", "run_id": plan["id"], "method": plan["method"], "lambda": plan.get("lambda"),
              "scientific_result": False, "completed_steps": len(rows), "selected_step": len(rows),
              "selected_epoch": 1, "selection_rule": "fixed_smoke_endpoint_not_best_checkpoint",
              "partial_epoch": True, "full_validation": False, "validation_samples": len(val_ids),
              "validation_ids": val_ids, "diagnostic_metrics": scores, "metric_scale": "fraction_0_to_1",
              "last": rows[-1], "losses": rows, "calibration": calibration,
              "student_initial_state_sha256": initial_student, "guidance_initial_state_sha256": initial_guide,
              "teacher_state_sha256": teacher_hash, "teacher_frozen_verified": teacher is not None,
              "teacher_diagnostic": teacher_diagnostic, "input_hashes": training_hashes,
              "input_sha256": json_hash(training_hashes), "calibration_state_restored": True,
              "source_sha256": source, "config_sha256": config_hash,
              "checkpoint": checkpoint, "student_parameters": sum(p.numel() for p in model.parameters()),
              "decoder_layers": len(model.decoder.blocks), "encoder_channels": model.encoder.d_model,
              "encoder_blocks": model.encoder.n_layers, "schedule_total_steps": scheduler.iter_max,
              "guidance_on": guide is not None, "guidance_stop_step": None,
              "controller_observation_epochs": 0, "natural_guidance_off_tested": False,
              "controller_synthetic_diagnostics": verify_controller_paths(effective) if controller else None,
              "fixed_loss_coefficients": config["fskd"]["coefficients"] if is_fskd else None,
              "method_provenance": config.get("fskd") if is_fskd else None,
              "soft_rank_execution": soft_rank_execution,
              "ibkd_deterministic_candidate": candidate_contract, "train_peak_allocated_bytes": peak,
              "median_step_seconds_excluding_first": statistics.median(row["seconds"] for row in rows[1:]),
              "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(), "precision": "fp32", "batch_size": 8},
              "test_used": False}
    checkpoint_path.unlink()
    save_json(output / "summary.json", result)
    return result


def compact_run(row):
    keys = ("status", "run_id", "method", "lambda", "completed_steps", "selected_step", "selected_epoch",
            "selection_rule", "partial_epoch", "validation_samples", "full_validation", "last", "calibration",
            "teacher_frozen_verified", "calibration_state_restored", "checkpoint", "decoder_layers",
            "encoder_channels", "encoder_blocks", "schedule_total_steps", "guidance_on", "guidance_stop_step",
            "natural_guidance_off_tested", "controller_synthetic_diagnostics", "train_peak_allocated_bytes", "median_step_seconds_excluding_first",
            "fixed_loss_coefficients", "method_provenance", "soft_rank_execution",
            "error", "summary_path", "deterministic_warning_count")
    result = {key: row[key] for key in keys if key in row}
    scores = row.get("diagnostic_metrics")
    result["diagnostic_metrics_percent"] = None if scores is None else {
        "pixel_accuracy": 100 * scores["pixel_accuracy"], "miou": 100 * scores["miou"],
        "class_iou": {key: None if value is None else 100 * value for key, value in scores["class_iou"].items()},
        "evaluated_classes": scores["evaluated_classes"], "valid_pixels": scores["valid_pixels"]}
    return result


def compare_runs(rows, expected_ids=("vanilla", "lg", "alg", "ibkd_lambda025", "ibkd_lambda050")):
    if (any(row["status"] != "passed" for row in rows) or len(rows) != len(expected_ids) or
            {row["run_id"] for row in rows} != set(expected_ids)):
        raise RuntimeError("All expected Tiny paths must pass")
    for key in ("student_initial_state_sha256", "input_sha256"):
        if len({row[key] for row in rows}) != 1:
            raise RuntimeError(f"Cross-method mismatch: {key}")
    if len({row["teacher_state_sha256"] for row in rows if row["method"] != "vanilla"}) != 1:
        raise RuntimeError("Different teachers across KD methods")
    by_id = {row["run_id"]: row for row in rows}
    left, right = by_id["lg"], by_id["alg"]
    if left["guidance_initial_state_sha256"] != right["guidance_initial_state_sha256"]:
        raise RuntimeError("LG/ALG adapter initial states differ")
    for key in ("ce_median", "guidance_median", "pilot_beta"):
        if not math.isclose(left["calibration"][key], right["calibration"][key], rel_tol=2e-5, abs_tol=2e-6):
            raise RuntimeError(f"LG/ALG calibration differs: {key}")
    if len(left["losses"]) != len(right["losses"]):
        raise RuntimeError("LG/ALG trajectory lengths differ")
    for a, b in zip(left["losses"], right["losses"], strict=True):
        for key in ("loss", "ce", "guidance", "grad_norm_unclipped"):
            if not math.isclose(a[key], b[key], rel_tol=2e-5, abs_tol=2e-6):
                raise RuntimeError(f"LG/ALG pre-controller trajectory differs: step={a['step']} key={key}")
    a, b = by_id["ibkd_lambda025"], by_id["ibkd_lambda050"]
    if a["guidance_initial_state_sha256"] != b["guidance_initial_state_sha256"]:
        raise RuntimeError("iBKD lambda conditions use different adapter initialization")
    return {"same_student_initial_state": True, "same_inputs_and_augmentations": True,
            "same_openmmlab_teacher": True, "lg_alg_25step_trajectory_matches": True,
            "ibkd_same_adapter_initialization": True, "input_sha256": rows[0]["input_sha256"],
            "student_initial_state_sha256": rows[0]["student_initial_state_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cache-root", "data-dir", "manifest", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("cache_root", "data_dir", "manifest", "output_dir", "config"):
        setattr(args, key, getattr(args, key).resolve())
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scientific_result": False, "test_used": False, "runs": [],
              "next_stage": "review_calibration_and_smoke_before_500_2000_beta_screen"}
    failed = False
    try:
        config = load_config(args.config)
        from .official_api import bootstrap
        bootstrap(args.cache_root)
        from .data import save_json
        from .official_assets import verify
        provenance = verify(args.cache_root, student="tiny")
        report.update(protocol_id=config["protocol_id"], teacher="OpenMMLab DeepLabV3-R101-D8 (frozen)",
                      student=config["student_backbone"], expected_runs=len(config["runs"]),
                      source_and_asset_verification="passed", assets=provenance["weights"])
        if config.get("fskd"):
            report.update(fskd_interpretation=config["fskd"]["interpretation"], c2vkd_status=config["c2vkd_status"])
        save_json(output / "provenance.json", provenance)
        save_json(output / "config.json", config)
        if args.run_id:
            plan = next(row for row in config["runs"] if row["id"] == args.run_id)
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                row = measure(args, config, plan, output)
            row["deterministic_warning_count"] = sum("deterministic" in str(w.message) for w in warning_records)
            save_json(output / "warnings.json", [str(w.message) for w in warning_records])
            save_json(output / "summary.json", row)
            report.update(status="passed", runs=[compact_run(row)])
        else:
            from .full_data import prepare_labels
            prepare_labels(args.data_dir, args.manifest, {"train": 2975, "val": 500}, output)
            rows = []
            for plan in config["runs"]:
                destination = output / plan["id"]
                command = [sys.executable, "-u", "-m", "ibkd_seg.cityscapes.tiny_smoke",
                           "--cache-root", str(args.cache_root), "--data-dir", str(args.data_dir),
                           "--manifest", str(args.manifest), "--output-dir", str(destination),
                           "--config", str(args.config), "--run-id", plan["id"]]
                completed = subprocess.run(command, check=False)
                result_path = destination / "summary.json"
                if completed.returncode == 0 and result_path.exists():
                    row = json.loads(result_path.read_text())
                else:
                    failure_path = destination / "smoke_summary.json"
                    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
                    progress = destination / "training_progress.json"
                    partial = json.loads(progress.read_text()) if progress.exists() else {}
                    row = {"status": "failed", "run_id": plan["id"], "method": plan["method"],
                           "lambda": plan.get("lambda"), "completed_steps": partial.get("completed_steps", 0),
                           "last": partial.get("last"), "selected_step": None, "selected_epoch": None,
                           "error": failure.get("error", f"child_exit={completed.returncode}"),
                           "summary_path": str(failure_path)}
                    if plan["method"] == "fskd":
                        row.update(method_provenance=config["fskd"], fixed_loss_coefficients=config["fskd"]["coefficients"])
                    calibration_path = destination / "calibration.json"
                    if calibration_path.exists():
                        calibration = json.loads(calibration_path.read_text())
                        row["calibration"] = {key: value for key, value in calibration.items()
                                              if key not in {"batches", "input_hashes"}}
                rows.append(row)
                report["runs"] = [compact_run(value) for value in rows]
                save_json(output / "smoke_summary.json", report)
            report["cross_method_checks"] = compare_runs(rows, [row["id"] for row in config["runs"]])
            report["status"] = "passed"
    except Exception as error:
        failed = True
        report.update(status="failed", error=repr(error), failed_run=args.run_id)
        (output / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report["summary_path"] = str(output / "smoke_summary.json")
        (output / "smoke_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("[CITYSCAPES_TI16_SMOKE_FINAL] " + json.dumps(report, allow_nan=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
