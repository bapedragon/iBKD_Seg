"""Run the locked Flowers-102 Phase 1A frozen spatial-probe protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ibkd_seg.phase0.checkpoints import load_manifest, sha256_file

from .config import effective_protocol, load_protocol, protocol_digest
from .flowers import (
    FlowersImageDataset,
    FlowersRecord,
    evenly_spaced_subset,
    load_flowers_records,
    load_targets,
)
from .probe import (
    confusion_from_tensors,
    evaluate_probe,
    probe_from_state,
    train_candidate,
)


SPLITS = ("train", "validation", "test")


def log(message: str) -> None:
    print(message, flush=True)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ids_digest(records: list[FlowersRecord]) -> str:
    payload = ",".join(str(record.image_id) for record in records).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_cache(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata", {})
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    return payload


def _target_cache(
    records: list[FlowersRecord],
    split: str,
    config: dict[str, Any],
    digest: str,
    cache_path: Path,
    refresh: bool,
) -> dict[str, Any]:
    input_size = int(config["dataset"]["input"]["size"])
    target_config = config["probe"]["target"]
    grid_size = (int(target_config["grid_height"]), int(target_config["grid_width"]))
    expected = {
        "kind": "flowers102_phase1a_targets",
        "protocol_digest": digest,
        "split": split,
        "ids_digest": _ids_digest(records),
        "count": len(records),
        "input_size": input_size,
        "grid_size": list(grid_size),
    }
    cached = None if refresh else _load_cache(cache_path, expected)
    if cached is not None:
        log(f"[CACHE] targets {split}: hit ({len(records)} samples)")
        return cached

    log(f"[CACHE] targets {split}: building {len(records)} samples")
    input_targets = torch.empty((len(records), input_size, input_size), dtype=torch.uint8)
    grid_targets = torch.empty((len(records), *grid_size), dtype=torch.uint8)
    ids = torch.empty(len(records), dtype=torch.int64)
    mask_config = config["dataset"]["mask"]
    for index, record in enumerate(records):
        input_target, grid_target = load_targets(
            record,
            input_size=input_size,
            grid_size=grid_size,
            alpha_threshold=float(mask_config["alpha_threshold"]),
            occupancy_threshold=float(target_config["occupancy_threshold"]),
        )
        input_targets[index] = input_target
        grid_targets[index] = grid_target
        ids[index] = record.image_id
        if (index + 1) % 1000 == 0 or index + 1 == len(records):
            log(f"[CACHE] targets {split}: {index + 1}/{len(records)}")

    payload = {
        "metadata": expected,
        "ids": ids,
        "input_targets": input_targets,
        "grid_targets": grid_targets,
    }
    _atomic_torch_save(payload, cache_path)
    return payload


def _load_encoder(
    entry: dict[str, Any],
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    import timm

    if checkpoint_path.stat().st_size != int(entry["size_bytes"]):
        raise RuntimeError(f"checkpoint size mismatch: {checkpoint_path}")
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != entry["sha256"]:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = timm.create_model(
        config["encoder"]["architecture"],
        pretrained=False,
        num_classes=int(entry["expected"]["num_classes"]),
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load returned incompatible keys")
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("encoder contains trainable parameters")
    return model.to(device)


@torch.inference_mode()
def _feature_cache(
    records: list[FlowersRecord],
    split: str,
    entry: dict[str, Any],
    checkpoint_path: Path,
    config: dict[str, Any],
    digest: str,
    cache_path: Path,
    device: torch.device,
    refresh: bool,
) -> dict[str, Any]:
    feature_config = config["encoder"]["feature"]
    feature_shape = [
        int(feature_config["channels"]),
        int(feature_config["height"]),
        int(feature_config["width"]),
    ]
    expected = {
        "kind": "flowers102_phase1a_frozen_features",
        "protocol_digest": digest,
        "split": split,
        "ids_digest": _ids_digest(records),
        "count": len(records),
        "checkpoint_id": entry["id"],
        "checkpoint_sha256": entry["sha256"],
        "feature_shape": feature_shape,
        "feature_dtype": "float32",
    }
    cached = None if refresh else _load_cache(cache_path, expected)
    if cached is not None:
        features = cached.get("features")
        if isinstance(features, torch.Tensor) and list(features.shape) == [len(records), *feature_shape]:
            log(f"[CACHE] features {entry['method']} {split}: hit ({len(records)} samples)")
            return cached

    log(f"[CACHE] features {entry['method']} {split}: building {len(records)} samples")
    model = _load_encoder(entry, checkpoint_path, config, device)
    dataset = FlowersImageDataset(records, config["dataset"]["input"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["runtime"]["feature_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["runtime"]["num_workers"]) > 0,
    )
    features = torch.empty((len(records), *feature_shape), dtype=torch.float32)
    ids = torch.empty(len(records), dtype=torch.int64)
    offset = 0
    for batch_index, (images, batch_ids) in enumerate(loader, start=1):
        _, intermediates = model.forward_intermediates(
            images.to(device),
            indices=[int(feature_config["block_index"])],
            norm=bool(feature_config["norm"]),
            output_fmt=feature_config["output_format"],
        )
        batch_features = intermediates[0].detach().to(device="cpu", dtype=torch.float32)
        if list(batch_features.shape[1:]) != feature_shape:
            raise RuntimeError(f"unexpected feature shape: {list(batch_features.shape)}")
        end = offset + len(batch_features)
        features[offset:end] = batch_features
        ids[offset:end] = batch_ids
        offset = end
        if batch_index % 25 == 0 or offset == len(records):
            log(f"[CACHE] features {entry['method']} {split}: {offset}/{len(records)}")
    if offset != len(records):
        raise RuntimeError("feature cache did not cover the complete split")
    expected_ids = torch.tensor([record.image_id for record in records], dtype=torch.int64)
    if not torch.equal(ids, expected_ids):
        raise RuntimeError("feature cache ID order mismatch")
    payload = {"metadata": expected, "ids": ids, "features": features}
    _atomic_torch_save(payload, cache_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _metrics_for_prediction(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, int | float]:
    return confusion_from_tensors(prediction, target).metrics()


def _baselines(targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train_grid = targets["train"]["grid_targets"]
    train_mean_score = train_grid.float().mean(dim=0)
    train_mean_grid = train_mean_score >= 0.5
    input_size = int(targets["train"]["input_targets"].shape[-1])
    train_mean_input = (
        F.interpolate(
            train_mean_score[None, None],
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        >= 0.5
    )
    report: dict[str, Any] = {}
    for name, grid_template, input_template in (
        (
            "all_background",
            torch.zeros_like(train_mean_grid),
            torch.zeros_like(train_mean_input),
        ),
        ("train_mean_mask", train_mean_grid, train_mean_input),
    ):
        split_metrics: dict[str, Any] = {}
        for split in ("validation", "test"):
            grid_target = targets[split]["grid_targets"]
            input_target = targets[split]["input_targets"]
            split_metrics[split] = {
                "grid": _metrics_for_prediction(
                    grid_template.expand_as(grid_target), grid_target
                ),
                "input_224": _metrics_for_prediction(
                    input_template.expand_as(input_target), input_target
                ),
            }
        report[name] = split_metrics
    return report


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


@torch.inference_mode()
def _save_qualitative_panels(
    probe: torch.nn.Module,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    test_records: list[FlowersRecord],
    qualitative_config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    input_size: int,
) -> list[str]:
    index_by_id = {record.image_id: index for index, record in enumerate(test_records)}
    record_by_id = {record.image_id: record for record in test_records}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for image_id in qualitative_config["test_ids"]:
        if int(image_id) not in index_by_id:
            continue
        index = index_by_id[int(image_id)]
        logits = probe(test_features[index : index + 1].to(device))
        logits = F.interpolate(
            logits,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        target = test_targets[index].numpy().astype(np.uint8)
        with Image.open(record_by_id[int(image_id)].image_path) as handle:
            resampling = getattr(Image, "Resampling", Image)
            image = handle.convert("RGB").resize(
                (input_size, input_size),
                resample=resampling.BILINEAR,
            )
        target_image = Image.fromarray(target * 255, mode="L").convert("RGB")
        prediction_image = Image.fromarray(prediction * 255, mode="L").convert("RGB")
        panel = Image.new("RGB", (input_size * 3, input_size))
        panel.paste(image, (0, 0))
        panel.paste(target_image, (input_size, 0))
        panel.paste(prediction_image, (input_size * 2, 0))
        path = output_dir / f"image_{int(image_id):05d}.png"
        panel.save(path)
        paths.append(str(path))
    return paths


def _checkpoint_result(
    entry: dict[str, Any],
    features: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    test_records: list[FlowersRecord],
    artifact_dir: Path,
    digest: str,
) -> dict[str, Any]:
    probe_config = config["probe"]
    batch_size = int(probe_config["batch_size"])
    seed_results: list[dict[str, Any]] = []
    for seed in probe_config["seeds"]:
        log(f"[PROBE] {entry['method']} seed={seed}")
        candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        for learning_rate in sorted(float(value) for value in probe_config["learning_rates"]):
            log(f"[PROBE] {entry['method']} seed={seed} lr={learning_rate:g}")
            state, candidate = train_candidate(
                features["train"]["features"],
                targets["train"]["grid_targets"],
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                probe_config=probe_config,
                learning_rate=learning_rate,
                seed=int(seed),
                device=device,
            )
            candidates.append((state, candidate))
        # Candidates were trained in ascending LR order. max() keeps the first
        # entry on an exact tie, implementing the locked lower-LR tie break.
        selected_index = max(
            range(len(candidates)),
            key=lambda index: candidates[index][1]["best_validation_grid_mean_iou"],
        )
        selected_state, selected = candidates[selected_index]
        probe = probe_from_state(probe_config, int(seed), selected_state, device)
        selected_artifact = artifact_dir / f"probe_seed_{int(seed)}.pt"
        _atomic_torch_save(
            {
                "protocol_id": config["protocol_id"],
                "protocol_sha256": digest,
                "checkpoint_id": entry["id"],
                "checkpoint_sha256": entry["sha256"],
                "probe_seed": int(seed),
                "selection": {
                    "learning_rate": selected["learning_rate"],
                    "epoch": selected["best_epoch"],
                    "validation_grid_mean_iou": selected[
                        "best_validation_grid_mean_iou"
                    ],
                },
                "model": selected_state,
            },
            selected_artifact,
        )
        evaluation: dict[str, Any] = {}
        for split in ("validation", "test"):
            evaluation[split] = {
                "grid": evaluate_probe(
                    probe,
                    features[split]["features"],
                    targets[split]["grid_targets"],
                    batch_size=batch_size,
                    device=device,
                ),
                "input_224": evaluate_probe(
                    probe,
                    features[split]["features"],
                    targets[split]["input_targets"],
                    batch_size=batch_size,
                    device=device,
                    output_size=(
                        int(config["dataset"]["input"]["size"]),
                        int(config["dataset"]["input"]["size"]),
                    ),
                ),
            }
        qualitative_paths: list[str] = []
        qualitative_config = config["report"]["qualitative"]
        if int(seed) == int(qualitative_config["probe_seed"]):
            qualitative_paths = _save_qualitative_panels(
                probe,
                features["test"]["features"],
                targets["test"]["input_targets"],
                test_records,
                qualitative_config,
                artifact_dir / "qualitative",
                device,
                int(config["dataset"]["input"]["size"]),
            )
        seed_results.append(
            {
                "probe_seed": int(seed),
                "selection": {
                    "learning_rate": selected["learning_rate"],
                    "epoch": selected["best_epoch"],
                    "validation_grid_mean_iou": selected[
                        "best_validation_grid_mean_iou"
                    ],
                },
                "candidates": [candidate for _, candidate in candidates],
                "evaluation": evaluation,
                "selected_probe_artifact": str(selected_artifact),
                "qualitative_panels": qualitative_paths,
            }
        )
        del probe

    metrics = config["report"]["metrics"]
    test_summary = {
        metric: _summary(
            [
                float(result["evaluation"]["test"]["input_224"][metric])
                for result in seed_results
            ]
        )
        for metric in metrics
    }
    return {
        "checkpoint_id": entry["id"],
        "method": entry["method"],
        "role": entry["role"],
        "protocol_family": entry["protocol_family"],
        "checkpoint_sha256": entry["sha256"],
        "encoder_seed": entry["expected"]["seed"],
        "seed_results": seed_results,
        "test_input_224_summary": test_summary,
    }


def _git_commit(repository_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    base_config = load_protocol(args.config)
    digest = protocol_digest(base_config)
    config = effective_protocol(base_config, smoke=args.smoke)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    device = _device(config["runtime"]["device"])
    log(f"[RUN] protocol={config['protocol_id']} mode={config['execution_mode']} device={device}")

    all_records = load_flowers_records(args.data_root)
    records = all_records
    if args.smoke:
        records = {
            split: evenly_spaced_subset(
                all_records[split], int(config["smoke"]["samples"][split])
            )
            for split in SPLITS
        }
    expected_counts = {"train": 1020, "validation": 1020, "test": 6149}
    if not args.smoke:
        actual_counts = {split: len(records[split]) for split in SPLITS}
        if actual_counts != expected_counts:
            raise RuntimeError(f"official split count mismatch: {actual_counts}")

    mode = config["execution_mode"]
    target_caches: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        target_caches[split] = _target_cache(
            records[split],
            split,
            base_config,
            digest,
            args.cache_root / "targets" / config["protocol_id"] / mode / f"{split}.pt",
            refresh=args.refresh_cache,
        )

    manifest_path = Path(config["encoder"]["checkpoint_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = args.repository_root / manifest_path
    manifest = load_manifest(manifest_path)
    entries = manifest["checkpoints"]
    if args.methods:
        requested_methods = set(args.methods)
    elif args.smoke:
        requested_methods = set(config["smoke"]["methods"])
    else:
        requested_methods = {entry["method"] for entry in entries}
    entries = [entry for entry in entries if entry["method"] in requested_methods]
    found_methods = {entry["method"] for entry in entries}
    if found_methods != requested_methods:
        raise ValueError(f"unknown or unavailable methods: {sorted(requested_methods - found_methods)}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "run": "flowers102_phase1a_frozen_spatial_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "execution_mode": mode,
        "scientific_interpretation": config["interpretation"],
        "comparable_result": not args.smoke,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": digest,
        "git_commit_at_start": _git_commit(args.repository_root),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
        "split_counts": {split: len(records[split]) for split in SPLITS},
        "baselines": _baselines(target_caches),
        "checkpoints": [],
    }
    _atomic_json_save(report, args.output)

    for entry in entries:
        checkpoint_path = args.source_root / entry["relative_path"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        feature_caches: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            feature_caches[split] = _feature_cache(
                records[split],
                split,
                entry,
                checkpoint_path,
                base_config,
                digest,
                args.cache_root
                / "features"
                / config["protocol_id"]
                / entry["id"]
                / mode
                / f"{split}.pt",
                device=device,
                refresh=args.refresh_cache,
            )
            if not torch.equal(feature_caches[split]["ids"], target_caches[split]["ids"]):
                raise RuntimeError(f"feature/target ID mismatch for {entry['method']} {split}")
        report["checkpoints"].append(
            _checkpoint_result(
                entry,
                feature_caches,
                target_caches,
                config,
                device,
                records["test"],
                args.artifact_root / config["protocol_id"] / mode / entry["id"],
                digest,
            )
        )
        _atomic_json_save(report, args.output)
        del feature_caches

    finite_values: list[float] = []
    gradient_checks: list[dict[str, int]] = []
    for checkpoint in report["checkpoints"]:
        for seed_result in checkpoint["seed_results"]:
            finite_values.extend(
                float(candidate["best_validation_grid_mean_iou"])
                for candidate in seed_result["candidates"]
            )
            finite_values.extend(
                float(item["train_loss"])
                for candidate in seed_result["candidates"]
                for item in candidate["history"]
            )
            gradient_checks.extend(
                candidate["gradient_contract"] for candidate in seed_result["candidates"]
            )
    gate = {
        "finite_training_and_validation": bool(finite_values)
        and all(math.isfinite(value) for value in finite_values),
        "cached_features_have_no_gradients": bool(gradient_checks)
        and all(item["cached_feature_gradient_tensor_count"] == 0 for item in gradient_checks),
        "probe_has_two_gradient_tensors": bool(gradient_checks)
        and all(item["probe_gradient_tensor_count"] == 2 for item in gradient_checks),
        "test_evaluated_only_after_selection": True,
        "qualitative_panels_saved": bool(report["checkpoints"])
        and all(
            bool(checkpoint["seed_results"][0]["qualitative_panels"])
            and (
                args.smoke
                or len(checkpoint["seed_results"][0]["qualitative_panels"])
                == len(config["report"]["qualitative"]["test_ids"])
            )
            for checkpoint in report["checkpoints"]
        ),
    }
    report["gate"] = gate
    report["status"] = "pass" if all(gate.values()) else "fail"
    report["elapsed_seconds"] = time.monotonic() - started
    _atomic_json_save(report, args.output)
    log(f"[DONE] status={report['status']} report={args.output}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("phase1/configs/flowers102_phase1a_v1.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("data/flowers102"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("phase1/results/raw/cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase1/reports/phase1a.local.json"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("phase1/results/runs"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("Ours", "ALG", "KD"))
    parser.add_argument("--device")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
