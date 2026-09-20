#!/usr/bin/env python3
"""Smoke-test spatial diagnostics on frozen CUB segmentation encoders."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ibkd_seg.cityscapes.data import json_hash, save_json, sha256
from ibkd_seg.cityscapes.models import Segmenter
from ibkd_seg.cityscapes.runtime import state_hash
from ibkd_seg.phase1.cub_direct_spatial import (
    GRID_SIZE,
    LinearCKAAccumulator,
    assert_probability,
    attention_gt_metrics,
    attention_rollout,
    build_part_probe,
    evaluate_part_probe,
    feature_map_to_observations,
    load_mask_views,
    load_part_supervision,
    load_spatial_annotations,
    masked_heatmap_mse,
    select_lowest_image_id_per_class,
    summarize_part_supervision,
)
from ibkd_seg.phase1.cub_probe_data import (
    CubImageDataset,
    CubProbeRecord,
    ids_sha256,
    load_train_validation_records,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub_Seg/configs/spatial_diagnostics_seed1_smoke_v1.json"
)
METHODS = ("vanilla", "lg", "alg", "ibkd")
PROBE_KINDS = ("linear", "nonlinear")


def log(message: str = "") -> None:
    print(message, flush=True)


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id",
        "status",
        "scientific_result",
        "scope",
        "checkpoint_source",
        "dataset",
        "part_probe",
        "spatial_cka",
        "attention_gt",
        "test_miou_reference",
        "execution",
        "completion_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise RuntimeError("CUB segmentation spatial-smoke config keys changed")

    scope = config["scope"]
    source = config["checkpoint_source"]
    dataset = config["dataset"]
    probe = config["part_probe"]
    cka = config["spatial_cka"]
    attention = config["attention_gt"]
    execution = config["execution"]
    gate = config["completion_gate"]
    checks = {
        "protocol": config["protocol_id"]
        == "cub200_direct_segmentation_seed1_spatial_diagnostics_smoke_v1",
        "locked": config["status"] == "locked_before_smoke_results_2026-09-20",
        "non_scientific": config["scientific_result"] is False,
        "scope": scope
        == {
            "encoder_seed": 1,
            "methods": list(METHODS),
            "checkpoint_selection": "job783_validation_selected_best",
            "method_or_protocol_selection_from_smoke": False,
        },
        "source": source["h200_job_id"] == 783
        and source["training_protocol_id"]
        == "cub200_direct_binary_segmentation_window30_exploratory_full_v1"
        and source["training_config_sha256"]
        == "be94f411f97ff162e1adacac4951cf7185b18d00bb8d09486b068105a38f8d93",
        "split": dataset["split"]
        == {
            "train": 5394,
            "validation": 600,
            "official_test": 5794,
            "validation_per_class": 3,
            "split_seed": 2027,
        },
        "subset": dataset["smoke_subset"]
        == {
            "rule": "lowest_image_id_per_class",
            "train_images": 200,
            "validation_images": 200,
        }
        and dataset["input_size"] == 224
        and dataset["part_count"] == 15
        and dataset["official_test_images_accessed"] is False,
        "probe": probe["student_feature"]
        == "pre_final_norm_block11_192x14x14"
        and probe["target"]
        == "visible_part_gaussian_heatmap_sigma_1_grid_pixel"
        and probe["loss"] == "visible_part_masked_mean_squared_error"
        and probe["probe_seed"] == 1
        and probe["learning_rates"] == [0.01, 0.03, 0.1]
        and probe["epochs"] == 2
        and probe["batch_size"] == 64
        and probe["linear"]
        == {
            "architecture": "Conv2d(192,15,1,bias=True)",
            "parameter_count": 2895,
        }
        and probe["nonlinear"]["parameter_count"] == 111695,
        "cka": cka["split"] == "fixed_validation_200"
        and cka["student_blocks"] == list(range(12))
        and cka["teacher_feature"] == "resnet50_layer3_1024x14x14"
        and cka["metric"] == "centered_linear_CKA"
        and cka["accumulator_dtype"] == "float64"
        and cka["reported_primary"] == "student_block11",
        "attention": attention["split"] == "fixed_validation_200"
        and attention["rollout_layers"] == 12
        and attention["head_fusion"] == "arithmetic_mean"
        and attention["reported_metrics"]
        == [
            "global_micro_patch_average_precision",
            "pointing_game_peak_inside_mask",
            "foreground_attention_mass_mean",
        ],
        "test_reference": config["test_miou_reference"]
        == {
            "source": "audited_job783_full_summary_only",
            "metric": "two_class_miou",
            "official_test_images_reevaluated": False,
        },
        "execution": execution
        == {
            "requested_mig_slices": 1,
            "feature_batch_size": 16,
            "cka_batch_size": 8,
            "attention_batch_size": 16,
            "num_workers": 4,
            "precision": "fp32",
        },
        "gate": gate
        == {
            "checkpoint_strict_loads": 5,
            "linear_probe_candidates": 12,
            "nonlinear_probe_candidates": 12,
            "probe_validation_selections": 8,
            "spatial_cka_values": 48,
            "attention_metric_rows": 4,
            "test_miou_references": 4,
            "official_test_image_evaluations": 0,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "invalid CUB segmentation spatial-smoke config: " + ", ".join(failures)
        )

    manifest_path = _repository_path(source["manifest_path"])
    if (
        not manifest_path.is_file()
        or sha256(manifest_path) != source["manifest_sha256"]
    ):
        raise RuntimeError("CUB segmentation checkpoint release manifest changed")
    return config


class NonlinearPartProbe(nn.Sequential):
    """A small local nonlinear head used only as a probe-capacity check."""

    def __init__(self, channels: int = 192, hidden: int = 64, parts: int = 15):
        super().__init__(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, parts, kernel_size=1, bias=True),
        )


def build_probe(kind: str, seed: int) -> nn.Module:
    if kind not in PROBE_KINDS:
        raise ValueError(f"unknown part-probe kind: {kind}")
    if kind == "linear":
        return build_part_probe(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        probe = NonlinearPartProbe()
        nn.init.kaiming_normal_(probe[0].weight, nonlinearity="relu")
        nn.init.ones_(probe[1].weight)
        nn.init.zeros_(probe[1].bias)
        nn.init.normal_(probe[3].weight, mean=0.0, std=0.01)
        nn.init.zeros_(probe[3].bias)
    return probe


def probe_parameter_count(kind: str) -> int:
    return sum(parameter.numel() for parameter in build_probe(kind, 1).parameters())


def train_probe_candidate(
    *,
    kind: str,
    train_features: torch.Tensor,
    train_supervision: dict[str, torch.Tensor],
    validation_features: torch.Tensor,
    validation_supervision: dict[str, torch.Tensor],
    learning_rate: float,
    epochs: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if learning_rate <= 0.0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("part probe LR, epochs, and batch size must be positive")
    probe = build_probe(kind, seed).to(device)
    initial_hash = state_hash(probe)
    optimizer = torch.optim.SGD(
        probe.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )
    generator = torch.Generator().manual_seed(seed)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        probe.train()
        order = torch.randperm(len(train_features), generator=generator)
        running_loss = 0.0
        visible_total = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            feature = train_features[indices].to(device)
            target = train_supervision["heatmaps"][indices].to(device)
            visible = train_supervision["visible"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(feature)
            loss = masked_heatmap_mse(prediction, target, visible)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {kind} probe loss")
            loss.backward()
            optimizer.step()
            count = int(visible.sum().item())
            running_loss += float(loss.detach().item()) * count
            visible_total += count
        metrics = evaluate_part_probe(
            probe,
            validation_features,
            validation_supervision,
            device=device,
            batch_size=batch_size,
        )
        row = {
            "epoch": epoch,
            "train_visible_weighted_loss": running_loss / visible_total,
            "learning_rate_before_scheduler_step": float(
                scheduler.get_last_lr()[0]
            ),
            "validation": metrics,
        }
        history.append(row)
        score = metrics["micro_pck_at_0.1"]
        if best is None or score > best["validation"]["micro_pck_at_0.1"]:
            best = {
                "epoch": epoch,
                "validation": metrics,
                "probe_state": {
                    key: value.detach().cpu().clone()
                    for key, value in probe.state_dict().items()
                },
            }
        scheduler.step()
    assert best is not None
    final_hash = state_hash(probe)
    if final_hash == initial_hash:
        raise RuntimeError(f"{kind} probe parameters did not update")
    if not all(
        math.isfinite(row["train_visible_weighted_loss"]) for row in history
    ):
        raise RuntimeError(f"{kind} probe history contains a non-finite loss")
    return {
        "kind": kind,
        "learning_rate": learning_rate,
        "seed": seed,
        "epochs": epochs,
        "parameter_count": sum(p.numel() for p in probe.parameters()),
        "initial_probe_state_sha256": initial_hash,
        "final_probe_state_sha256": final_hash,
        "parameter_update_verified": True,
        "best_epoch": best["epoch"],
        "best_validation": best["validation"],
        "probe_state": best["probe_state"],
        "history": history,
    }


def _select_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("cannot select from empty probe candidates")
    return min(
        candidates,
        key=lambda value: (
            -value["best_validation"]["micro_pck_at_0.1"],
            value["learning_rate"],
            value["best_epoch"],
        ),
    )


@torch.inference_mode()
def extract_block11_features(
    model: Segmenter,
    records: Sequence[CubProbeRecord],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        CubImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    values: list[torch.Tensor] = []
    for images, _image_ids in loader:
        feature = model.features(images.to(device, non_blocking=True))[11]
        if feature.shape[1:] != (192, GRID_SIZE, GRID_SIZE):
            raise RuntimeError(
                f"unexpected student block11 feature shape: {tuple(feature.shape)}"
            )
        values.append(feature.detach().cpu())
    result = torch.cat(values)
    if result.shape != (len(records), 192, GRID_SIZE, GRID_SIZE):
        raise RuntimeError("student block11 feature cache shape changed")
    return result


@torch.inference_mode()
def run_cka(
    student: Segmenter,
    teacher: Segmenter,
    records: Sequence[CubProbeRecord],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[float]:
    loader = DataLoader(
        CubImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    accumulators = [
        LinearCKAAccumulator(192, 1024, device=device, dtype=torch.float64)
        for _ in range(12)
    ]
    images_seen = 0
    for images, _image_ids in loader:
        images = images.to(device, non_blocking=True)
        student_features = student.features(images)
        teacher_feature = teacher.features(images)[2]
        if teacher_feature.shape[1:] != (1024, GRID_SIZE, GRID_SIZE):
            raise RuntimeError(
                f"unexpected teacher layer3 feature shape: {tuple(teacher_feature.shape)}"
            )
        teacher_observations = feature_map_to_observations(teacher_feature)
        for accumulator, feature in zip(
            accumulators, student_features, strict=True
        ):
            accumulator.update(
                feature_map_to_observations(feature), teacher_observations
            )
        images_seen += images.shape[0]
    if images_seen != len(records):
        raise RuntimeError("spatial CKA image count mismatch")
    values = [accumulator.compute() for accumulator in accumulators]
    for block, value in enumerate(values):
        assert_probability(value, name=f"block{block}/centered_linear_cka")
    return values


@torch.inference_mode()
def run_attention(
    student: Segmenter,
    records: Sequence[CubProbeRecord],
    masks: torch.Tensor,
    occupancies: torch.Tensor,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        CubImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    values: list[torch.Tensor] = []
    for images, _image_ids in loader:
        values.append(
            attention_rollout(
                student.encoder, images.to(device, non_blocking=True)
            ).cpu()
        )
    rollout = torch.cat(values)
    if len(rollout) != len(records):
        raise RuntimeError("attention rollout image count mismatch")
    metrics = attention_gt_metrics(rollout, masks, occupancies)
    for key in (
        "global_micro_patch_average_precision",
        "pointing_game_peak_inside_mask",
        "foreground_attention_mass_mean",
    ):
        assert_probability(metrics[key], name=key)
    return metrics


def _load_checkpoint_model(
    release_dir: Path,
    release_manifest: dict[str, Any],
    name: str,
    training_config: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[Segmenter, dict[str, Any]]:
    contract = release_manifest["checkpoints"][name]
    checkpoint = release_dir / contract["relative_path"]
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != contract["bytes"]
        or sha256(checkpoint) != contract["checkpoint_sha256"]
    ):
        raise RuntimeError(f"checkpoint byte contract failed: {name}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    role = "teacher" if name == "teacher" else "student"
    method = None if name == "teacher" else name
    identity = payload.get("identity", {})
    if (
        payload.get("artifact") != "cub_direct_segmentation_validation_selected"
        or payload.get("scientific_result") is not False
        or payload.get("epoch") != contract["selected_epoch"]
        or payload.get("config") != training_config
        or identity.get("role") != role
        or identity.get("method") != method
    ):
        raise RuntimeError(f"checkpoint metadata contract failed: {name}")
    model = Segmenter(role, training_config)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    loaded_hash = state_hash(model)
    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
        or loaded_hash != payload.get("model_state_sha256")
        or loaded_hash != contract["model_state_sha256"]
        or not all(
            bool(torch.isfinite(value).all())
            for value in payload["model"].values()
            if value.is_floating_point()
        )
    ):
        raise RuntimeError(f"checkpoint strict-load audit failed: {name}")
    model.to(device).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"frozen-model contract failed: {name}")
    audit = {
        "name": name,
        "role": role,
        "method": method,
        "selected_epoch": int(payload["epoch"]),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "model_state_sha256": loaded_hash,
        "strict_load": True,
        "all_floating_tensors_finite": True,
        "eval_mode": True,
        "trainable_parameters": 0,
    }
    del payload
    return model, audit


def _run_probe_kind(
    *,
    method: str,
    kind: str,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    train_supervision: dict[str, torch.Tensor],
    validation_supervision: dict[str, torch.Tensor],
    protocol: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    candidates: list[dict[str, Any]] = []
    for learning_rate in protocol["learning_rates"]:
        candidate = train_probe_candidate(
            kind=kind,
            train_features=train_features,
            train_supervision=train_supervision,
            validation_features=validation_features,
            validation_supervision=validation_supervision,
            learning_rate=float(learning_rate),
            epochs=int(protocol["epochs"]),
            seed=int(protocol["probe_seed"]),
            batch_size=int(protocol["batch_size"]),
            device=device,
        )
        candidates.append(candidate)
        save_json(
            output_dir
            / "probe_histories"
            / f"{method}_{kind}_lr{format(float(learning_rate), 'g').replace('.', 'p')}.json",
            {
                key: value
                for key, value in candidate.items()
                if key != "probe_state"
            },
        )
        log(
            "[CUB_SEG_SPATIAL_PROBE_CANDIDATE] "
            f"method={method} kind={kind} lr={learning_rate} "
            f"best_epoch={candidate['best_epoch']} "
            "val_pck="
            f"{candidate['best_validation']['micro_pck_at_0.1']:.6f} "
            "val_error="
            f"{candidate['best_validation']['mean_normalized_localization_error']:.6f}"
        )
    initial_hashes = {row["initial_probe_state_sha256"] for row in candidates}
    if len(initial_hashes) != 1:
        raise RuntimeError(f"probe initialization changed across LR: {method}/{kind}")
    selected = _select_candidate(candidates)
    selected_probe = build_probe(kind, int(protocol["probe_seed"]))
    incompatible = selected_probe.load_state_dict(selected["probe_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"selected probe strict reload failed: {method}/{kind}")
    selected_probe.to(device).eval()
    reloaded_validation = evaluate_part_probe(
        selected_probe,
        validation_features,
        validation_supervision,
        device=device,
        batch_size=int(protocol["batch_size"]),
    )
    if (
        reloaded_validation["micro_pck_at_0.1"]
        != selected["best_validation"]["micro_pck_at_0.1"]
        or reloaded_validation["mean_normalized_localization_error"]
        != selected["best_validation"]["mean_normalized_localization_error"]
    ):
        raise RuntimeError(f"selected probe validation changed: {method}/{kind}")

    candidate_summary = [
        {
            "learning_rate": row["learning_rate"],
            "best_epoch": row["best_epoch"],
            "best_validation_micro_pck_at_0.1": row["best_validation"][
                "micro_pck_at_0.1"
            ],
            "best_validation_mean_normalized_localization_error": row[
                "best_validation"
            ]["mean_normalized_localization_error"],
            "first_epoch_train_loss": row["history"][0][
                "train_visible_weighted_loss"
            ],
            "last_epoch_train_loss": row["history"][-1][
                "train_visible_weighted_loss"
            ],
            "loss_decreased_across_smoke": row["history"][-1][
                "train_visible_weighted_loss"
            ]
            < row["history"][0]["train_visible_weighted_loss"],
            "parameter_update_verified": row["parameter_update_verified"],
            "selected": row is selected,
        }
        for row in candidates
    ]
    result = {
        "kind": kind,
        "architecture": protocol[kind]["architecture"],
        "parameter_count": selected["parameter_count"],
        "probe_seed": int(protocol["probe_seed"]),
        "selected_learning_rate": selected["learning_rate"],
        "selected_epoch": selected["best_epoch"],
        "validation": reloaded_validation,
        "initial_probe_state_sha256": next(iter(initial_hashes)),
        "selected_probe_state_sha256": state_hash(selected_probe),
        "strict_reload": True,
        "candidates": candidate_summary,
    }
    del selected_probe, candidates
    return result, len(candidate_summary)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config = load_config(args.config)
    expected_execution = config["execution"]
    actual_execution = {
        "feature_batch_size": args.feature_batch_size,
        "cka_batch_size": args.cka_batch_size,
        "attention_batch_size": args.attention_batch_size,
        "num_workers": args.num_workers,
    }
    if actual_execution != {
        key: expected_execution[key] for key in actual_execution
    }:
        raise RuntimeError("spatial-smoke CLI differs from locked execution settings")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUB segmentation spatial smoke requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty smoke output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    release_manifest_path = _repository_path(
        config["checkpoint_source"]["manifest_path"]
    )
    release_manifest = json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    if (
        release_manifest.get("asset_type")
        != "phase1_cub_direct_segmentation_window30_seed1_v1"
        or release_manifest.get("source", {}).get("h200_job_id") != 783
        or set(release_manifest.get("checkpoints", {}))
        != {"teacher", *METHODS}
    ):
        raise RuntimeError("unexpected CUB segmentation checkpoint release manifest")

    full_summary_path = args.release_dir / "full_summary.json"
    training_config_path = args.release_dir / "config.json"
    if (
        not full_summary_path.is_file()
        or not training_config_path.is_file()
        or sha256(full_summary_path)
        != release_manifest["source"]["full_summary_sha256"]
    ):
        raise RuntimeError("audited issue-783 summary is missing or changed")
    full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if (
        full_summary.get("status") != "complete"
        or full_summary.get("scientific_result") is not False
        or full_summary.get("config") != training_config
        or full_summary.get("config_sha256")
        != config["checkpoint_source"]["training_config_sha256"]
    ):
        raise RuntimeError("audited issue-783 completion contract failed")

    records, split_manifest, data_source = load_train_validation_records(
        args.data_dir, download=True
    )
    if {key: len(value) for key, value in records.items()} != {
        "train": 5394,
        "validation": 600,
    }:
        raise RuntimeError("CUB train/validation counts changed")
    full_identity = full_summary["data_identity"]
    if (
        json_hash(split_manifest) != full_identity["split_manifest_sha256"]
        or ids_sha256(records["train"])
        != full_identity["train_image_ids_sha256"]
        or ids_sha256(records["validation"])
        != full_identity["validation_image_ids_sha256"]
    ):
        raise RuntimeError("CUB data identity differs from issue 783")

    train_records = select_lowest_image_id_per_class(records["train"])
    validation_records = select_lowest_image_id_per_class(records["validation"])
    _part_names, annotations = load_spatial_annotations(
        Path(data_source["dataset_root"])
    )
    train_supervision = load_part_supervision(train_records, annotations)
    validation_supervision = load_part_supervision(validation_records, annotations)
    train_part_audit = summarize_part_supervision(
        train_records, train_supervision
    )
    validation_part_audit = summarize_part_supervision(
        validation_records, validation_supervision
    )
    mask_views = [load_mask_views(record) for record in validation_records]
    validation_masks = torch.stack([value[0] for value in mask_views])
    validation_occupancies = torch.stack([value[1] for value in mask_views])
    dataset_audit = {
        "full_train_count": len(records["train"]),
        "full_validation_count": len(records["validation"]),
        "full_train_ids_sha256": ids_sha256(records["train"]),
        "full_validation_ids_sha256": ids_sha256(records["validation"]),
        "smoke_subset_rule": "lowest_image_id_per_class",
        "smoke_train_count": len(train_records),
        "smoke_validation_count": len(validation_records),
        "smoke_train_ids_sha256": ids_sha256(train_records),
        "smoke_validation_ids_sha256": ids_sha256(validation_records),
        "train_part_supervision": train_part_audit,
        "validation_part_supervision": validation_part_audit,
        "official_test_images_accessed": False,
    }
    save_json(args.output_dir / "dataset_audit.json", dataset_audit)

    teacher, teacher_audit = _load_checkpoint_model(
        args.release_dir,
        release_manifest,
        "teacher",
        training_config,
        device=device,
    )
    checkpoint_audits = [teacher_audit]
    method_results: dict[str, Any] = {}
    counts = {
        "checkpoint_strict_loads": 1,
        "linear_probe_candidates": 0,
        "nonlinear_probe_candidates": 0,
        "probe_validation_selections": 0,
        "spatial_cka_values": 0,
        "attention_metric_rows": 0,
        "test_miou_references": 0,
        "official_test_image_evaluations": 0,
    }
    full_result_methods = full_summary["final_results"]["methods"]

    for method in METHODS:
        method_started = time.monotonic()
        log(f"[CUB_SEG_SPATIAL_METHOD_START] method={method}")
        student, checkpoint_audit = _load_checkpoint_model(
            args.release_dir,
            release_manifest,
            method,
            training_config,
            device=device,
        )
        checkpoint_audits.append(checkpoint_audit)
        counts["checkpoint_strict_loads"] += 1

        train_features = extract_block11_features(
            student,
            train_records,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        validation_features = extract_block11_features(
            student,
            validation_records,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        probe_results: dict[str, Any] = {}
        for kind in PROBE_KINDS:
            probe_result, candidate_count = _run_probe_kind(
                method=method,
                kind=kind,
                train_features=train_features,
                validation_features=validation_features,
                train_supervision=train_supervision,
                validation_supervision=validation_supervision,
                protocol=config["part_probe"],
                output_dir=args.output_dir,
                device=device,
            )
            probe_results[kind] = probe_result
            counts[f"{kind}_probe_candidates"] += candidate_count
            counts["probe_validation_selections"] += 1

        cka_values = run_cka(
            student,
            teacher,
            validation_records,
            batch_size=args.cka_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        counts["spatial_cka_values"] += len(cka_values)
        attention_metrics = run_attention(
            student,
            validation_records,
            validation_masks,
            validation_occupancies,
            batch_size=args.attention_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        counts["attention_metric_rows"] += 1
        reference = full_result_methods[method]
        test_metrics = reference["test"]
        counts["test_miou_references"] += 1
        table_row = {
            "method": method,
            "part_probe_primary": "linear",
            "part_pck_at_0.1": probe_results["linear"]["validation"][
                "micro_pck_at_0.1"
            ],
            "normalized_localization_error": probe_results["linear"][
                "validation"
            ]["mean_normalized_localization_error"],
            "nonlinear_part_pck_at_0.1": probe_results["nonlinear"][
                "validation"
            ]["micro_pck_at_0.1"],
            "nonlinear_normalized_localization_error": probe_results[
                "nonlinear"
            ]["validation"]["mean_normalized_localization_error"],
            "cka_block11": cka_values[11],
            "attention_ap": attention_metrics[
                "global_micro_patch_average_precision"
            ],
            "pointing": attention_metrics["pointing_game_peak_inside_mask"],
            "foreground_mass": attention_metrics[
                "foreground_attention_mass_mean"
            ],
            "test_miou_reference": test_metrics["two_class_miou"],
        }
        method_result = {
            "table_row": table_row,
            "checkpoint": checkpoint_audit,
            "linear_probe": probe_results["linear"],
            "nonlinear_probe": probe_results["nonlinear"],
            "spatial_cka": {
                "validation_images": len(validation_records),
                "spatial_observations": len(validation_records)
                * GRID_SIZE
                * GRID_SIZE,
                "teacher_feature": "resnet50_layer3",
                "student_blocks_0_through_11": cka_values,
                "block11": cka_values[11],
            },
            "attention": attention_metrics,
            "test_miou_reference": test_metrics["two_class_miou"],
            "test_metrics_reference": test_metrics,
            "test_reference_selected_epoch": reference["selected_epoch"],
            "official_test_images_reevaluated": False,
            "elapsed_seconds": time.monotonic() - method_started,
        }
        method_results[method] = method_result
        save_json(args.output_dir / "methods" / method / "summary.json", method_result)
        log(
            "[CUB_SEG_SPATIAL_METHOD_DONE] "
            f"method={method} "
            "linear_pck="
            f"{probe_results['linear']['validation']['micro_pck_at_0.1']:.6f} "
            "nonlinear_pck="
            f"{probe_results['nonlinear']['validation']['micro_pck_at_0.1']:.6f} "
            f"cka_block11={cka_values[11]:.6f} "
            "attention_ap="
            f"{attention_metrics['global_micro_patch_average_precision']:.6f} "
            f"test_miou_reference={test_metrics['two_class_miou']:.6f}"
        )
        del student, train_features, validation_features, probe_results
        gc.collect()
        torch.cuda.empty_cache()

    expected_counts = config["completion_gate"]
    if counts != expected_counts:
        raise RuntimeError(f"smoke completion gate failed: {counts} != {expected_counts}")
    if probe_parameter_count("linear") != config["part_probe"]["linear"][
        "parameter_count"
    ] or probe_parameter_count("nonlinear") != config["part_probe"]["nonlinear"][
        "parameter_count"
    ]:
        raise RuntimeError("part-probe parameter count changed")

    summary = {
        "status": "passed",
        "protocol_id": config["protocol_id"],
        "scientific_result": False,
        "config_sha256": sha256(args.config),
        "checkpoint_release_manifest_sha256": sha256(release_manifest_path),
        "source_h200_job_id": 783,
        "encoder_seed": 1,
        "dataset": dataset_audit,
        "checkpoint_audits": checkpoint_audits,
        "methods": method_results,
        "table_rows": [method_results[method]["table_row"] for method in METHODS],
        "completion": {
            "actual": counts,
            "expected": expected_counts,
            "passed": True,
        },
        "official_test_images_accessed": False,
        "official_test_image_evaluations": 0,
        "test_miou_source": "audited_job783_full_summary_only",
        "elapsed_seconds": time.monotonic() - started,
    }
    save_json(args.output_dir / "smoke_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--cka-batch-size", type=int, default=8)
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    log(
        "[CUB_SEG_SPATIAL_SMOKE_FINAL_RESULTS] "
        + json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


if __name__ == "__main__":
    main()
