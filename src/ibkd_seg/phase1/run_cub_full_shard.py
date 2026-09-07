#!/usr/bin/env python3
"""Run one locked CUB Phase 1 classification-to-frozen-probe full shard."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .cub_data import (
    ARCHIVE_BYTES,
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    DATASET_NAME,
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
    NUM_CLASSES,
    OFFICIAL_TEST_COUNT,
    file_digest,
)
from .cub_probe_data import (
    SEGMENTATION_ARCHIVE_BYTES,
    SEGMENTATION_ARCHIVE_MD5,
    SEGMENTATION_ARCHIVE_NAME,
    SEGMENTATION_ARCHIVE_SHA256,
    CubProbeRecord,
    load_official_test_records,
    load_train_validation_records,
)
from .models import create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .run_cub_combined_smoke import (
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
    _atomic_json_save,
    _atomic_torch_save,
    _feature_cache,
    _finite_metrics,
    _load_json,
    _runtime,
    _target_cache,
)
from .train_timing import file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "phase1/phase1_cub/configs/cub200_b128_full_v1.json"
)
EXPECTED_VALIDATION_HASH = (
    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
)
ENCODER_SEEDS = (1, 2, 3)
PROBE_SEEDS = (1, 2, 3, 4, 5)
EXPECTED_SHARDS = {
    "a": ("vanilla", "lg", "ibkd_lambda_0.5"),
    "b": ("kd", "alg_warmup20", "ibkd_lambda_0.25"),
}


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-shard", action="store_true", required=True)
    parser.add_argument("--shard", choices=tuple(EXPECTED_SHARDS), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _validate_config(config: dict[str, Any]) -> None:
    classification = config.get("classification", {})
    student = classification.get("student", {})
    methods = classification.get("methods", {})
    probe = config.get("frozen_probe", {}).get("probe", {})
    split = config.get("dataset", {}).get("split", {})
    execution = config.get("execution", {})
    configured_shards = execution.get("shards", {})
    checks = {
        "protocol": config.get("protocol_id")
        == "cub200_phase1_b128_frozen_spatial_probe_full_v1",
        "locked": str(config.get("status", "")).startswith("locked_before_full_results"),
        "scientific": config.get("scientific_result") is True,
        "dataset": config.get("dataset", {}).get("name") == DATASET_NAME,
        "classes": config.get("dataset", {}).get("num_classes") == NUM_CLASSES,
        "image_archive": config.get("dataset", {}).get("image_archive")
        == {
            "url": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
            "expected_bytes": ARCHIVE_BYTES,
            "md5": ARCHIVE_MD5,
            "sha256": ARCHIVE_SHA256,
        },
        "segmentation_archive": config.get("dataset", {}).get(
            "segmentation_archive"
        )
        == {
            "url": "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1",
            "expected_bytes": SEGMENTATION_ARCHIVE_BYTES,
            "md5": SEGMENTATION_ARCHIVE_MD5,
            "sha256": SEGMENTATION_ARCHIVE_SHA256,
        },
        "split_counts": split.get("expected_counts")
        == {"train": 5394, "validation": 600, "test": 5794},
        "split_rule": split.get("validation_per_class") == 3
        and split.get("split_seed") == 2027
        and split.get("validation_image_ids_sha256") == EXPECTED_VALIDATION_HASH,
        "classification": classification.get("epochs") == 300
        and classification.get("teacher", {}).get("architecture")
        == "cifar_style_resnet56_6n_plus_2_n9"
        and student.get("architecture") == "deit_tiny_patch16_224"
        and student.get("train_batch_size") == 128
        and student.get("encoder_seeds") == [1, 2, 3],
        "variants": tuple(classification.get("variants", ())) == EXPECTED_VARIANTS,
        "alg_w20": methods.get("alg_warmup20", {}).get(
            "controller_warmup_epochs"
        )
        == 20
        and methods.get("alg_warmup20", {}).get(
            "canonical_alg_warmup0_included"
        )
        is False,
        "ibkd": methods.get("ibkd", {}).get("fusion_ratio_lambdas") == [0.25, 0.5]
        and methods.get("ibkd", {}).get("controller_warmup_epochs") == 20,
        "probe": probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 100
        and probe.get("batch_size") == 64
        and probe.get("probe_seeds") == [1, 2, 3, 4, 5],
        "shards": {
            key: tuple(value.get("variants", ())) for key, value in configured_shards.items()
        }
        == EXPECTED_SHARDS,
        "shard_cover": sorted(
            variant
            for shard_variants in EXPECTED_SHARDS.values()
            for variant in shard_variants
        )
        == sorted(EXPECTED_VARIANTS),
        "artifacts": config.get("artifacts", {}).get(
            "retain_classification_best_checkpoints"
        )
        is True
        and config.get("artifacts", {}).get("retain_selected_probe_checkpoints")
        is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid CUB full config: " + ", ".join(failures))


def _write_status(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    shard: str,
    classification_complete: int,
    probe_candidates_complete: int,
    probe_selections_complete: int,
    test_evaluations_complete: int,
    active: str | None = None,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "shard": shard,
            "classification_complete": classification_complete,
            "classification_expected": 9,
            "probe_candidates_complete": probe_candidates_complete,
            "probe_candidates_expected": 135,
            "probe_selections_complete": probe_selections_complete,
            "probe_selections_expected": 45,
            "official_test_evaluations_complete": test_evaluations_complete,
            "official_test_evaluations_expected": 45,
            "active": active,
            "failure": failure,
        },
        output_dir / "sequence_status.json",
    )


def _run_or_resume(command: list[str], summary_path: Path, *, label: str) -> dict[str, Any]:
    if summary_path.is_file():
        payload = _load_json(summary_path)
        if payload.get("status") == "complete":
            log(f"[CUB_FULL_RESUME] {label} summary={summary_path}")
            return payload
    log(f"[CUB_FULL_TASK_START] {label} command={' '.join(command)}")
    subprocess.run(command, check=True)
    payload = _load_json(summary_path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"CUB full subprocess produced incomplete summary: {label}")
    log(f"[CUB_FULL_TASK_DONE] {label}")
    return payload


def _variant_run_name(variant: str, seed: int) -> str:
    return f"cub_{variant}_deit_tiny_b128_full_300ep_seed{seed}"


def _classification_rows(
    args: argparse.Namespace,
    *,
    variants: Sequence[str],
    validation_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classification_root = args.output_dir / "classification"
    teacher_root = classification_root / "teacher"
    teacher_name = "cub_teacher_resnet56_32_b128_full_300ep_seed1"
    teacher_dir = teacher_root / teacher_name
    teacher_checkpoint = teacher_dir / "teacher_best_validation.pt"
    teacher_command = [
        sys.executable,
        "-m",
        "ibkd_seg.phase1.train_full",
        "--full-run",
        "--dataset",
        "cub",
        "--kind",
        "teacher",
        "--batch-size",
        "128",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(teacher_root),
        "--run-name",
        teacher_name,
        "--num-workers",
        str(args.num_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--seed",
        "1",
    ]
    teacher = _run_or_resume(
        teacher_command,
        teacher_dir / "summary.json",
        label="teacher_resnet56_seed1",
    )
    if (
        teacher.get("dataset") != DATASET_NAME
        or teacher.get("kind") != "teacher"
        or teacher.get("scientific_result") is not True
        or teacher.get("split_manifest", {}).get("validation_image_ids_sha256")
        != validation_hash
        or teacher.get("official_test_evaluations") != 1
        or not teacher_checkpoint.is_file()
        or teacher.get("checkpoint_sha256") != file_sha256(teacher_checkpoint)
    ):
        raise RuntimeError("CUB full teacher contract failed")

    student_root = classification_root / "students"
    rows: list[dict[str, Any]] = []
    for variant in variants:
        method, fusion_ratio, controller_warmup = VARIANT_ARGUMENTS[variant]
        for seed in ENCODER_SEEDS:
            run_name = _variant_run_name(variant, seed)
            run_dir = student_root / run_name
            checkpoint_path = run_dir / "student_best_validation.pt"
            command = [
                sys.executable,
                "-m",
                "ibkd_seg.phase1.train_full",
                "--full-run",
                "--dataset",
                "cub",
                "--kind",
                "student",
                "--method",
                method,
                "--batch-size",
                "128",
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(student_root),
                "--run-name",
                run_name,
                "--num-workers",
                str(args.num_workers),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--seed",
                str(seed),
            ]
            if method != "vanilla":
                command.extend(["--teacher-checkpoint", str(teacher_checkpoint)])
            if fusion_ratio is not None:
                command.extend(["--fusion-ratio", str(fusion_ratio)])
            if method == "alg":
                command.extend(["--alg-controller-warmup-epochs", "20"])
            summary = _run_or_resume(
                command,
                run_dir / "summary.json",
                label=f"classification_{variant}_seed{seed}",
            )
            expected_warmup = controller_warmup
            if (
                summary.get("dataset") != DATASET_NAME
                or summary.get("kind") != "student"
                or summary.get("method") != method
                or summary.get("fusion_ratio_lambda") != fusion_ratio
                or summary.get("batch_size") != 128
                or summary.get("seed") != seed
                or summary.get("epochs") != 300
                or summary.get("scientific_result") is not True
                or summary.get("confirmatory_main_result") is not True
                or summary.get("guidance_controller_warmup_epochs") != expected_warmup
                or summary.get("split_manifest", {}).get(
                    "validation_image_ids_sha256"
                )
                != validation_hash
                or summary.get("official_test_evaluations") != 1
                or summary.get("official_test_used_for_training_or_selection") is not False
                or not checkpoint_path.is_file()
                or summary.get("checkpoint_sha256") != file_sha256(checkpoint_path)
            ):
                raise RuntimeError(
                    f"CUB full classification contract failed: {variant}/seed{seed}"
                )
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "fusion_ratio_lambda": fusion_ratio,
                    "controller_warmup_epochs": controller_warmup,
                    "encoder_seed": seed,
                    "summary_path": str((run_dir / "summary.json").resolve()),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "summary": summary,
                }
            )
            _write_status(
                args.output_dir,
                status="running",
                phase="classification_students",
                shard=args.shard,
                classification_complete=len(rows),
                probe_candidates_complete=0,
                probe_selections_complete=0,
                test_evaluations_complete=0,
                active=f"{variant}/encoder_seed{seed}",
            )
    return teacher, rows


def _load_full_encoder(
    row: dict[str, Any],
    *,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(row["checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": "phase1_scientific_full_student",
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": row["method"],
        "fusion_ratio_lambda": row["fusion_ratio_lambda"],
        "batch_size": 128,
        "epochs": 300,
        "seed": row["encoder_seed"],
        "validation_image_ids_sha256": validation_hash,
        "guidance_controller_warmup_epochs": row["controller_warmup_epochs"],
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"CUB full encoder mismatch {row['variant']}/seed{row['encoder_seed']}:"
                f"{key} expected={value!r} got={metadata.get(key)!r}"
            )
    if row["summary"].get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise RuntimeError("CUB full encoder checkpoint SHA-256 changed")
    model = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("CUB full encoder strict load returned incompatible keys")
    state_hash = state_dict_sha256(model)
    if state_hash != metadata.get("student_state_sha256"):
        raise RuntimeError("CUB full encoder state SHA-256 changed")
    model.to(device).eval().requires_grad_(False)
    audit = {
        "strict_load": True,
        "eval_mode": not model.training,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "student_state_sha256": state_hash,
    }
    if audit["trainable_parameter_count"] != 0 or not audit["eval_mode"]:
        raise RuntimeError("CUB full encoder freeze contract failed")
    return model, audit


def _archive_audit(data_dir: Path) -> dict[str, Any]:
    identities = (
        (
            "image",
            data_dir / ARCHIVE_NAME,
            ARCHIVE_BYTES,
            ARCHIVE_MD5,
            ARCHIVE_SHA256,
        ),
        (
            "segmentation",
            data_dir / SEGMENTATION_ARCHIVE_NAME,
            SEGMENTATION_ARCHIVE_BYTES,
            SEGMENTATION_ARCHIVE_MD5,
            SEGMENTATION_ARCHIVE_SHA256,
        ),
    )
    result: dict[str, Any] = {}
    for name, path, expected_bytes, expected_md5, expected_sha256 in identities:
        if not path.is_file():
            raise RuntimeError(f"CUB {name} archive missing after download: {path}")
        actual = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "md5": file_digest(path, "md5"),
            "sha256": file_digest(path),
        }
        if actual["bytes"] != expected_bytes:
            raise RuntimeError(f"CUB {name} archive byte-size mismatch")
        if actual["md5"] != expected_md5 or actual["sha256"] != expected_sha256:
            raise RuntimeError(f"CUB {name} archive digest mismatch")
        result[name] = actual
    return result


def _write_csv(rows: Sequence[dict[str, Any]], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _classification_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    flattened = []
    for row in rows:
        summary = row["summary"]
        flattened.append(
            {
                "variant": row["variant"],
                "method": row["method"],
                "fusion_ratio_lambda": row["fusion_ratio_lambda"],
                "controller_warmup_epochs": row["controller_warmup_epochs"],
                "encoder_seed": row["encoder_seed"],
                "selected_epoch": summary["selected_epoch"],
                "validation_macro_top1": summary["selected_validation"]["macro_top1"],
                "test_macro_top1": summary["official_test"]["macro_top1"],
                "test_overall_top1": summary["official_test"]["overall_top1"],
                "test_top5": summary["official_test"]["top5"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
            }
        )
    _write_csv(
        flattened,
        path,
        (
            "variant",
            "method",
            "fusion_ratio_lambda",
            "controller_warmup_epochs",
            "encoder_seed",
            "selected_epoch",
            "validation_macro_top1",
            "test_macro_top1",
            "test_overall_top1",
            "test_top5",
            "checkpoint_path",
            "checkpoint_sha256",
        ),
    )


def _select_probe_candidates(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    config_sha256: str,
    records: dict[str, list[CubProbeRecord]],
    targets: dict[str, dict[str, Any]],
    classification_rows: Sequence[dict[str, Any]],
    validation_hash: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probe_config = config["frozen_probe"]["probe"]
    learning_rates = [float(value) for value in probe_config["learning_rates"]]
    probe_seeds = [int(value) for value in probe_config["probe_seeds"]]
    probe_epochs = int(probe_config["epochs"])
    selections: list[dict[str, Any]] = []
    initial_hashes: dict[int, set[str]] = defaultdict(set)
    batch_orders: dict[tuple[int, int], set[str]] = defaultdict(set)
    encoder_audits: list[dict[str, Any]] = []

    log(
        "[CUB_FULL_PROBE_SELECTION_COUNT] encoders=9 probe_seeds=5 "
        "lr_candidates=135 selections=45 epochs_per_candidate=100"
    )
    for classification in classification_rows:
        variant = classification["variant"]
        encoder_seed = int(classification["encoder_seed"])
        model, encoder_audit = _load_full_encoder(
            classification,
            validation_hash=validation_hash,
            device=device,
        )
        encoder_audits.append(encoder_audit)
        features: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "validation"):
            features[split_name], _ = _feature_cache(
                model,
                records[split_name],
                split=split_name,
                variant=variant,
                encoder_seed=encoder_seed,
                checkpoint_sha256=encoder_audit["checkpoint_sha256"],
                state_sha256=encoder_audit["student_state_sha256"],
                config=config,
                config_sha256=config_sha256,
                cache_path=args.cache_dir
                / "features"
                / variant
                / f"encoder_seed{encoder_seed}"
                / f"{split_name}.pt",
                device=device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for probe_seed in probe_seeds:
            candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
            for learning_rate in learning_rates:
                state, candidate = train_candidate(
                    features["train"]["features"],
                    targets["train"]["grid_targets"],
                    features["validation"]["features"],
                    targets["validation"]["grid_targets"],
                    probe_config=probe_config,
                    learning_rate=learning_rate,
                    seed=probe_seed,
                    device=device,
                    epochs=probe_epochs,
                )
                candidates.append((state, candidate))
                initial_hashes[probe_seed].add(candidate["initial_probe_state_sha256"])
                for epoch, digest in enumerate(
                    candidate["batch_order_sha256_by_epoch"], start=1
                ):
                    batch_orders[(probe_seed, epoch)].add(digest)
                completed = len(selections) * len(learning_rates) + len(candidates)
                log(
                    f"[CUB_FULL_PROBE_CANDIDATE] variant={variant} "
                    f"encoder_seed={encoder_seed} probe_seed={probe_seed} "
                    f"lr={learning_rate:g} best_epoch={candidate['best_epoch']} "
                    "validation_grid_miou="
                    f"{candidate['best_validation_grid_mean_iou']:.6f} "
                    f"completed_at_least={completed}/135"
                )
            selected_state, selected = min(
                candidates,
                key=lambda item: (
                    -float(item[1]["best_validation_grid_mean_iou"]),
                    float(item[1]["learning_rate"]),
                    int(item[1]["best_epoch"]),
                ),
            )
            probe = probe_from_state(probe_config, probe_seed, selected_state, device)
            validation_metrics, _ = evaluate_probe_both_resolutions(
                probe,
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                targets["validation"]["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=int(config["frozen_probe"]["image_input"]["size"]),
                ignore_index=int(probe_config["loss"]["ignore_index"]),
            )
            if not all(_finite_metrics(metrics) for metrics in validation_metrics.values()):
                raise RuntimeError("non-finite CUB full validation probe metrics")
            probe_path = (
                args.output_dir
                / "probe"
                / "checkpoints"
                / variant
                / f"encoder_seed{encoder_seed}"
                / f"probe_seed{probe_seed}_best_validation.pt"
            )
            _atomic_torch_save(
                {
                    "metadata": {
                        "purpose": "phase1_cub_scientific_frozen_probe",
                        "scientific_result": True,
                        "dataset": DATASET_NAME,
                        "config_sha256": config_sha256,
                        "shard": args.shard,
                        "variant": variant,
                        "encoder_seed": encoder_seed,
                        "encoder_checkpoint_sha256": encoder_audit[
                            "checkpoint_sha256"
                        ],
                        "probe_seed": probe_seed,
                        "selection_split": "validation",
                        "selected_learning_rate": selected["learning_rate"],
                        "selected_epoch": selected["best_epoch"],
                        "validation_grid_mean_iou": selected[
                            "best_validation_grid_mean_iou"
                        ],
                        "official_test_evaluations_at_checkpoint_write": 0,
                    },
                    "model": selected_state,
                },
                probe_path,
            )
            saved = torch.load(probe_path, map_location="cpu", weights_only=True)
            strict_probe = probe_from_state(
                probe_config, probe_seed, saved["model"], device
            )
            del strict_probe, probe
            selections.append(
                {
                    "variant": variant,
                    "method": classification["method"],
                    "fusion_ratio_lambda": classification["fusion_ratio_lambda"],
                    "controller_warmup_epochs": classification[
                        "controller_warmup_epochs"
                    ],
                    "encoder_seed": encoder_seed,
                    "encoder_checkpoint_path": classification["checkpoint_path"],
                    "encoder_checkpoint_sha256": encoder_audit["checkpoint_sha256"],
                    "encoder_state_sha256": encoder_audit["student_state_sha256"],
                    "probe_seed": probe_seed,
                    "candidates": [candidate for _, candidate in candidates],
                    "selected_learning_rate": selected["learning_rate"],
                    "selected_epoch": selected["best_epoch"],
                    "validation": validation_metrics,
                    "probe_checkpoint_path": str(probe_path.resolve()),
                    "probe_checkpoint_sha256": file_sha256(probe_path),
                    "selected_probe_strict_reloaded": True,
                    "official_test_evaluations": 0,
                }
            )
            _atomic_json_save(
                {
                    "status": "selection_in_progress",
                    "shard": args.shard,
                    "completed": len(selections),
                    "expected": 45,
                    "official_test_masks_accessed": False,
                    "selections": selections,
                },
                args.output_dir / "probe" / "validation_selections.json",
            )
            _write_status(
                args.output_dir,
                status="running",
                phase="probe_validation_selection",
                shard=args.shard,
                classification_complete=9,
                probe_candidates_complete=len(selections) * 3,
                probe_selections_complete=len(selections),
                test_evaluations_complete=0,
                active=f"{variant}/encoder_seed{encoder_seed}/probe_seed{probe_seed}",
            )
            del candidates
        del features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contracts = {
        "selection_count": len(selections) == 45,
        "same_initial_probe_state_per_probe_seed": all(
            len(initial_hashes[seed]) == 1 for seed in PROBE_SEEDS
        ),
        "same_probe_batch_order_per_seed_and_epoch": all(
            len(values) == 1 for values in batch_orders.values()
        ),
        "all_encoders_strict_loaded_frozen_eval": len(encoder_audits) == 9
        and all(
            audit["strict_load"]
            and audit["eval_mode"]
            and audit["trainable_parameter_count"] == 0
            for audit in encoder_audits
        ),
        "all_selected_probes_strict_reloaded": all(
            row["selected_probe_strict_reloaded"] for row in selections
        ),
        "no_official_test_masks_accessed_before_all_selections": True,
    }
    if not all(contracts.values()):
        raise RuntimeError("CUB full probe selection contract failed")
    return selections, contracts


def _evaluate_probe_test(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    config_sha256: str,
    test_records: Sequence[CubProbeRecord],
    test_targets: dict[str, Any],
    classification_rows: Sequence[dict[str, Any]],
    selections: list[dict[str, Any]],
    validation_hash: str,
    device: torch.device,
) -> None:
    probe_config = config["frozen_probe"]["probe"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[(row["variant"], int(row["encoder_seed"]))].append(row)
    by_encoder = {
        (row["variant"], int(row["encoder_seed"])): row
        for row in classification_rows
    }
    for key, selection_rows in grouped.items():
        classification = by_encoder[key]
        model, encoder_audit = _load_full_encoder(
            classification,
            validation_hash=validation_hash,
            device=device,
        )
        test_features, _ = _feature_cache(
            model,
            test_records,
            split="official_test",
            variant=classification["variant"],
            encoder_seed=int(classification["encoder_seed"]),
            checkpoint_sha256=encoder_audit["checkpoint_sha256"],
            state_sha256=encoder_audit["student_state_sha256"],
            config=config,
            config_sha256=config_sha256,
            cache_path=args.cache_dir
            / "features"
            / classification["variant"]
            / f"encoder_seed{classification['encoder_seed']}"
            / "official_test.pt",
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        del model
        for row in sorted(selection_rows, key=lambda item: int(item["probe_seed"])):
            saved = torch.load(
                row["probe_checkpoint_path"], map_location="cpu", weights_only=True
            )
            metadata = saved.get("metadata", {})
            expected = {
                "config_sha256": config_sha256,
                "shard": args.shard,
                "variant": row["variant"],
                "encoder_seed": row["encoder_seed"],
                "probe_seed": row["probe_seed"],
                "official_test_evaluations_at_checkpoint_write": 0,
            }
            if any(metadata.get(key_name) != value for key_name, value in expected.items()):
                raise RuntimeError("CUB selected probe metadata mismatch before test")
            probe = probe_from_state(
                probe_config, int(row["probe_seed"]), saved["model"], device
            )
            metrics, _ = evaluate_probe_both_resolutions(
                probe,
                test_features["features"],
                test_targets["grid_targets"],
                test_targets["input_targets"],
                batch_size=int(probe_config["batch_size"]),
                device=device,
                input_size=int(config["frozen_probe"]["image_input"]["size"]),
                ignore_index=int(probe_config["loss"]["ignore_index"]),
            )
            if not all(_finite_metrics(item) for item in metrics.values()):
                raise RuntimeError("non-finite CUB full official-test probe metrics")
            row["official_test"] = metrics
            row["official_test_evaluations"] = 1
            log(
                f"[CUB_FULL_PROBE_TEST] variant={row['variant']} "
                f"encoder_seed={row['encoder_seed']} probe_seed={row['probe_seed']} "
                f"selected_lr={row['selected_learning_rate']:g} "
                f"selected_epoch={row['selected_epoch']} "
                "test_input_miou="
                f"{metrics['input_224']['mean_iou']:.6f}"
            )
            del probe
            _write_status(
                args.output_dir,
                status="running",
                phase="probe_official_test",
                shard=args.shard,
                classification_complete=9,
                probe_candidates_complete=135,
                probe_selections_complete=45,
                test_evaluations_complete=sum(
                    int(item["official_test_evaluations"]) for item in selections
                ),
                active=(
                    f"{row['variant']}/encoder_seed{row['encoder_seed']}"
                    f"/probe_seed{row['probe_seed']}"
                ),
            )
        del test_features
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _probe_flat_rows(selections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in selections:
        validation = item["validation"]
        test = item["official_test"]
        rows.append(
            {
                "variant": item["variant"],
                "method": item["method"],
                "fusion_ratio_lambda": item["fusion_ratio_lambda"],
                "controller_warmup_epochs": item["controller_warmup_epochs"],
                "encoder_seed": item["encoder_seed"],
                "probe_seed": item["probe_seed"],
                "selected_learning_rate": item["selected_learning_rate"],
                "selected_epoch": item["selected_epoch"],
                "validation_grid_mean_iou": validation["grid_14x14"]["mean_iou"],
                "validation_input_224_mean_iou": validation["input_224"]["mean_iou"],
                "test_grid_mean_iou": test["grid_14x14"]["mean_iou"],
                "test_input_224_mean_iou": test["input_224"]["mean_iou"],
                "test_input_224_foreground_iou": test["input_224"]["foreground_iou"],
                "test_input_224_background_iou": test["input_224"]["background_iou"],
                "test_input_224_foreground_dice": test["input_224"]["foreground_dice"],
                "test_input_224_pixel_accuracy": test["input_224"]["pixel_accuracy"],
                "encoder_checkpoint_path": item["encoder_checkpoint_path"],
                "encoder_checkpoint_sha256": item["encoder_checkpoint_sha256"],
                "probe_checkpoint_path": item["probe_checkpoint_path"],
                "probe_checkpoint_sha256": item["probe_checkpoint_sha256"],
                "official_test_evaluations": item["official_test_evaluations"],
                "scientific_result": True,
            }
        )
    return rows


def _aggregate_probe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant_encoder: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_variant_encoder[(row["variant"], int(row["encoder_seed"]))].append(
            float(row["test_input_224_mean_iou"])
        )
    by_variant: dict[str, dict[int, float]] = defaultdict(dict)
    for (variant, encoder_seed), values in by_variant_encoder.items():
        if len(values) != 5:
            raise RuntimeError("expected five probe seeds per CUB encoder")
        by_variant[variant][encoder_seed] = statistics.mean(values)
    aggregates = []
    for variant in sorted(by_variant):
        encoder_means = by_variant[variant]
        values = [encoder_means[seed] for seed in ENCODER_SEEDS]
        aggregates.append(
            {
                "variant": variant,
                "encoder_seed_means": {
                    str(seed): encoder_means[seed] for seed in ENCODER_SEEDS
                },
                "mean_over_encoder_seed_means": statistics.mean(values),
                "sample_standard_deviation_over_encoder_seed_means": statistics.stdev(
                    values
                ),
                "independent_n": 3,
                "probe_seeds_per_encoder": 5,
            }
        )
    return aggregates


def _checkpoint_manifest(
    teacher: dict[str, Any],
    classification_rows: Sequence[dict[str, Any]],
    probe_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    entries = [
        {
            "kind": "teacher",
            "path": teacher["checkpoint"],
            "sha256": teacher["checkpoint_sha256"],
        }
    ]
    entries.extend(
        {
            "kind": "classification_encoder",
            "variant": row["variant"],
            "encoder_seed": row["encoder_seed"],
            "path": row["checkpoint_path"],
            "sha256": row["summary"]["checkpoint_sha256"],
        }
        for row in classification_rows
    )
    entries.extend(
        {
            "kind": "selected_probe",
            "variant": row["variant"],
            "encoder_seed": row["encoder_seed"],
            "probe_seed": row["probe_seed"],
            "path": row["probe_checkpoint_path"],
            "sha256": row["probe_checkpoint_sha256"],
        }
        for row in probe_rows
    )
    return {"count": len(entries), "entries": entries}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUB full shard requires CUDA")
    if args.feature_batch_size <= 0 or args.eval_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid CUB full loader setting")
    device = torch.device("cuda")
    config = _load_json(args.config)
    _validate_config(config)
    config_sha256 = file_sha256(args.config)
    variants = EXPECTED_SHARDS[args.shard]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_status(
        args.output_dir,
        status="running",
        phase="dataset_and_mask_setup",
        shard=args.shard,
        classification_complete=0,
        probe_candidates_complete=0,
        probe_selections_complete=0,
        test_evaluations_complete=0,
    )
    log("=" * 104)
    log(
        f"CUB-200-2011 PHASE 1 — FULL CLASSIFICATION TO FROZEN PROBE SHARD {args.shard.upper()}"
    )
    log("=" * 104)
    log(
        f"[CUB_FULL_POLICY] protocol_sha256={config_sha256} shard={args.shard} "
        f"variants={','.join(variants)} scientific_result=true"
    )

    records, split_manifest, source = load_train_validation_records(
        args.data_dir, download=True
    )
    validation_hash = split_manifest["validation_image_ids_sha256"]
    if (
        len(records["train"]) != DERIVED_TRAIN_COUNT
        or len(records["validation"]) != DERIVED_VALIDATION_COUNT
        or validation_hash != EXPECTED_VALIDATION_HASH
    ):
        raise RuntimeError("CUB full train/validation split contract failed")
    archives = _archive_audit(args.data_dir)
    data_audit = {
        "status": "train_validation_pass_test_sealed",
        "source": source,
        "archives": archives,
        "split_manifest": split_manifest,
        "counts": {"train": 5394, "validation": 600, "test": 5794},
        "train_validation_mask_pairs_present": 5994,
        "official_test_masks_decoded": False,
    }
    _atomic_json_save(data_audit, args.output_dir / "dataset_audit.json")

    teacher, classification_rows = _classification_rows(
        args, variants=variants, validation_hash=validation_hash
    )
    if len(classification_rows) != 9:
        raise RuntimeError("CUB full shard classification count changed")
    initial_by_seed = {
        seed: {
            row["summary"]["initial_student_state_sha256"]
            for row in classification_rows
            if row["encoder_seed"] == seed
        }
        for seed in ENCODER_SEEDS
    }
    if not all(len(values) == 1 for values in initial_by_seed.values()):
        raise RuntimeError("CUB full shard paired student initialization failed")
    if {
        row["summary"].get("teacher_model_state_sha256")
        for row in classification_rows
        if row["method"] != "vanilla"
    } != {teacher["model_state_sha256"]}:
        raise RuntimeError("CUB full shard shared teacher contract failed")
    _classification_csv(
        classification_rows, args.output_dir / "classification_results.csv"
    )

    targets = {}
    for split_name in ("train", "validation"):
        targets[split_name], _ = _target_cache(
            records[split_name],
            split=split_name,
            config=config,
            config_sha256=config_sha256,
            cache_path=args.cache_dir / "targets" / f"{split_name}.pt",
        )
    target_values = {
        int(value)
        for payload in targets.values()
        for key in ("input_targets", "grid_targets")
        for value in torch.unique(payload[key])
    }
    if target_values != {0, 1}:
        raise RuntimeError(f"CUB full binary target values changed: {target_values}")

    selections, selection_contracts = _select_probe_candidates(
        args,
        config=config,
        config_sha256=config_sha256,
        records=records,
        targets=targets,
        classification_rows=classification_rows,
        validation_hash=validation_hash,
        device=device,
    )
    selection_marker = {
        "status": "complete",
        "shard": args.shard,
        "completed_probe_selections": len(selections),
        "expected_probe_selections": 45,
        "official_test_masks_accessed": False,
        "selection_contracts": selection_contracts,
    }
    _atomic_json_save(
        selection_marker,
        args.output_dir / "probe" / "selection_complete_before_test.json",
    )
    log(
        "[CUB_FULL_SELECTION_COMPLETE] selections=45/45 "
        "official_test_masks_accessed=false opening_test=true"
    )

    test_records, test_source = load_official_test_records(args.data_dir, download=True)
    if len(test_records) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("CUB full official-test probe count changed")
    test_targets, _ = _target_cache(
        test_records,
        split="official_test",
        config=config,
        config_sha256=config_sha256,
        cache_path=args.cache_dir / "targets" / "official_test.pt",
    )
    _evaluate_probe_test(
        args,
        config=config,
        config_sha256=config_sha256,
        test_records=test_records,
        test_targets=test_targets,
        classification_rows=classification_rows,
        selections=selections,
        validation_hash=validation_hash,
        device=device,
    )
    if len(selections) != 45 or any(
        row["official_test_evaluations"] != 1 for row in selections
    ):
        raise RuntimeError("CUB full probe test-once contract failed")

    flat_probe_rows = _probe_flat_rows(selections)
    probe_fields = tuple(flat_probe_rows[0].keys())
    _write_csv(
        flat_probe_rows,
        args.output_dir / "probe" / "raw_results.csv",
        probe_fields,
    )
    aggregates = _aggregate_probe(flat_probe_rows)
    _atomic_json_save(
        {
            "status": "complete",
            "shard": args.shard,
            "scientific_result": True,
            "rows": flat_probe_rows,
            "aggregates": aggregates,
        },
        args.output_dir / "probe" / "results.json",
    )
    qualitative_ids = config["frozen_probe"]["qualitative"][
        "official_test_image_ids"
    ]
    test_by_id = {record.image_id: record for record in test_records}
    if any(image_id not in test_by_id for image_id in qualitative_ids):
        raise RuntimeError("preselected CUB qualitative ID is not in official test")
    _atomic_json_save(
        {
            "selection_before_results": True,
            "image_ids": qualitative_ids,
            "records": [
                {
                    "image_id": image_id,
                    "relative_path": test_by_id[image_id].relative_path,
                }
                for image_id in qualitative_ids
            ],
            "encoder_seed": config["frozen_probe"]["qualitative"][
                "encoder_seed"
            ],
            "probe_seed": config["frozen_probe"]["qualitative"]["probe_seed"],
            "predictions_can_be_rendered_from_retained_checkpoints": True,
        },
        args.output_dir / "probe" / "qualitative_manifest.json",
    )
    data_audit.update(
        {
            "status": "complete",
            "test_source": test_source,
            "official_test_mask_pairs_present": len(test_records),
            "all_image_mask_pairs_present": 11788,
            "official_test_masks_decoded": True,
            "official_test_masks_first_decoded_after_selection_marker": True,
        }
    )
    _atomic_json_save(data_audit, args.output_dir / "dataset_audit.json")

    manifest = _checkpoint_manifest(teacher, classification_rows, flat_probe_rows)
    if manifest["count"] != 55:
        raise RuntimeError("CUB full shard checkpoint manifest count changed")
    _atomic_json_save(manifest, args.output_dir / "checkpoint_manifest.json")
    contracts = {
        "config_locked": True,
        "archive_bytes_md5_sha256_match": True,
        "fixed_5394_600_5794_split": True,
        "validation_image_ids_sha256_match": validation_hash
        == EXPECTED_VALIDATION_HASH,
        "classification_3variants_x3seeds_complete": len(classification_rows) == 9,
        "same_initial_student_state_within_each_seed": all(
            len(values) == 1 for values in initial_by_seed.values()
        ),
        "one_teacher_shared_by_guided_students": True,
        **selection_contracts,
        "probe_135_lr_candidates_complete": len(selections) * 3 == 135,
        "probe_45_selections_complete": len(selections) == 45,
        "official_test_evaluated_once_per_selected_probe": all(
            row["official_test_evaluations"] == 1 for row in selections
        ),
        "classification_and_probe_checkpoints_retained": manifest["count"] == 55,
    }
    if not all(contracts.values()):
        raise RuntimeError("CUB full shard final contract failed")
    elapsed = time.monotonic() - started
    summary = {
        "status": "complete",
        "scientific_result": True,
        "protocol_id": config["protocol_id"],
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "shard": args.shard,
        "variants": list(variants),
        "counts": {
            "teacher": 1,
            "classification_students": 9,
            "probe_lr_candidates": 135,
            "selected_probes": 45,
            "probe_official_test_evaluations": 45,
            "retained_checkpoints": 55,
        },
        "teacher": {
            "checkpoint": teacher["checkpoint"],
            "checkpoint_sha256": teacher["checkpoint_sha256"],
            "model_state_sha256": teacher["model_state_sha256"],
        },
        "classification_rows": [
            {key: value for key, value in row.items() if key != "summary"}
            for row in classification_rows
        ],
        "probe_aggregates": aggregates,
        "contracts": {"all_passed": True, **contracts},
        "elapsed_seconds": elapsed,
        "runtime": _runtime(device),
        "checkpoint_manifest": str(
            (args.output_dir / "checkpoint_manifest.json").resolve()
        ),
    }
    _atomic_json_save(summary, args.output_dir / "combined_full_summary.json")
    _write_status(
        args.output_dir,
        status="complete",
        phase="complete",
        shard=args.shard,
        classification_complete=9,
        probe_candidates_complete=135,
        probe_selections_complete=45,
        test_evaluations_complete=45,
    )
    log("[CUB_FULL_FINAL_CLASSIFICATION_RESULTS]")
    for row in classification_rows:
        result = row["summary"]
        log(
            f"[CUB_FULL_CLASSIFICATION_RESULT] variant={row['variant']} "
            f"encoder_seed={row['encoder_seed']} selected_epoch={result['selected_epoch']} "
            f"test_macro_top1={result['official_test']['macro_top1']:.4f}"
        )
    log("[CUB_FULL_FINAL_PROBE_RESULTS]")
    for aggregate in aggregates:
        log(
            f"[CUB_FULL_PROBE_AGGREGATE] variant={aggregate['variant']} "
            "test_input_miou_mean="
            f"{aggregate['mean_over_encoder_seed_means']:.6f} "
            "test_input_miou_encoder_seed_sd="
            f"{aggregate['sample_standard_deviation_over_encoder_seed_means']:.6f} "
            "independent_n=3 probe_seeds_per_encoder=5"
        )
    log(
        f"[CUB_FULL_SHARD_DONE] status=complete shard={args.shard} "
        "classification=9/9 probe_candidates=135/135 selections=45/45 "
        f"test_once=45/45 checkpoints=55 elapsed={format_duration(elapsed)} "
        f"summary={(args.output_dir / 'combined_full_summary.json').resolve()}"
    )
    return summary


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_save(
            {
                "status": "failed",
                "scientific_result": False,
                "shard": args.shard,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            args.output_dir / "failure.json",
        )
        log(
            f"[CUB_FULL_SHARD_FAILED] shard={args.shard} "
            f"error={type(error).__name__}:{error}"
        )
        raise


if __name__ == "__main__":
    main()
