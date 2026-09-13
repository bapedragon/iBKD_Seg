#!/usr/bin/env python3
"""Run one locked, validation-only CUB loader-pilot full shard.

Each invocation executes exactly one of L0/L1/L2.  All three independently
completed shards are required before loader selection; the official CUB test
split is never constructed by this runner.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .cub_data import DATASET_NAME, NUM_CLASSES, resolve_dataset_root
from .cub_direct_spatial import (
    build_part_probe,
    load_part_supervision,
    load_spatial_annotations,
    select_lowest_image_id_per_class,
    summarize_part_supervision,
)
from .cub_loader_profiles import LOADER_PROFILE_ORDER, loader_profile_contract
from .cub_probe_data import CubProbeRecord, load_train_validation_records
from .models import create_student
from .probe import evaluate_probe_both_resolutions, probe_from_state, train_candidate
from .run_cub_combined_smoke import (
    _atomic_json_save,
    _atomic_torch_save,
    _feature_cache,
    _finite_metrics,
    _runtime,
    _target_cache,
)
from .run_cub_direct_spatial_full import _run_part_validation
from .run_cub_direct_spatial_smoke import (
    _attention_smoke,
    _cka_smoke,
    _save_cka_heatmap,
    _write_csv,
)
from .run_cub_loader_pilot_smoke import (
    EXPECTED_TEACHER_SHA256,
    EXPECTED_TEACHER_STATE_SHA256,
    EXPECTED_VALIDATION_SHA256,
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
)
from .run_cub_r50_batch_profile_full import _archive_audit
from .run_cub_r50_teacher_full import load_scientific_teacher
from .train_timing import file_sha256, format_duration, state_dict_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/image_loader_experiment/configs/"
    "cub200_r50_224_b128_seed1_loader_pilot_full_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "97adb9274a4f1a996932915dbb8a715a719b2ae6777aceb40201e96174cbe5a4"
)
EXPECTED_PROTOCOL_ID = "cub200_phase1_r50_224_b128_seed1_loader_pilot_full_v1"


def log(message: str = "") -> None:
    print(message, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _validate_provenance(config: dict[str, Any]) -> None:
    provenance = config["protocol_provenance"]
    sources = (
        (provenance["base_v3"], "path", "sha256"),
        (provenance["loader_damage_audit_v1"], "config_path", "config_sha256"),
        (provenance["loader_damage_audit_v1"], "result_path", "result_sha256"),
        (provenance["loader_pilot_smoke_v2"], "config_path", "config_sha256"),
        (provenance["loader_pilot_smoke_v2"], "result_path", "result_sha256"),
    )
    for source, path_key, hash_key in sources:
        path = _repository_path(source[path_key])
        if not path.is_file() or file_sha256(path) != source[hash_key]:
            raise RuntimeError(f"loader-pilot provenance changed: {path_key}")
    teacher = config["teacher"]
    release = _repository_path(teacher["release_manifest"])
    if (
        not release.is_file()
        or file_sha256(release) != teacher["release_manifest_sha256"]
    ):
        raise RuntimeError("audited issue-722 teacher release manifest changed")


def _validate_config(config: dict[str, Any], path: Path) -> None:
    scope = config.get("scope", {})
    classification = config.get("classification", {})
    probe = config.get("frozen_probe", {}).get("probe", {})
    direct = config.get("direct_spatial", {})
    execution = config.get("execution", {})
    checks = {
        "config_hash": file_sha256(path) == EXPECTED_CONFIG_SHA256,
        "protocol_id": config.get("protocol_id") == EXPECTED_PROTOCOL_ID,
        "locked": config.get("status")
        == "locked_after_loader_pilot_smoke_v2_pass_before_full_results_2026-09-12",
        "scientific_exploratory": config.get("scientific_result") is True
        and config.get("exploratory_loader_pilot") is True
        and config.get("confirmatory_main_result") is False,
        "test_closed": config.get("official_test_accessed") is False,
        "smoke_selection_forbidden": config.get(
            "selection_from_smoke_metrics_forbidden"
        )
        is True,
        "scope": tuple(scope.get("loader_profiles", ())) == LOADER_PROFILE_ORDER
        and tuple(scope.get("variants", ())) == EXPECTED_VARIANTS
        and scope.get("student_batch_size") == 128
        and scope.get("encoder_seeds") == [1]
        and scope.get("one_loader_profile_per_h200_job") is True
        and scope.get("all_three_shards_required_before_loader_selection") is True,
        "dataset": config.get("dataset", {}).get("name") == DATASET_NAME
        and config.get("dataset", {}).get("num_classes") == NUM_CLASSES
        and config.get("dataset", {}).get("split")
        == {
            "train": 5394,
            "validation": 600,
            "official_test_loaded": 0,
            "validation_per_class": 3,
            "split_seed": 2027,
            "validation_image_ids_sha256": EXPECTED_VALIDATION_SHA256,
        },
        "teacher": config.get("teacher", {}).get("checkpoint_sha256")
        == EXPECTED_TEACHER_SHA256
        and config.get("teacher", {}).get("model_state_sha256")
        == EXPECTED_TEACHER_STATE_SHA256,
        "classification": classification.get("student_architecture")
        == "deit_tiny_patch16_224"
        and classification.get("batch_size") == 128
        and classification.get("encoder_seed") == 1
        and classification.get("epochs") == 300
        and classification.get("selection_split") == "validation"
        and classification.get("official_test_evaluations") == 0,
        "segmentation": probe.get("probe_seeds") == [1, 2, 3, 4, 5]
        and probe.get("learning_rates") == [0.01, 0.03, 0.1]
        and probe.get("epochs") == 100
        and probe.get("batch_size") == 64
        and probe.get("official_test_evaluations") == 0,
        "direct": direct.get("subset")
        == {
            "train": "lowest_image_id_per_class_200",
            "validation": "lowest_image_id_per_class_200",
        }
        and direct.get("part_localization_probe", {}).get("probe_seeds")
        == [1, 2, 3, 4, 5]
        and direct.get("part_localization_probe", {}).get("epochs") == 100
        and direct.get("part_localization_probe", {}).get("checkpoint_purpose")
        == "phase1_cub_loader_pilot_full_part_probe_v1",
        "selection": config.get("loader_selection", {}).get("primary")
        == "arithmetic_mean_validation_part_micro_PCK_at_0.1_over_4_variants_x5_probe_seeds"
        and config.get("loader_selection", {}).get("tie_break_1")
        == "arithmetic_mean_validation_frozen_probe_input224_mIoU_over_4_variants_x5_probe_seeds"
        and config.get("loader_selection", {}).get("official_test_used_for_selection")
        is False,
        "execution": execution.get("requested_mig_slices") == 7
        and execution.get("feature_batch_size") == 16
        and execution.get("cka_batch_size") == 8
        and execution.get("attention_batch_size") == 16
        and execution.get("classification_eval_batch_size") == 200
        and execution.get("classification_num_workers") == 4
        and execution.get("single_all_profile_job_forbidden_by_10h_limit") is True,
        "gate": config.get("completion_gate_per_shard")
        == {
            "loader_profiles": 1,
            "classification_students": 4,
            "classification_checkpoints": 4,
            "segmentation_probe_lr_candidates": 60,
            "segmentation_probe_validation_selections": 20,
            "part_probe_lr_candidates": 60,
            "part_probe_validation_selections": 20,
            "spatial_cka_values": 48,
            "attention_metric_rows": 4,
            "attention_qualitative_pngs": 16,
            "new_checkpoints": 44,
            "official_test_evaluations": 0,
        },
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("invalid loader-pilot full config: " + ", ".join(failures))
    for profile in LOADER_PROFILE_ORDER:
        contract = loader_profile_contract(profile)
        stored = config["loader_profiles"][profile]
        expected = {
            "random_resized_crop_scale": contract["random_resized_crop"]["scale"],
            "random_resized_crop_ratio": contract["random_resized_crop"]["ratio"],
            "horizontal_flip_probability": contract["horizontal_flip_probability"],
            "color_jitter_argument": contract["color_jitter_argument"],
            "auto_augment": contract["auto_augment"],
            "random_erasing_probability": contract["random_erasing_probability"],
        }
        if stored != expected:
            raise RuntimeError(f"loader profile contract changed: {profile}")
    _validate_provenance(config)


def _validate_cli(args: argparse.Namespace, config: dict[str, Any]) -> None:
    expected = config["execution"]
    actual = {
        "feature_batch_size": args.feature_batch_size,
        "cka_batch_size": args.cka_batch_size,
        "attention_batch_size": args.attention_batch_size,
        "classification_eval_batch_size": args.eval_batch_size,
        "classification_num_workers": args.num_workers,
    }
    if actual != {key: expected[key] for key in actual}:
        raise RuntimeError(f"loader-pilot full runtime changed: {actual}")
    if args.profile not in LOADER_PROFILE_ORDER:
        raise RuntimeError(f"unexpected loader profile: {args.profile}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUB loader-pilot full requires H200 CUDA")


def _status(
    path: Path,
    *,
    status: str,
    phase: str,
    profile: str,
    classification: int = 0,
    segmentation_candidates: int = 0,
    segmentation_selections: int = 0,
    part_candidates: int = 0,
    part_selections: int = 0,
    cka_values: int = 0,
    attention_rows: int = 0,
    active: str | None = None,
    failure: str | None = None,
) -> None:
    _atomic_json_save(
        {
            "status": status,
            "phase": phase,
            "loader_profile": profile,
            "classification": classification,
            "segmentation_candidates": segmentation_candidates,
            "segmentation_selections": segmentation_selections,
            "part_candidates": part_candidates,
            "part_selections": part_selections,
            "cka_values": cka_values,
            "attention_rows": attention_rows,
            "active": active,
            "official_test_accessed": False,
            "failure": failure,
        },
        path,
    )


def _variant_metadata(variant: str) -> tuple[str, float | None, int]:
    try:
        return VARIANT_ARGUMENTS[variant]
    except KeyError as error:
        raise RuntimeError(f"unexpected loader-pilot variant: {variant}") from error


def _run_or_resume(command: list[str], summary_path: Path, label: str) -> dict[str, Any]:
    if summary_path.is_file():
        summary = _load_json(summary_path)
        if summary.get("status") == "complete":
            log(f"[LOADER_PILOT_FULL_RESUME] {label} summary={summary_path}")
            return summary
    log(f"[LOADER_PILOT_FULL_TASK_START] {label} command={' '.join(command)}")
    subprocess.run(command, check=True)
    summary = _load_json(summary_path)
    if summary.get("status") != "complete":
        raise RuntimeError(f"incomplete classification summary: {label}")
    return summary


def _classification_command(
    args: argparse.Namespace,
    *,
    variant: str,
    student_root: Path,
) -> tuple[list[str], str]:
    method, fusion_ratio, warmup = _variant_metadata(variant)
    run_name = f"cub_loader_pilot_{args.profile}_{variant}_b128_seed1_full_300ep"
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
        "--teacher-architecture",
        "resnet50_224_scratch",
        "--scientific-cub-r50-teacher",
        "--loader-pilot-full",
        "--cub-loader-profile",
        args.profile,
        "--batch-profile-role",
        "exploratory_loader_pilot",
        "--protocol-config",
        str(args.config),
        "--batch-size",
        "128",
        "--seed",
        "1",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(student_root),
        "--run-name",
        run_name,
        "--teacher-checkpoint",
        str(args.teacher_checkpoint),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if fusion_ratio is not None:
        command.extend(["--fusion-ratio", str(fusion_ratio)])
    if method == "alg":
        command.extend(["--alg-controller-warmup-epochs", str(warmup)])
    return command, run_name


def _classification_rows(
    args: argparse.Namespace,
    *,
    config_sha256: str,
    validation_hash: str,
    status_path: Path,
) -> list[dict[str, Any]]:
    student_root = args.output_dir / "classification/students"
    rows: list[dict[str, Any]] = []
    for variant in EXPECTED_VARIANTS:
        method, fusion_ratio, warmup = _variant_metadata(variant)
        command, run_name = _classification_command(
            args, variant=variant, student_root=student_root
        )
        run_dir = student_root / run_name
        summary = _run_or_resume(command, run_dir / "summary.json", variant)
        checkpoint = run_dir / "student_best_validation.pt"
        expected = {
            "status": "complete",
            "scientific_result": True,
            "confirmatory_main_result": False,
            "exploratory_loader_pilot": True,
            "loader_pilot_full": True,
            "cub_loader_profile": args.profile,
            "kind": "student",
            "dataset": DATASET_NAME,
            "num_classes": NUM_CLASSES,
            "method": method,
            "fusion_ratio_lambda": fusion_ratio,
            "batch_size": 128,
            "epochs": 300,
            "seed": 1,
            "official_test": None,
            "official_test_evaluations": 0,
            "official_test_accessed": False,
            "selected_checkpoint_strict_reloaded": True,
            "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
            "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
            "teacher_architecture": "resnet50_224_scratch",
            "protocol_config_sha256": config_sha256,
            "batch_profile_role": "exploratory_loader_pilot",
            "eligible_locked_v3_matrix_cell": False,
            "guidance_controller_warmup_epochs": warmup,
        }
        failures = [key for key, value in expected.items() if summary.get(key) != value]
        split = summary.get("split_manifest", {})
        if split.get("validation_image_ids_sha256") != validation_hash:
            failures.append("validation_image_ids_sha256")
        if split.get("student_train_loader") != loader_profile_contract(args.profile):
            failures.append("student_train_loader")
        if (
            not checkpoint.is_file()
            or summary.get("checkpoint_sha256") != file_sha256(checkpoint)
        ):
            failures.append("checkpoint")
        if failures:
            raise RuntimeError(
                f"classification contract failed {args.profile}/{variant}: "
                + ",".join(failures)
            )
        rows.append(
            {
                "profile": args.profile,
                "variant": variant,
                "method": method,
                "fusion_ratio_lambda": fusion_ratio,
                "controller_warmup_epochs": warmup,
                "run_name": run_name,
                "summary_path": str((run_dir / "summary.json").resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "summary": summary,
            }
        )
        _status(
            status_path,
            status="running",
            phase="classification",
            profile=args.profile,
            classification=len(rows),
            active=variant,
        )
        log(f"[LOADER_PILOT_FULL_TASK_DONE] profile={args.profile} variant={variant}")
    initial_hashes = {row["summary"]["initial_student_state_sha256"] for row in rows}
    if len(initial_hashes) != 1:
        raise RuntimeError("student initialization changed across variants")
    return rows


def _load_encoder(
    row: dict[str, Any],
    *,
    profile: str,
    config_sha256: str,
    validation_hash: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = Path(row["checkpoint_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    expected = {
        "purpose": "phase1_cub_r50_224_loader_pilot_full_student_v1",
        "loader_pilot_full": True,
        "cub_loader_profile": profile,
        "exploratory_loader_pilot": True,
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "architecture": "deit_tiny_patch16_224",
        "method": row["method"],
        "fusion_ratio_lambda": row["fusion_ratio_lambda"],
        "batch_size": 128,
        "epochs": 300,
        "seed": 1,
        "validation_image_ids_sha256": validation_hash,
        "guidance_controller_warmup_epochs": row["controller_warmup_epochs"],
        "teacher_checkpoint_sha256": EXPECTED_TEACHER_SHA256,
        "teacher_model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
        "protocol_config_sha256": config_sha256,
        "batch_profile_role": "exploratory_loader_pilot",
        "eligible_locked_v3_matrix_cell": False,
        "official_test_policy": "not_accessed_validation_only_loader_pilot",
        "official_test_evaluations_at_checkpoint_write": 0,
    }
    failures = [key for key, value in expected.items() if metadata.get(key) != value]
    if failures:
        raise RuntimeError(
            f"encoder metadata mismatch {profile}/{row['variant']}: "
            + ",".join(failures)
        )
    model = create_student(num_classes=NUM_CLASSES, drop_path_rate=0.1)
    incompatible = model.load_state_dict(payload["student"], strict=True)
    state_hash = state_dict_sha256(model)
    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
        or state_hash != metadata.get("student_state_sha256")
        or file_sha256(checkpoint) != row["summary"]["checkpoint_sha256"]
        or not all(
            bool(torch.isfinite(value).all())
            for value in payload["student"].values()
            if value.is_floating_point()
        )
    ):
        raise RuntimeError(f"encoder state audit failed: {profile}/{row['variant']}")
    model.to(device).eval().requires_grad_(False)
    audit = {
        "profile": profile,
        "variant": row["variant"],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "student_state_sha256": state_hash,
        "strict_load": True,
        "all_floating_tensors_finite": True,
        "eval_mode": not model.training,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    if not audit["eval_mode"] or audit["trainable_parameters"] != 0:
        raise RuntimeError("frozen encoder contract failed")
    return model, audit


def _segmentation_validation(
    *,
    profile: str,
    row: dict[str, Any],
    model: torch.nn.Module,
    audit: dict[str, Any],
    records: dict[str, list[CubProbeRecord]],
    targets: dict[str, dict[str, Any]],
    config: dict[str, Any],
    config_sha256: str,
    output_dir: Path,
    cache_dir: Path,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, set[str]], dict[tuple[int, int], set[str]]]:
    variant = row["variant"]
    features: dict[str, dict[str, Any]] = {}
    cache_paths: list[Path] = []
    for split in ("train", "validation"):
        cache_path = cache_dir / "features" / variant / f"{split}.pt"
        cache_paths.append(cache_path)
        features[split], _ = _feature_cache(
            model,
            records[split],
            split=split,
            variant=f"{profile}__{variant}",
            encoder_seed=1,
            checkpoint_sha256=audit["checkpoint_sha256"],
            state_sha256=audit["student_state_sha256"],
            config=config,
            config_sha256=config_sha256,
            cache_path=cache_path,
            device=device,
            batch_size=feature_batch_size,
            num_workers=num_workers,
        )
    probe_config = config["frozen_probe"]["probe"]
    candidates_out: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    initial_hashes: dict[int, set[str]] = defaultdict(set)
    batch_orders: dict[tuple[int, int], set[str]] = defaultdict(set)
    for probe_seed in probe_config["probe_seeds"]:
        candidates: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        for learning_rate in probe_config["learning_rates"]:
            state, candidate = train_candidate(
                features["train"]["features"],
                targets["train"]["grid_targets"],
                features["validation"]["features"],
                targets["validation"]["grid_targets"],
                probe_config=probe_config,
                learning_rate=float(learning_rate),
                seed=int(probe_seed),
                device=device,
                epochs=int(probe_config["epochs"]),
            )
            candidates.append((state, candidate))
            initial_hashes[int(probe_seed)].add(candidate["initial_probe_state_sha256"])
            for epoch, digest in enumerate(
                candidate["batch_order_sha256_by_epoch"], start=1
            ):
                batch_orders[(int(probe_seed), epoch)].add(digest)
            _atomic_json_save(
                {
                    "profile": profile,
                    "variant": variant,
                    "encoder_seed": 1,
                    "probe_seed": probe_seed,
                    "learning_rate": learning_rate,
                    "candidate": candidate,
                    "scientific_result": True,
                    "exploratory_loader_pilot": True,
                    "official_test_accessed": False,
                    "config_sha256": config_sha256,
                },
                output_dir
                / "segmentation_probe/histories"
                / variant
                / f"probe_seed{probe_seed}_lr{str(learning_rate).replace('.', 'p')}.json",
            )
            log(
                f"[LOADER_PILOT_SEG_FULL_CANDIDATE] profile={profile} "
                f"variant={variant} probe_seed={probe_seed} lr={learning_rate:g} "
                f"best_epoch={candidate['best_epoch']} validation_grid_miou="
                f"{candidate['best_validation_grid_mean_iou']:.6f}"
            )
        selected_state, selected = min(
            candidates,
            key=lambda item: (
                -float(item[1]["best_validation_grid_mean_iou"]),
                float(item[1]["learning_rate"]),
                int(item[1]["best_epoch"]),
            ),
        )
        probe = probe_from_state(probe_config, int(probe_seed), selected_state, device)
        validation, _ = evaluate_probe_both_resolutions(
            probe,
            features["validation"]["features"],
            targets["validation"]["grid_targets"],
            targets["validation"]["input_targets"],
            batch_size=int(probe_config["batch_size"]),
            device=device,
            input_size=int(config["frozen_probe"]["image_input"]["size"]),
            ignore_index=int(probe_config["loss"]["ignore_index"]),
        )
        if not all(_finite_metrics(value) for value in validation.values()):
            raise RuntimeError(f"non-finite segmentation metric: {variant}/{probe_seed}")
        checkpoint = (
            output_dir
            / "segmentation_probe/checkpoints"
            / variant
            / f"probe_seed{probe_seed}_best_validation.pt"
        )
        _atomic_torch_save(
            {
                "model": selected_state,
                "metadata": {
                    "purpose": "phase1_cub_loader_pilot_full_segmentation_probe_v1",
                    "scientific_result": True,
                    "exploratory_loader_pilot": True,
                    "official_test_accessed": False,
                    "config_sha256": config_sha256,
                    "loader_profile": profile,
                    "variant": variant,
                    "encoder_seed": 1,
                    "encoder_checkpoint_sha256": audit["checkpoint_sha256"],
                    "probe_seed": probe_seed,
                    "selection_split": "validation",
                    "selected_learning_rate": selected["learning_rate"],
                    "selected_epoch": selected["best_epoch"],
                    "official_test_evaluations_at_checkpoint_write": 0,
                },
            },
            checkpoint,
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        strict_probe = probe_from_state(
            probe_config, int(probe_seed), saved["model"], device
        )
        del strict_probe, probe
        for _state, candidate in candidates:
            candidates_out.append(
                {
                    "profile": profile,
                    "variant": variant,
                    "encoder_seed": 1,
                    "probe_seed": probe_seed,
                    "learning_rate": candidate["learning_rate"],
                    "best_epoch": candidate["best_epoch"],
                    "validation_grid_mean_iou": candidate[
                        "best_validation_grid_mean_iou"
                    ],
                    "selected": candidate is selected,
                    "scientific_result": True,
                    "official_test_evaluations": 0,
                }
            )
        selections.append(
            {
                "profile": profile,
                "variant": variant,
                "encoder_seed": 1,
                "probe_seed": probe_seed,
                "selected_learning_rate": selected["learning_rate"],
                "selected_epoch": selected["best_epoch"],
                "validation": validation,
                "initial_probe_state_sha256": selected[
                    "initial_probe_state_sha256"
                ],
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "selected_probe_strict_reloaded": True,
                "scientific_result": True,
                "exploratory_loader_pilot": True,
                "official_test_evaluations": 0,
            }
        )
        del candidates
    del features
    if not config["execution"]["retain_dataset_or_feature_cache_in_output"]:
        for path in cache_paths:
            path.unlink(missing_ok=True)
    return candidates_out, selections, initial_hashes, batch_orders


def _part_adapter(config: dict[str, Any], profile: str) -> dict[str, Any]:
    protocol = dict(config["direct_spatial"]["part_localization_probe"])
    protocol["loader_profile"] = profile
    return {
        "scope": {"encoder_seeds_in_this_run": [1]},
        "part_localization_probe": protocol,
    }


def _audit_part_checkpoints(
    selections: Sequence[dict[str, Any]],
    *,
    profile: str,
    config_sha256: str,
    output_dir: Path,
) -> None:
    for selection in selections:
        checkpoint = output_dir / selection["checkpoint_relative_path"]
        if (
            not checkpoint.is_file()
            or file_sha256(checkpoint) != selection["checkpoint_sha256"]
        ):
            raise RuntimeError("selected part-probe checkpoint bytes changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        expected = {
            "purpose": "phase1_cub_loader_pilot_full_part_probe_v1",
            "loader_profile": profile,
            "variant": selection["variant"],
            "encoder_seed": 1,
            "probe_seed": selection["probe_seed"],
            "selection_metric": "validation_micro_PCK_at_0.1",
            "official_test_evaluations_at_checkpoint_write": 0,
            "config_sha256": config_sha256,
        }
        failures = [
            key for key, value in expected.items() if metadata.get(key) != value
        ]
        probe = build_part_probe(int(selection["probe_seed"]))
        incompatible = probe.load_state_dict(payload["probe"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            failures.append("strict_load")
        if failures:
            raise RuntimeError(
                "selected part-probe checkpoint contract failed: "
                + ",".join(failures)
            )


def _classification_flat(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "profile": row["profile"],
            "variant": row["variant"],
            "method": row["method"],
            "fusion_ratio_lambda": row["fusion_ratio_lambda"],
            "controller_warmup_epochs": row["controller_warmup_epochs"],
            "encoder_seed": 1,
            "selected_epoch": row["summary"]["selected_epoch"],
            "validation_macro_top1": row["summary"]["selected_validation"][
                "macro_top1"
            ],
            "validation_overall_top1": row["summary"]["selected_validation"][
                "overall_top1"
            ],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["summary"]["checkpoint_sha256"],
            "official_test_evaluations": 0,
        }
        for row in rows
    ]


def _checkpoint_manifest(
    teacher_checkpoint: Path,
    classifications: Sequence[dict[str, Any]],
    segmentation: Sequence[dict[str, Any]],
    part: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    entries = [
        {
            "kind": "external_shared_teacher",
            "source_h200_issue": 722,
            "path": str(teacher_checkpoint.resolve()),
            "sha256": file_sha256(teacher_checkpoint),
        }
    ]
    entries.extend(
        {
            "kind": "classification_encoder",
            "variant": row["variant"],
            "path": row["checkpoint_path"],
            "sha256": row["summary"]["checkpoint_sha256"],
        }
        for row in classifications
    )
    entries.extend(
        {
            "kind": "selected_segmentation_probe",
            "variant": row["variant"],
            "probe_seed": row["probe_seed"],
            "path": row["checkpoint_path"],
            "sha256": row["checkpoint_sha256"],
        }
        for row in segmentation
    )
    entries.extend(
        {
            "kind": "selected_part_probe",
            "variant": row["variant"],
            "probe_seed": row["probe_seed"],
            "path": str((output_dir / row["checkpoint_relative_path"]).resolve()),
            "sha256": row["checkpoint_sha256"],
        }
        for row in part
    )
    return {
        "count": len(entries),
        "external_shared_teacher_count": 1,
        "new_checkpoint_count": len(entries) - 1,
        "entries": entries,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _utc_now()
    args.config = args.config.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    config = _load_json(args.config)
    _validate_config(config, args.config)
    _validate_cli(args, config)
    config_sha256 = file_sha256(args.config)

    import timm

    if timm.__version__ != "1.0.27":
        raise RuntimeError(f"expected timm==1.0.27, found {timm.__version__}")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "sequence_status.json"
    _status(
        status_path,
        status="running",
        phase="dataset_and_teacher_audit",
        profile=args.profile,
    )
    log("=" * 100)
    log(f"CUB PHASE 1 — LOADER PILOT FULL SHARD {args.profile}")
    log("=" * 100)
    log(
        "[LOADER_PILOT_FULL_POLICY] scientific_result=true exploratory=true "
        "all_three_shards_required=true official_test_accessed=false"
    )

    records, split_manifest, source = load_train_validation_records(
        args.data_dir, download=True
    )
    validation_hash = split_manifest.get("validation_image_ids_sha256")
    if (
        {key: len(value) for key, value in records.items()}
        != {"train": 5394, "validation": 600}
        or validation_hash != EXPECTED_VALIDATION_SHA256
    ):
        raise RuntimeError("CUB train/validation split changed")
    archives = _archive_audit(args.data_dir)
    teacher, teacher_metadata, teacher_hash, teacher_state_hash = (
        load_scientific_teacher(
            args.teacher_checkpoint,
            validation_hash=validation_hash,
            device=device,
        )
    )
    if (
        teacher_hash != EXPECTED_TEACHER_SHA256
        or teacher_state_hash != EXPECTED_TEACHER_STATE_SHA256
    ):
        raise RuntimeError("issue-722 teacher identity changed")
    del teacher
    torch.cuda.empty_cache()

    targets: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        targets[split], _ = _target_cache(
            records[split],
            split=split,
            config=config,
            config_sha256=config_sha256,
            cache_path=args.cache_dir / "targets" / f"{split}.pt",
        )
    target_values = {
        int(value)
        for payload in targets.values()
        for key in ("input_targets", "grid_targets")
        for value in torch.unique(payload[key])
    }
    if target_values != {0, 1}:
        raise RuntimeError(f"binary segmentation mask values changed: {target_values}")

    direct_train = select_lowest_image_id_per_class(records["train"])
    direct_validation = select_lowest_image_id_per_class(records["validation"])
    dataset_root = resolve_dataset_root(args.data_dir)
    part_names, annotations = load_spatial_annotations(dataset_root)
    train_supervision = load_part_supervision(direct_train, annotations)
    validation_supervision = load_part_supervision(direct_validation, annotations)
    train_part_audit = summarize_part_supervision(direct_train, train_supervision)
    validation_part_audit = summarize_part_supervision(
        direct_validation, validation_supervision
    )
    if len(part_names) != 15 or len(annotations) != 11788:
        raise RuntimeError("CUB part annotation inventory changed")
    _atomic_json_save(
        {
            "status": "pass",
            "dataset": DATASET_NAME,
            "source": source,
            "archives": archives,
            "split_manifest": split_manifest,
            "counts": {"train": 5394, "validation": 600, "official_test_loaded": 0},
            "binary_mask_values": sorted(target_values),
            "direct_subset_counts": {"train": 200, "validation": 200},
            "part_names": list(part_names),
            "part_annotation_validity": {
                "train": train_part_audit,
                "validation": validation_part_audit,
            },
            "global_part_annotation_index_parsed_for_integrity": True,
            "spatial_annotations_used_for_student_training": False,
            "official_test_images_masks_or_metrics_accessed": False,
        },
        args.output_dir / "dataset_audit.json",
    )

    classifications = _classification_rows(
        args,
        config_sha256=config_sha256,
        validation_hash=str(validation_hash),
        status_path=status_path,
    )
    classification_flat = _classification_flat(classifications)
    _write_csv(classification_flat, args.output_dir / "classification/results.csv")

    teacher, _teacher_metadata_again, _teacher_hash_again, _teacher_state_again = (
        load_scientific_teacher(
            args.teacher_checkpoint,
            validation_hash=str(validation_hash),
            device=device,
        )
    )
    encoder_audits: list[dict[str, Any]] = []
    segmentation_candidates: list[dict[str, Any]] = []
    segmentation_selections: list[dict[str, Any]] = []
    part_candidates: list[dict[str, Any]] = []
    part_selections: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    seg_initial: dict[int, set[str]] = defaultdict(set)
    seg_orders: dict[tuple[int, int], set[str]] = defaultdict(set)
    part_initial: dict[int, set[str]] = defaultdict(set)
    part_config = _part_adapter(config, args.profile)

    for classification in classifications:
        variant = classification["variant"]
        model, audit = _load_encoder(
            classification,
            profile=args.profile,
            config_sha256=config_sha256,
            validation_hash=str(validation_hash),
            device=device,
        )
        encoder_audits.append(audit)
        seg_candidates, seg_selections, initial, orders = _segmentation_validation(
            profile=args.profile,
            row=classification,
            model=model,
            audit=audit,
            records=records,
            targets=targets,
            config=config,
            config_sha256=config_sha256,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device=device,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        segmentation_candidates.extend(seg_candidates)
        segmentation_selections.extend(seg_selections)
        for key, values in initial.items():
            seg_initial[key].update(values)
        for key, values in orders.items():
            seg_orders[key].update(values)

        candidates, selections, initial_hashes = _run_part_validation(
            variant=variant,
            encoder_seed=1,
            model=model,
            train_records=direct_train,
            validation_records=direct_validation,
            train_supervision=train_supervision,
            validation_supervision=validation_supervision,
            config=part_config,
            config_sha256=config_sha256,
            output_dir=args.output_dir,
            device=device,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
        )
        for item in candidates:
            item.update({"profile": args.profile, "official_test_evaluations": 0})
        for item in selections:
            item.update(
                {
                    "profile": args.profile,
                    "exploratory_loader_pilot": True,
                    "official_test_evaluations": 0,
                }
            )
        part_candidates.extend(candidates)
        part_selections.extend(selections)
        for key, value in initial_hashes.items():
            part_initial[key].add(value)

        variant_cka = _cka_smoke(
            variant=variant,
            encoder_seed=1,
            student=model,
            teacher=teacher,
            records=direct_validation,
            batch_size=args.cka_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        for item in variant_cka:
            item.update(
                {
                    "profile": args.profile,
                    "scientific_result": True,
                    "exploratory_loader_pilot": True,
                }
            )
        cka_rows.extend(variant_cka)
        attention = _attention_smoke(
            variant=variant,
            encoder_seed=1,
            student=model,
            records=direct_validation,
            batch_size=args.attention_batch_size,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            device=device,
        )
        attention.update(
            {
                "profile": args.profile,
                "scientific_result": True,
                "exploratory_loader_pilot": True,
            }
        )
        attention_rows.append(attention)
        del model
        torch.cuda.empty_cache()
        _status(
            status_path,
            status="running",
            phase="frozen_spatial_probes",
            profile=args.profile,
            classification=4,
            segmentation_candidates=len(segmentation_candidates),
            segmentation_selections=len(segmentation_selections),
            part_candidates=len(part_candidates),
            part_selections=len(part_selections),
            cka_values=len(cka_rows),
            attention_rows=len(attention_rows),
            active=variant,
        )
    del teacher
    torch.cuda.empty_cache()

    _audit_part_checkpoints(
        part_selections,
        profile=args.profile,
        config_sha256=config_sha256,
        output_dir=args.output_dir,
    )

    if (
        set(seg_initial) != {1, 2, 3, 4, 5}
        or any(len(values) != 1 for values in seg_initial.values())
        or len(seg_orders) != 500
        or any(len(values) != 1 for values in seg_orders.values())
        or set(part_initial) != {1, 2, 3, 4, 5}
        or any(len(values) != 1 for values in part_initial.values())
    ):
        raise RuntimeError("matched probe initialization or batch-order contract failed")

    qualitative_pngs = sorted(
        (args.output_dir / "attention_gt/qualitative").glob("*.png")
    )
    manifest = _checkpoint_manifest(
        args.teacher_checkpoint,
        classifications,
        segmentation_selections,
        part_selections,
        args.output_dir,
    )
    gate = {
        "loader_profiles": 1,
        "classification_students": len(classifications),
        "classification_checkpoints": sum(
            Path(row["checkpoint_path"]).is_file() for row in classifications
        ),
        "segmentation_probe_lr_candidates": len(segmentation_candidates),
        "segmentation_probe_validation_selections": len(segmentation_selections),
        "part_probe_lr_candidates": len(part_candidates),
        "part_probe_validation_selections": len(part_selections),
        "spatial_cka_values": len(cka_rows),
        "attention_metric_rows": len(attention_rows),
        "attention_qualitative_pngs": len(qualitative_pngs),
        "new_checkpoints": manifest["new_checkpoint_count"],
        "official_test_evaluations": 0,
    }
    if gate != config["completion_gate_per_shard"]:
        raise RuntimeError(f"loader-pilot full completion gate failed: {gate}")

    _write_csv(
        segmentation_candidates,
        args.output_dir / "segmentation_probe/candidates.csv",
    )
    _atomic_json_save(
        {
            "status": "complete",
            "profile": args.profile,
            "selections": segmentation_selections,
            "matched_initialization_and_batch_order": True,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "official_test_evaluations": 0,
        },
        args.output_dir / "segmentation_probe/results.json",
    )
    _write_csv(part_candidates, args.output_dir / "part_probe/candidates.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "profile": args.profile,
            "selections": part_selections,
            "matched_initialization": True,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "official_test_evaluations": 0,
        },
        args.output_dir / "part_probe/results.json",
    )
    _write_csv(cka_rows, args.output_dir / "spatial_cka/results.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "profile": args.profile,
            "rows": cka_rows,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "official_test_used": False,
        },
        args.output_dir / "spatial_cka/results.json",
    )
    _save_cka_heatmap(cka_rows, args.output_dir / "spatial_cka/layerwise_heatmap.png")
    _write_csv(attention_rows, args.output_dir / "attention_gt/results.csv")
    _atomic_json_save(
        {
            "status": "complete",
            "profile": args.profile,
            "rows": attention_rows,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "official_test_evaluations": 0,
        },
        args.output_dir / "attention_gt/results.json",
    )
    _atomic_json_save(manifest, args.output_dir / "checkpoint_manifest.json")
    _atomic_json_save(
        {
            "status": "pass",
            "teacher": {
                "metadata": teacher_metadata,
                "checkpoint_sha256": teacher_hash,
                "model_state_sha256": teacher_state_hash,
                "strict_load": True,
            },
            "students": encoder_audits,
        },
        args.output_dir / "checkpoint_audit.json",
    )

    part_score = statistics.mean(
        float(row["validation"]["micro_pck_at_0.1"])
        for row in part_selections
    )
    segmentation_tie_score = statistics.mean(
        float(row["validation"]["input_224"]["mean_iou"])
        for row in segmentation_selections
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config_path": str(args.config),
        "config_sha256": config_sha256,
        "loader_profile": args.profile,
        "loader_profile_contract": loader_profile_contract(args.profile),
        "scientific_result": True,
        "exploratory_loader_pilot": True,
        "confirmatory_main_result": False,
        "all_three_shards_required_before_loader_selection": True,
        "this_shard_alone_must_not_select_loader": True,
        "classification": classification_flat,
        "loader_selection_scores": {
            "primary_mean_validation_part_micro_pck_at_0.1": part_score,
            "tie_break_1_mean_validation_frozen_segmentation_input224_miou": segmentation_tie_score,
            "method_count": 4,
            "probe_seed_count_per_method": 5,
            "value_count_per_score": 20,
        },
        "completion_gate": gate,
        "checkpoint_manifest": str(
            (args.output_dir / "checkpoint_manifest.json").resolve()
        ),
        "runtime": {
            **_runtime(device),
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "official_test_accessed": False,
        "official_test_evaluations": 0,
    }
    _atomic_json_save(summary, args.output_dir / "summary.json")
    _status(
        status_path,
        status="complete",
        phase="complete",
        profile=args.profile,
        classification=4,
        segmentation_candidates=60,
        segmentation_selections=20,
        part_candidates=60,
        part_selections=20,
        cka_values=48,
        attention_rows=4,
    )

    seg_by_variant: dict[str, list[float]] = defaultdict(list)
    part_by_variant: dict[str, list[float]] = defaultdict(list)
    for row in segmentation_selections:
        seg_by_variant[row["variant"]].append(
            float(row["validation"]["input_224"]["mean_iou"])
        )
    for row in part_selections:
        part_by_variant[row["variant"]].append(
            float(row["validation"]["micro_pck_at_0.1"])
        )
    cka_lookup = {
        (row["variant"], int(row["student_block"])): row for row in cka_rows
    }
    attention_lookup = {row["variant"]: row for row in attention_rows}
    log("[LOADER_PILOT_FULL_RESULTS]")
    for row in classifications:
        variant = row["variant"]
        attention = attention_lookup[variant]
        log(
            f"[LOADER_PILOT_FULL_RESULT] profile={args.profile} variant={variant} "
            f"class_val_macro={row['summary']['selected_validation']['macro_top1']:.4f} "
            f"seg_val_miou_probe_seed_mean={statistics.mean(seg_by_variant[variant]):.6f} "
            f"part_val_pck_probe_seed_mean={statistics.mean(part_by_variant[variant]):.6f} "
            f"cka_block11={cka_lookup[(variant, 11)]['centered_linear_cka']:.6f} "
            "attention_patch_ap="
            f"{attention['global_micro_patch_average_precision']:.6f} "
            f"attention_pointing={attention['pointing_game_peak_inside_mask']:.6f}"
        )
    log(
        f"[LOADER_PILOT_FULL_SCORE] profile={args.profile} "
        f"primary_part_pck={part_score:.6f} tie1_seg_miou={segmentation_tie_score:.6f} "
        "loader_selection_deferred_until_3_shards=true"
    )
    log(
        f"[LOADER_PILOT_FULL_DONE] status=complete profile={args.profile} "
        "classification=4/4 segmentation_candidates=60/60 "
        "segmentation_selections=20/20 part_candidates=60/60 "
        "part_selections=20/20 cka_values=48/48 attention_rows=4/4 "
        "qualitative_pngs=16/16 new_checkpoints=44/44 official_test=0 "
        f"elapsed={format_duration(elapsed)} summary={args.output_dir / 'summary.json'}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", required=True)
    parser.add_argument("--profile", choices=LOADER_PROFILE_ORDER, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--cka-batch-size", type=int, default=8)
    parser.add_argument("--attention-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        _status(
            output_dir / "sequence_status.json",
            status="failed",
            phase="failed",
            profile=args.profile,
            failure=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
