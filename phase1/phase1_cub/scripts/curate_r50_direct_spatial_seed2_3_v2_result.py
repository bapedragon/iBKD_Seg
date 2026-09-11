#!/usr/bin/env python3
"""Audit issue 739 and combine it with issue 737 into a 3-seed report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

import torch

from ibkd_seg.phase1.cub_direct_spatial import build_part_probe
from ibkd_seg.phase1.train_timing import state_dict_sha256


PROTOCOL_ID = "cub200_phase1_r50_224_b128_seed2_3_direct_spatial_full_v2"
CONFIG_SHA256 = "54980771cf910543a3aba24c0a5ff86de6a0dce34866c662025409ab3abd0691"
VALIDATION_SHA256 = "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
TEACHER_CHECKPOINT_SHA256 = "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
TEACHER_STATE_SHA256 = "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
VARIANTS = ("lg", "alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5")
ENCODER_SEEDS = (2, 3)
LABELS = {
    "lg": "LG",
    "alg_warmup20": "ALG-w20",
    "ibkd_lambda_0.25": "iBKD λ=0.25",
    "ibkd_lambda_0.5": "iBKD λ=0.5",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def save_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def finite_state(state: dict[str, torch.Tensor]) -> bool:
    return all(
        bool(torch.isfinite(tensor).all())
        for tensor in state.values()
        if tensor.is_floating_point()
    )


def verify_import(raw_dir: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("status") != "pass"
        or manifest.get("experiment_kind")
        != "resnet50-direct-spatial-v2-seeds2-3"
        or manifest.get("h200_issue_id") != "739"
        or manifest.get("checkpoint_count") != 40
        or manifest.get("imported_file_count_excluding_this_manifest") != 240
        or len(manifest.get("files", [])) != 240
    ):
        raise RuntimeError("wrong or incomplete issue 739 import manifest")
    for item in manifest["files"]:
        path = raw_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"imported issue 739 artifact digest mismatch: {path}")


def validate_completion(raw_dir: Path, summary: dict[str, Any]) -> None:
    expected_gate = {
        "attention_metric_rows": 8,
        "attention_official_test_evaluations": 8,
        "checkpoint_strict_loads": 8,
        "official_test_evaluations": 48,
        "part_probe_lr_candidates": 120,
        "part_probe_official_test_evaluations": 40,
        "part_probe_validation_selections": 40,
        "qualitative_pngs": 64,
        "spatial_cka_values": 96,
    }
    if (
        summary.get("status") != "complete"
        or summary.get("scientific_result") is not True
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("config_sha256") != CONFIG_SHA256
        or summary.get("student_batch_size") != 128
        or summary.get("encoder_seeds") != [2, 3]
        or summary.get("independent_encoder_seed_n") != 2
        or summary.get("final_encoder_seed_inference") is not False
        or summary.get("settings_unchanged_after_seed1_and_smoke") is not True
        or summary.get("variants") != list(VARIANTS)
        or summary.get("completion_gate") != expected_gate
        or summary.get("official_test_evaluations") != 48
        or summary.get("official_test_used_for_selection") is not False
    ):
        raise RuntimeError("issue 739 completion contract failed")

    sequence = load_json(raw_dir / "sequence_status.json")
    if (
        sequence.get("status") != "complete"
        or sequence.get("phase") != "complete"
        or sequence.get("failure") is not None
        or sequence.get("official_test_accessed") is not True
        or sequence.get("official_test_used_for_selection") is not False
        or sequence.get("completion_gate") != expected_gate
    ):
        raise RuntimeError("issue 739 sequence gate failed")

    dataset = load_json(raw_dir / "dataset_audit.json")
    validity = dataset.get("part_annotation_validity_audit", {})
    train = validity.get("train", {})
    validation = validity.get("validation", {})
    test = validity.get("official_test", {})
    if (
        dataset.get("status") != "pass"
        or dataset.get("counts")
        != {"train": 5394, "validation": 600, "official_test": 5794}
        or dataset.get("split_manifest", {}).get("validation_image_ids_sha256")
        != VALIDATION_SHA256
        or dataset.get("official_test_opened_after_all_validation_selections")
        is not True
        or dataset.get("validation_selections_complete_before_test") != 40
        or dataset.get("part_coordinate_validity", {}).get("coordinate_clipping")
        is not False
        or train.get("official_visible_keypoints") != 64697
        or train.get("valid_visible_keypoints") != 64696
        or train.get("affected_image_ids") != [5007]
        or validation.get("valid_visible_keypoints") != 7164
        or test.get("valid_visible_keypoints") != 69546
    ):
        raise RuntimeError("issue 739 dataset or keypoint-validity audit failed")


def audit_reused_encoders(
    raw_dir: Path,
    guided_report_dir: Path,
) -> dict[str, Any]:
    source = load_json(raw_dir / "checkpoint_audit.json")
    guided = load_json(guided_report_dir / "checkpoint_audit.json")
    expected = {
        (row["variant"], int(row["encoder_seed"])): row
        for row in guided["entries"]
        if row.get("kind") == "classification_encoder"
        and int(row["encoder_seed"]) in ENCODER_SEEDS
    }
    students = source.get("students", [])
    actual = {
        (row["variant"], int(row["encoder_seed"])): row for row in students
    }
    expected_keys = {
        (variant, seed) for variant in VARIANTS for seed in ENCODER_SEEDS
    }
    teacher = source.get("teacher", {})
    if (
        source.get("status") != "pass"
        or set(expected) != expected_keys
        or set(actual) != expected_keys
        or teacher.get("checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256
        or teacher.get("model_state_sha256") != TEACHER_STATE_SHA256
        or teacher.get("strict_load") is not True
        or teacher.get("eval_mode") is not True
        or teacher.get("trainable_parameters") != 0
    ):
        raise RuntimeError("issue 739 reused teacher/encoder audit is incomplete")

    entries: list[dict[str, Any]] = []
    for key in sorted(expected_keys, key=lambda item: (item[1], VARIANTS.index(item[0]))):
        expected_row = expected[key]
        actual_row = actual[key]
        if (
            actual_row.get("checkpoint_sha256")
            != expected_row.get("checkpoint_sha256")
            or actual_row.get("model_state_sha256")
            != expected_row.get("model_state_sha256")
            or actual_row.get("all_floating_tensors_finite") is not True
            or actual_row.get("strict_load") is not True
            or actual_row.get("eval_mode") is not True
            or actual_row.get("trainable_parameters") != 0
        ):
            raise RuntimeError(f"issue 739 encoder differs from issue 730: {key}")
        entries.append(
            {
                "variant": key[0],
                "encoder_seed": key[1],
                "checkpoint_sha256": actual_row["checkpoint_sha256"],
                "model_state_sha256": actual_row["model_state_sha256"],
                "hashes_match_audited_issue730": True,
                "strict_load_eval_frozen_on_h200": True,
            }
        )
    return {
        "source_h200_issue": 730,
        "encoder_checkpoint_count": 8,
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "teacher_state_sha256": teacher["model_state_sha256"],
        "hashes_match_audited_issue730": True,
        "classification_checkpoints_reused_not_duplicated": True,
        "entries": entries,
    }


def audit_part_probes(
    raw_dir: Path,
    results: dict[str, Any],
) -> dict[str, Any]:
    selections = results.get("selections", [])
    test_rows = results.get("official_test_rows", [])
    expected = {
        (variant, encoder_seed, probe_seed)
        for variant in VARIANTS
        for encoder_seed in ENCODER_SEEDS
        for probe_seed in range(1, 6)
    }
    by_key = {
        (row["variant"], int(row["encoder_seed"]), int(row["probe_seed"])): row
        for row in selections
    }
    test_keys = {
        (row["variant"], int(row["encoder_seed"]), int(row["probe_seed"]))
        for row in test_rows
    }
    if set(by_key) != expected or test_keys != expected:
        raise RuntimeError("issue 739 part-probe seed/variant matrix is incomplete")

    entries: list[dict[str, Any]] = []
    for variant, encoder_seed, probe_seed in sorted(
        expected,
        key=lambda key: (key[1], VARIANTS.index(key[0]), key[2]),
    ):
        selection = by_key[(variant, encoder_seed, probe_seed)]
        checkpoint = raw_dir / selection["checkpoint_relative_path"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        probe = build_part_probe(probe_seed)
        incompatible = probe.load_state_dict(payload["probe"], strict=True)
        metadata = payload.get("metadata", {})
        digest = file_sha256(checkpoint)
        if (
            digest != selection["checkpoint_sha256"]
            or metadata.get("config_sha256") != CONFIG_SHA256
            or metadata.get("variant") != variant
            or metadata.get("encoder_seed") != encoder_seed
            or metadata.get("probe_seed") != probe_seed
            or metadata.get("official_test_evaluations_at_checkpoint_write") != 0
            or incompatible.missing_keys
            or incompatible.unexpected_keys
            or not finite_state(payload["probe"])
        ):
            raise RuntimeError(
                "issue 739 part-probe checkpoint audit failed: "
                f"{variant} encoder_seed={encoder_seed} probe_seed={probe_seed}"
            )
        entries.append(
            {
                "variant": variant,
                "encoder_seed": encoder_seed,
                "probe_seed": probe_seed,
                "path_under_ignored_raw_root": selection[
                    "checkpoint_relative_path"
                ],
                "bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": digest,
                "model_state_sha256": state_dict_sha256(probe),
                "strict_load": True,
                "all_floating_tensors_finite": True,
            }
        )
        del payload, probe
    return {
        "selected_part_probe_checkpoint_count": 40,
        "all_file_hashes_match": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "entries": entries,
    }


def aggregate(
    rows: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        cells = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: int(row["encoder_seed"]),
        )
        seeds = [int(row["encoder_seed"]) for row in cells]
        values = [float(row[metric]) for row in cells]
        if seeds != [1, 2, 3] or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"invalid 3-seed aggregate input: {variant} {metric}")
        output.append(
            {
                "variant": variant,
                "metric": metric,
                "encoder_seeds": seeds,
                "encoder_seed_values": values,
                "independent_encoder_n": 3,
                "mean": statistics.mean(values),
                "sample_standard_deviation": statistics.stdev(values),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed1-report-dir", type=Path, required=True)
    parser.add_argument("--guided-report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    seed1_dir = args.seed1_report_dir.resolve()
    guided_dir = args.guided_report_dir.resolve()
    repository_root = Path(__file__).resolve().parents[3]

    manifest = load_json(raw_dir / "artifact_manifest.json")
    verify_import(raw_dir, manifest)
    summary = load_json(raw_dir / "summary.json")
    validate_completion(raw_dir, summary)
    config = (
        repository_root
        / "phase1/phase1_cub/configs/"
        "cub200_r50_224_b128_seed2_3_direct_spatial_full_v2.json"
    )
    if file_sha256(config) != CONFIG_SHA256:
        raise RuntimeError("repository seed-2/3 direct-spatial config differs")

    part = load_json(raw_dir / "part_probe/results.json")
    selection = load_json(
        raw_dir / "part_probe/selection_complete_before_test.json"
    )
    cka = load_json(raw_dir / "spatial_cka/results.json")
    attention = load_json(raw_dir / "attention_gt/results.json")
    if (
        part.get("status") != "complete"
        or part.get("official_test_evaluations") != 40
        or part.get("final_encoder_seed_inference") is not False
        or len(part.get("candidates", [])) != 120
        or len(part.get("selections", [])) != 40
        or len(part.get("official_test_rows", [])) != 40
        or len(part.get("aggregates", [])) != 8
        or selection.get("status") != "complete"
        or selection.get("config_sha256") != CONFIG_SHA256
        or selection.get("official_test_accessed") is not False
        or selection.get("selection_uses_official_test") is not False
        or selection.get("part_probe_validation_selections") != 40
        or cka.get("status") != "complete"
        or cka.get("official_test_used") is not False
        or len(cka.get("rows", [])) != 96
        or attention.get("status") != "complete"
        or attention.get("official_test_evaluations") != 8
        or len(attention.get("rows", [])) != 8
    ):
        raise RuntimeError("issue 739 metric-level completion contract failed")

    expected_cells = {
        (variant, seed) for variant in VARIANTS for seed in ENCODER_SEEDS
    }
    part_new = part["aggregates"]
    cka_new = [row for row in cka["rows"] if row["student_block"] == 11]
    attention_new = attention["rows"]
    for label, rows in (
        ("part", part_new),
        ("CKA block11", cka_new),
        ("attention", attention_new),
    ):
        keys = {(row["variant"], int(row["encoder_seed"])) for row in rows}
        if keys != expected_cells:
            raise RuntimeError(f"issue 739 {label} matrix is incomplete")

    reused_encoders = audit_reused_encoders(raw_dir, guided_dir)
    part_probe_audit = audit_part_probes(raw_dir, part)
    seed1 = load_json(seed1_dir / "direct_spatial_summary.json")
    if (
        seed1.get("status") != "complete_audited_seed1_only"
        or seed1.get("encoder_seed") != 1
        or seed1.get("independent_encoder_n") != 1
    ):
        raise RuntimeError("seed-1 direct-spatial source is not the audited issue 737 report")

    part_cells = seed1["part_pck_primary"] + part_new
    cka_cells = seed1["spatial_cka_block11_secondary"] + cka_new
    attention_cells = seed1["attention_gt_secondary"] + attention_new
    key_order = lambda row: (VARIANTS.index(row["variant"]), int(row["encoder_seed"]))
    part_cells.sort(key=key_order)
    cka_cells.sort(key=key_order)
    attention_cells.sort(key=key_order)

    part_pck_aggregates = aggregate(
        part_cells, "test_micro_pck_at_0.1_probe_seed_mean"
    )
    part_error_aggregates = aggregate(
        part_cells, "test_mean_normalized_error_probe_seed_mean"
    )
    cka_aggregates = aggregate(cka_cells, "centered_linear_cka")
    attention_ap_aggregates = aggregate(
        attention_cells, "global_micro_patch_average_precision"
    )
    attention_pointing_aggregates = aggregate(
        attention_cells, "pointing_game_peak_inside_mask"
    )
    attention_mass_aggregates = aggregate(
        attention_cells, "foreground_attention_mass_mean"
    )

    curated = {
        "schema_version": 1,
        "status": "complete_audited_guided_3seed_v2",
        "protocol_family": "cub200_phase1_r50_224_b128_direct_spatial_v2",
        "source_protocol_ids": [
            seed1["protocol_id"],
            PROTOCOL_ID,
        ],
        "source_h200_issues": [737, 739],
        "batch_size": 128,
        "variants": list(VARIANTS),
        "encoder_seeds": [1, 2, 3],
        "independent_unit": "classification_encoder_seed",
        "independent_encoder_n": 3,
        "probe_seeds_per_encoder": 5,
        "probe_seed_is_not_independent_replication": True,
        "final_encoder_seed_inference": True,
        "guided_four_method_direct_spatial_block_complete": True,
        "part_pck_primary": {
            "cells": part_cells,
            "aggregates": part_pck_aggregates,
        },
        "part_normalized_error_secondary": {
            "cells": part_cells,
            "aggregates": part_error_aggregates,
        },
        "spatial_cka_block11_secondary": {
            "cells": cka_cells,
            "aggregates": cka_aggregates,
            "evaluation_split": "validation_600",
        },
        "attention_gt_secondary": {
            "cells": attention_cells,
            "patch_ap_aggregates": attention_ap_aggregates,
            "pointing_aggregates": attention_pointing_aggregates,
            "foreground_mass_aggregates": attention_mass_aggregates,
        },
        "official_test_evaluations": {
            "part_probe": 60,
            "attention_gt": 12,
            "total": 72,
        },
        "official_test_used_for_selection": False,
    }
    checkpoint_audit = {
        "schema_version": 1,
        "status": "pass",
        "issue739_new_checkpoint_count": 40,
        "reused_encoders": reused_encoders,
        "part_probes": part_probe_audit,
        "seed1_source": {
            "h200_issue": 737,
            "report": "../resnet50_224_b128_seed1_v2",
            "selected_part_probe_checkpoints": 20,
            "checkpoint_audit_reused_not_duplicated": True,
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, report_dir / "source_manifest.json")
    save_json(checkpoint_audit, report_dir / "checkpoint_audit.json")
    save_json(curated, report_dir / "direct_spatial_summary.json")

    part_csv = [
        {
            "variant": row["variant"],
            "encoder_seed": row["encoder_seed"],
            "probe_seeds": 5,
            "test_pck_at_0.1_probe_seed_mean": row[
                "test_micro_pck_at_0.1_probe_seed_mean"
            ],
            "test_pck_at_0.1_probe_seed_sample_sd": row[
                "test_micro_pck_at_0.1_probe_seed_sample_sd"
            ],
            "test_normalized_error_probe_seed_mean": row[
                "test_mean_normalized_error_probe_seed_mean"
            ],
        }
        for row in part_cells
    ]
    cka_csv = [
        {
            "variant": row["variant"],
            "encoder_seed": row["encoder_seed"],
            "student_block": row["student_block"],
            "teacher_feature": row["teacher_feature"],
            "centered_linear_cka": row["centered_linear_cka"],
        }
        for row in cka_cells
    ]
    attention_csv = [
        {
            "variant": row["variant"],
            "encoder_seed": row["encoder_seed"],
            "patch_average_precision": row[
                "global_micro_patch_average_precision"
            ],
            "pointing_game": row["pointing_game_peak_inside_mask"],
            "foreground_attention_mass": row["foreground_attention_mass_mean"],
        }
        for row in attention_cells
    ]
    aggregate_by_metric = {
        "part_pck_at_0.1": part_pck_aggregates,
        "part_normalized_error": part_error_aggregates,
        "spatial_cka_block11": cka_aggregates,
        "attention_patch_ap": attention_ap_aggregates,
        "attention_pointing": attention_pointing_aggregates,
        "attention_foreground_mass": attention_mass_aggregates,
    }
    aggregate_csv: list[dict[str, Any]] = []
    for metric, rows in aggregate_by_metric.items():
        for row in rows:
            aggregate_csv.append(
                {
                    "metric": metric,
                    "variant": row["variant"],
                    "independent_encoder_n": 3,
                    "mean": row["mean"],
                    "sample_standard_deviation": row[
                        "sample_standard_deviation"
                    ],
                }
            )
    write_csv(part_csv, report_dir / "part_pck_per_encoder_seed.csv")
    write_csv(cka_csv, report_dir / "spatial_cka_block11_per_encoder_seed.csv")
    write_csv(attention_csv, report_dir / "attention_gt_per_encoder_seed.csv")
    write_csv(aggregate_csv, report_dir / "three_seed_aggregates.csv")

    for source, destination in {
        "summary.json": "h200_seed2_3_summary.json",
        "dataset_audit.json": "h200_dataset_audit.json",
        "sequence_status.json": "h200_sequence_status.json",
        "checkpoint_audit.json": "h200_checkpoint_audit.json",
        "part_probe/annotation_validity_audit.json": "part_annotation_validity_audit.json",
        "part_probe/candidates.csv": "h200_part_probe_candidates.csv",
        "part_probe/official_test_results.csv": "h200_part_probe_official_test_results.csv",
        "part_probe/results.json": "h200_part_probe_results.json",
        "part_probe/selection_complete_before_test.json": "h200_part_probe_selection_complete_before_test.json",
        "spatial_cka/results.csv": "h200_spatial_cka_results.csv",
        "spatial_cka/results.json": "h200_spatial_cka_results.json",
        "spatial_cka/layerwise_heatmap.png": "h200_spatial_cka_layerwise_heatmap_seed2_3.png",
        "attention_gt/results.csv": "h200_attention_gt_results.csv",
        "attention_gt/results.json": "h200_attention_gt_results.json",
    }.items():
        copy_file(raw_dir / source, report_dir / destination)
    qualitative = sorted((raw_dir / "attention_gt/qualitative").glob("*.png"))
    if len(qualitative) != 64:
        raise RuntimeError("issue 739 does not contain 64 qualitative PNGs")
    for image in qualitative:
        copy_file(image, report_dir / "attention_qualitative_seed2_3" / image.name)

    def indexed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {row["variant"]: row for row in rows}

    pck = indexed(part_pck_aggregates)
    error = indexed(part_error_aggregates)
    cka11 = indexed(cka_aggregates)
    ap = indexed(attention_ap_aggregates)
    pointing = indexed(attention_pointing_aggregates)
    mass = indexed(attention_mass_aggregates)
    table = "\n".join(
        f"| {LABELS[variant]} | "
        f"{100*pck[variant]['mean']:.3f} ± {100*pck[variant]['sample_standard_deviation']:.3f}% | "
        f"{error[variant]['mean']:.4f} ± {error[variant]['sample_standard_deviation']:.4f} | "
        f"{cka11[variant]['mean']:.4f} ± {cka11[variant]['sample_standard_deviation']:.4f} | "
        f"{ap[variant]['mean']:.4f} ± {ap[variant]['sample_standard_deviation']:.4f} | "
        f"{pointing[variant]['mean']:.4f} ± {pointing[variant]['sample_standard_deviation']:.4f} | "
        f"{mass[variant]['mean']:.4f} ± {mass[variant]['sample_standard_deviation']:.4f} |"
        for variant in VARIANTS
    )
    report_dir.joinpath("RESULTS.md").write_text(
        f"""# CUB ResNet-50/224 batch128 직접 공간정보 진단 v2 — 3 seeds

상태: **H200 issue 737·739 완료 · encoder seed 1·2·3 결합 · 독립 감사 통과**

각 part-probe 셀에서 probe seed 5개를 먼저 평균한 뒤, 아래 `±`는 독립
classification encoder seed `[1,2,3]`에 대한 sample SD로 계산했습니다. Part
PCK@0.1이 주 직접지표이고 CKA와 attention–GT는 보조지표입니다.

| 방법 | Part PCK@0.1 ↑ | 정규화 위치오차 ↓ | CKA block11 ↑ | Attention AP ↑ | Pointing ↑ | FG mass ↑ |
|---|---:|---:|---:|---:|---:|---:|
{table}

## 감사

- issue 739 실행시간: {summary['runtime']['elapsed_seconds'] / 60:.2f}분
- issue 739 encoder 8개는 감사된 issue 730 classification checkpoint의 파일·state hash와 일치
- 새 part-probe checkpoint 40개 모두 SHA-256, `weights_only=True`, strict load, 유한값 검사 통과
- issue 737의 seed-1 part-probe 20개 감사 결과를 합쳐 encoder seed 3개 × probe seed 5개 구성
- seed 2·3의 validation 후보 120개와 선택 40개를 끝낸 뒤 official test를 열었으며 test는 선택에 사용하지 않음
- 세 seed 전체 official-test 평가는 part probe 60회 + attention 12회 = 72회
- CKA는 validation 600장만 사용했고 official test를 사용하지 않음
- train의 공식 visible keypoint 64,697개 중 image 5007의 프레임 밖 1개만 사전 규칙대로 제외; validation 7,164개와 test 69,546개는 모두 유효
- issue 737의 32개와 issue 739의 64개, 총 96개 attention 정성 이미지를 두 보고서에 보존

## 해석

- **주 지표 Part PCK는 LG가 가장 높습니다.** LG는 {100*pck['lg']['mean']:.3f}%, ALG-w20은 {100*pck['alg_warmup20']['mean']:.3f}%, iBKD λ=0.25는 {100*pck['ibkd_lambda_0.25']['mean']:.3f}%, iBKD λ=0.5는 {100*pck['ibkd_lambda_0.5']['mean']:.3f}%입니다.
- CKA block11도 LG가 {cka11['lg']['mean']:.4f}로 가장 높고, 세 encoder seed 모두 `LG > ALG-w20 > iBKD-0.25 > iBKD-0.5` 순서입니다.
- Attention의 3-seed 평균은 AP·pointing·foreground mass 모두 iBKD λ=0.25가 가장 높습니다. 다만 seed별 1위가 달라지고, 주 Part PCK 및 CKA와 방향이 일치하지 않으므로 보조적인 국소화 신호로만 해석합니다.
- Frozen segmentation probe에서도 LG가 가장 높았던 결과와 함께 보면, 현재 CUB 프로토콜은 **iBKD가 LG/ALG보다 공간정보를 전반적으로 더 잘 보존한다는 가설을 지지하지 않습니다.**
- 이 결론은 잠긴 ResNet-50/224 scratch·현재 image-loader 조건의 guided 네 방법에 한정합니다. 이후 loader를 바꾸면 기존 결과는 그대로 보존하고 새 버전에서 모든 비교 방법을 동일 조건으로 다시 실행해야 합니다.
""",
        encoding="utf-8",
    )
    print(
        "[CUB_DIRECT_SPATIAL_SEED23_V2_CURATE_DONE] status=pass "
        f"issue739_checkpoints=40 encoder_seeds=1,2,3 report={report_dir}"
    )


if __name__ == "__main__":
    run(parse_args())
