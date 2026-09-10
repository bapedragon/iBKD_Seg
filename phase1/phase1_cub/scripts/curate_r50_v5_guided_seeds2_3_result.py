#!/usr/bin/env python3
"""Audit issue 730 and combine it with audited seed 1 into the CUB guided 3-seed report."""

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
import torch.nn as nn

from ibkd_seg.phase1.models import create_student
from ibkd_seg.phase1.train_timing import state_dict_sha256


PROTOCOL_ID = "cub200_phase1_r50_224_guided_b128_s23_b64_s2_full_v5"
CONFIG_SHA256 = "f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9"
VALIDATION_SHA256 = "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
TEACHER_CHECKPOINT_SHA256 = "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
TEACHER_STATE_SHA256 = "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
VARIANTS = ("lg", "alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5")
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        or manifest.get("experiment_kind") != "resnet50-v5-guided-seeds2-3"
        or manifest.get("h200_issue_id") != "730"
        or manifest.get("checkpoint_count") != 48
        or manifest.get("imported_file_count_excluding_this_manifest") != 77
    ):
        raise RuntimeError("wrong or incomplete issue 730 import manifest")
    for item in manifest.get("files", []):
        path = raw_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"imported artifact digest mismatch: {path}")


def validate_completion(raw_dir: Path, combined: dict[str, Any]) -> None:
    expected_counts = {
        "classification_official_test_evaluations": 8,
        "classification_students": 8,
        "new_checkpoints": 48,
        "probe_lr_candidates": 120,
        "probe_official_test_evaluations": 40,
        "selected_probes": 40,
        "teacher_reused": 1,
    }
    teacher = combined.get("teacher", {})
    if (
        combined.get("status") != "complete"
        or combined.get("scientific_result") is not True
        or combined.get("protocol_id") != PROTOCOL_ID
        or combined.get("config_sha256") != CONFIG_SHA256
        or combined.get("encoder_seeds") != [2, 3]
        or combined.get("variants") != list(VARIANTS)
        or combined.get("counts") != expected_counts
        or combined.get("partition") != "batch128_encoder_seeds_2_3"
        or teacher.get("source_h200_issue") != 722
        or teacher.get("checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256
        or teacher.get("model_state_sha256") != TEACHER_STATE_SHA256
    ):
        raise RuntimeError("issue 730 combined completion contract failed")
    sequence = load_json(raw_dir / "sequence_status.json")
    expected = {
        "classification_complete": 8,
        "classification_expected": 8,
        "probe_candidates_complete": 120,
        "probe_candidates_expected": 120,
        "probe_selections_complete": 40,
        "probe_selections_expected": 40,
        "official_test_evaluations_complete": 40,
        "official_test_evaluations_expected": 40,
    }
    if (
        sequence.get("status") != "complete"
        or sequence.get("failure") is not None
        or any(sequence.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError("issue 730 sequence is incomplete")
    dataset = load_json(raw_dir / "dataset_audit.json")
    if (
        dataset.get("status") != "pass"
        or dataset.get("counts") != {"train": 5394, "validation": 600, "test": 5794}
        or dataset.get("split_manifest", {}).get("validation_image_ids_sha256")
        != VALIDATION_SHA256
    ):
        raise RuntimeError("issue 730 dataset/split audit failed")


def parse_classification(raw_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for seed in (2, 3):
        with (raw_dir / f"batch128_seed{seed}/classification_results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            sources = list(csv.DictReader(handle))
        if len(sources) != 4:
            raise RuntimeError(f"seed {seed} classification does not have four rows")
        for source in sources:
            variant = source["variant"]
            row = {
                "variant": variant,
                "encoder_seed": int(source["encoder_seed"]),
                "selected_epoch": int(source["selected_epoch"]),
                "validation_macro_top1": float(source["validation_macro_top1"]),
                "test_macro_top1": float(source["test_macro_top1"]),
                "test_overall_top1": float(source["test_overall_top1"]),
                "test_top5": float(source["test_top5"]),
                "checkpoint_sha256": source["checkpoint_sha256"],
            }
            if (
                variant not in VARIANTS
                or row["encoder_seed"] != seed
                or source["batch_size"] != "128"
                or source["confirmatory_main_result"] != "True"
                or any(not math.isfinite(row[key]) for key in ("validation_macro_top1", "test_macro_top1", "test_overall_top1", "test_top5"))
            ):
                raise RuntimeError(f"invalid classification row for seed {seed}: {variant}")
            run_name = f"cub_r50_224_{variant}_deit_tiny_b128_full_300ep_seed{seed}"
            summary_path = (
                raw_dir
                / f"batch128_seed{seed}/classification/students"
                / run_name
                / "summary.json"
            )
            if not summary_path.is_file():
                raise RuntimeError(f"student summary is missing: seed={seed} variant={variant}")
            summary = load_json(summary_path)
            checkpoint = summary_path.parent / "student_best_validation.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = create_student(num_classes=200, drop_path_rate=0.1)
            incompatible = model.load_state_dict(payload["student"], strict=True)
            state_hash = state_dict_sha256(model)
            if (
                summary.get("status") != "complete"
                or summary.get("protocol_config_sha256") != CONFIG_SHA256
                or summary.get("seed") != seed
                or summary.get("official_test_evaluations") != 1
                or summary.get("official_test_used_for_training_or_selection") is not False
                or file_sha256(checkpoint) != row["checkpoint_sha256"]
                or state_hash != summary.get("student_state_sha256")
                or incompatible.missing_keys
                or incompatible.unexpected_keys
                or not finite_state(payload["student"])
            ):
                raise RuntimeError(f"student checkpoint audit failed: seed={seed} variant={variant}")
            audit.append({
                "kind": "classification_encoder",
                "variant": variant,
                "encoder_seed": seed,
                "path_under_ignored_raw_root": checkpoint.relative_to(raw_dir).as_posix(),
                "bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "model_state_sha256": state_hash,
                "strict_load": True,
                "all_floating_tensors_finite": True,
            })
            rows.append(row)
            del payload, model
    if {(row["variant"], row["encoder_seed"]) for row in rows} != {
        (variant, seed) for variant in VARIANTS for seed in (2, 3)
    }:
        raise RuntimeError("classification seed/variant matrix is incomplete")
    return rows, audit


def parse_probes(raw_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_encoder: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for seed in (2, 3):
        with (raw_dir / f"batch128_seed{seed}/probe/raw_results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 20:
            raise RuntimeError(f"seed {seed} selected probe CSV does not have 20 rows")
        selection = load_json(raw_dir / f"batch128_seed{seed}/probe/selection_complete_before_test.json")
        if (
            selection.get("status") != "complete"
            or selection.get("config_sha256") != CONFIG_SHA256
            or selection.get("official_test_masks_accessed") is not False
            or selection.get("completed_probe_selections") != 20
        ):
            raise RuntimeError(f"seed {seed} selection-before-test contract failed")
        for variant in VARIANTS:
            subset = [row for row in rows if row["variant"] == variant]
            if sorted(int(row["probe_seed"]) for row in subset) != [1, 2, 3, 4, 5]:
                raise RuntimeError(f"seed {seed} {variant} probe seeds are incomplete")
            values: list[float] = []
            for row in subset:
                probe_seed = int(row["probe_seed"])
                value = float(row["test_input_224_mean_iou"])
                checkpoint = raw_dir / f"batch128_seed{seed}/probe/checkpoints/{variant}/probe_seed{probe_seed}_best_validation.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                probe = nn.Conv2d(192, 2, kernel_size=1, bias=True)
                incompatible = probe.load_state_dict(payload["model"], strict=True)
                digest = file_sha256(checkpoint)
                if (
                    row["official_test_evaluations"] != "1"
                    or row["scientific_result"] != "True"
                    or digest != row["probe_checkpoint_sha256"]
                    or incompatible.missing_keys
                    or incompatible.unexpected_keys
                    or not finite_state(payload["model"])
                ):
                    raise RuntimeError(f"probe checkpoint audit failed: seed={seed} variant={variant} probe={probe_seed}")
                audit.append({
                    "kind": "selected_probe",
                    "variant": variant,
                    "encoder_seed": seed,
                    "probe_seed": probe_seed,
                    "path_under_ignored_raw_root": checkpoint.relative_to(raw_dir).as_posix(),
                    "bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": digest,
                    "model_state_sha256": state_dict_sha256(probe),
                    "strict_load": True,
                    "all_floating_tensors_finite": True,
                })
                values.append(value)
                del payload, probe
            per_encoder.append({
                "variant": variant,
                "encoder_seed": seed,
                "probe_seeds_per_encoder": 5,
                "test_input_224_mean_iou_probe_seed_values": values,
                "test_input_224_mean_iou_probe_seed_mean": statistics.mean(values),
                "test_input_224_mean_iou_probe_seed_sample_sd": statistics.stdev(values),
            })
    return per_encoder, audit


def aggregate(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANTS:
        values = [float(row[metric]) for row in rows if row["variant"] == variant]
        if len(values) != 3:
            raise RuntimeError(f"expected three encoder seeds for {variant} {metric}")
        output.append({
            "variant": variant,
            "encoder_seed_values": values,
            "independent_encoder_n": 3,
            "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values),
        })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed1-report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    seed1_dir = args.seed1_report_dir.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    manifest = load_json(raw_dir / "artifact_manifest.json")
    verify_import(raw_dir, manifest)
    combined = load_json(raw_dir / "combined_full_summary.json")
    validate_completion(raw_dir, combined)
    if file_sha256(repository_root / "phase1/phase1_cub/configs/cub200_r50_224_b128_s23_b64_s2_guided_full_v5.json") != CONFIG_SHA256:
        raise RuntimeError("repository v5 config does not match issue 730")

    new_class, class_audit = parse_classification(raw_dir)
    new_probe, probe_audit = parse_probes(raw_dir)
    seed1_class = load_json(seed1_dir / "classification_summary.json")
    seed1_probe = load_json(seed1_dir / "probe_summary.json")
    class_rows = [
        {key: cell[key] for key in ("variant", "encoder_seed", "selected_epoch", "validation_macro_top1", "test_macro_top1", "test_overall_top1", "test_top5", "checkpoint_sha256")}
        for cell in seed1_class["cells"] if cell["batch_size"] == 128
    ] + new_class
    probe_rows = [
        {key: cell[key] for key in ("variant", "encoder_seed", "probe_seeds_per_encoder", "test_input_224_mean_iou_probe_seed_values", "test_input_224_mean_iou_probe_seed_mean", "test_input_224_mean_iou_probe_seed_sample_sd")}
        for cell in seed1_probe["cells"] if cell["batch_size"] == 128
    ] + new_probe
    class_rows.sort(key=lambda row: (VARIANTS.index(row["variant"]), row["encoder_seed"]))
    probe_rows.sort(key=lambda row: (VARIANTS.index(row["variant"]), row["encoder_seed"]))

    h200_manifest = load_json(raw_dir / "checkpoint_manifest.json")
    expected_hashes = {entry["sha256"] for entry in h200_manifest["entries"] if entry["kind"] != "external_shared_teacher"}
    audited = class_audit + probe_audit
    if len(audited) != 48 or {entry["checkpoint_sha256"] for entry in audited} != expected_hashes:
        raise RuntimeError("issue 730 checkpoint manifest and independent audit differ")

    class_summary = {
        "schema_version": 1,
        "status": "complete_audited_guided_3seed_v5",
        "protocol_id": PROTOCOL_ID,
        "batch_size": 128,
        "independent_unit": "encoder_seed",
        "encoder_seeds": [1, 2, 3],
        "independent_n": 3,
        "cells": class_rows,
        "aggregates": aggregate(class_rows, "test_macro_top1"),
        "official_test_evaluations": 12,
        "final_six_method_three_seed_matrix_complete": False,
    }
    probe_summary = {
        "schema_version": 1,
        "status": "complete_audited_guided_3seed_v5",
        "protocol_id": PROTOCOL_ID,
        "batch_size": 128,
        "primary_metric": "official_test_input_224_two_class_mean_iou",
        "independent_unit": "encoder_seed",
        "encoder_seeds": [1, 2, 3],
        "independent_n": 3,
        "probe_seeds_per_encoder": 5,
        "probe_seed_is_not_independent_replication": True,
        "cells": probe_rows,
        "aggregates": aggregate(probe_rows, "test_input_224_mean_iou_probe_seed_mean"),
        "official_test_evaluations": 60,
        "final_six_method_three_seed_matrix_complete": False,
    }
    checkpoint_audit = {
        "schema_version": 1,
        "status": "pass",
        "issue730_new_checkpoint_count": 48,
        "classification_encoder_checkpoint_count": 8,
        "selected_probe_checkpoint_count": 40,
        "all_file_hashes_match": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "seed1_source": {
            "h200_issue": 727,
            "report": "../resnet50_224_b128_b64_guided_seed1_v4",
            "checkpoint_release_reused_not_duplicated": True,
        },
        "entries": sorted(audited, key=lambda row: (row["kind"], VARIANTS.index(row["variant"]), row["encoder_seed"], row.get("probe_seed", 0))),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, report_dir / "source_manifest.json")
    save_json(checkpoint_audit, report_dir / "checkpoint_audit.json")
    save_json(class_summary, report_dir / "classification_summary.json")
    save_json(probe_summary, report_dir / "probe_summary.json")
    write_csv(class_rows, report_dir / "classification_per_encoder_seed.csv")
    write_csv([{k: v for k, v in row.items() if k != "test_input_224_mean_iou_probe_seed_values"} for row in probe_rows], report_dir / "probe_per_encoder_seed.csv")
    for source, destination in {
        "combined_full_summary.json": "h200_combined_full_summary.json",
        "dataset_audit.json": "h200_dataset_audit.json",
        "sequence_status.json": "h200_sequence_status.json",
        "checkpoint_manifest.json": "h200_checkpoint_manifest.json",
    }.items():
        copy_file(raw_dir / source, report_dir / destination)
    for seed in (2, 3):
        for source, destination in {
            "classification_results.csv": "classification_results.csv",
            "profile_full_summary.json": "profile_full_summary.json",
            "probe/results.json": "probe_results.json",
            "probe/selection_complete_before_test.json": "probe_selection_complete_before_test.json",
        }.items():
            copy_file(raw_dir / f"batch128_seed{seed}/{source}", report_dir / f"h200_seed{seed}_{destination}")

    class_agg = {row["variant"]: row for row in class_summary["aggregates"]}
    probe_agg = {row["variant"]: row for row in probe_summary["aggregates"]}
    table = "\n".join(
        f"| {LABELS[v]} | {class_agg[v]['mean']:.3f} ± {class_agg[v]['sample_standard_deviation']:.3f}% | "
        f"{100 * probe_agg[v]['mean']:.3f} ± {100 * probe_agg[v]['sample_standard_deviation']:.3f}% |"
        for v in VARIANTS
    )
    p_i025_lg = 100 * (probe_agg["ibkd_lambda_0.25"]["mean"] - probe_agg["lg"]["mean"])
    p_i025_alg = 100 * (probe_agg["ibkd_lambda_0.25"]["mean"] - probe_agg["alg_warmup20"]["mean"])
    report_dir.joinpath("RESULTS.md").write_text(f"""# CUB ResNet-50/224 batch128 guided 3-seed 결과

상태: **H200 issue 730 완료 · issue 727 seed 1과 결합 · 독립 감사 통과**

LG, ALG-w20, iBKD λ=0.25/0.5의 encoder seed `[1,2,3]`가 모두 채워졌습니다.
각 frozen probe 셀은 5개 probe seed를 먼저 평균했고, 아래 `±`는 그 셀 평균을
독립 encoder seed 3개에 대해 계산한 sample SD입니다.

| 방법 | 분류 test macro top-1 (3 encoder seeds) | frozen probe mIoU (3 encoder seeds) |
|---|---:|---:|
{table}

## 감사 및 범위

- issue 730 실행시간: {combined['elapsed_seconds'] / 3600:.2f}시간
- seed 2·3 분류 8/8, probe 후보 120/120, validation 선택 40/40, official test 40/40
- issue 730의 새 checkpoint 48개 모두 SHA-256, `weights_only=True`, strict load, 유한값 검사 통과
- seed 1 checkpoint는 issue 727 Release를 재사용하며 이 Release에 중복하지 않음
- 분류 official test 총 12회(4방법×3seed), probe official test 총 60회(4×3×5)

## 해석

- frozen probe 평균은 **LG가 가장 높습니다**.
- iBKD λ=0.25는 LG 대비 `{p_i025_lg:+.3f}%p`, ALG-w20 대비 `{p_i025_alg:+.3f}%p`입니다.
- 따라서 이 CUB guided 4방법 블록은 “iBKD가 LG/ALG보다 공간정보를 더 보존한다”는
  가설을 지지하지 않습니다. seed 1에서 보였던 iBKD λ=0.25의 ALG 대비 우위도
  seed 2·3을 포함하면 유지되지 않습니다.
- 이는 **guided 4방법의 3-seed 결과**입니다. Vanilla/KD가 아직 없으므로 최종
  6방법×3seed 매트릭스가 완성됐다고 표기하면 안 됩니다.
""", encoding="utf-8")
    print(f"[CUB_R50_V5_CURATE_DONE] status=pass checkpoints=48 report={report_dir}")


if __name__ == "__main__":
    run(parse_args())
