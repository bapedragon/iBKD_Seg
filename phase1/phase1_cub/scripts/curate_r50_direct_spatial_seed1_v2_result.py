#!/usr/bin/env python3
"""Audit and curate the issue 737 CUB direct-spatial seed-1 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch

from ibkd_seg.phase1.cub_direct_spatial import build_part_probe
from ibkd_seg.phase1.train_timing import state_dict_sha256


PROTOCOL_ID = "cub200_phase1_r50_224_b128_seed1_direct_spatial_full_v2"
CONFIG_SHA256 = "90f7dc92b7e1ad27b6a4a4b68e91bb5e72ea021389304d87dda5950fac8e6017"
VALIDATION_SHA256 = "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
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
        or manifest.get("experiment_kind") != "resnet50-direct-spatial-v2-seed1"
        or manifest.get("h200_issue_id") != "737"
        or manifest.get("checkpoint_count") != 20
        or manifest.get("imported_file_count_excluding_this_manifest") != 128
    ):
        raise RuntimeError("wrong or incomplete issue 737 import manifest")
    if len(manifest.get("files", [])) != 128:
        raise RuntimeError("issue 737 manifest file count differs")
    for item in manifest["files"]:
        path = raw_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"imported direct-spatial artifact digest mismatch: {path}")


def validate_completion(raw_dir: Path, summary: dict[str, Any]) -> None:
    expected_gate = {
        "attention_metric_rows": 4,
        "attention_official_test_evaluations": 4,
        "checkpoint_strict_loads": 4,
        "official_test_evaluations": 24,
        "part_probe_lr_candidates": 60,
        "part_probe_official_test_evaluations": 20,
        "part_probe_validation_selections": 20,
        "qualitative_pngs": 32,
        "spatial_cka_values": 48,
    }
    if (
        summary.get("status") != "complete"
        or summary.get("scientific_result") is not True
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("config_sha256") != CONFIG_SHA256
        or summary.get("student_batch_size") != 128
        or summary.get("encoder_seed") != 1
        or summary.get("independent_encoder_seed_n") != 1
        or summary.get("final_encoder_seed_inference") is not False
        or summary.get("variants") != list(VARIANTS)
        or summary.get("completion_gate") != expected_gate
    ):
        raise RuntimeError("issue 737 completion contract failed")
    sequence = load_json(raw_dir / "sequence_status.json")
    if (
        sequence.get("status") != "complete"
        or sequence.get("phase") != "complete"
        or sequence.get("failure") is not None
        or sequence.get("official_test_used_for_selection") is not False
        or sequence.get("completion_gate") != expected_gate
    ):
        raise RuntimeError("issue 737 sequence gate failed")
    dataset = load_json(raw_dir / "dataset_audit.json")
    split = dataset.get("split_manifest", {})
    train = dataset.get("part_annotation_validity_audit", {}).get("train", {})
    validation = dataset.get("part_annotation_validity_audit", {}).get("validation", {})
    test = dataset.get("part_annotation_validity_audit", {}).get("official_test", {})
    if (
        dataset.get("counts") != {"train": 5394, "validation": 600, "official_test": 5794}
        or split.get("validation_image_ids_sha256") != VALIDATION_SHA256
        or dataset.get("official_test_opened_after_all_20_validation_selections") is not True
        or train.get("official_visible_keypoints") != 64697
        or train.get("valid_visible_keypoints") != 64696
        or train.get("affected_image_ids") != [5007]
        or validation.get("valid_visible_keypoints") != 7164
        or test.get("valid_visible_keypoints") != 69546
    ):
        raise RuntimeError("issue 737 dataset or keypoint-validity audit failed")


def audit_reused_encoders(raw_dir: Path, seed1_report: Path) -> dict[str, Any]:
    source = load_json(raw_dir / "checkpoint_audit.json")
    seed1 = load_json(seed1_report / "classification_summary.json")
    expected = {
        cell["variant"]: cell["checkpoint_sha256"]
        for cell in seed1["cells"]
        if cell["batch_size"] == 128
    }
    students = source.get("students", [])
    actual = {row["variant"]: row["checkpoint_sha256"] for row in students}
    if (
        source.get("status") != "pass"
        or actual != expected
        or len(students) != 4
        or not all(
            row.get("strict_load") is True
            and row.get("all_floating_tensors_finite") is True
            and row.get("eval_mode") is True
            and row.get("trainable_parameters") == 0
            for row in students
        )
    ):
        raise RuntimeError("issue 737 reused encoder audit differs from issue 727")
    return {
        "source_h200_issue": 727,
        "encoder_checkpoint_count": 4,
        "hashes_match_audited_issue727_batch128_seed1": True,
        "strict_load_eval_frozen_on_h200": True,
        "checkpoints_reused_from_issue727_release_not_duplicated": True,
        "students": students,
    }


def audit_part_probes(raw_dir: Path, results: dict[str, Any]) -> dict[str, Any]:
    selections = results.get("selections", [])
    rows = results.get("official_test_rows", [])
    expected = {(variant, seed) for variant in VARIANTS for seed in range(1, 6)}
    by_key = {(row["variant"], int(row["probe_seed"])): row for row in selections}
    test_keys = {(row["variant"], int(row["probe_seed"])) for row in rows}
    if set(by_key) != expected or test_keys != expected or len(selections) != 20 or len(rows) != 20:
        raise RuntimeError("issue 737 part-probe matrix is incomplete")
    entries = []
    for variant, probe_seed in sorted(expected, key=lambda key: (VARIANTS.index(key[0]), key[1])):
        selection = by_key[(variant, probe_seed)]
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
            or metadata.get("encoder_seed") != 1
            or metadata.get("probe_seed") != probe_seed
            or metadata.get("official_test_evaluations_at_checkpoint_write") != 0
            or incompatible.missing_keys
            or incompatible.unexpected_keys
            or not finite_state(payload["probe"])
        ):
            raise RuntimeError(f"part-probe checkpoint audit failed: {variant} seed {probe_seed}")
        entries.append({
            "variant": variant,
            "encoder_seed": 1,
            "probe_seed": probe_seed,
            "path_under_ignored_raw_root": selection["checkpoint_relative_path"],
            "bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": digest,
            "model_state_sha256": state_dict_sha256(probe),
            "strict_load": True,
            "all_floating_tensors_finite": True,
        })
        del payload, probe
    return {
        "selected_part_probe_checkpoint_count": 20,
        "all_file_hashes_match": True,
        "all_strict_loads_passed": True,
        "all_floating_tensors_finite": True,
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seed1-report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    seed1_report = args.seed1_report_dir.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    manifest = load_json(raw_dir / "artifact_manifest.json")
    verify_import(raw_dir, manifest)
    summary = load_json(raw_dir / "summary.json")
    validate_completion(raw_dir, summary)
    if file_sha256(repository_root / "phase1/phase1_cub/configs/cub200_r50_224_b128_seed1_direct_spatial_full_v2.json") != CONFIG_SHA256:
        raise RuntimeError("repository direct-spatial v2 config does not match issue 737")

    part = load_json(raw_dir / "part_probe/results.json")
    cka = load_json(raw_dir / "spatial_cka/results.json")
    attention = load_json(raw_dir / "attention_gt/results.json")
    selection = load_json(raw_dir / "part_probe/selection_complete_before_test.json")
    if (
        part.get("status") != "complete"
        or part.get("official_test_evaluations") != 20
        or part.get("final_encoder_seed_inference") is not False
        or len(part.get("candidates", [])) != 60
        or selection.get("official_test_accessed") is not False
        or selection.get("selection_uses_official_test") is not False
        or cka.get("status") != "complete"
        or cka.get("official_test_used") is not False
        or len(cka.get("rows", [])) != 48
        or attention.get("status") != "complete"
        or attention.get("official_test_evaluations") != 4
        or len(attention.get("rows", [])) != 4
    ):
        raise RuntimeError("issue 737 metric-level completion contract failed")

    checkpoint_audit = {
        "schema_version": 1,
        "status": "pass",
        "safe_torch_load": "weights_only_true",
        "reused_encoders": audit_reused_encoders(raw_dir, seed1_report),
        "part_probes": audit_part_probes(raw_dir, part),
    }
    part_aggregates = part["aggregates"]
    cka_block11 = [row for row in cka["rows"] if row["student_block"] == 11]
    if len(cka_block11) != 4:
        raise RuntimeError("spatial CKA block-11 matrix is incomplete")
    curated = {
        "schema_version": 1,
        "status": "complete_audited_seed1_only",
        "protocol_id": PROTOCOL_ID,
        "batch_size": 128,
        "encoder_seed": 1,
        "independent_encoder_n": 1,
        "final_encoder_seed_inference": False,
        "part_pck_primary": part_aggregates,
        "spatial_cka_block11_secondary": cka_block11,
        "attention_gt_secondary": attention["rows"],
        "official_test_evaluations": 24,
        "seed2_3_settings_remain_locked_regardless_of_seed1_results": True,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, report_dir / "source_manifest.json")
    save_json(checkpoint_audit, report_dir / "checkpoint_audit.json")
    save_json(curated, report_dir / "direct_spatial_summary.json")
    for source, destination in {
        "summary.json": "h200_summary.json",
        "dataset_audit.json": "h200_dataset_audit.json",
        "sequence_status.json": "h200_sequence_status.json",
        "checkpoint_audit.json": "h200_checkpoint_audit.json",
        "part_probe/annotation_validity_audit.json": "part_annotation_validity_audit.json",
        "part_probe/candidates.csv": "part_probe_candidates.csv",
        "part_probe/official_test_results.csv": "part_probe_official_test_results.csv",
        "part_probe/results.json": "part_probe_results.json",
        "part_probe/selection_complete_before_test.json": "part_probe_selection_complete_before_test.json",
        "spatial_cka/results.csv": "spatial_cka_results.csv",
        "spatial_cka/results.json": "spatial_cka_results.json",
        "spatial_cka/layerwise_heatmap.png": "spatial_cka_layerwise_heatmap.png",
        "attention_gt/results.csv": "attention_gt_results.csv",
        "attention_gt/results.json": "attention_gt_results.json",
    }.items():
        copy_file(raw_dir / source, report_dir / destination)
    for image in sorted((raw_dir / "attention_gt/qualitative").glob("*.png")):
        copy_file(image, report_dir / "attention_qualitative" / image.name)

    pck = {row["variant"]: row for row in part_aggregates}
    cka11 = {row["variant"]: row for row in cka_block11}
    attn = {row["variant"]: row for row in attention["rows"]}
    table = "\n".join(
        f"| {LABELS[v]} | {100*pck[v]['test_micro_pck_at_0.1_probe_seed_mean']:.3f} ± "
        f"{100*pck[v]['test_micro_pck_at_0.1_probe_seed_sample_sd']:.3f}% | "
        f"{pck[v]['test_mean_normalized_error_probe_seed_mean']:.4f} | "
        f"{cka11[v]['centered_linear_cka']:.4f} | "
        f"{attn[v]['global_micro_patch_average_precision']:.4f} | "
        f"{attn[v]['pointing_game_peak_inside_mask']:.4f} | "
        f"{attn[v]['foreground_attention_mass_mean']:.4f} |"
        for v in VARIANTS
    )
    report_dir.joinpath("RESULTS.md").write_text(f"""# CUB ResNet-50/224 seed-1 직접 공간정보 진단 v2

상태: **H200 issue 737 완료 · 독립 감사 통과 · encoder seed 1 전용**

| 방법 | Part PCK@0.1 ↑ | 정규화 위치오차 ↓ | CKA block11 ↑ | Attention AP ↑ | Pointing ↑ | FG mass ↑ |
|---|---:|---:|---:|---:|---:|---:|
{table}

Part PCK가 주 직접지표이고 CKA와 attention–GT는 보조지표입니다. `±`는 동일
encoder에서 학습한 5개 part-probe seed의 sample SD이며 독립 encoder 반복이
아닙니다.

## 감사

- 실행시간: {summary['runtime']['elapsed_seconds'] / 60:.2f}분
- encoder 4개는 issue 727의 감사된 batch128 seed-1 checkpoint hash와 일치
- part-probe checkpoint 20개 모두 SHA-256, `weights_only=True`, strict load, 유한값 검사 통과
- validation으로 20개 선택을 끝낸 뒤 official test를 열었으며 test는 선택에 사용하지 않음
- Part probe 20회 + attention 4회 = official test 평가 총 24회
- CKA는 validation 600장만 사용해 official test를 사용하지 않음
- train의 공식 visible keypoint 64,697개 중 이미지 5007의 프레임 밖 1개만 규칙대로 제외;
  validation 7,164개와 test 69,546개는 모두 유효
- 32개 attention 정성 이미지와 layerwise CKA heatmap을 함께 보존

## 해석

- **주 지표 Part PCK는 LG가 28.174%로 가장 높습니다.** iBKD λ=0.25는
  22.393%로 ALG-w20(19.945%)보다 높지만 LG보다 낮습니다.
- CKA block11도 LG가 0.4420으로 가장 높고 iBKD λ=0.25는 0.3341입니다.
  따라서 teacher spatial feature와의 유사도 관점에서도 iBKD 우위가 아닙니다.
- Attention AP는 iBKD λ=0.25가 0.2765로 가장 높고, foreground mass는
  iBKD λ=0.5가 0.2206으로 가장 높습니다. 반면 pointing은 LG가 0.5247로
  가장 높아 attention 신호는 지표별로 엇갈립니다.
- 결론적으로 이 seed-1 직접진단은 iBKD λ=0.25에 일부 attention/ALG 대비
  국소화 신호는 보여주지만, **“LG/ALG보다 공간정보를 전반적으로 더 잘
  보존한다”는 핵심 주장을 지지하지 않습니다.**
- encoder seed가 하나뿐이므로 최종 통계 결론은 아닙니다. 사전 고정된 seed 2·3
  실행과 설정을 이 결과를 보고 바꾸면 안 됩니다.
""", encoding="utf-8")
    print(f"[CUB_DIRECT_SPATIAL_V2_CURATE_DONE] status=pass checkpoints=20 report={report_dir}")


if __name__ == "__main__":
    run(parse_args())
