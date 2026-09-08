#!/usr/bin/env python3
"""Audit an imported CUB ResNet-50/224 teacher and write tracked reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch

from ibkd_seg.phase1.run_cub_r50_teacher_full import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_VALIDATION_HASH,
    _validate_config,
    load_scientific_teacher,
)


TRACKED_SOURCE_ARTIFACTS = {
    "teacher_summary.json": "h200_teacher_summary.json",
    "teacher_history.csv": "h200_teacher_history.csv",
    "validation_split.json": "h200_validation_split.json",
    "protocol_snapshot.json": "h200_protocol_snapshot.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if source.suffix == ".csv":
        temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _verify_import(raw_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "pass":
        raise RuntimeError("raw import manifest did not pass")
    if manifest.get("experiment_kind") != "resnet50-v3-teacher":
        raise RuntimeError("wrong CUB experiment kind for teacher curator")
    if manifest.get("h200_issue_id") != "722":
        raise RuntimeError("expected H200 issue 722")
    files = manifest.get("files", [])
    if len(files) != 7 or manifest.get("checkpoint_count") != 1:
        raise RuntimeError("unexpected v3 teacher import file count")
    for item in files:
        path = raw_dir / item["path"]
        if not path.is_file():
            raise RuntimeError(f"imported artifact is missing: {path}")
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"imported artifact digest mismatch: {path}")


def _history_audit(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or [int(row["epoch"]) for row in rows] != list(range(1, 201)):
        raise RuntimeError("teacher history is not exactly epochs 1..200")
    numeric_columns = tuple(name for name in rows[0] if name != "epoch")
    if any(
        not math.isfinite(float(row[column]))
        for row in rows
        for column in numeric_columns
    ):
        raise RuntimeError("teacher history contains non-finite values")
    selected = max(rows, key=lambda row: float(row["validation_macro_top1"]))
    selected_epoch = int(selected["epoch"])
    if selected_epoch != summary["selected_epoch"]:
        raise RuntimeError("reported teacher checkpoint is not the history validation maximum")
    if not math.isclose(
        float(selected["validation_macro_top1"]),
        float(summary["selected_validation"]["macro_top1"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("selected validation metric disagrees with history")
    learning_rates = [float(row["learning_rate"]) for row in rows]
    if learning_rates[:5] != [0.0, 0.01, 0.02, 0.03, 0.04]:
        raise RuntimeError("teacher warm-up learning rates changed")
    if any(
        learning_rates[index] <= learning_rates[index + 1]
        for index in range(5, len(learning_rates) - 1)
    ):
        raise RuntimeError("teacher post-warm-up cosine learning rates are not decreasing")
    return {
        "status": "pass",
        "epoch_rows": 200,
        "epochs_are_exactly_1_through_200": True,
        "all_numeric_values_finite": True,
        "validation_maximum_recomputed": True,
        "earlier_epoch_tie_break_recomputed": True,
        "selected_epoch": selected_epoch,
        "selected_epoch_train_top1": float(selected["train_top1"]),
        "selected_epoch_train_loss": float(selected["train_loss"]),
        "warmup_learning_rates": learning_rates[:5],
        "post_warmup_learning_rate_strictly_decreases": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = args.raw_dir.resolve()
    report_dir = args.report_dir.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    manifest = _load_json(raw_dir / "artifact_manifest.json")
    _verify_import(raw_dir, manifest)

    summary = _load_json(raw_dir / "teacher_summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("scientific_result") is not True
        or summary.get("official_test_evaluations") != 1
        or summary.get("official_test_used_for_training_or_selection") is not False
        or summary.get("selected_checkpoint_strict_reloaded") is not True
    ):
        raise RuntimeError("CUB v3 teacher completion/test contract failed")
    if summary.get("full_config_sha256") != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("CUB v3 teacher full-config SHA-256 mismatch")
    if summary.get("split_manifest", {}).get("validation_image_ids_sha256") != EXPECTED_VALIDATION_HASH:
        raise RuntimeError("CUB v3 teacher validation split SHA-256 mismatch")

    config_path = (
        repository_root
        / "phase1/phase1_cub/configs/cub200_r50_224_b128_full_v3.json"
    )
    config = _load_json(config_path)
    _validate_config(config, config_path=config_path)
    snapshot = _load_json(raw_dir / "protocol_snapshot.json")
    if (
        snapshot.get("protocol_config_sha256") != EXPECTED_CONFIG_SHA256
        or snapshot.get("protocol") != config
    ):
        raise RuntimeError("H200 protocol snapshot differs from the locked repository config")

    split = _load_json(raw_dir / "validation_split.json")
    if (
        split.get("train_samples") != 5394
        or split.get("validation_samples") != 600
        or split.get("validation_image_ids_sha256") != EXPECTED_VALIDATION_HASH
    ):
        raise RuntimeError("H200 teacher split manifest mismatch")

    checkpoint_path = raw_dir / "teacher_best_validation.pt"
    model, metadata, checkpoint_hash, model_state_hash = load_scientific_teacher(
        checkpoint_path, device=torch.device("cpu")
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    all_finite = all(
        bool(torch.isfinite(tensor).all())
        for tensor in payload["model"].values()
        if tensor.is_floating_point()
    )
    if not all_finite:
        raise RuntimeError("CUB v3 teacher checkpoint contains non-finite tensors")
    if (
        checkpoint_hash != summary.get("checkpoint_sha256")
        or model_state_hash != summary.get("model_state_sha256")
        or metadata.get("selected_epoch") != summary.get("selected_epoch")
    ):
        raise RuntimeError("CUB v3 teacher checkpoint disagrees with its summary")
    del model, payload

    history_audit = _history_audit(raw_dir / "teacher_history.csv", summary)
    archive = manifest["source_archive"]
    audit = {
        "schema_version": 1,
        "status": "pass",
        "h200_issue_id": "722",
        "protocol_id": summary["protocol_id"],
        "protocol_config_sha256": EXPECTED_CONFIG_SHA256,
        "validation_image_ids_sha256": EXPECTED_VALIDATION_HASH,
        "archive_crc_verified": archive["all_member_crc_verified"],
        "archive_sha256": archive["sha256"],
        "checkpoint": {
            "path_under_ignored_raw_root": "teacher_best_validation.pt",
            "bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "model_state_sha256": model_state_hash,
            "safe_torch_load": "weights_only_true_for_tensor_finiteness_audit",
            "strict_load": True,
            "all_floating_tensors_finite": True,
            "frozen_after_load": True,
        },
        "history": history_audit,
        "official_test": {
            "evaluations": 1,
            "used_for_training_or_selection": False,
            "performed_after_validation_selection_and_strict_reload": True,
        },
        "all_checks_passed": True,
    }

    selected_validation = summary["selected_validation"]
    official_test = summary["official_test"]
    compact = {
        "schema_version": 1,
        "status": "complete_audited",
        "role": "single_teacher_shared_by_all_v3_guided_students",
        "protocol_id": summary["protocol_id"],
        "architecture": summary["architecture"],
        "initialization": summary["initialization"],
        "external_pretraining": summary["external_pretraining"],
        "input_size": summary["input_size"],
        "train_validation_test_counts": [5394, 600, 5794],
        "epochs": summary["epochs"],
        "seed": summary["seed"],
        "batch_size": summary["train_batch_size"],
        "selected_epoch": summary["selected_epoch"],
        "selected_validation": selected_validation,
        "official_test": official_test,
        "validation_to_test_macro_top1_gap_percentage_points": (
            selected_validation["macro_top1"] - official_test["macro_top1"]
        ),
        "training_and_test_seconds": summary["training_and_test_seconds"],
        "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
        "peak_cuda_memory_reserved_bytes": summary[
            "peak_cuda_memory_reserved_bytes"
        ],
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": model_state_hash,
        "protocol_config_sha256": EXPECTED_CONFIG_SHA256,
        "validation_image_ids_sha256": EXPECTED_VALIDATION_HASH,
        "runtime": summary["runtime"],
        "interpretation_scope": (
            "valid_v3_teacher_artifact_only; no student or frozen-probe claim yet"
        ),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in TRACKED_SOURCE_ARTIFACTS.items():
        _copy(raw_dir / source_name, report_dir / destination_name)
    _save_json(manifest, report_dir / "source_manifest.json")
    _save_json(audit, report_dir / "checkpoint_audit.json")
    _save_json(compact, report_dir / "summary.json")

    result_markdown = f"""# CUB ResNet-50/224 scratch Teacher v3 결과

상태: **H200 issue 722 완료 · 독립 감사 통과 · 후속 v3 공용 Teacher로 고정**

## 프로토콜

- TorchVision ResNet-50, `weights=None`, external pretraining 없음
- 입력 224×224, batch 128, seed 1, 200 epoch
- CUB train/validation/official test: 5,394 / 600 / 5,794
- validation macro top-1 최대 checkpoint 선택, 동률이면 이른 epoch
- 선택 완료 및 새 모델 strict reload 뒤 official test 정확히 1회

## 결과

| 항목 | 값 |
|---|---:|
| 선택 epoch | {summary['selected_epoch']} |
| 선택 epoch train top-1 | {history_audit['selected_epoch_train_top1']:.3f}% |
| validation top-1 / macro top-1 | {selected_validation['overall_top1']:.3f}% / {selected_validation['macro_top1']:.3f}% |
| official test top-1 / macro top-1 | {official_test['overall_top1']:.3f}% / {official_test['macro_top1']:.3f}% |
| official test top-5 | {official_test['top5']:.3f}% |
| validation−test macro gap | {compact['validation_to_test_macro_top1_gap_percentage_points']:.3f}%p |
| 총 실행시간 | {summary['training_and_test_seconds'] / 60:.2f}분 |
| peak allocated / reserved | {summary['peak_cuda_memory_bytes'] / 10**9:.3f} / {summary['peak_cuda_memory_reserved_bytes'] / 10**9:.3f} GB |

## 판정

Teacher 학습, validation-only 선택, checkpoint strict reload, official-test 1회 규칙이
모두 지켜졌습니다. Checkpoint 파일 SHA-256은 `{checkpoint_hash}`, model-state
SHA-256은 `{model_state_hash}`입니다. 이 두 hash를 후속 LG, ALG-w20, iBKD 실행에서
동시에 검사해 정확히 같은 Teacher 하나를 공유해야 합니다.

Scratch 소규모 fine-grained 분류라 선택 epoch의 train top-1과 validation/test 사이
간격이 큽니다. 이는 pretrained 결과와 비교할 수 없으며, 현재 결과만으로 iBKD의
공간정보 보존 성능을 판단할 수도 없습니다. 이 결과가 확정하는 것은 **v3 Teacher
artifact가 정상적으로 준비됐다**는 점까지입니다.

원본 checkpoint와 로그는 Git 이력이 아닌 ignored raw 경로와
[검증된 GitHub Release](checkpoint_release.json)에 보존하고, 이 폴더에는 검증
가능한 소형 결과와 hash만 추적합니다.
"""
    (report_dir / "RESULTS.md").write_text(result_markdown, encoding="utf-8")
    print(
        "[CUB_R50_TEACHER_CURATE_DONE] status=pass "
        f"selected_epoch={summary['selected_epoch']} "
        f"test_macro_top1={official_test['macro_top1']:.6f} "
        f"checkpoint_sha256={checkpoint_hash} report={report_dir}",
        flush=True,
    )
    return compact


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
