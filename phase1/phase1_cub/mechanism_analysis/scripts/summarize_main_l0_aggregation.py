#!/usr/bin/env python3
"""Audit and summarize learned iBKD aggregation weights from main-L0 v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED: tuple[dict[str, Any], ...] = (
    {
        "lambda": 0.25,
        "seed": 1,
        "release": "issue727",
        "relative_path": "batch128/classification/students/cub_r50_224_ibkd_lambda_0.25_deit_tiny_b128_full_300ep_seed1/student_best_validation.pt",
        "bytes": 116028605,
        "sha256": "fb8c141be32e101f8ad91689ea809d8ecc8d9701ef7b843efdcd88c3b3ac7bd8",
    },
    {
        "lambda": 0.5,
        "seed": 1,
        "release": "issue727",
        "relative_path": "batch128/classification/students/cub_r50_224_ibkd_lambda_0.5_deit_tiny_b128_full_300ep_seed1/student_best_validation.pt",
        "bytes": 116025213,
        "sha256": "d9dd2e97e39783cb813467c2f263aa7b41bf4b5c8f6bf866dfa8e7afa60f4979",
    },
    {
        "lambda": 0.25,
        "seed": 2,
        "release": "issue730",
        "relative_path": "batch128_seed2/classification/students/cub_r50_224_ibkd_lambda_0.25_deit_tiny_b128_full_300ep_seed2/student_best_validation.pt",
        "bytes": 116028477,
        "sha256": "1c1bf6f6141f1efc0e21fe48e60be9e6a1ef3a126c60d81b6903d462fdc86410",
    },
    {
        "lambda": 0.5,
        "seed": 2,
        "release": "issue730",
        "relative_path": "batch128_seed2/classification/students/cub_r50_224_ibkd_lambda_0.5_deit_tiny_b128_full_300ep_seed2/student_best_validation.pt",
        "bytes": 116028669,
        "sha256": "bf3eade3a6ee2fca0b9d0e1335a2ab89468a0abb97f04620670a8b3a6c03cf09",
    },
    {
        "lambda": 0.25,
        "seed": 3,
        "release": "issue730",
        "relative_path": "batch128_seed3/classification/students/cub_r50_224_ibkd_lambda_0.25_deit_tiny_b128_full_300ep_seed3/student_best_validation.pt",
        "bytes": 116028925,
        "sha256": "87f73dfbc0cb9b8dc6475c61ecbd9dcf270269a3f746105aa6d2de3aa8fd2980",
    },
    {
        "lambda": 0.5,
        "seed": 3,
        "release": "issue730",
        "relative_path": "batch128_seed3/classification/students/cub_r50_224_ibkd_lambda_0.5_deit_tiny_b128_full_300ep_seed3/student_best_validation.pt",
        "bytes": 116028477,
        "sha256": "fea3f7a6512e0670db906877a4b764f2e7ab8cd86964b3a45b9b1c66663bd043",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty aggregation CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_path(
    entry: dict[str, Any], issue727_dir: Path, issue730_dir: Path
) -> Path:
    root = issue727_dir if entry["release"] == "issue727" else issue730_dir
    return root / str(entry["relative_path"])


def _row_metrics(probabilities: torch.Tensor) -> dict[str, Any]:
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum() / math.log(12)
    uniform = torch.full_like(probabilities, 1.0 / 12.0)
    return {
        "top_block": int(probabilities.argmax()),
        "top_probability": float(probabilities.max()),
        "minimum_probability": float(probabilities.min()),
        "normalized_entropy": float(entropy),
        "l1_distance_from_uniform": float((probabilities - uniform).abs().sum()),
        "expected_block_index": float(
            (probabilities * torch.arange(12, dtype=probabilities.dtype)).sum()
        ),
        "early_block_mass_0_3": float(probabilities[:4].sum()),
        "middle_block_mass_4_7": float(probabilities[4:8].sum()),
        "late_block_mass_8_11": float(probabilities[8:].sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    probabilities_by_lambda: dict[float, list[torch.Tensor]] = {0.25: [], 0.5: []}
    for entry in EXPECTED:
        path = _checkpoint_path(entry, args.issue727_dir, args.issue730_dir)
        if not path.is_file():
            raise FileNotFoundError(f"main-L0 checkpoint is missing: {path}")
        if path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"main-L0 checkpoint byte contract mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata", {})
        expected_metadata = {
            "dataset": "CUB-200-2011",
            "architecture": "deit_tiny_patch16_224",
            "method": "ibkd",
            "fusion_ratio_lambda": entry["lambda"],
            "batch_size": 128,
            "epochs": 300,
            "seed": entry["seed"],
            "validation_image_ids_sha256": "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854",
            "teacher_checkpoint_sha256": "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3",
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"main-L0 metadata mismatch {path}:{key}: "
                    f"{metadata.get(key)!r} != {value!r}"
                )
        guidance = payload.get("guidance")
        if not isinstance(guidance, dict) or "aggregation.weights" not in guidance:
            raise RuntimeError(f"learned aggregation weights are missing: {path}")
        logits = guidance["aggregation.weights"].detach().float().cpu()
        if logits.shape != (3, 12) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"invalid aggregation tensor: {path}")
        probabilities = torch.softmax(logits, dim=-1)
        probabilities_by_lambda[float(entry["lambda"])].append(probabilities)
        stages: list[dict[str, Any]] = []
        for stage in range(3):
            row_metrics = _row_metrics(probabilities[stage])
            stage_payload = {
                "teacher_stage": stage,
                "teacher_feature": ("layer2", "layer3", "layer4")[stage],
                "raw_logits": logits[stage].tolist(),
                "probabilities": probabilities[stage].tolist(),
                **row_metrics,
            }
            stages.append(stage_payload)
            csv_rows.append(
                {
                    "fusion_ratio_lambda": entry["lambda"],
                    "encoder_seed": entry["seed"],
                    "teacher_stage": stage,
                    "teacher_feature": stage_payload["teacher_feature"],
                    **{
                        f"block_{block}_probability": float(probabilities[stage, block])
                        for block in range(12)
                    },
                    **row_metrics,
                }
            )
        checkpoint_rows.append(
            {
                "fusion_ratio_lambda": entry["lambda"],
                "encoder_seed": entry["seed"],
                "source_h200_issue": 727 if entry["release"] == "issue727" else 730,
                "relative_path": entry["relative_path"],
                "checkpoint_bytes": entry["bytes"],
                "checkpoint_sha256": entry["sha256"],
                "selected_epoch": metadata["selected_epoch"],
                "stages": stages,
            }
        )

    aggregate_rows: list[dict[str, Any]] = []
    for fusion_ratio, tensors in probabilities_by_lambda.items():
        stacked = torch.stack(tensors)
        mean = stacked.mean(dim=0)
        sample_sd = stacked.std(dim=0, unbiased=True)
        for stage in range(3):
            metrics = _row_metrics(mean[stage])
            aggregate_rows.append(
                {
                    "fusion_ratio_lambda": fusion_ratio,
                    "teacher_stage": stage,
                    "teacher_feature": ("layer2", "layer3", "layer4")[stage],
                    "encoder_seeds": [1, 2, 3],
                    "mean_probabilities": mean[stage].tolist(),
                    "sample_sd_probabilities": sample_sd[stage].tolist(),
                    **metrics,
                }
            )

    summary = {
        "status": "pass",
        "analysis_id": "cub200_main_l0_v3_ibkd_aggregation_checkpoint_audit_v1",
        "scientific_role": "observational_mechanism_diagnostic_not_causal_evidence",
        "input_lineage_id": "main_l0_v3",
        "loader_profile": "l0_current_strong",
        "source_h200_issues": [727, 730],
        "checkpoints_audited": len(checkpoint_rows),
        "checkpoint_rows": checkpoint_rows,
        "three_seed_aggregates": aggregate_rows,
        "finding": (
            "All learned aggregation rows remain close to a uniform 1/12 mixture; "
            "lambda 0.25 teacher layer3 has the clearest but still modest late-block tilt."
        ),
        "causal_claim_allowed": False,
        "next_test": (
            "paired learned_all versus fixed_uniform_all, fixed_stage_match, and "
            "fixed_last retraining under the locked main-L0 protocol"
        ),
    }
    _write_json(summary, args.output_dir / "aggregation_checkpoint_audit.json")
    _write_csv(csv_rows, args.output_dir / "aggregation_per_checkpoint_stage.csv")
    _write_json(
        {
            "analysis_id": summary["analysis_id"],
            "source_h200_issues": [727, 730],
            "input_lineage_id": "main_l0_v3",
            "checkpoint_count": len(checkpoint_rows),
            "checkpoint_sha256": [entry["sha256"] for entry in EXPECTED],
        },
        args.output_dir / "source_manifest.json",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue727-dir", type=Path, required=True)
    parser.add_argument("--issue730-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        "[MAIN_L0_AGGREGATION_AUDIT] "
        f"status={result['status']} checkpoints={result['checkpoints_audited']} "
        "causal_claim=false",
        flush=True,
    )
