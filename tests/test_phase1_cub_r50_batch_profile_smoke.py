from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ibkd_seg.phase1.run_cub_r50_guided_smoke import (
    EXPECTED_BATCH_PROFILE_CONFIG_SHA256,
    EXPECTED_SCIENTIFIC_TEACHER_SHA256,
    EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
    _validate_batch_profile_configs,
)
from ibkd_seg.phase1.summarize_cub_r50_batch_profile_smoke import run as summarize
from ibkd_seg.phase1.train_timing import file_sha256, validate_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/cub200_r50_224_b64_b128_guided_smoke_v4.json"
)
FULL_CONFIG = (
    ROOT / "phase1/phase1_cub/configs/cub200_r50_224_b128_full_v3.json"
)
TEACHER_MANIFEST = (
    ROOT
    / "phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3/checkpoint_release.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/scripts/run_r50_224_guided_probe_smoke_b128_b64.sh"
)


def _synthetic_summary(batch_size: int) -> dict:
    classification = []
    probes = []
    peak = {}
    reserved = {}
    for index, variant in enumerate(EXPECTED_VARIANTS, start=1):
        method, fusion_ratio, _ = VARIANT_ARGUMENTS[variant]
        classification.append(
            {
                "variant": variant,
                "method": method,
                "fusion_ratio_lambda": fusion_ratio,
                "summary": {
                    "batch_size": batch_size,
                    "official_test_accessed": True,
                    "initial_student_state_sha256": "a" * 64,
                    "teacher_checkpoint_sha256": (
                        EXPECTED_SCIENTIFIC_TEACHER_SHA256
                    ),
                    "avg_epoch_seconds": 10.0 + index,
                    "epochs": [
                        {"validation": {"macro_top1": 1.0}},
                        {"validation": {"macro_top1": 2.0}},
                    ],
                    "official_test": {"macro_top1": 2.5},
                },
            }
        )
        probes.append(
            {
                "variant": variant,
                "selection": {"learning_rate": 0.1, "epoch": 2},
                "validation": {"input_224": {"mean_iou": 0.2}},
                "official_test": {"input_224": {"mean_iou": 0.21}},
                "official_test_evaluations": 1,
                "timing": {
                    "feature_cache_seconds": 5.0,
                    "probe_training_and_validation_seconds": 6.0,
                    "official_test_evaluation_seconds": 1.0,
                },
            }
        )
        peak[variant] = index * 100
        reserved[variant] = index * 200
    return {
        "status": "pass",
        "contracts": {"all_passed": True},
        "smoke_id": (
            "cub200_phase1_r50_224_b64_b128_guided_classification_to_probe_smoke_v4"
        ),
        "scientific_result": False,
        "batch_profile_mode": True,
        "student_batch_size": batch_size,
        "teacher": {
            "checkpoint_sha256": EXPECTED_SCIENTIFIC_TEACHER_SHA256,
            "model_state_sha256": EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
            "reused": True,
            "trained_in_smoke": False,
        },
        "classification": classification,
        "frozen_probe": probes,
        "capacity": {
            "requested_batch_size": batch_size,
            "peak_cuda_memory_bytes_by_student": peak,
            "peak_cuda_memory_reserved_bytes_by_student": reserved,
        },
        "timing": {
            "smoke_suite_seconds": 100.0,
            "rough_one_encoder_seed_four_variant_estimate": {
                "total_seconds": 1000.0 + batch_size,
            },
        },
    }


class Phase1CubR50BatchProfileSmokeTest(unittest.TestCase):
    def test_locked_config_validates_for_both_batches_and_bound_assets(self) -> None:
        smoke = json.loads(CONFIG.read_text(encoding="utf-8"))
        full = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
        for batch_size in (128, 64):
            _validate_batch_profile_configs(smoke, full, batch_size=batch_size)
        self.assertEqual(file_sha256(CONFIG), EXPECTED_BATCH_PROFILE_CONFIG_SHA256)
        self.assertEqual(
            file_sha256(FULL_CONFIG), smoke["full_protocol_config_sha256"]
        )
        self.assertEqual(
            file_sha256(TEACHER_MANIFEST),
            smoke["classification"]["teacher"]["release_manifest_sha256"],
        )

    def test_scientific_teacher_student_timing_is_valid_for_both_batches(self) -> None:
        for batch_size in (128, 64):
            for variant in EXPECTED_VARIANTS:
                method, fusion_ratio, warmup = VARIANT_ARGUMENTS[variant]
                with self.subTest(batch_size=batch_size, variant=variant):
                    validate_args(
                        argparse.Namespace(
                            dataset="cub",
                            kind="student",
                            method=method,
                            teacher_architecture="resnet50_224_scratch",
                            access_official_test=True,
                            batch_size=batch_size,
                            fusion_ratio=fusion_ratio,
                            teacher_checkpoint=Path("teacher.pt"),
                            scientific_cub_r50_teacher=True,
                            eval_batch_size=200,
                            num_workers=4,
                            seed=1,
                            alg_controller_warmup_epochs=(
                                20 if method == "alg" else 0
                            ),
                            save_student_checkpoint=True,
                        )
                    )
                    self.assertEqual(warmup, 20 if method != "lg" else 0)

    def test_h200_script_downloads_teacher_and_runs_both_batches(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("checkpoint_release.json", script)
        self.assertIn("for student_batch_size in 128 64", script)
        self.assertIn("--student-batch-size", script)
        self.assertIn("--teacher-checkpoint", script)
        self.assertIn("summarize_cub_r50_batch_profile_smoke", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_combined_summary_validates_and_lists_both_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for batch_size in (128, 64):
                batch_dir = root / f"batch{batch_size}"
                batch_dir.mkdir()
                (batch_dir / "combined_smoke_summary.json").write_text(
                    json.dumps(_synthetic_summary(batch_size)), encoding="utf-8"
                )
            output = StringIO()
            with redirect_stdout(output):
                summary = summarize(root)
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                [row["student_batch_size"] for row in summary["batches"]],
                [128, 64],
            )
            self.assertIn("batch_profiles=2/2", output.getvalue())
            self.assertTrue((root / "batch_profile_smoke_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
