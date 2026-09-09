from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ibkd_seg.phase1.run_cub_r50_guided_smoke import (
    EXPECTED_SCIENTIFIC_TEACHER_SHA256,
    EXPECTED_SCIENTIFIC_TEACHER_STATE_SHA256,
    EXPECTED_SEED_EXTENSION_CONFIG_SHA256,
    EXPECTED_SEED_EXTENSION_FULL_CONFIG_SHA256,
    EXPECTED_VARIANTS,
    SEED_EXTENSION_SMOKE_ID,
    VARIANT_ARGUMENTS,
    _validate_seed_extension_configs,
)
from ibkd_seg.phase1.summarize_cub_r50_seed_extension_smoke import run as summarize
from ibkd_seg.phase1.train_timing import file_sha256, validate_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_s23_b64_s2_guided_smoke_v5.json"
)
FULL_CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_s23_b64_s2_guided_full_v5.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/scripts/"
    "run_r50_224_guided_probe_seed_extension_smoke.sh"
)


def _synthetic_summary(batch_size: int, encoder_seed: int) -> dict:
    initial_hash = ("b" if encoder_seed == 2 else "c") * 64
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
                    "seed": encoder_seed,
                    "official_test_accessed": True,
                    "initial_student_state_sha256": initial_hash,
                    "teacher_checkpoint_sha256": EXPECTED_SCIENTIFIC_TEACHER_SHA256,
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
                "encoder_seed": encoder_seed,
                "selection": {"learning_rate": 0.1, "epoch": 2},
                "validation": {"input_224": {"mean_iou": 0.2}},
                "official_test": {"input_224": {"mean_iou": 0.21}},
                "official_test_evaluations": 1,
            }
        )
        peak[variant] = index * 100
        reserved[variant] = index * 200
    return {
        "status": "pass",
        "contracts": {"all_passed": True},
        "smoke_id": SEED_EXTENSION_SMOKE_ID,
        "scientific_result": False,
        "batch_profile_mode": False,
        "seed_extension_mode": True,
        "student_batch_size": batch_size,
        "encoder_seed": encoder_seed,
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


class Phase1CubR50SeedExtensionSmokeTest(unittest.TestCase):
    def test_locked_configs_validate_for_exact_three_profiles(self) -> None:
        smoke = json.loads(CONFIG.read_text(encoding="utf-8"))
        full = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
        for batch_size, encoder_seed in ((128, 2), (128, 3), (64, 2)):
            _validate_seed_extension_configs(
                smoke,
                full,
                batch_size=batch_size,
                encoder_seed=encoder_seed,
            )
        self.assertEqual(file_sha256(CONFIG), EXPECTED_SEED_EXTENSION_CONFIG_SHA256)
        self.assertEqual(
            file_sha256(FULL_CONFIG), EXPECTED_SEED_EXTENSION_FULL_CONFIG_SHA256
        )

    def test_unplanned_batch_seed_pair_is_rejected(self) -> None:
        smoke = json.loads(CONFIG.read_text(encoding="utf-8"))
        full = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(RuntimeError, "requested_profile"):
            _validate_seed_extension_configs(
                smoke, full, batch_size=64, encoder_seed=3
            )

    def test_student_timing_accepts_each_new_profile(self) -> None:
        for batch_size, encoder_seed in ((128, 2), (128, 3), (64, 2)):
            for variant in EXPECTED_VARIANTS:
                method, fusion_ratio, _ = VARIANT_ARGUMENTS[variant]
                with self.subTest(
                    batch_size=batch_size,
                    encoder_seed=encoder_seed,
                    variant=variant,
                ):
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
                            seed_extension_smoke=True,
                            eval_batch_size=200,
                            num_workers=4,
                            seed=encoder_seed,
                            alg_controller_warmup_epochs=(
                                20 if method == "alg" else 0
                            ),
                            save_student_checkpoint=True,
                        )
                    )

    def test_h200_script_runs_exact_profiles_and_reuses_teacher(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn('"128:2" "128:3" "64:2"', script)
        self.assertIn("--encoder-seed", script)
        self.assertIn("summarize_cub_r50_seed_extension_smoke", script)
        self.assertIn('tee "${output_root}/run.log"', script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_combined_summary_lists_all_profiles_and_job_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for batch_size, encoder_seed in ((128, 2), (128, 3), (64, 2)):
                profile_dir = root / f"batch{batch_size}_seed{encoder_seed}"
                profile_dir.mkdir()
                (profile_dir / "combined_smoke_summary.json").write_text(
                    json.dumps(_synthetic_summary(batch_size, encoder_seed)),
                    encoding="utf-8",
                )
            output = StringIO()
            with redirect_stdout(output):
                summary = summarize(root)
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                [
                    (row["student_batch_size"], row["encoder_seed"])
                    for row in summary["profiles"]
                ],
                [(128, 2), (128, 3), (64, 2)],
            )
            self.assertIn("profiles=3/3", output.getvalue())
            self.assertTrue((root / "seed_extension_smoke_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
