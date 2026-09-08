from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.run_cub_r50_batch_profile_full import (
    EXPECTED_BATCHES,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
    _validate_config,
    aggregate_probe_seed_results,
)
from ibkd_seg.phase1.train_full import validate_args
from ibkd_seg.phase1.train_timing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_b64_guided_seed1_full_v4.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/scripts/"
    "run_r50_224_guided_probe_full_b128_b64_seed1.sh"
)


class Phase1CubR50BatchProfileFullTest(unittest.TestCase):
    def test_locked_profile_validates_and_has_exact_task_counts(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        _validate_config(config, CONFIG)
        self.assertEqual(file_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        self.assertEqual(tuple(config["result_scope"]["variants"]), EXPECTED_VARIANTS)
        self.assertEqual(tuple(config["result_scope"]["batch_order"]), EXPECTED_BATCHES)
        self.assertEqual(config["task_count"]["classification_students_total"], 8)
        self.assertEqual(config["task_count"]["probe_lr_candidates_total"], 120)
        self.assertEqual(config["task_count"]["selected_probes_total"], 40)
        self.assertFalse(
            config["result_scope"]["final_six_variant_three_seed_matrix_complete"]
        )

    def test_full_student_cli_contract_accepts_all_eight_runs(self) -> None:
        for batch_size in EXPECTED_BATCHES:
            for variant in EXPECTED_VARIANTS:
                method, fusion_ratio, _ = VARIANT_ARGUMENTS[variant]
                with self.subTest(batch_size=batch_size, variant=variant):
                    validate_args(
                        argparse.Namespace(
                            dataset="cub",
                            kind="student",
                            method=method,
                            batch_size=batch_size,
                            fusion_ratio=fusion_ratio,
                            teacher_checkpoint=Path("teacher.pt"),
                            teacher_architecture="resnet50_224_scratch",
                            scientific_cub_r50_teacher=True,
                            protocol_config=CONFIG,
                            batch_profile_role=(
                                "locked_v3_partial_cell"
                                if batch_size == 128
                                else "batch64_sensitivity"
                            ),
                            eval_batch_size=200,
                            num_workers=4,
                            seed=1,
                            alg_controller_warmup_epochs=(
                                20 if method == "alg" else 0
                            ),
                            posthoc_diagnostic_id=None,
                        )
                    )

    def test_batch64_cannot_be_mislabeled_as_confirmatory(self) -> None:
        with self.assertRaisesRegex(ValueError, "role disagree"):
            validate_args(
                argparse.Namespace(
                    dataset="cub",
                    kind="student",
                    method="lg",
                    batch_size=64,
                    fusion_ratio=None,
                    teacher_checkpoint=Path("teacher.pt"),
                    teacher_architecture="resnet50_224_scratch",
                    scientific_cub_r50_teacher=True,
                    protocol_config=CONFIG,
                    batch_profile_role="locked_v3_partial_cell",
                    eval_batch_size=200,
                    num_workers=4,
                    seed=1,
                    alg_controller_warmup_epochs=0,
                    posthoc_diagnostic_id=None,
                )
            )

    def test_probe_aggregate_reports_probe_seed_not_encoder_uncertainty(self) -> None:
        rows = [
            {
                "variant": variant,
                "test_input_224_mean_iou": 0.1 * index + 0.001 * probe_seed,
            }
            for index, variant in enumerate(EXPECTED_VARIANTS, start=1)
            for probe_seed in range(1, 6)
        ]
        aggregates = aggregate_probe_seed_results(rows)
        self.assertEqual(len(aggregates), 4)
        self.assertTrue(
            all(row["independent_encoder_n"] == 1 for row in aggregates)
        )
        self.assertTrue(
            all(
                row["encoder_seed_standard_deviation_not_estimable"]
                for row in aggregates
            )
        )

    def test_h200_entry_reuses_teacher_and_runs_one_combined_orchestrator(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_r50_batch_profile_full", script)
        self.assertIn("--batch-profile-full", script)
        self.assertIn("teacher_best_validation.pt", script)
        self.assertIn("2>&1 | tee", script)
        self.assertNotIn("run_cub_r50_teacher_full", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
