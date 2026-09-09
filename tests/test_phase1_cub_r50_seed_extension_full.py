from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.run_cub_r50_batch_profile_full import (
    EXPECTED_SEED_EXTENSION_CONFIG_SHA256,
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
    _execution_profiles,
    _validate_seed_extension_config,
)
from ibkd_seg.phase1.train_full import validate_args
from ibkd_seg.phase1.train_timing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_s23_b64_s2_guided_full_v5.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/scripts/"
    "run_r50_224_guided_probe_full_b128_seeds2_3.sh"
)


def _student_args(batch_size: int, encoder_seed: int, variant: str) -> argparse.Namespace:
    method, fusion_ratio, _ = VARIANT_ARGUMENTS[variant]
    return argparse.Namespace(
        dataset="cub",
        kind="student",
        method=method,
        batch_size=batch_size,
        fusion_ratio=fusion_ratio,
        teacher_checkpoint=Path("teacher.pt"),
        teacher_architecture="resnet50_224_scratch",
        scientific_cub_r50_teacher=True,
        seed_extension_full=True,
        protocol_config=CONFIG,
        batch_profile_role=(
            "locked_v3_confirmatory_continuation"
            if batch_size == 128
            else "posthoc_exploratory_batch_sensitivity"
        ),
        eval_batch_size=200,
        num_workers=4,
        seed=encoder_seed,
        alg_controller_warmup_epochs=20 if method == "alg" else 0,
        posthoc_diagnostic_id=None,
    )


class Phase1CubR50SeedExtensionFullTest(unittest.TestCase):
    def test_locked_v5_config_validates_for_both_exact_partitions(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        for partition in (
            "batch128_encoder_seeds_2_3",
            "batch64_encoder_seed_2",
        ):
            _validate_seed_extension_config(
                config, CONFIG, partition=partition
            )
        self.assertEqual(
            file_sha256(CONFIG), EXPECTED_SEED_EXTENSION_CONFIG_SHA256
        )

    def test_partition_profiles_are_exact_and_disjoint(self) -> None:
        batch128 = _execution_profiles(
            argparse.Namespace(
                batch_profile_full=False,
                seed_extension_full=True,
                partition="batch128_encoder_seeds_2_3",
            )
        )
        batch64 = _execution_profiles(
            argparse.Namespace(
                batch_profile_full=False,
                seed_extension_full=True,
                partition="batch64_encoder_seed_2",
            )
        )
        self.assertEqual(batch128, ((128, 2), (128, 3)))
        self.assertEqual(batch64, ((64, 2),))
        self.assertFalse(set(batch128) & set(batch64))

    def test_full_student_cli_accepts_all_locked_v5_cells(self) -> None:
        for batch_size, encoder_seed in ((128, 2), (128, 3), (64, 2)):
            for variant in EXPECTED_VARIANTS:
                with self.subTest(
                    batch_size=batch_size,
                    encoder_seed=encoder_seed,
                    variant=variant,
                ):
                    validate_args(
                        _student_args(batch_size, encoder_seed, variant)
                    )

    def test_unplanned_v5_cell_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the locked v5"):
            validate_args(_student_args(64, 3, "lg"))

    def test_h200_entry_runs_only_batch128_seed2_and_seed3(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_r50_batch_profile_full", script)
        self.assertIn("--seed-extension-full", script)
        self.assertIn("--partition batch128_encoder_seeds_2_3", script)
        self.assertIn("teacher_best_validation.pt", script)
        self.assertIn("2>&1 | tee", script)
        self.assertNotIn("run_cub_r50_teacher_full", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
