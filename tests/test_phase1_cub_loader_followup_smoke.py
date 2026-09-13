from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.cub_loader_profiles import L2_CONSERVATIVE_SPATIAL
from ibkd_seg.phase1.run_cub_r50_guided_smoke import (
    EXPECTED_LOADER_FOLLOWUP_CONFIG_SHA256,
    EXPECTED_LOADER_FOLLOWUP_FULL_CONFIG_SHA256,
    EXPECTED_VARIANTS,
    LOADER_FOLLOWUP_SMOKE_ID,
    _validate_loader_followup_configs,
)
from ibkd_seg.phase1.train_timing import file_sha256, validate_args


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "phase1/phase1_cub/image_loader_experiment"
SMOKE_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_seed1_l2_guided_preliminary_smoke_v1.json"
)
FULL_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_seed1_l2_guided_preliminary_full_v1.json"
)
SCRIPT = (
    EXPERIMENT
    / "scripts/run_r50_224_l2_guided_preliminary_smoke_b128_seed1.sh"
)


def _timing_args() -> argparse.Namespace:
    return argparse.Namespace(
        dataset="cub",
        kind="student",
        method="lg",
        teacher_architecture="resnet50_224_scratch",
        access_official_test=True,
        batch_size=128,
        fusion_ratio=None,
        teacher_checkpoint=Path("teacher.pt"),
        scientific_cub_r50_teacher=True,
        seed_extension_smoke=False,
        eval_batch_size=200,
        num_workers=4,
        seed=1,
        alg_controller_warmup_epochs=0,
        save_student_checkpoint=True,
        cub_loader_profile=L2_CONSERVATIVE_SPATIAL,
        loader_pilot_smoke=False,
        loader_followup_smoke=True,
    )


class Phase1CubLoaderFollowupSmokeTest(unittest.TestCase):
    def test_locked_smoke_and_preliminary_full_contract(self) -> None:
        smoke = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        full = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
        _validate_loader_followup_configs(smoke, full)
        self.assertEqual(smoke["smoke_id"], LOADER_FOLLOWUP_SMOKE_ID)
        self.assertEqual(
            file_sha256(SMOKE_CONFIG),
            EXPECTED_LOADER_FOLLOWUP_CONFIG_SHA256,
        )
        self.assertEqual(
            file_sha256(FULL_CONFIG),
            EXPECTED_LOADER_FOLLOWUP_FULL_CONFIG_SHA256,
        )
        self.assertEqual(
            tuple(smoke["classification"]["variants"]), EXPECTED_VARIANTS
        )
        self.assertEqual(smoke["loader"]["profile"], L2_CONSERVATIVE_SPATIAL)
        self.assertEqual(full["classification"]["encoder_seeds"], [1])
        self.assertFalse(full["confirmatory_main_result"])
        self.assertFalse(full["full_execution_gate"]["current_state"] == "ready")

    def test_timing_mode_requires_exact_l2_official_test_scope(self) -> None:
        args = _timing_args()
        validate_args(args)

        args.cub_loader_profile = "l1_matched_weak"
        with self.assertRaisesRegex(ValueError, "locked L2 loader"):
            validate_args(args)

        args = _timing_args()
        args.access_official_test = False
        with self.assertRaisesRegex(ValueError, "official-test"):
            validate_args(args)

    def test_h200_entry_uses_audited_teacher_and_selected_loader_config(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_r50_guided_smoke", script)
        self.assertIn("l2_guided_preliminary_smoke_v1.json", script)
        self.assertIn("--student-batch-size 128", script)
        self.assertIn("--encoder-seed 1", script)
        self.assertIn('tee "${output_root}/run.log"', script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
