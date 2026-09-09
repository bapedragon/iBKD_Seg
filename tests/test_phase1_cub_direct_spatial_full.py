from __future__ import annotations

import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.run_cub_direct_spatial_full import (
    EXPECTED_CONFIG_SHA256,
    _select_candidate,
    _validate_config,
)
from ibkd_seg.phase1.train_timing import file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_seed1_direct_spatial_full_v1.json"
)
SHELL_PATH = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/scripts/"
    "run_r50_224_direct_spatial_full_b128_seed1.sh"
)
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "src/ibkd_seg/phase1/run_cub_direct_spatial_full.py"
)


class CubDirectSpatialSeed1FullProtocolTest(unittest.TestCase):
    def test_full_config_is_locked_and_valid(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(CONFIG_PATH), EXPECTED_CONFIG_SHA256)
        metric_config = _validate_config(config, CONFIG_PATH)
        self.assertTrue(config["scientific_result"])
        self.assertEqual(config["scope"]["encoder_seeds_in_this_run"], [1])
        self.assertEqual(config["scope"]["eventual_encoder_seeds"], [1, 2, 3])
        self.assertEqual(len(metric_config["checkpoint_inputs"]), 4)
        self.assertEqual(
            config["completion_gate"],
            {
                "checkpoint_strict_loads": 4,
                "part_probe_lr_candidates": 60,
                "part_probe_validation_selections": 20,
                "part_probe_official_test_evaluations": 20,
                "spatial_cka_values": 48,
                "attention_metric_rows": 4,
                "attention_official_test_evaluations": 4,
                "qualitative_pngs": 32,
                "official_test_evaluations": 24,
            },
        )

    def test_h200_entry_downloads_audited_inputs_and_runs_full_entrypoint(self) -> None:
        text = SHELL_PATH.read_text(encoding="utf-8")
        self.assertEqual(text.count("python -m ibkd_seg.phase1.release_asset"), 2)
        self.assertIn("python -m ibkd_seg.phase1.run_cub_direct_spatial_full", text)
        self.assertIn("--full", text)
        self.assertIn("--feature-batch-size 16", text)
        self.assertIn("--cka-batch-size 8", text)
        self.assertIn("--attention-batch-size 16", text)

    def test_official_test_is_opened_after_all_twenty_selections(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        selection_marker = 'output_dir / "part_probe/selection_complete_before_test.json"'
        test_open = "test_records, test_source = load_official_test_records("
        self.assertIn(selection_marker, text)
        self.assertIn(test_open, text)
        self.assertLess(text.index(selection_marker), text.index(test_open))
        self.assertIn("final_encoder_seed_inference=false", text)

    def test_candidate_selection_uses_pck_then_lower_lr_then_earlier_epoch(self) -> None:
        candidates = [
            {
                "learning_rate": 0.1,
                "best_epoch": 9,
                "best_validation": {"micro_pck_at_0.1": 0.4},
            },
            {
                "learning_rate": 0.03,
                "best_epoch": 8,
                "best_validation": {"micro_pck_at_0.1": 0.4},
            },
            {
                "learning_rate": 0.03,
                "best_epoch": 5,
                "best_validation": {"micro_pck_at_0.1": 0.4},
            },
        ]
        self.assertIs(_select_candidate(candidates), candidates[2])


if __name__ == "__main__":
    unittest.main()
