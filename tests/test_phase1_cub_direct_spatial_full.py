from __future__ import annotations

import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.run_cub_direct_spatial_full import (
    EXPECTED_CONFIG_SHA256,
    SEED23_CONFIG_SHA256,
    _part_aggregates,
    _select_candidate,
    _validate_config,
)
from ibkd_seg.phase1.train_timing import file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_seed1_direct_spatial_full_v2.json"
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
SEED23_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_seed2_3_direct_spatial_full_v2.json"
)
SEED23_SHELL_PATH = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/scripts/"
    "run_r50_224_direct_spatial_full_b128_seeds2_3.sh"
)


class CubDirectSpatialSeed1FullProtocolTest(unittest.TestCase):
    def test_full_config_is_locked_and_valid(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(CONFIG_PATH), EXPECTED_CONFIG_SHA256)
        metric_config = _validate_config(config, CONFIG_PATH)
        self.assertTrue(config["scientific_result"])
        self.assertEqual(config["scope"]["encoder_seeds_in_this_run"], [1])
        self.assertEqual(config["scope"]["eventual_encoder_seeds"], [1, 2, 3])
        self.assertFalse(
            config["dataset"]["part_coordinate_validity"]["coordinate_clipping"]
        )
        self.assertEqual(
            config["dataset"]["part_coordinate_validity"]["validity"],
            "official_visible_and_in_image_bounds",
        )
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
        self.assertIn("direct_spatial_full_v2", text)
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

    def test_seed23_full_config_is_locked_and_doubles_seed1_gate(self) -> None:
        config = json.loads(SEED23_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(SEED23_CONFIG_PATH), SEED23_CONFIG_SHA256)
        metric_config = _validate_config(config, SEED23_CONFIG_PATH)
        self.assertTrue(config["scientific_result"])
        self.assertEqual(config["scope"]["encoder_seeds_in_this_run"], [2, 3])
        self.assertFalse(
            config["protocol_inheritance"][
                "method_lambda_metric_or_protocol_changed_after_seed1_or_smoke"
            ]
        )
        self.assertEqual(len(metric_config["checkpoint_inputs"]), 8)
        self.assertEqual(
            config["completion_gate"],
            {
                "checkpoint_strict_loads": 8,
                "part_probe_lr_candidates": 120,
                "part_probe_validation_selections": 40,
                "part_probe_official_test_evaluations": 40,
                "spatial_cka_values": 96,
                "attention_metric_rows": 8,
                "attention_official_test_evaluations": 8,
                "qualitative_pngs": 64,
                "official_test_evaluations": 48,
            },
        )

    def test_seed23_h200_entry_downloads_issue730_and_runs_full_entrypoint(self) -> None:
        text = SEED23_SHELL_PATH.read_text(encoding="utf-8")
        self.assertEqual(text.count("python -m ibkd_seg.phase1.release_asset"), 2)
        self.assertIn("resnet50_224_b128_guided_3seed_v5/artifact_release.json", text)
        self.assertIn("python -m ibkd_seg.phase1.run_cub_direct_spatial_full", text)
        self.assertIn("seed2_3_direct_spatial_full_v2.json", text)
        self.assertIn("--full", text)
        self.assertIn("--feature-batch-size 16", text)
        self.assertIn("--cka-batch-size 8", text)
        self.assertIn("--attention-batch-size 16", text)

    def test_runner_passes_seed_and_metric_config_to_shared_loaders(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("config=metric_config"), 2)
        self.assertGreaterEqual(text.count("encoder_seed=encoder_seed"), 4)
        self.assertIn("expected_selections = expected_gate", text)
        self.assertIn("DIRECT_SPATIAL_FULL_SEED23_DONE", text)

    def test_part_aggregates_keep_encoder_and_probe_seeds_separate(self) -> None:
        rows = []
        for encoder_seed in (2, 3):
            for variant in (
                "lg",
                "alg_warmup20",
                "ibkd_lambda_0.25",
                "ibkd_lambda_0.5",
            ):
                for probe_seed in (1, 2, 3, 4, 5):
                    rows.append(
                        {
                            "variant": variant,
                            "encoder_seed": encoder_seed,
                            "probe_seed": probe_seed,
                            "test": {
                                "micro_pck_at_0.1": encoder_seed / 10 + probe_seed / 100,
                                "mean_normalized_localization_error": probe_seed / 10,
                            },
                        }
                    )
        aggregates = _part_aggregates(rows)
        self.assertEqual(len(aggregates), 8)
        self.assertEqual(
            [(row["encoder_seed"], row["variant"]) for row in aggregates],
            [
                (seed, variant)
                for seed in (2, 3)
                for variant in (
                    "lg",
                    "alg_warmup20",
                    "ibkd_lambda_0.25",
                    "ibkd_lambda_0.5",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
