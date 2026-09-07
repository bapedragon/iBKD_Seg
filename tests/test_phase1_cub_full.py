from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.cub_data import ARCHIVE_BYTES, ARCHIVE_SHA256
from ibkd_seg.phase1.cub_probe_data import (
    SEGMENTATION_ARCHIVE_BYTES,
    SEGMENTATION_ARCHIVE_SHA256,
)
from ibkd_seg.phase1.run_cub_full_shard import (
    EXPECTED_SHARDS,
    EXPECTED_VARIANTS,
    _validate_config,
)
from ibkd_seg.phase1.train_full import validate_args as validate_full_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT / "phase1/phase1_cub/configs/cub200_b128_full_v1.json"
)
SCRIPT_A = REPOSITORY_ROOT / "phase1/phase1_cub/scripts/run_full_shard_a_b128.sh"
SCRIPT_B = REPOSITORY_ROOT / "phase1/phase1_cub/scripts/run_full_shard_b_b128.sh"


def _cub_full_args(**updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dataset": "cub",
        "kind": "student",
        "method": "alg",
        "batch_size": 128,
        "fusion_ratio": None,
        "teacher_checkpoint": Path("teacher.pt"),
        "eval_batch_size": 200,
        "num_workers": 4,
        "seed": 1,
        "alg_controller_warmup_epochs": 20,
        "posthoc_diagnostic_id": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class Phase1CubFullProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_locked_config_and_balanced_shards(self) -> None:
        _validate_config(self.config)
        self.assertEqual(
            sorted(variant for values in EXPECTED_SHARDS.values() for variant in values),
            sorted(EXPECTED_VARIANTS),
        )
        self.assertEqual(len(EXPECTED_SHARDS["a"]), 3)
        self.assertEqual(len(EXPECTED_SHARDS["b"]), 3)
        self.assertFalse(set(EXPECTED_SHARDS["a"]) & set(EXPECTED_SHARDS["b"]))
        self.assertEqual(
            self.config["execution"]["shards"]["a"]["probe_lr_candidates"],
            135,
        )
        self.assertEqual(
            self.config["execution"]["shards"]["b"]["selected_probes"],
            45,
        )

    def test_archive_identities_are_fully_locked(self) -> None:
        image = self.config["dataset"]["image_archive"]
        segmentation = self.config["dataset"]["segmentation_archive"]
        self.assertEqual(image["expected_bytes"], ARCHIVE_BYTES)
        self.assertEqual(image["sha256"], ARCHIVE_SHA256)
        self.assertEqual(segmentation["expected_bytes"], SEGMENTATION_ARCHIVE_BYTES)
        self.assertEqual(segmentation["sha256"], SEGMENTATION_ARCHIVE_SHA256)

    def test_cub_alg_warmup20_is_confirmatory_and_warmup0_is_rejected(self) -> None:
        validate_full_args(_cub_full_args())
        with self.assertRaisesRegex(ValueError, "fixed to warm-up 20"):
            validate_full_args(_cub_full_args(alg_controller_warmup_epochs=0))
        with self.assertRaisesRegex(ValueError, "not a Pet diagnostic"):
            validate_full_args(_cub_full_args(posthoc_diagnostic_id="pet-only"))

    def test_both_h200_scripts_run_one_distinct_full_shard(self) -> None:
        for shard, script_path in (("a", SCRIPT_A), ("b", SCRIPT_B)):
            script = script_path.read_text(encoding="utf-8")
            self.assertIn("ibkd_seg.phase1.run_cub_full_shard", script)
            self.assertIn("--full-shard", script)
            self.assertIn(f"--shard {shard}", script)
            self.assertIn("/app/output/phase1_cub_b128_full_v1_shard_", script)
            self.assertTrue(script_path.stat().st_mode & 0o111)

    def test_full_probe_matches_pet_repeat_depth(self) -> None:
        probe = self.config["frozen_probe"]["probe"]
        self.assertEqual(probe["epochs"], 100)
        self.assertEqual(probe["learning_rates"], [0.01, 0.03, 0.1])
        self.assertEqual(probe["probe_seeds"], [1, 2, 3, 4, 5])
        self.assertEqual(
            self.config["dataset"]["split"]["expected_counts"],
            {"train": 5394, "validation": 600, "test": 5794},
        )


if __name__ == "__main__":
    unittest.main()
