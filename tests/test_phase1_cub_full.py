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
    RUNNABLE_SHARDS,
    _validate_config,
)
from ibkd_seg.phase1.train_full import validate_args as validate_full_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT / "phase1/phase1_cub/configs/cub200_b128_full_v2.json"
)
GUIDED_SCRIPT = REPOSITORY_ROOT / "phase1/phase1_cub/scripts/run_full_guided_b128.sh"


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

    def test_locked_config_and_shared_teacher_shards(self) -> None:
        _validate_config(self.config)
        self.assertEqual(
            sorted(variant for values in EXPECTED_SHARDS.values() for variant in values),
            sorted(EXPECTED_VARIANTS),
        )
        self.assertEqual(
            EXPECTED_SHARDS["guided"],
            ("alg_warmup20", "ibkd_lambda_0.25", "ibkd_lambda_0.5"),
        )
        self.assertEqual(EXPECTED_SHARDS["baseline"], ("vanilla", "kd", "lg"))
        self.assertFalse(
            set(EXPECTED_SHARDS["guided"]) & set(EXPECTED_SHARDS["baseline"])
        )
        self.assertEqual(
            self.config["execution"]["shards"]["guided"]["probe_lr_candidates"],
            135,
        )
        self.assertEqual(
            self.config["execution"]["shards"]["baseline"]["selected_probes"],
            45,
        )
        self.assertEqual(
            self.config["execution"]["shards"]["guided"]["teacher_runs"], 1
        )
        self.assertEqual(
            self.config["execution"]["shards"]["baseline"]["teacher_runs"], 0
        )
        self.assertEqual(RUNNABLE_SHARDS, ("guided",))

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

    def test_only_shared_teacher_producer_is_currently_runnable(self) -> None:
        script = GUIDED_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.run_cub_full_shard", script)
        self.assertIn("--full-shard", script)
        self.assertIn("--shard guided", script)
        self.assertIn("/app/output/phase1_cub_b128_full_v2_guided", script)
        self.assertTrue(GUIDED_SCRIPT.stat().st_mode & 0o111)
        self.assertFalse(
            (
                REPOSITORY_ROOT
                / "phase1/phase1_cub/scripts/run_full_shard_b_b128.sh"
            ).exists()
        )

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
