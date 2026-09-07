from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ibkd_seg.phase1.cub_data import (
    NUM_CLASSES,
    CubRecord,
    build_stratified_split,
)
from ibkd_seg.phase1.cub_probe_data import CubProbeRecord, load_targets
from ibkd_seg.phase1.run_cub_combined_smoke import (
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
    _validate_config,
)
from ibkd_seg.phase1.train_timing import validate_args as validate_timing_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub/configs/cub200_b128_combined_smoke_v2.json"
)
SCRIPT_PATH = (
    REPOSITORY_ROOT / "phase1/phase1_cub/scripts/run_combined_smoke_b128.sh"
)


class Phase1CubSplitTest(unittest.TestCase):
    def test_three_validation_images_per_class_are_deterministic(self) -> None:
        records = [
            CubRecord(
                image_id=label * 100 + offset + 1,
                relative_path=f"{label:03d}/image_{offset}.jpg",
                label=label,
                is_train=True,
            )
            for label in range(NUM_CLASSES)
            for offset in range(4)
        ]
        first = build_stratified_split(records)
        second = build_stratified_split(records)
        self.assertEqual(first, second)
        train_indices, validation_indices, manifest = first
        self.assertEqual(len(train_indices), 200)
        self.assertEqual(len(validation_indices), 600)
        self.assertEqual(manifest["validation_per_class"], 3)
        self.assertEqual(manifest["split_seed"], 2027)
        self.assertEqual(len(manifest["validation_image_ids_sha256"]), 64)


class Phase1CubMaskTest(unittest.TestCase):
    def test_positive_grayscale_values_are_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "bird.jpg"
            mask_path = root / "bird.png"
            Image.new("RGB", (4, 4), color=(100, 120, 140)).save(image_path)
            mask = np.array(
                [
                    [1, 1, 0, 0],
                    [255, 127, 0, 0],
                    [0, 0, 255, 0],
                    [0, 0, 1, 255],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(mask, mode="L").save(mask_path)
            record = CubProbeRecord(
                image_id=1,
                label=0,
                relative_path="001/bird.jpg",
                image_path=image_path,
                mask_path=mask_path,
            )
            input_target, grid_target = load_targets(
                record,
                input_size=4,
                grid_size=(2, 2),
            )
        self.assertEqual(set(int(v) for v in torch.unique(input_target)), {0, 1})
        self.assertTrue(
            torch.equal(
                grid_target,
                torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8),
            )
        )


class Phase1CubSmokeContractTest(unittest.TestCase):
    def test_committed_smoke_config_is_non_scientific(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        _validate_config(config)
        self.assertFalse(config["scientific_result"])
        self.assertFalse(config["official_test_accessed"])
        self.assertEqual(config["full_protocol_status"], "not_locked")
        self.assertEqual(
            tuple(config["classification"]["variants"]), EXPECTED_VARIANTS
        )
        self.assertEqual(len(EXPECTED_VARIANTS), 6)
        self.assertEqual(VARIANT_ARGUMENTS["alg_warmup20"], ("alg", None, 20))
        self.assertEqual(VARIANT_ARGUMENTS["ibkd_lambda_0.25"][2], 20)
        self.assertEqual(VARIANT_ARGUMENTS["ibkd_lambda_0.5"][2], 20)
        self.assertEqual(config["classification"]["student"]["batch_size"], 128)
        self.assertNotIn("alg", config["classification"]["variants"])
        self.assertIn("alg_warmup20", config["classification"]["variants"])
        self.assertEqual(
            config["classification"]["controller"]["alg_warmup_epochs"], 20
        )
        self.assertFalse(
            config["classification"]["controller"][
                "canonical_alg_warmup0_included"
            ]
        )
        self.assertEqual(config["task_count"]["probe_lr_candidates"], 18)

    def test_cub_timing_export_allows_each_smoke_student(self) -> None:
        for method, fusion_ratio, alg_warmup in (
            ("vanilla", None, 0),
            ("kd", None, 0),
            ("lg", None, 0),
            ("alg", None, 20),
            ("ibkd", 0.25, 0),
            ("ibkd", 0.5, 0),
        ):
            with self.subTest(method=method, fusion_ratio=fusion_ratio):
                args = argparse.Namespace(
                    dataset="cub",
                    kind="student",
                    method=method,
                    batch_size=128,
                    fusion_ratio=fusion_ratio,
                    teacher_checkpoint=(
                        None if method == "vanilla" else Path("teacher.pt")
                    ),
                    eval_batch_size=200,
                    num_workers=4,
                    seed=1,
                    alg_controller_warmup_epochs=alg_warmup,
                    save_student_checkpoint=True,
                )
                validate_timing_args(args)

    def test_h200_script_runs_the_combined_smoke(self) -> None:
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.run_cub_combined_smoke", script)
        self.assertIn("--smoke", script)
        self.assertIn("--device cuda", script)
        self.assertTrue(SCRIPT_PATH.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
