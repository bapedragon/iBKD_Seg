from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

import torch

from ibkd_seg.phase1.models import (
    IBKD,
    LocalityGuidance,
    RESNET50_TEACHER_CHANNELS,
    ResNet50CUB,
)
from ibkd_seg.phase1.run_cub_r50_guided_smoke import (
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
    _validate_configs,
)
from ibkd_seg.phase1.train_timing import validate_args


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/cub200_r50_224_b128_guided_smoke_v3.json"
)
FULL_CONFIG = (
    ROOT / "phase1/phase1_cub/configs/cub200_r50_224_b128_full_v3.json"
)
SCRIPT = (
    ROOT / "phase1/phase1_cub/scripts/run_r50_224_guided_smoke_b128.sh"
)


class Phase1CubR50SmokeTest(unittest.TestCase):
    def test_locked_smoke_and_full_configs_match(self) -> None:
        smoke = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        full = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
        _validate_configs(smoke, full)
        self.assertFalse(smoke["scientific_result"])
        self.assertTrue(smoke["official_test_accessed"])
        self.assertTrue(full["matrix_commitment"]["unconditional"])
        self.assertEqual(len(full["matrix_commitment"]["variants"]), 6)
        self.assertEqual(full["matrix_commitment"]["encoder_seeds"], [1, 2, 3])
        self.assertEqual(tuple(smoke["classification"]["variants"]), EXPECTED_VARIANTS)

    def test_resnet50_feature_contract_and_dynamic_guidance_channels(self) -> None:
        model = ResNet50CUB(num_classes=200).eval()
        with torch.inference_mode():
            features = model.forward_features(torch.zeros(1, 3, 224, 224))
            logits = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(
            [tuple(value.shape) for value in features],
            [
                (1, 512, 28, 28),
                (1, 1024, 14, 14),
                (1, 2048, 7, 7),
            ],
        )
        self.assertEqual(tuple(logits.shape), (1, 200))
        lg = LocalityGuidance(teacher_channels=RESNET50_TEACHER_CHANNELS)
        ibkd = IBKD(teacher_channels=RESNET50_TEACHER_CHANNELS)
        self.assertEqual(lg.teacher_channels, RESNET50_TEACHER_CHANNELS)
        self.assertEqual(ibkd.teacher_channels, RESNET50_TEACHER_CHANNELS)

    def test_each_guided_timing_command_is_valid(self) -> None:
        for variant in EXPECTED_VARIANTS:
            method, fusion_ratio, warmup = VARIANT_ARGUMENTS[variant]
            with self.subTest(variant=variant):
                args = argparse.Namespace(
                    dataset="cub",
                    kind="student",
                    method=method,
                    teacher_architecture="resnet50_224_scratch",
                    access_official_test=True,
                    batch_size=128,
                    fusion_ratio=fusion_ratio,
                    teacher_checkpoint=Path("teacher.pt"),
                    eval_batch_size=200,
                    num_workers=4,
                    seed=1,
                    alg_controller_warmup_epochs=20 if method == "alg" else 0,
                    save_student_checkpoint=True,
                )
                validate_args(args)
                self.assertEqual(warmup, 20 if method != "lg" else 0)

    def test_h200_script_runs_v3_entrypoint(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.run_cub_r50_guided_smoke", script)
        self.assertIn("cub200_r50_224_b128_guided_smoke_v3.json", script)
        self.assertIn("--device cuda", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
