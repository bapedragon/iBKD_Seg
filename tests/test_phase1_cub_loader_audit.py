from __future__ import annotations

import json
import importlib.util
import unittest
from pathlib import Path

import torch
from PIL import Image

from ibkd_seg.phase1.cub_loader_profiles import (
    L0_CURRENT_STRONG,
    L1_MATCHED_WEAK,
    L2_CONSERVATIVE_SPATIAL,
    LOADER_PROFILE_ORDER,
    build_cub_student_train_transform,
    crop_geometry,
    loader_profile_contract,
    sample_random_resized_crop,
)
from ibkd_seg.phase1.run_cub_loader_audit import (
    EXPECTED_CONFIG_SHA256,
    _box_coverage,
    _paired_seed,
    _validate_config,
)
from ibkd_seg.phase1.train_timing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "phase1/phase1_cub/image_loader_experiment/configs/"
    "cub200_r50_224_loader_damage_audit_v1.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/image_loader_experiment/scripts/"
    "run_r50_224_loader_damage_audit.sh"
)


class CubLoaderAuditTest(unittest.TestCase):
    def test_locked_config_and_script(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        _validate_config(config, CONFIG)
        self.assertEqual(file_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        self.assertEqual(
            tuple(item["id"] for item in config["loader_profiles"]),
            LOADER_PROFILE_ORDER,
        )
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.run_cub_loader_audit", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_l0_l1_share_geometry_and_l2_is_conservative(self) -> None:
        self.assertEqual(
            crop_geometry(L0_CURRENT_STRONG), crop_geometry(L1_MATCHED_WEAK)
        )
        self.assertEqual(
            crop_geometry(L2_CONSERVATIVE_SPATIAL).scale, (0.5, 1.0)
        )
        self.assertEqual(
            loader_profile_contract(L0_CURRENT_STRONG)["auto_augment"],
            "rand-m9-mstd0.5-inc1",
        )
        self.assertIsNone(
            loader_profile_contract(L1_MATCHED_WEAK)["auto_augment"]
        )

    def test_crop_sampling_is_paired_and_deterministic(self) -> None:
        image = Image.new("RGB", (320, 240))
        geometry_id = crop_geometry(L0_CURRENT_STRONG).identity
        seed = _paired_seed(20270912, geometry_id, 7, 0)
        first = sample_random_resized_crop(
            image,
            profile=L0_CURRENT_STRONG,
            generator=torch.Generator().manual_seed(seed),
        )
        second = sample_random_resized_crop(
            image,
            profile=L1_MATCHED_WEAK,
            generator=torch.Generator().manual_seed(seed),
        )
        self.assertEqual(first, second)
        top, left, height, width = first
        self.assertGreater(height, 0)
        self.assertGreater(width, 0)
        self.assertLessEqual(top + height, image.height)
        self.assertLessEqual(left + width, image.width)

    def test_profile_transforms_produce_expected_tensor(self) -> None:
        image = Image.new("RGB", (320, 240), color=(100, 120, 140))
        profiles = [L1_MATCHED_WEAK, L2_CONSERVATIVE_SPATIAL]
        if importlib.util.find_spec("timm") is not None:
            profiles.insert(0, L0_CURRENT_STRONG)
        for profile in profiles:
            with self.subTest(profile=profile):
                output = build_cub_student_train_transform(profile)(image)
                self.assertEqual(tuple(output.shape), (3, 224, 224))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_box_coverage(self) -> None:
        full = _box_coverage(
            (10.0, 10.0, 20.0, 20.0),
            top=0,
            left=0,
            height=40,
            width=40,
        )
        quarter = _box_coverage(
            (10.0, 10.0, 20.0, 20.0),
            top=20,
            left=20,
            height=20,
            width=20,
        )
        self.assertAlmostEqual(full, 1.0)
        self.assertAlmostEqual(quarter, 0.25)


if __name__ == "__main__":
    unittest.main()
