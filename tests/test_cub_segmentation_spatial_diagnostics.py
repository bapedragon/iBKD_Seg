from __future__ import annotations

import json
import stat
import unittest
from pathlib import Path

import torch

from ibkd_seg.cub_segmentation.spatial_diagnostics import (
    ProbeCandidateDiverged,
    build_probe,
    load_config,
    probe_parameter_count,
    train_probe_candidate,
)
from ibkd_seg.cub_segmentation.spatial_diagnostics_full import (
    _mean_and_sample_sd,
    load_config as load_full_config,
)
from ibkd_seg.phase1.release_asset import _asset_kind


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub_Seg/configs/spatial_diagnostics_seed1_smoke_v1.json"
)
RELEASE = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub_Seg/reports/window30_seed1_v1/checkpoint_release.json"
)
FULL_CONFIG = (
    REPOSITORY_ROOT
    / "phase1/phase1_cub_Seg/configs/spatial_diagnostics_seed1_full_v1.json"
)


class CubSegmentationSpatialDiagnosticsTest(unittest.TestCase):
    def test_locked_config_and_release_are_supported(self) -> None:
        config = load_config(CONFIG)
        manifest = json.loads(RELEASE.read_text(encoding="utf-8"))
        self.assertEqual(
            config["scope"]["methods"], ["vanilla", "lg", "alg", "ibkd"]
        )
        self.assertEqual(_asset_kind(manifest), "cub_seg_window30_seed1_v1")

    def test_entrypoint_is_executable_and_keeps_final_json_last(self) -> None:
        script = (
            REPOSITORY_ROOT
            / "phase1/phase1_cub_Seg/scripts/run_spatial_diagnostics_seed1_smoke.sh"
        )
        source = script.read_text(encoding="utf-8")
        self.assertTrue(bool(script.stat().st_mode & stat.S_IXUSR))
        self.assertIn(str(CONFIG.relative_to(REPOSITORY_ROOT)), source)
        self.assertIn("ibkd_seg.phase1.release_asset", source)
        self.assertIn("ibkd_seg.cub_segmentation.spatial_diagnostics", source)
        module_source = (
            REPOSITORY_ROOT
            / "src/ibkd_seg/cub_segmentation/spatial_diagnostics.py"
        ).read_text(encoding="utf-8")
        self.assertIn("[CUB_SEG_SPATIAL_SMOKE_FINAL_RESULTS]", module_source)

    def test_full_config_and_entrypoint_are_locked_after_smoke(self) -> None:
        config = load_full_config(FULL_CONFIG)
        self.assertEqual(config["scope"]["encoder_seed"], 1)
        self.assertEqual(config["scope"]["probe_seeds"], [1])
        self.assertEqual(config["part_probe"]["epochs"], 100)
        self.assertEqual(config["dataset"]["part_probe_train_images"], 5394)
        self.assertEqual(config["dataset"]["part_probe_validation_images"], 600)
        self.assertEqual(config["part_probe"]["divergence_loss_ceiling"], 10.0)
        script = (
            REPOSITORY_ROOT
            / "phase1/phase1_cub_Seg/scripts/run_spatial_diagnostics_seed1_full.sh"
        )
        source = script.read_text(encoding="utf-8")
        self.assertTrue(bool(script.stat().st_mode & stat.S_IXUSR))
        self.assertIn(str(FULL_CONFIG.relative_to(REPOSITORY_ROOT)), source)
        self.assertIn("spatial_diagnostics_full", source)
        module_source = (
            REPOSITORY_ROOT
            / "src/ibkd_seg/cub_segmentation/spatial_diagnostics_full.py"
        ).read_text(encoding="utf-8")
        self.assertIn("[CUB_SEG_SPATIAL_FULL_FINAL_RESULTS]", module_source)

    def test_probe_parameter_counts_and_initialization_are_fixed(self) -> None:
        self.assertEqual(probe_parameter_count("linear"), 2895)
        self.assertEqual(probe_parameter_count("nonlinear"), 111695)
        for kind in ("linear", "nonlinear"):
            left = build_probe(kind, 1).state_dict()
            right = build_probe(kind, 1).state_dict()
            self.assertEqual(set(left), set(right))
            self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))

    def test_both_probe_types_update_on_synthetic_features(self) -> None:
        generator = torch.Generator().manual_seed(7)
        sample_count = 4
        features = torch.randn(
            sample_count, 192, 14, 14, generator=generator
        )
        heatmaps = torch.rand(
            sample_count, 15, 14, 14, generator=generator
        )
        visible = torch.ones(sample_count, 15, dtype=torch.bool)
        points = torch.full((sample_count, 15, 2), 112.0)
        image_sizes = torch.full((sample_count, 2), 224.0)
        boxes = torch.tensor([[0.0, 0.0, 224.0, 224.0]]).repeat(
            sample_count, 1
        )
        supervision = {
            "heatmaps": heatmaps,
            "visible": visible,
            "points": points,
            "image_sizes": image_sizes,
            "boxes": boxes,
        }
        for kind in ("linear", "nonlinear"):
            result = train_probe_candidate(
                kind=kind,
                train_features=features,
                train_supervision=supervision,
                validation_features=features,
                validation_supervision=supervision,
                learning_rate=0.01,
                epochs=1,
                seed=1,
                batch_size=2,
                device=torch.device("cpu"),
            )
            self.assertTrue(result["parameter_update_verified"])
            self.assertNotEqual(
                result["initial_probe_state_sha256"],
                result["final_probe_state_sha256"],
            )
            self.assertEqual(result["parameter_count"], probe_parameter_count(kind))
            self.assertEqual(len(result["history"]), 1)

    def test_probe_divergence_isolated_by_locked_loss_ceiling(self) -> None:
        features = torch.full((2, 192, 14, 14), 1000.0)
        supervision = {
            "heatmaps": torch.zeros(2, 15, 14, 14),
            "visible": torch.ones(2, 15, dtype=torch.bool),
            "points": torch.zeros(2, 15, 2),
            "image_sizes": torch.full((2, 2), 224.0),
            "boxes": torch.tensor([[0.0, 0.0, 224.0, 224.0]]).repeat(2, 1),
        }
        with self.assertRaises(ProbeCandidateDiverged) as context:
            train_probe_candidate(
                kind="linear",
                train_features=features,
                train_supervision=supervision,
                validation_features=features,
                validation_supervision=supervision,
                learning_rate=0.1,
                epochs=2,
                seed=1,
                batch_size=2,
                device=torch.device("cpu"),
                divergence_loss_ceiling=10.0,
            )
        self.assertGreater(context.exception.observed_loss, 10.0)

    def test_single_probe_seed_aggregate_does_not_fake_standard_deviation(self) -> None:
        self.assertEqual(
            _mean_and_sample_sd([0.25]),
            {"mean": 0.25, "sample_standard_deviation": None, "n": 1},
        )


if __name__ == "__main__":
    unittest.main()
