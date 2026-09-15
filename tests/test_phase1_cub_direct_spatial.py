from __future__ import annotations

import json
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from ibkd_seg.phase1.cub_direct_spatial import (
    CubSpatialAnnotation,
    LinearCKAAccumulator,
    attention_rollout,
    binary_average_precision,
    gaussian_part_targets,
    load_part_supervision,
    masked_heatmap_mse,
    part_localization_metrics,
    summarize_part_supervision,
)
from ibkd_seg.phase1.cub_probe_data import CubProbeRecord
from ibkd_seg.phase1.models import create_student
from ibkd_seg.phase1.release_asset import _asset_kind


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
class CubDirectSpatialProtocolTest(unittest.TestCase):
    def test_guided_issue727_release_kind_is_supported(self) -> None:
        manifest_path = (
            REPOSITORY_ROOT
            / "phase1/phase1_cub/reports/frozen_probe/"
            "resnet50_224_b128_b64_guided_seed1_v4/artifact_release.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(_asset_kind(manifest), "cub_r50_guided_seed1_v4")

    def test_guided_issue730_release_kind_is_supported(self) -> None:
        manifest_path = (
            REPOSITORY_ROOT
            / "phase1/phase1_cub/reports/frozen_probe/"
            "resnet50_224_b128_guided_3seed_v5/artifact_release.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(_asset_kind(manifest), "cub_r50_guided_seed23_v5")


class CubDirectSpatialMetricTest(unittest.TestCase):
    def test_gaussian_target_masks_invisible_parts(self) -> None:
        annotation = CubSpatialAnnotation(
            image_id=1,
            bounding_box=(0.0, 0.0, 224.0, 224.0),
            points=tuple((8.0, 8.0) for _ in range(15)),
            visible=(True,) + (False,) * 14,
        )
        heatmaps, visible, points, image_size, box = gaussian_part_targets(
            image_size=(224, 224), annotation=annotation
        )
        self.assertEqual(tuple(heatmaps.shape), (15, 14, 14))
        self.assertTrue(visible[0])
        self.assertEqual(float(heatmaps[0, 0, 0]), 1.0)
        self.assertEqual(int(torch.count_nonzero(heatmaps[1:])), 0)
        self.assertEqual(points.shape, (15, 2))
        self.assertEqual(image_size.tolist(), [224.0, 224.0])
        self.assertEqual(box.tolist(), [0.0, 0.0, 224.0, 224.0])

    def test_masked_mse_ignores_occluded_heatmaps(self) -> None:
        prediction = torch.zeros(1, 15, 14, 14)
        target = torch.zeros_like(prediction)
        target[:, 1:] = 100.0
        visible = torch.zeros(1, 15, dtype=torch.bool)
        visible[:, 0] = True
        self.assertEqual(float(masked_heatmap_mse(prediction, target, visible)), 0.0)

    def test_out_of_frame_visible_part_is_excluded_without_clipping(self) -> None:
        annotation = CubSpatialAnnotation(
            image_id=5007,
            bounding_box=(42.0, 30.0, 440.0, 303.0),
            points=(
                (0.0, 0.0),
                (185.0, 163.0),
                (0.0, 0.0),
                (405.0, 344.0),
            )
            + tuple((0.0, 0.0) for _ in range(11)),
            visible=(False, True, False, True) + (False,) * 11,
        )
        heatmaps, valid, points, image_size, _box = gaussian_part_targets(
            image_size=(500, 333), annotation=annotation
        )
        self.assertFalse(valid[3])
        self.assertTrue(valid[1])
        self.assertEqual(points[3].tolist(), [405.0, 344.0])
        self.assertEqual(int(torch.count_nonzero(heatmaps[3])), 0)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (500, 333)).save(image_path)
            record = CubProbeRecord(
                image_id=5007,
                label=85,
                relative_path="086.Pacific_Loon/Pacific_Loon_0034_75438.jpg",
                image_path=image_path,
                mask_path=Path(directory) / "unused.png",
            )
            supervision = load_part_supervision([record], {5007: annotation})
            audit = summarize_part_supervision([record], supervision)
        self.assertEqual(audit["excluded_out_of_frame_visible_keypoints"], 1)
        self.assertEqual(audit["affected_image_ids"], [5007])
        self.assertEqual(audit["excluded_landmarks"][0]["part_ids"], [4])
        self.assertEqual(
            audit["excluded_landmarks"][0]["coordinates"], [[405.0, 344.0]]
        )
        self.assertEqual(audit["excluded_landmarks"][0]["image_size"], [500.0, 333.0])
        self.assertEqual(audit["images_without_valid_keypoints"], [])

    def test_pck_uses_patch_center_and_bbox_max_side(self) -> None:
        logits = torch.zeros(1, 15, 14, 14)
        logits[:, :, 0, 0] = 1.0
        visible = torch.zeros(1, 15, dtype=torch.bool)
        visible[:, 0] = True
        points = torch.zeros(1, 15, 2)
        points[:, 0] = torch.tensor([8.0, 8.0])
        image_sizes = torch.tensor([[224.0, 224.0]])
        boxes = torch.tensor([[12.0, 9.0, 100.0, 80.0]])
        metrics = part_localization_metrics(
            logits,
            visible=visible,
            points=points,
            image_sizes=image_sizes,
            boxes=boxes,
        )
        self.assertEqual(metrics["visible_keypoints"], 1)
        self.assertEqual(metrics["micro_pck_at_0.1"], 1.0)
        self.assertEqual(metrics["mean_normalized_localization_error"], 0.0)

    def test_streaming_linear_cka_is_one_for_identical_representations(self) -> None:
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(31, 5, generator=generator)
        accumulator = LinearCKAAccumulator(5, 5, device=torch.device("cpu"))
        accumulator.update(values[:13], values[:13])
        accumulator.update(values[13:], values[13:])
        self.assertTrue(math.isclose(accumulator.compute(), 1.0, abs_tol=1e-12))

    def test_binary_average_precision_matches_hand_calculation(self) -> None:
        score = binary_average_precision(
            torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 0, 1])
        )
        self.assertTrue(math.isclose(score, 5.0 / 6.0, abs_tol=1e-12))

    @unittest.skipUnless(importlib.util.find_spec("timm"), "timm is not installed")
    def test_deit_attention_rollout_is_normalized(self) -> None:
        model = create_student(num_classes=200, drop_path_rate=0.1).eval()
        rollout = attention_rollout(model, torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(rollout.shape), (1, 14, 14))
        self.assertTrue(torch.isfinite(rollout).all())
        self.assertTrue(torch.allclose(rollout.sum(dim=(1, 2)), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
