from __future__ import annotations

import tempfile
import unittest
import stat
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ibkd_seg.cityscapes.models import MultiLevelDecoder
from ibkd_seg.cub_segmentation.data import load_pair
from ibkd_seg.cub_segmentation.smoke import CONFIG, binary_metrics, load_config
from ibkd_seg.phase1.cub_probe_data import CubProbeRecord


class CubDirectSegmentationSmokeContract(unittest.TestCase):
    def test_entrypoint_stays_in_the_separate_phase1_cub_seg_folder(self):
        repository = Path(__file__).resolve().parents[1]
        expected_config = (
            repository
            / "phase1/phase1_cub_Seg/configs/direct_segmentation_smoke_v1.json"
        )
        script = (
            repository
            / "phase1/phase1_cub_Seg/scripts/run_direct_segmentation_smoke.sh"
        )
        self.assertEqual(CONFIG, expected_config)
        self.assertTrue(script.is_file())
        self.assertTrue(bool(script.stat().st_mode & stat.S_IXUSR))

    def test_config_is_bounded_binary_four_method_smoke(self):
        config = load_config(CONFIG)
        self.assertFalse(config["scientific_result"])
        self.assertFalse(config["full_training_authorized"])
        self.assertEqual(config["methods"], ["vanilla", "lg", "alg", "ibkd"])
        self.assertEqual(config["num_classes"], 2)
        self.assertEqual(config["primary_metric"], "two_class_miou")
        self.assertEqual(config["train_samples"], config["steps"] * config["batch_size"])

    def test_shared_decoder_supports_binary_output_without_changing_default(self):
        features = [
            torch.randn(2, channels, size, size)
            for channels, size in ((8, 16), (16, 8), (32, 4), (64, 2))
        ]
        binary = MultiLevelDecoder((8, 16, 32, 64), 8, num_classes=2)
        cityscapes_default = MultiLevelDecoder((8, 16, 32, 64), 8)
        self.assertEqual(tuple(binary(features).shape), (2, 2, 16, 16))
        self.assertEqual(tuple(cityscapes_default(features).shape), (2, 19, 16, 16))

    def test_binary_metrics_use_global_confusion(self):
        result = binary_metrics(torch.tensor([[2, 1], [1, 2]]))
        self.assertAlmostEqual(result["background_iou"], 0.5)
        self.assertAlmostEqual(result["foreground_iou"], 0.5)
        self.assertAlmostEqual(result["two_class_miou"], 0.5)
        self.assertAlmostEqual(result["foreground_dice"], 2 / 3)
        self.assertAlmostEqual(result["pixel_accuracy"], 2 / 3)

    def test_rgb_and_mask_receive_the_same_horizontal_flip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "bird.jpg"
            mask_path = root / "bird.png"
            mask = np.array([[0, 255], [0, 255]], dtype=np.uint8)
            rgb = np.repeat(mask[:, :, None], 3, axis=2)
            Image.fromarray(rgb).save(image_path)
            Image.fromarray(mask).save(mask_path)
            record = CubProbeRecord(1, 0, "bird.jpg", image_path, mask_path)
            _, target = load_pair(record, input_size=2, horizontal_flip=True)
        self.assertTrue(torch.equal(target, torch.tensor([[1, 0], [1, 0]])))


if __name__ == "__main__":
    unittest.main()
