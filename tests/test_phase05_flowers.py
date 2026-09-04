from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ibkd_seg.phase05.flowers import FlowersRecord, evenly_spaced_subset, load_targets


class Phase05FlowersTest(unittest.TestCase):
    def test_evenly_spaced_subset_is_deterministic_and_keeps_endpoints(self) -> None:
        records = [
            FlowersRecord(index, Path(f"image-{index}"), Path(f"mask-{index}"))
            for index in range(10)
        ]
        selected = evenly_spaced_subset(records, 4)
        self.assertEqual([record.image_id for record in selected], [0, 3, 6, 9])

    def test_paired_target_resize_and_patch_majority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.full((8, 8, 3), (200, 100, 50), dtype=np.uint8)
            foreground = np.zeros((8, 8), dtype=bool)
            foreground[:, :4] = True
            composite = np.empty_like(original)
            composite[foreground] = original[foreground]
            composite[~foreground] = np.asarray((0, 0, 255), dtype=np.uint8)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(original).save(image_path)
            Image.fromarray(composite).save(mask_path)
            record = FlowersRecord(1, image_path, mask_path)

            input_target, grid_target = load_targets(
                record,
                input_size=8,
                grid_size=(2, 2),
                alpha_threshold=0.5,
                occupancy_threshold=0.5,
            )
            self.assertEqual(tuple(input_target.shape), (8, 8))
            self.assertEqual(tuple(grid_target.shape), (2, 2))
            self.assertTrue(grid_target[:, 0].bool().all())
            self.assertFalse(grid_target[:, 1].bool().any())


if __name__ == "__main__":
    unittest.main()
