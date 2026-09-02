from __future__ import annotations

import unittest

from ibkd_seg.metrics import binary_segmentation_metrics


class BinarySegmentationMetricsTest(unittest.TestCase):
    def test_known_confusion_counts(self) -> None:
        metrics = binary_segmentation_metrics(
            prediction=[1, 1, 1, 0, 0, 0],
            target=[1, 1, 0, 1, 0, 0],
        )
        self.assertEqual(metrics.true_positive, 2)
        self.assertEqual(metrics.true_negative, 2)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertAlmostEqual(metrics.flower_iou, 0.5)
        self.assertAlmostEqual(metrics.background_iou, 0.5)
        self.assertAlmostEqual(metrics.mean_iou, 0.5)
        self.assertAlmostEqual(metrics.flower_dice, 2 / 3)
        self.assertAlmostEqual(metrics.pixel_accuracy, 2 / 3)

    def test_dice_iou_identity(self) -> None:
        metrics = binary_segmentation_metrics(
            prediction=[1, 1, 0, 0],
            target=[1, 0, 1, 0],
        )
        expected_dice = 2 * metrics.flower_iou / (1 + metrics.flower_iou)
        self.assertAlmostEqual(metrics.flower_dice, expected_dice)

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            binary_segmentation_metrics([1], [1, 0])

    def test_all_background_baseline(self) -> None:
        metrics = binary_segmentation_metrics(
            prediction=[0, 0, 0, 0],
            target=[1, 0, 0, 0],
        )
        self.assertEqual(metrics.flower_iou, 0.0)
        self.assertAlmostEqual(metrics.background_iou, 0.75)
        self.assertAlmostEqual(metrics.mean_iou, 0.375)


if __name__ == "__main__":
    unittest.main()
