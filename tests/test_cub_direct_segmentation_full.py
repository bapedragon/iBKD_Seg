from __future__ import annotations

import stat
import unittest
from pathlib import Path

from ibkd_seg.cub_segmentation.full import (
    CONFIG,
    final_results,
    learning_rate,
    load_config,
)


class CubDirectSegmentationFullContract(unittest.TestCase):
    def test_locked_full_config(self):
        config = load_config(CONFIG)
        self.assertFalse(config["scientific_result"])
        self.assertTrue(config["full_training_authorized"])
        self.assertEqual(config["methods"], ["vanilla", "lg", "alg", "ibkd"])
        self.assertEqual(config["split"]["train"], 5394)
        self.assertEqual(config["split"]["validation"], 600)
        self.assertEqual(config["split"]["test"], 5794)
        self.assertEqual(config["teacher"]["epochs"], 100)
        self.assertEqual(config["student"]["epochs"], 100)
        self.assertEqual(config["selection_metric"], "two_class_miou")

    def test_full_entrypoint_is_executable_and_pins_its_config(self):
        repository = Path(__file__).resolve().parents[1]
        script = (
            repository
            / "phase1/phase1_cub_Seg/scripts/run_direct_segmentation_full.sh"
        )
        self.assertTrue(script.is_file())
        self.assertTrue(bool(script.stat().st_mode & stat.S_IXUSR))
        self.assertIn(str(CONFIG.relative_to(repository)), script.read_text())

    def test_warmup_then_cosine_schedule_hits_contract_points(self):
        role = {
            "epochs": 4,
            "warmup_epochs": 1,
            "learning_rate": 1.0,
            "minimum_learning_rate": 0.1,
        }
        self.assertAlmostEqual(learning_rate(0, 2, role), 0.5)
        self.assertAlmostEqual(learning_rate(1, 2, role), 1.0)
        self.assertAlmostEqual(learning_rate(2, 2, role), 1.0)
        self.assertAlmostEqual(learning_rate(7, 2, role), 0.1)

    def test_final_results_contains_every_final_metric(self):
        metrics = {
            "two_class_miou": 0.5,
            "foreground_iou": 0.4,
            "background_iou": 0.6,
            "foreground_dice": 0.57,
            "pixel_accuracy": 0.7,
            "valid_pixels": 10,
            "confusion_matrix_truth_rows_prediction_columns": [[4, 1], [2, 3]],
        }

        def summary(method):
            return {
                "method": method,
                "selected_epoch": 3,
                "selected_validation": metrics,
                "official_test": metrics,
                "last_epoch_losses": {"total": 1.0, "ce": 0.5, "guidance": 0.2},
                "guidance_stop_epoch": None,
                "elapsed_seconds": 1.0,
                "peak_cuda_allocated_bytes": 100,
            }

        report = {
            "config": {"protocol_id": "test"},
            "teacher": summary(None),
            "runs": [summary(method) for method in ("vanilla", "lg", "alg", "ibkd")],
        }
        result = final_results(report)
        self.assertEqual(set(result["methods"]), {"vanilla", "lg", "alg", "ibkd"})
        for row in [result["teacher"], *result["methods"].values()]:
            self.assertEqual(set(row["validation"]), set(metrics))
            self.assertEqual(set(row["test"]), set(metrics))


if __name__ == "__main__":
    unittest.main()
