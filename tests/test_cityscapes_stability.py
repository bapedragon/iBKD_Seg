import math
import json
import unittest
from pathlib import Path

from ibkd_seg.cityscapes.official_stability import (
    stability_decision,
    guidance_beta_for,
    terminal_result,
    validate_config,
)


class CityscapesStabilityTests(unittest.TestCase):
    def config(self):
        return {
            "reference_steps": 1,
            "tail_steps": 2,
            "max_peak_ratio": 100.0,
            "max_tail_ratio": 10.0,
        }

    @staticmethod
    def rows(guidance):
        return [
            {"loss": 3.0 + value, "ce": 3.0, "guidance": value,
             "grad_norm_unclipped": 2.0}
            for value in guidance
        ]

    def test_stable_sequence_passes(self):
        rows = self.rows([1.0, 0.9, 0.8, 0.7])
        result = stability_decision(rows, self.config(), expected_steps=4)
        self.assertTrue(result["stable"])
        self.assertEqual(result["reasons"], [])

    def test_exploding_guidance_fails(self):
        rows = self.rows([4.61, 43.57, 527.91])
        result = stability_decision(rows, self.config(), expected_steps=3)
        self.assertFalse(result["stable"])
        self.assertTrue(any(reason.startswith("guidance:peak_ratio=") for reason in result["reasons"]))

    def test_incomplete_or_nonfinite_run_fails(self):
        rows = self.rows([1.0])
        rows[0]["loss"] = math.inf
        result = stability_decision(
            rows, self.config(), expected_steps=4, runtime_error="FloatingPointError()")
        self.assertFalse(result["stable"])
        self.assertTrue(any(reason.startswith("runtime_error:") for reason in result["reasons"]))
        self.assertIn("incomplete_steps:1/4", result["reasons"])

    def test_locked_protocol_rejects_silent_crop_change(self):
        config = {
            "protocol_id": "cityscapes_segmenter_l16_crop512_stability_v1",
            "methods": ["vanilla", "lg", "alg", "ibkd"],
            "stability_steps": 500,
            "train_samples": 2975,
            "val_samples": 20,
            "batch_size": 8,
            "image_size": 1024,
            "crop_size": 512,
            "window_size": 512,
            "window_stride": 512,
            "decoder_layers": 1,
            "total_steps": 80000,
            "precision": "fp32",
            "gradient_clipping": False,
            "automatic_hyperparameter_changes": False,
            "test_used": False,
            "guidance_beta": 2.5,
            "guidance_beta_by_method": None,
            "reference_steps": 1,
            "tail_steps": 50,
            "max_peak_ratio": 100.0,
            "max_tail_ratio": 10.0,
        }
        validate_config(config)
        config["crop_size"] = 768
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_beta_screen_locks_method_specific_values(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "phase4/phase4_cityscapes/configs/paper_l16_crop512_beta_screen100_v2.json"
        config = json.loads(path.read_text())
        validate_config(config)
        self.assertEqual(guidance_beta_for("lg", config), 0.05)
        self.assertEqual(guidance_beta_for("ibkd", config), 0.5)
        with self.assertRaises(ValueError):
            guidance_beta_for("alg", config)

    def test_terminal_result_contains_every_method_result(self):
        runs = []
        for method in ("vanilla", "lg", "alg", "ibkd"):
            runs.append({
                "method": method,
                "status": "stable",
                "completed_steps": 500,
                "expected_steps": 500,
                "first_step": {"step": 1, "loss": 3.0},
                "final_step": {"step": 500, "loss": 1.0},
                "decision": {"diagnostics": {"loss": {}}, "reasons": []},
                "diagnostic_validation": {"pixel_accuracy": 0.5, "miou": 0.2},
                "validation_samples": 20,
                "effective_guidance_beta": 0.0 if method == "vanilla" else 2.5,
                "decoder_layers": 1,
                "schedule_total_steps": 80_000,
                "train_seconds": 10.0,
                "invocation_seconds": 12.0,
                "peak_cuda_allocated_bytes": 100,
                "teacher_frozen_verified": True,
                "parameters_finite": True,
                "optimizer_state_finite": True,
                "runtime_error": None,
            })
        config = {
            "methods": ["vanilla", "lg", "alg", "ibkd"],
            "protocol_id": "test",
            "crop_size": 512,
            "batch_size": 8,
            "total_steps": 80_000,
            "guidance_beta": 2.5,
            "guidance_beta_by_method": None,
        }
        result = terminal_result(runs, config, [])
        self.assertTrue(result["all_methods_stable"])
        self.assertEqual(set(result["methods"]), {"vanilla", "lg", "alg", "ibkd"})
        self.assertEqual(result["methods"]["ibkd"]["final_step"]["step"], 500)
        self.assertEqual(result["methods"]["lg"]["diagnostic_miou"], 0.2)
        self.assertEqual(result["methods"]["vanilla"]["diagnostic_validation"]["pixel_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
