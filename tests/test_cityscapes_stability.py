import math
import unittest

from ibkd_seg.cityscapes.official_stability import stability_decision, validate_config


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
            "reference_steps": 1,
            "tail_steps": 50,
            "max_peak_ratio": 100.0,
            "max_tail_ratio": 10.0,
        }
        validate_config(config)
        config["crop_size"] = 768
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
