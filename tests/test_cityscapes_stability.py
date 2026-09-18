import math
import json
import unittest
from pathlib import Path

from ibkd_seg.cityscapes.official_stability import (
    beta_run_id,
    stability_decision,
    guidance_beta_for,
    online_divergence_reason,
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

    def test_online_guard_stops_a_clearly_exploding_run(self):
        config = {"max_peak_ratio": 100.0}
        rows = self.rows([1.0, 101.0])
        rows[0]["step"] = 1
        rows[1]["step"] = 2
        self.assertEqual(
            online_divergence_reason(rows, config),
            "guidance:step=2:ratio=101.0",
        )

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

    def test_beta_confirmation_locks_all_guided_methods(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "phase4/phase4_cityscapes/configs/paper_l16_crop512_beta_confirm500_v3.json"
        config = json.loads(path.read_text())
        validate_config(config)
        self.assertEqual(config["stability_steps"], 500)
        self.assertEqual(config["methods"], ["lg", "alg", "ibkd"])
        self.assertEqual(guidance_beta_for("lg", config), 0.05)
        self.assertEqual(guidance_beta_for("alg", config), 0.05)
        self.assertEqual(guidance_beta_for("ibkd", config), 0.5)

    def test_reproducibility_profile_requires_strict_determinism(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "phase4/phase4_cityscapes/configs/paper_l16_crop512_repro25_v4.json"
        config = json.loads(path.read_text())
        validate_config(config)
        self.assertEqual(config["stability_steps"], 25)
        self.assertEqual(config["methods"], ["lg", "alg"])
        self.assertTrue(config["strict_determinism"])
        self.assertEqual(guidance_beta_for("lg", config), 0.05)
        self.assertEqual(guidance_beta_for("alg", config), 0.05)
        config["strict_determinism"] = False
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_beta_grid_locks_all_twelve_runs(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "phase4/phase4_cityscapes/configs/paper_l16_crop512_beta_grid500_v5.json"
        config = json.loads(path.read_text())
        validate_config(config)
        expected = {
            "lg": [0.02, 0.05, 0.1, 0.2],
            "alg": [0.02, 0.05, 0.1, 0.2],
            "ibkd": [0.1, 0.25, 0.5, 1.0],
        }
        self.assertEqual(config["guidance_beta_candidates_by_method"], expected)
        for method, values in expected.items():
            for beta in values:
                self.assertEqual(guidance_beta_for(method, config, beta), beta)
        with self.assertRaises(ValueError):
            guidance_beta_for("lg", config)
        with self.assertRaises(ValueError):
            guidance_beta_for("lg", config, 2.5)
        self.assertEqual(beta_run_id("ibkd", 0.25), "ibkd_beta_0p25")

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

    def test_beta_grid_terminal_result_keeps_all_runs_and_survivors(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((
            root / "phase4/phase4_cityscapes/configs/paper_l16_crop512_beta_grid500_v5.json"
        ).read_text())
        runs = []
        for method, candidates in config["guidance_beta_candidates_by_method"].items():
            for beta in candidates:
                stable = not (method == "ibkd" and beta == 1.0)
                runs.append({
                    "run_id": beta_run_id(method, beta),
                    "method": method,
                    "status": "stable" if stable else "unstable",
                    "completed_steps": 500 if stable else 20,
                    "expected_steps": 500,
                    "first_step": {"step": 1, "loss": 3.0},
                    "final_step": {"step": 500 if stable else 20, "loss": 1.0},
                    "decision": {"diagnostics": {"loss": {}}, "reasons": []},
                    "diagnostic_validation": (
                        {"pixel_accuracy": 0.5, "miou": 0.2} if stable else None
                    ),
                    "validation_samples": 20 if stable else 0,
                    "effective_guidance_beta": beta,
                    "decoder_layers": 1,
                    "schedule_total_steps": 80_000,
                    "train_seconds": 10.0,
                    "invocation_seconds": 12.0,
                    "peak_cuda_allocated_bytes": 100,
                    "teacher_frozen_verified": True,
                    "parameters_finite": stable,
                    "optimizer_state_finite": stable,
                    "runtime_error": None if stable else "online_divergence",
                })
        result = terminal_result(runs, config, [])
        self.assertEqual(len(result["methods"]), 12)
        self.assertEqual(result["stable_methods"], 11)
        self.assertFalse(result["all_methods_stable"])
        self.assertEqual(result["stable_beta_candidates_by_method"]["lg"],
                         [0.02, 0.05, 0.1, 0.2])
        self.assertEqual(result["unstable_beta_candidates_by_method"]["ibkd"], [1.0])


if __name__ == "__main__":
    unittest.main()
