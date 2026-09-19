import unittest

from ibkd_seg.cityscapes.warn_only_reproducibility import (
    compact_run_status,
    diagnose,
    warn_only_controls_ok,
)


class CityscapesWarnOnlyReproducibilityTests(unittest.TestCase):
    def test_warn_only_environment_is_locked(self):
        summary = {
            "environment": {
                "strict_determinism_requested": False,
                "determinism_warn_only_requested": True,
                "torch_deterministic_algorithms": True,
                "torch_deterministic_warn_only": True,
                "cublas_workspace_config": ":4096:8",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            }
        }
        self.assertTrue(warn_only_controls_ok(summary))
        summary["environment"]["torch_deterministic_warn_only"] = False
        self.assertFalse(warn_only_controls_ok(summary))

    def test_all_three_update_comparisons_authorize_next_stage(self):
        comparisons = {
            name: {
                "passed": True,
                "scalar_values_exact": name == "ibkd_repeat",
                "scalar_fields": {"ce": {"exact": name == "ibkd_repeat"}},
            }
            for name in (
                "lg_repeat",
                "lg_vs_alg_before_controller_action",
                "ibkd_repeat",
            )
        }
        result = diagnose(comparisons, all_completed=True, controls_verified=True)
        self.assertTrue(result["passed"])
        self.assertTrue(result["scalar_variation_observed"])
        self.assertTrue(result["ce_scalar_variation_observed"])

    def test_gradient_or_state_mismatch_blocks_next_stage(self):
        comparisons = {
            name: {"passed": name != "ibkd_repeat", "scalar_values_exact": True}
            for name in (
                "lg_repeat",
                "lg_vs_alg_before_controller_action",
                "ibkd_repeat",
            )
        }
        result = diagnose(comparisons, all_completed=True, controls_verified=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_comparisons"], ["ibkd_repeat"])

    def test_terminal_run_status_keeps_full_final_step_and_validation(self):
        validation = {
            "miou": 0.2,
            "pixel_accuracy": 0.5,
            "class_iou": {"road": 0.7},
            "confusion_matrix": [[1]],
        }
        final_step = {"step": 25, "loss": 1.0, "gradient_sha256": "gradient"}
        result = compact_run_status({
            "method": "lg",
            "final_step": final_step,
            "diagnostic_validation": validation,
            "decision": {"stable": True, "reasons": [], "diagnostics": {}},
        }, returncode=0, controls_ok=True)
        self.assertEqual(result["final_step"], final_step)
        self.assertEqual(result["diagnostic_validation"], validation)
        self.assertEqual(result["stability_decision"]["stable"], True)


if __name__ == "__main__":
    unittest.main()
