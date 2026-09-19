import copy
import unittest

from ibkd_seg.cityscapes.ibkd_candidate_reproducibility import (
    candidate_controls_ok,
    diagnose,
    diagnose_repro2000,
    timing_summary,
)


class CityscapesIBKDCandidateReproducibilityTests(unittest.TestCase):
    @staticmethod
    def summary():
        stages = [
            {
                "stage": index,
                "wrapper": "DiagnosticCBAM",
                "channel_mode": "flatmax",
                "spatial_mode": "cpu_deform",
                "applied": True,
            }
            for index in range(3)
        ]
        return {
            "nondeterministic_operators": ["nll_loss2d_forward_out_cuda_template"],
            "ibkd_deterministic_candidate": {
                "candidate_id": "flatmax_cpu_deform_v1",
                "applied": True,
                "cpu_threads": 1,
                "stages": stages,
            },
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
                "cpu_threads": 1,
            },
        }

    def test_candidate_and_warn_only_controls_pass(self):
        self.assertTrue(candidate_controls_ok(self.summary()))

    def test_original_ibkd_nondeterministic_operator_blocks(self):
        summary = self.summary()
        summary["nondeterministic_operators"].append("compute_grad_input")
        self.assertFalse(candidate_controls_ok(summary))

    def test_incomplete_candidate_application_blocks(self):
        summary = copy.deepcopy(self.summary())
        summary["ibkd_deterministic_candidate"]["stages"][1]["applied"] = False
        self.assertFalse(candidate_controls_ok(summary))

    def test_exact_update_path_authorizes_2000_step_grid(self):
        result = diagnose({"passed": True}, all_completed=True, controls_verified=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["code"], "ibkd_candidate_warn25_reproducible")

    def test_gradient_mismatch_blocks_next_stage(self):
        result = diagnose({"passed": False}, all_completed=True, controls_verified=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["code"], "ibkd_candidate_update_path_not_reproducible")

    def test_exact_2000_step_update_path_authorizes_remaining_grid(self):
        result = diagnose_repro2000(
            {"passed": True}, all_completed=True, controls_verified=True
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["code"], "ibkd_candidate_repro2000_reproducible")

    def test_2000_step_gradient_mismatch_blocks_remaining_grid(self):
        result = diagnose_repro2000(
            {"passed": False}, all_completed=True, controls_verified=True
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["code"],
            "ibkd_candidate_repro2000_update_path_not_reproducible",
        )

    def test_timing_summary_keeps_step_distribution(self):
        result = timing_summary([
            {"seconds": 1.0},
            {"seconds": 3.0},
            {"seconds": 2.0},
        ])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["total_seconds"], 6.0)
        self.assertEqual(result["mean_step_seconds"], 2.0)
        self.assertEqual(result["median_step_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
