import copy
import json
import unittest
from pathlib import Path

import torch

from ibkd_seg.cityscapes.ibkd_reproducibility_forensics import (
    DiagnosticCBAM,
    _operator_names,
    _validate_config,
    compare_pair,
    diagnose,
    gradient_report,
)
from ibkd_seg.phase1.models import DeformableCBAM


class CityscapesIBKDReproducibilityForensicsTests(unittest.TestCase):
    @staticmethod
    def run_summary():
        environment = {
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cpu_threads": 1,
        }
        gradient = {
            "sha256": "gradient",
            "parameter_sha256": {"weight": "weight-gradient"},
            "nonfinite_gradient_names": [],
        }
        return {
            "status": "completed",
            "sample_ids": ["sample"],
            "input_sha256": "input",
            "student_initial_state_sha256": "student-initial",
            "base_guidance_initial_state_sha256": "guide-initial",
            "variant_guidance_state_sha256": "variant-initial",
            "teacher_state_sha256": "teacher",
            "logits_sha256": "logits",
            "student_forward_sha256": "student-forward",
            "teacher_forward_sha256": "teacher-forward",
            "alignment_sha256": "alignment",
            "fusion_sha256": "fusion",
            "objective_sha256": "objective",
            "gradients": {
                "student": copy.deepcopy(gradient),
                "guidance": copy.deepcopy(gradient),
            },
            "nondeterministic_operators": [],
            "environment": environment,
        }

    def test_exact_pair_passes(self):
        run = self.run_summary()
        result = compare_pair(run, copy.deepcopy(run))
        self.assertTrue(result["passed"])
        self.assertEqual(result["student_gradient_mismatch_names"], [])

    def test_parameter_gradient_mismatch_is_reported(self):
        left = self.run_summary()
        right = copy.deepcopy(left)
        right["gradients"]["student"]["sha256"] = "different"
        right["gradients"]["student"]["parameter_sha256"]["weight"] = "different"
        result = compare_pair(left, right)
        self.assertFalse(result["passed"])
        self.assertEqual(result["student_gradient_mismatch_names"], ["weight"])

    def test_candidate_authorizes_only_next_25_step_audit(self):
        comparisons = {
            name: {"passed": True, "left_operators": [], "right_operators": []}
            for name in (
                "alignment_only",
                "attention_only",
                "adaptive_regular",
                "flatmax_regular",
                "flatmax_gpu_deform",
                "adaptive_cpu_deform",
                "full_original",
                "full_deterministic_candidate",
            )
        }
        comparisons["adaptive_regular"]["passed"] = False
        comparisons["flatmax_gpu_deform"]["passed"] = False
        comparisons["full_original"]["passed"] = False
        result = diagnose(comparisons, all_completed=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["code"], "deterministic_candidate_passed")
        self.assertFalse(result["adaptive_max_pool_reproducible"])
        self.assertFalse(result["gpu_deform_reproducible"])

    def test_candidate_warning_blocks_next_audit(self):
        comparisons = {
            name: {"passed": True, "left_operators": [], "right_operators": []}
            for name in (
                "alignment_only",
                "attention_only",
                "flatmax_regular",
                "full_deterministic_candidate",
            )
        }
        comparisons["full_deterministic_candidate"]["left_operators"] = [
            "unexpected_cuda_backward"
        ]
        result = diagnose(comparisons, all_completed=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["code"], "deterministic_candidate_failed")

    def test_flattened_max_preserves_adaptive_pool_tie_gradient(self):
        inputs = torch.tensor([[[[3.0, 3.0], [1.0, 0.0]]]], requires_grad=True)
        original = DeformableCBAM(1)
        diagnostic = DiagnosticCBAM(
            copy.deepcopy(original), channel_mode="flatmax", spatial_mode="regular"
        )
        adaptive = original.channel.maximum(inputs)
        flattened = inputs.flatten(2).max(dim=2).values[:, :, None, None]
        self.assertTrue(torch.equal(adaptive, flattened))
        adaptive_gradient = torch.autograd.grad(adaptive.sum(), inputs, retain_graph=True)[0]
        flattened_gradient = torch.autograd.grad(flattened.sum(), inputs)[0]
        self.assertTrue(torch.equal(adaptive_gradient, flattened_gradient))
        self.assertEqual(diagnostic.channel_mode, "flatmax")

    def test_cpu_deform_candidate_matches_original_on_cpu(self):
        torch.manual_seed(7)
        original = DeformableCBAM(8).eval()
        candidate = DiagnosticCBAM(
            copy.deepcopy(original).eval(),
            channel_mode="flatmax",
            spatial_mode="cpu_deform",
        )
        inputs = torch.randn(2, 8, 7, 9, requires_grad=True)
        expected = original(inputs)
        actual = candidate(inputs)
        self.assertTrue(torch.equal(expected, actual))
        expected_gradient = torch.autograd.grad(
            expected.sum(), inputs, retain_graph=True
        )[0]
        actual_gradient = torch.autograd.grad(actual.sum(), inputs)[0]
        self.assertTrue(torch.equal(expected_gradient, actual_gradient))

        original.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        original(inputs.detach().clone().requires_grad_(True)).sum().backward()
        candidate(inputs.detach().clone().requires_grad_(True)).sum().backward()
        for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
            original.named_parameters(),
            candidate.original.named_parameters(),
            strict=True,
        ):
            self.assertEqual(expected_name, actual_name)
            self.assertIsNotNone(expected_parameter.grad)
            self.assertIsNotNone(actual_parameter.grad)
            self.assertTrue(torch.equal(expected_parameter.grad, actual_parameter.grad))

    def test_gradient_report_is_repeatable(self):
        layer = torch.nn.Linear(3, 2)
        named = list(layer.named_parameters())
        loss = layer(torch.ones(2, 3)).square().mean()
        first = torch.autograd.grad(loss, [parameter for _, parameter in named], retain_graph=True)
        second = torch.autograd.grad(loss, [parameter for _, parameter in named])
        self.assertEqual(
            gradient_report(named, first)["sha256"],
            gradient_report(named, second)["sha256"],
        )

    def test_deterministic_warning_operator_is_extracted(self):
        messages = [
            "adaptive_max_pool2d_backward_cuda does not have a deterministic implementation"
        ]
        self.assertEqual(
            _operator_names(messages), ["adaptive_max_pool2d_backward_cuda"]
        )

    def test_forensics_config_is_locked(self):
        root = Path(__file__).resolve().parents[1]
        path = (
            root
            / "phase4/phase4_cityscapes/configs/"
            "paper_l16_crop512_ibkd_forensics1_v10.json"
        )
        config = json.loads(path.read_text())
        _validate_config(config)
        config["independent_repeats_per_variant"] = 1
        with self.assertRaises(ValueError):
            _validate_config(config)


if __name__ == "__main__":
    unittest.main()
