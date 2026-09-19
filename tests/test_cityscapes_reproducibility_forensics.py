import copy
import unittest

import torch

from ibkd_seg.cityscapes.reproducibility_forensics import (
    _component_gradients,
    _operator_names,
    compare_cases,
    diagnose,
)


class CityscapesReproducibilityForensicsTests(unittest.TestCase):
    @staticmethod
    def runs():
        common = {
            "status": "completed",
            "input_sha256": "input",
            "student_initial_state_sha256": "student",
            "rng": {"before_forward": {"torch_cuda_sha256": "rng"}},
            "within_run": {
                "ce_backward_repeat_exact": True,
                "guidance_student_backward_repeat_exact": None,
                "guidance_module_backward_repeat_exact": None,
                "vanilla_ce_vs_total_gradient_exact": True,
            },
            "nondeterministic_operators": [],
        }
        runs = {case: copy.deepcopy(common) for case in ("vanilla_a", "vanilla_b", "lg_a", "lg_b")}
        for case in ("lg_a", "lg_b"):
            runs[case]["within_run"].update(
                guidance_student_backward_repeat_exact=True,
                guidance_module_backward_repeat_exact=True,
                vanilla_ce_vs_total_gradient_exact=None,
            )
        return runs

    @staticmethod
    def comparisons():
        return {
            name: {"passed": True, "checks": {}, "mismatched_fields": []}
            for name in (
                "vanilla_repeat_forward",
                "vanilla_repeat_ce_scalar",
                "vanilla_repeat_ce_backward",
                "vanilla_vs_lg_student_forward",
                "lg_repeat_forward",
                "lg_repeat_ce_scalar",
                "lg_repeat_ce_backward",
                "lg_repeat_guidance_backward",
                "lg_repeat_total_backward",
            )
        }

    def test_nested_case_comparison_reports_field(self):
        left = {"forward": {"logits_sha256": "a", "features": ["x", "y"]}}
        right = {"forward": {"logits_sha256": "b", "features": ["x", "z"]}}
        result = compare_cases(
            left, right, ["forward.logits_sha256", "forward.features"]
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["mismatched_fields"],
            ["forward.logits_sha256", "forward.features"],
        )
        self.assertEqual(result["mismatched_indices_by_field"]["forward.features"], [1])

    def test_exact_components_advance_to_optimizer_diagnostic(self):
        result = diagnose(self.runs(), self.comparisons())
        self.assertEqual(result["code"], "one_batch_components_exact")

    def test_vanilla_within_run_gradient_difference_is_base_backward(self):
        runs = self.runs()
        runs["vanilla_a"]["within_run"]["ce_backward_repeat_exact"] = False
        result = diagnose(runs, self.comparisons())
        self.assertEqual(result["code"], "base_ce_or_student_backward_nondeterminism")

    def test_guidance_gradient_difference_is_isolated(self):
        runs = self.runs()
        runs["lg_b"]["within_run"]["guidance_student_backward_repeat_exact"] = False
        result = diagnose(runs, self.comparisons())
        self.assertEqual(result["code"], "lg_guidance_backward_nondeterminism")

    def test_ce_scalar_only_difference_is_not_called_student_forward(self):
        comparisons = self.comparisons()
        comparisons["vanilla_repeat_ce_scalar"]["passed"] = False
        comparisons["lg_repeat_ce_scalar"]["passed"] = False
        result = diagnose(self.runs(), comparisons)
        self.assertEqual(result["code"], "ce_forward_scalar_reduction_nondeterminism")

    def test_deterministic_warning_operator_is_extracted(self):
        messages = [
            "nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but warn_only=True"
        ]
        self.assertEqual(_operator_names(messages), ["nll_loss2d_forward_out_cuda_template"])

    def test_component_gradient_hash_is_repeatable(self):
        layer = torch.nn.Linear(3, 2)
        named = list(layer.named_parameters())
        loss = layer(torch.ones(2, 3)).square().mean()
        first = _component_gradients(
            torch, loss, {"student": named}, retain_graph=True
        )
        second = _component_gradients(
            torch, loss, {"student": named}, retain_graph=False
        )
        self.assertEqual(first["student"]["sha256"], second["student"]["sha256"])


if __name__ == "__main__":
    unittest.main()
