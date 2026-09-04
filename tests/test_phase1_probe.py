from __future__ import annotations

import unittest

import torch

from ibkd_seg.phase1.probe import build_probe, confusion_from_tensors, evaluate_probe


PROBE_CONFIG = {
    "parameter_count": 386,
    "initialization": {"weight_std": 0.01, "bias": 0.0},
}


class Phase1ProbeTest(unittest.TestCase):
    def test_probe_contract_and_seed_are_deterministic(self) -> None:
        device = torch.device("cpu")
        first = build_probe(PROBE_CONFIG, seed=3, device=device)
        second = build_probe(PROBE_CONFIG, seed=3, device=device)
        self.assertEqual(sum(parameter.numel() for parameter in first.parameters()), 386)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(torch.equal(first.bias, second.bias))

    def test_global_confusion_counts(self) -> None:
        confusion = confusion_from_tensors(
            torch.tensor([[1, 1], [0, 0]]),
            torch.tensor([[1, 0], [1, 0]]),
        )
        self.assertEqual(confusion.true_positive, 1)
        self.assertEqual(confusion.true_negative, 1)
        self.assertEqual(confusion.false_positive, 1)
        self.assertEqual(confusion.false_negative, 1)
        self.assertEqual(confusion.metrics()["mean_iou"], 1 / 3)

    def test_input_resolution_evaluation_upsamples_logits(self) -> None:
        device = torch.device("cpu")
        probe = build_probe(PROBE_CONFIG, seed=1, device=device)
        with torch.no_grad():
            probe.weight.zero_()
            probe.bias[:] = torch.tensor([1.0, 0.0])
        features = torch.zeros(2, 192, 2, 2)
        targets = torch.zeros(2, 8, 8, dtype=torch.uint8)
        metrics = evaluate_probe(
            probe,
            features,
            targets,
            batch_size=1,
            device=device,
            output_size=(8, 8),
        )
        self.assertEqual(metrics["pixel_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
