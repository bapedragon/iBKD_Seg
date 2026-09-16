import unittest

import torch

from ibkd_seg.phase1.models import LocalityGuidance, TransformerAggregationPooling


class LargeStudentGuidance(unittest.TestCase):
    def test_lg_preserves_first_middle_last_rule_at_both_depths(self):
        for depth, width, expected in ((12, 192, (0, 6, 11)), (24, 1024, (0, 12, 23))):
            guide = LocalityGuidance((2, 3, 4), student_channels=width, student_blocks=depth)
            self.assertEqual(guide.selected_blocks, expected)
            features = [torch.randn(1, width, 2, 2, requires_grad=True) for _ in range(depth)]
            teachers = [torch.randn(1, c, 3, 3) for c in (2, 3, 4)]
            guide(features, teachers).backward()
            active = tuple(i for i, x in enumerate(features) if x.grad is not None and x.grad.abs().sum() > 0)
            self.assertEqual(active, expected)

    def test_aggregation_uses_all_24_blocks_and_learns_weights(self):
        pool = TransformerAggregationPooling(student_blocks=24)
        features = [torch.full((1, 2, 2, 2), float(i), requires_grad=True) for i in range(24)]
        output = pool(features)
        torch.testing.assert_close(output, torch.full((1, 3, 2, 2, 2), 11.5))
        output.square().sum().backward()
        self.assertTrue(all(x.grad is not None and x.grad.abs().sum() > 0 for x in features))
        self.assertEqual(tuple(pool.weights.shape), (3, 24))
        self.assertGreater(float(pool.weights.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            pool(features[:12])

    def test_default_checkpoint_contract_is_unchanged(self):
        guide = LocalityGuidance()
        restored = LocalityGuidance(student_channels=192, student_blocks=12)
        restored.load_state_dict(guide.state_dict(), strict=True)
        self.assertEqual(tuple(guide.projections[0].weight.shape), (16, 192, 1, 1))
        self.assertEqual(set(TransformerAggregationPooling().state_dict()), {"weights"})
        self.assertEqual(tuple(TransformerAggregationPooling().weights.shape), (3, 12))


if __name__ == "__main__":
    unittest.main()
