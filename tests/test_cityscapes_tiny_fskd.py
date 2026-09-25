import copy
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from ibkd_seg.cityscapes.tiny_fskd import (TinyFSKD, attention_rank_loss, pixel_logit_kd,
                                       weighted_components, verify_soft_rank)
from ibkd_seg.cityscapes.tiny_smoke import fixed_fskd_calibration, load_config


class TinyFSKDTests(unittest.TestCase):
    def test_v2_preserves_five_paths_and_only_adds_fskd(self):
        configs = Path(__file__).resolve().parents[1] / "phase4/Cityscapes_Segmenter-Ti16/configs"
        old = load_config(configs / "smoke25_v1.json")
        new = load_config(configs / "smoke25_fskd_v2.json")
        for key, value in old.items():
            if key not in {"protocol_id", "runs"}:
                self.assertEqual(new[key], value, key)
        self.assertEqual(new["runs"][:-1], old["runs"])
        self.assertEqual(new["runs"][-1]["method"], "fskd")
        self.assertFalse(new["fskd"]["beta_search"])
        self.assertIn("deferred_by_user", new["c2vkd_status"])
        raw = {key: torch.tensor(1.) for key in ("global", "patch", "attention", "logit_kd")}
        for key, value in weighted_components(raw).items():
            self.assertEqual(float(value), new["fskd"]["coefficients"][key])

    def test_pixel_kd_ignores_void_and_is_not_multiplied_by_image_area(self):
        s = torch.tensor([[[[.5]], [[1.2]]]], requires_grad=True)
        t = torch.tensor([[[[1.]], [[.2]]]], requires_grad=True)
        a = pixel_logit_kd(s, t, torch.zeros(1, 1, 1, dtype=torch.long))
        large_s = s.detach().expand(-1, -1, 4, 4).clone().requires_grad_()
        labels = torch.zeros(1, 4, 4, dtype=torch.long)
        labels[:, 0, 0] = 255
        b = pixel_logit_kd(large_s, t, labels)
        torch.testing.assert_close(a, b)
        b.backward()
        self.assertTrue((large_s.grad[:, :, 0, 0] == 0).all())
        self.assertIsNone(t.grad)
        with self.assertRaises(ValueError):
            pixel_logit_kd(s, t, torch.full((1, 1, 1), 255))

    def test_attention_keeps_soft_rank_gradient_to_cls_patch_probabilities(self):
        import torchsort
        torch.manual_seed(3)
        pre = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
        cls_patches = pre.softmax(-1)[:, :, 1:]
        teacher = torch.randn(2, 7, 2, 2, dtype=torch.double, requires_grad=True)
        actual = attention_rank_loss(cls_patches, teacher, (2, 2))
        s = torchsort.soft_rank(cls_patches.mean(1), regularization_strength=1.)
        t = torchsort.soft_rank(teacher.detach().square().mean(1).flatten(1).softmax(-1),
                                regularization_strength=1.)
        expected = 1 - (1 - 6 * (s - t).square().sum(-1) / (4 * 15)).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(pre.grad).all())
        self.assertGreater(float(pre.grad.abs().max()), 0.)
        self.assertIsNone(teacher.grad)
        self.assertEqual(verify_soft_rank(torch.device("cpu"))["forward_backward"], "passed")

    def test_actual_tiny_widths_feature_pairs_and_fixed_loss_gradients(self):
        torch.manual_seed(4)
        features = [torch.randn(2, 192, 2, 2, requires_grad=True) for _ in range(12)]
        teacher = [torch.randn(2, c, size, size, requires_grad=True)
                   for c, size in ((256, 8), (512, 4), (1024, 4), (2048, 4))]
        attention = torch.randn(2, 3, 5, requires_grad=True).softmax(-1)[:, :, 1:]
        logits = torch.randn(2, 19, 32, 32, requires_grad=True)
        tlogits = torch.randn_like(logits, requires_grad=True)
        labels = torch.randint(19, (2, 32, 32)); labels[:, 0, 0] = 255
        guide = TinyFSKD(crop=32)
        raw = guide(features, attention, teacher, logits, tlogits, labels)
        weighted = weighted_components(raw)
        (F.cross_entropy(logits, labels, ignore_index=255) + sum(weighted.values())).backward()
        for index in (0, 3):
            self.assertGreater(float(features[index].grad.abs().max()), 0.)
        self.assertIsNone(features[8].grad)  # Do not accidentally use a PiT/B0 stage mapping.
        self.assertIsNone(features[1].grad)  # Generic timm wrapper has a different stage1 index.
        for adapter in guide.align:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()))
        self.assertTrue(all(x.grad is None for x in teacher))
        self.assertIsNone(tlogits.grad)
        row = {"ce": 3., "guidance": float(sum(weighted.values()).detach()),
               "components": {k: float(v.detach()) for k, v in raw.items()}}
        result = fixed_fskd_calibration([row, copy.deepcopy(row)], {"coefficients": {"ce": 1.}})
        self.assertEqual(result["beta_candidates"], [])
        self.assertIsNone(result["pilot_beta"])
        self.assertFalse(result["validation_used"])


if __name__ == "__main__":
    unittest.main()
