import unittest
from pathlib import Path

import torch

from ibkd_seg.cityscapes.tiny_c2vkd import (COEFFICIENTS, pixel_pdd, training_loss,
                                         weighted_components, verify_pool_asset)
from ibkd_seg.cityscapes.tiny_smoke import load_config, fixed_fskd_calibration


class TinyC2VKDTests(unittest.TestCase):
    def test_v3_preserves_six_paths_and_common_protocol(self):
        configs = Path(__file__).resolve().parents[1] / "phase4/Cityscapes_Segmenter-Ti16/configs"
        old = load_config(configs / "smoke25_fskd_v2.json")
        new = load_config(configs / "smoke25_fskd_c2vkd_v3.json")
        for key, value in old.items():
            if key not in {"protocol_id", "runs", "c2vkd_status"}:
                self.assertEqual(new[key], value, key)
        self.assertEqual(new["runs"][:-1], old["runs"])
        self.assertEqual(new["c2vkd"]["coefficients"], COEFFICIENTS)
        self.assertFalse(new["c2vkd"]["author_exact_reproduction"])
        self.assertEqual(new["runs"][-1]["comparison_group"], "supplementary_extra_pretraining")

    def test_pdd_equation_void_pixels_and_area_reduction(self):
        s = torch.tensor([[[[.5]], [[1.2]], [[-.2]]]], requires_grad=True)
        t = torch.tensor([[[[1.]], [[.2]], [[.8]]]], requires_grad=True)
        a = pixel_pdd(s, t, torch.zeros(1, 1, 1, dtype=torch.long))
        sp, tp = s.flatten().softmax(0), t.detach().flatten().softmax(0)
        sb = torch.stack((sp[0], sp[1:].sum()))
        q = torch.stack(((tp[0] + 1) / 2, tp[1:].sum() / 2))
        torch.testing.assert_close(a, (.5 * sb * (sb.log() - q.log())).sum())
        large = s.detach().expand(-1, -1, 4, 4).clone().requires_grad_()
        labels = torch.zeros(1, 4, 4, dtype=torch.long); labels[:, 0, 0] = 255
        b = pixel_pdd(large, t, labels)
        torch.testing.assert_close(a, b)
        b.backward()
        self.assertTrue((large.grad[:, :, 0, 0] == 0).all())
        self.assertGreater(float(large.grad.abs().max()), 0.)
        self.assertIsNone(t.grad)
        with self.assertRaises(ValueError):
            pixel_pdd(s, t, torch.full((1, 1, 1), 255))

    def test_no_extra_ce_and_no_beta_tuning(self):
        ce = torch.tensor(100., requires_grad=True)
        raw = {k: torch.tensor(2., requires_grad=True) for k in COEFFICIENTS if k != "ce"}
        weighted = weighted_components(raw)
        loss = training_loss(ce, sum(weighted.values()))
        torch.testing.assert_close(loss, torch.tensor(3.4))
        loss.backward()
        self.assertEqual(float(ce.grad), 0.)
        for name, component in raw.items():
            self.assertAlmostEqual(float(component.grad), COEFFICIENTS[name])
        result = fixed_fskd_calibration([{"ce": 100., "guidance": 3.4, "components": {"pdd": 2.}}],
                                        {"coefficients": COEFFICIENTS}, "c2vkd")
        self.assertEqual(result["beta_candidates"], [])
        self.assertIsNone(result["pilot_beta"])
        self.assertEqual(result["status"], "fixed_c2vkd_recipe_no_beta_search")

    def test_unverified_pool_is_rejected_before_deserialization(self):
        with self.assertRaises(RuntimeError):
            verify_pool_asset(Path(__file__))


if __name__ == "__main__":
    unittest.main()
