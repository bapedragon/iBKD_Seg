import copy
import json
import math
import unittest
from pathlib import Path

import torch
from torch import nn

from ibkd_seg.cityscapes.official_api import FeatureCapture, student_spec
from ibkd_seg.cityscapes.official_assets import WEIGHTS, weight_manifest
from ibkd_seg.cityscapes.tiny_smoke import assert_state_close, beta_candidates, compare_runs


class TinyProtocol(unittest.TestCase):
    def test_common_protocol_matches_the_existing_large_track(self):
        root = Path(__file__).resolve().parents[1]
        tiny = json.loads((root / "phase4/Cityscapes_Segmenter-Ti16/configs/smoke25_v1.json").read_text())
        large = json.loads((root / "phase4/phase4_cityscapes/configs/paper_l16_crop512_final80000_v19.json").read_text())
        for key in ("seed", "train_samples", "val_samples", "batch_size", "drop_last", "image_size",
                    "crop_size", "window_size", "window_stride", "decoder_layers", "precision",
                    "optimizer", "learning_rate", "momentum", "weight_decay", "poly_power", "min_lr",
                    "total_steps", "gradient_checkpointing", "gradient_clipping", "attention_query_chunk",
                    "alg_window", "alg_threshold", "alg_warmup_epochs", "ibkd_warmup_epochs",
                    "controller_epoch_mean", "sampling_policy", "ibkd_deterministic_candidate_id"):
            with self.subTest(key=key):
                self.assertEqual(tiny[key], large[key])

    def test_beta_grid_uses_unweighted_segmentation_loss_and_doubles(self):
        rows = [{"ce": 4., "guidance": 2.}, {"ce": 2., "guidance": 1.},
                {"ce": 6., "guidance": 3.}]
        result = beta_candidates(rows)
        self.assertEqual(result["beta_candidates"], [.06, .12, .24, .48])
        self.assertFalse(result["validation_used"])
        self.assertEqual(result["optimizer_updates"], 0)
        for candidate, ratio in zip(result["measured_ratios"], [.03, .06, .12, .24]):
            self.assertAlmostEqual(candidate["weighted_guidance_to_seg_loss_median"], ratio)
        for bad in [[], [{"ce": 1, "guidance": 0}], [{"ce": 1, "guidance": math.inf}],
                    [{"ce": 0, "guidance": 1}], [{"ce": math.nan, "guidance": 1}]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                beta_candidates(bad)

    def test_default_assets_still_require_the_original_large_and_teacher(self):
        self.assertEqual(weight_manifest(), WEIGHTS)
        tiny = weight_manifest("tiny")
        self.assertNotIn("vit_large_384.npz", tiny)
        self.assertEqual(tiny["deeplabv3_r101.pth"], WEIGHTS["deeplabv3_r101.pth"])
        self.assertEqual(student_spec()["blocks"], 24)
        self.assertEqual(student_spec("vit_tiny_patch16_384")["channels"], 192)
        with self.assertRaises(ValueError):
            student_spec("unverified_random_fallback")

    def test_feature_capture_uses_encoder_width_and_retains_gradients(self):
        class Encoder(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.d_model = width
                self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])

        class Model(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.encoder = Encoder(width)

            def forward(self, image):
                tokens = image.mean((1, 2, 3))[:, None, None].expand(-1, 5, self.encoder.d_model)
                for block in self.encoder.blocks:
                    tokens = block(tokens)
                return tokens

        for width in (192, 384, 1024):
            model = Model(width)
            capture = FeatureCapture(model)
            logits, features = capture.forward(model, torch.ones(1, 3, 32, 32))
            self.assertEqual([tuple(x.shape) for x in features], [(1, width, 2, 2)] * 3)
            features[0].square().mean().backward()
            self.assertIsNotNone(model.encoder.blocks[0].weight.grad)
            self.assertFalse(capture.enabled)
            self.assertTrue(all(x is None for x in capture.features))

    def test_resume_comparison_detects_optimizer_and_metadata_changes(self):
        original = {"state": [{"momentum_buffer": torch.tensor([1., 2.])}], "steps": 24}
        assert_state_close(original, copy.deepcopy(original), rtol=0, atol=0)
        changed = copy.deepcopy(original)
        changed["state"][0]["momentum_buffer"][0] = 3
        with self.assertRaises(AssertionError):
            assert_state_close(original, changed, rtol=2e-5, atol=2e-6)
        changed = copy.deepcopy(original)
        changed["steps"] = 0
        with self.assertRaises(AssertionError):
            assert_state_close(original, changed, rtol=2e-5, atol=2e-6)

    def test_cross_method_gate_rejects_mismatched_inputs_and_lg_alg_trajectory(self):
        rows = []
        for run_id, method in [("vanilla", "vanilla"), ("lg", "lg"), ("alg", "alg"),
                               ("ibkd_lambda025", "ibkd"), ("ibkd_lambda050", "ibkd")]:
            rows.append({"status": "passed", "run_id": run_id, "method": method,
                         "student_initial_state_sha256": "student", "input_sha256": "inputs",
                         "teacher_state_sha256": None if method == "vanilla" else "teacher",
                         "guidance_initial_state_sha256": "guide",
                         "calibration": {"ce_median": 2., "guidance_median": 1., "pilot_beta": .06},
                         "losses": [{"step": 1, "loss": 2.06, "ce": 2., "guidance": 1., "grad_norm_unclipped": 1.}]})
        self.assertTrue(compare_runs(rows)["lg_alg_25step_trajectory_matches"])
        bad = copy.deepcopy(rows)
        bad[2]["input_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "input_sha256"):
            compare_runs(bad)
        bad = copy.deepcopy(rows)
        bad[2]["losses"][0]["ce"] = 10.
        with self.assertRaisesRegex(RuntimeError, "trajectory differs"):
            compare_runs(bad)


if __name__ == "__main__":
    unittest.main()
