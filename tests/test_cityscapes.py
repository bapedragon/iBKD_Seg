from __future__ import annotations

import copy
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ibkd_seg.cityscapes.data import Cityscapes, LABEL_IDS, audit, encode_mask, verify_manifest
from ibkd_seg.cityscapes.evaluation import confusion_update, metrics, sliding_logits
from ibkd_seg.cityscapes.models import ChunkedCrossAttention
from ibkd_seg.cityscapes.run import build_commands
from ibkd_seg.cityscapes.runtime import DEFAULT_CONFIG, load_config, train, validate_config, validate_manifest_contract
from ibkd_seg.cityscapes.smoke import make_fixture
from ibkd_seg.phase1.models import ConvCrossAttention


def small_config():
    config = load_config(DEFAULT_CONFIG)
    config.update({"protocol_id": "cityscapes_resume_test", "status": "synthetic_smoke_not_scientific",
                   "crop_size": [32, 64], "eval_stride": [32, 64], "scale_range": [1.0, 1.0],
                   "decoder_channels": 8, "epochs": 2, "warmup_epochs": 0,
                   "validation_every": 1, "precision": "fp32", "num_workers": 0,
                   "batch_size": 2, "student_seeds": [1]})
    return config


class CityscapesContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_official_label_ids_and_void(self):
        mask = np.array([list(LABEL_IDS) + [0, 9, 10, 14, 15, 16, 18, 29, 30, 255]], dtype=np.uint8)
        self.assertEqual(encode_mask(mask).tolist(), [list(range(19)) + [255] * 10])
        with self.assertRaises(ValueError):
            encode_mask(np.array([[99]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            encode_mask(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_global_iou_and_ignore(self):
        matrix = torch.zeros((19, 19), dtype=torch.int64)
        confusion_update(matrix, torch.tensor([0, 0, 1, 18]), torch.tensor([0, 1, 1, 255]))
        confusion_update(matrix, torch.tensor([1, 1]), torch.tensor([0, 1]))
        result = metrics(matrix)
        self.assertAlmostEqual(result["class_iou"]["road"], 1 / 3)
        self.assertAlmostEqual(result["class_iou"]["sidewalk"], 2 / 4)
        self.assertAlmostEqual(result["miou"], (1 / 3 + 2 / 4) / 2)
        self.assertEqual(result["valid_pixels"], 5)
        self.assertAlmostEqual(result["pixel_accuracy"], 3 / 5)
        self.assertIsNone(result["class_iou"]["car"])

    def test_native_sliding_windows_cover_edges_and_padding(self):
        class Pointwise(nn.Module):
            def forward(self, image):
                return torch.cat([image[:, :1] * (i + 1) for i in range(19)], dim=1)
        config = small_config()
        config["eval_stride"] = [16, 32]
        model = Pointwise()
        for shape in ((1, 3, 35, 99), (1, 3, 17, 25)):
            inputs = torch.randn(*shape)
            actual = sliding_logits(model, inputs, config, torch.device("cpu"))
            torch.testing.assert_close(actual, model(inputs), rtol=1e-6, atol=1e-5)

    def test_chunked_attention_matches_original_outputs_and_gradients(self):
        torch.manual_seed(5)
        original = ConvCrossAttention(16, num_heads=4)
        chunked = ChunkedCrossAttention(16, chunk_size=7, recompute=True)
        chunked.load_state_dict(original.state_dict(), strict=True)
        student1 = torch.randn(2, 16, 3, 5, requires_grad=True)
        teacher1 = torch.randn(2, 16, 3, 5, requires_grad=True)
        student2, teacher2 = student1.detach().clone().requires_grad_(), teacher1.detach().clone().requires_grad_()
        output1, output2 = original(student1, teacher1), chunked(student2, teacher2)
        torch.testing.assert_close(output1, output2, atol=1e-6, rtol=1e-5)
        output1.square().mean().backward()
        output2.square().mean().backward()
        torch.testing.assert_close(student1.grad, student2.grad, atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(teacher1.grad, teacher2.grad, atol=1e-7, rtol=1e-4)
        for (name1, parameter1), (name2, parameter2) in zip(original.named_parameters(), chunked.named_parameters(), strict=True):
            self.assertEqual(name1, name2)
            torch.testing.assert_close(parameter1.grad, parameter2.grad, atol=1e-7, rtol=1e-4)

    def test_dataset_audit_detects_changed_bytes_and_rejects_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            manifest = audit(root, synthetic=True)
            verify_manifest(root, manifest)
            with self.assertRaises(ValueError):
                audit(root)
            image_path = root / manifest["splits"]["train"][0]["image"]
            image_path.write_bytes(image_path.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                verify_manifest(root, manifest)

    def test_paired_geometry_deterministic_across_method_and_epoch_restarts(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            manifest = audit(root, synthetic=True)
            row = manifest["splits"]["train"][0]
            with Image.open(root / row["mask"]) as source:
                labels = np.array(source)
            # Red channel equals raw labelId, so any geometry mismatch is observable.
            Image.fromarray(np.repeat(labels[:, :, None], 3, axis=2)).save(root / row["image"])
            config = small_config()
            first = Cityscapes(root, [row], config, training=True, seed=1)
            other = Cityscapes(root, [row], config, training=True, seed=1)
            first.epoch = other.epoch = 3
            image, target, _ = first[0]
            image2, target2, _ = other[0]
            self.assertTrue(torch.equal(image, image2))
            self.assertTrue(torch.equal(target, target2))
            raw_red = ((image[0] * 0.229 + 0.485) * 255).round().byte().numpy()
            self.assertTrue(np.array_equal(encode_mask(raw_red), target.numpy()))

    def test_protocol_rejects_bad_geometry_and_synthetic_data_in_real_run(self):
        config = load_config(DEFAULT_CONFIG)
        invalid = copy.deepcopy(config)
        invalid["eval_stride"] = [1024, 512]
        with self.assertRaises(ValueError):
            validate_config(invalid)
        with self.assertRaises(ValueError):
            validate_manifest_contract({"synthetic": True, "splits": {"train": [], "val": []}}, config)

    def test_matrix_has_one_shared_teacher_and_four_methods_three_seeds(self):
        config = load_config(DEFAULT_CONFIG)
        args = Namespace(seeds=None, config=DEFAULT_CONFIG, data_dir=Path("/tmp/cityscapes"),
                         manifest=Path("/tmp/manifest.json"), device="cuda", output_dir=Path("/tmp/new_results"), resume=False)
        jobs = build_commands(args, config)
        self.assertEqual(len(jobs), 13)
        checkpoints = []
        for name, command, _ in jobs:
            if "--teacher-checkpoint" in command:
                checkpoints.append(command[command.index("--teacher-checkpoint") + 1])
            if name.startswith("vanilla"):
                self.assertNotIn("--teacher-checkpoint", command)
        self.assertEqual(len(checkpoints), 9)
        self.assertEqual(len(set(checkpoints)), 1)

    def test_epoch_resume_reproduces_uninterrupted_student_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root / "data")
            manifest = audit(root / "data", synthetic=True)
            kwargs = dict(config=small_config(), data_root=root / "data", manifest=manifest,
                          kind="student", method="vanilla", seed=1, device=torch.device("cpu"))
            full = train(output=root / "full", **kwargs)
            paused = train(output=root / "resumed", stop_after_epoch=1, **kwargs)
            self.assertEqual(paused["status"], "paused_at_epoch_boundary")
            resumed = train(output=root / "resumed", resume=True, **kwargs)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(full["selected_val_miou"], resumed["selected_val_miou"])
            self.assertEqual(full["selected_val_pixel_accuracy"], resumed["selected_val_pixel_accuracy"])
            self.assertEqual(resumed["selection_metric"], "pixel_accuracy")
            a = torch.load(root / "full/latest.pt", weights_only=True)
            b = torch.load(root / "resumed/latest.pt", weights_only=True)
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            # Refuse changed hyperparameters when resuming an existing checkpoint.
            changed = copy.deepcopy(kwargs)
            changed["config"]["learning_rate"] /= 2
            with self.assertRaises(ValueError):
                train(output=root / "resumed", resume=True, **changed)

    def test_accuracy_selects_checkpoint_even_when_iou_is_lower(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root / "data")
            manifest = audit(root / "data", synthetic=True)
            # The later epoch has higher accuracy but lower mIoU: both reported
            # metrics must come from the accuracy-selected epoch, without mixing maxima.
            values = [{"pixel_accuracy": 0.8, "miou": 0.7}, {"pixel_accuracy": 0.9, "miou": 0.6}]
            with patch("ibkd_seg.cityscapes.runtime.evaluate", side_effect=values):
                result = train(config=small_config(), data_root=root / "data", manifest=manifest,
                               output=root / "run", kind="student", method="vanilla", seed=1, device=torch.device("cpu"))
            self.assertEqual(result["selected_epoch"], 2)
            self.assertEqual(result["selected_val_pixel_accuracy"], 0.9)
            self.assertEqual(result["selected_val_miou"], 0.6)
            selected = torch.load(root / "run/best.pt", weights_only=True)
            self.assertEqual(selected["epoch"], 2)
            self.assertEqual(selected["validation"], values[1])


if __name__ == "__main__":
    unittest.main()
