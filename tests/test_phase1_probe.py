from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ibkd_seg.phase1.models import create_student
from ibkd_seg.phase1.probe import (
    build_probe,
    confusion_from_tensors,
    module_state_sha256,
    train_candidate,
)
from ibkd_seg.phase1.probe_data import PetRecord, load_targets
from ibkd_seg.phase1.run_probe_smoke import _validate_smoke_config, _variant


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_phase1_v1.json"
SMOKE_PATH = (
    REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_probe_smoke_b64_v1.json"
)


class Phase1ProbeTargetTest(unittest.TestCase):
    def test_trimap_mapping_boundary_ignore_and_half_foreground_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "pet.jpg"
            trimap_path = root / "pet.png"
            Image.new("RGB", (4, 4), color=(128, 64, 32)).save(image_path)
            trimap = np.array(
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 1, 2],
                    [3, 3, 1, 2],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(trimap, mode="L").save(trimap_path)
            record = PetRecord("pet", 0, image_path, trimap_path)
            input_target, grid_target = load_targets(
                record,
                input_size=4,
                grid_size=(2, 2),
            )
        self.assertEqual(set(int(value) for value in torch.unique(input_target)), {0, 1, 255})
        self.assertTrue(
            torch.equal(
                grid_target,
                torch.tensor([[1, 0], [255, 1]], dtype=torch.uint8),
            )
        )

    def test_confusion_excludes_boundary_pixels(self) -> None:
        confusion = confusion_from_tensors(
            torch.tensor([[1, 0], [1, 0]]),
            torch.tensor([[1, 0], [255, 1]]),
        )
        self.assertEqual(confusion.true_positive, 1)
        self.assertEqual(confusion.true_negative, 1)
        self.assertEqual(confusion.false_positive, 0)
        self.assertEqual(confusion.false_negative, 1)
        self.assertEqual(confusion.ignored, 1)
        self.assertEqual(confusion.metrics()["mean_iou"], 0.5)


class Phase1ProbeTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.probe_config = cls.protocol["frozen_spatial_probe"]["probe"]

    def test_probe_initialization_and_batch_order_are_deterministic(self) -> None:
        device = torch.device("cpu")
        first = build_probe(self.probe_config, seed=1, device=device)
        second = build_probe(self.probe_config, seed=1, device=device)
        self.assertEqual(sum(parameter.numel() for parameter in first.parameters()), 386)
        self.assertEqual(module_state_sha256(first), module_state_sha256(second))

        generator = torch.Generator().manual_seed(7)
        train_features = torch.randn(6, 192, 2, 2, generator=generator)
        validation_features = torch.randn(4, 192, 2, 2, generator=generator)
        train_targets = torch.tensor(
            [
                [[0, 0], [1, 1]],
                [[1, 255], [0, 1]],
                [[0, 1], [0, 1]],
                [[1, 1], [0, 0]],
                [[0, 255], [1, 0]],
                [[1, 0], [1, 0]],
            ],
            dtype=torch.uint8,
        )
        validation_targets = train_targets[:4].clone()
        _, first_result = train_candidate(
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            probe_config=self.probe_config,
            learning_rate=0.03,
            seed=1,
            device=device,
            epochs=2,
        )
        _, second_result = train_candidate(
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            probe_config=self.probe_config,
            learning_rate=0.1,
            seed=1,
            device=device,
            epochs=2,
        )
        self.assertEqual(
            first_result["initial_probe_state_sha256"],
            second_result["initial_probe_state_sha256"],
        )
        self.assertEqual(
            first_result["batch_order_sha256_by_epoch"],
            second_result["batch_order_sha256_by_epoch"],
        )
        self.assertEqual(
            first_result["gradient_contract"],
            {
                "cached_feature_gradient_tensor_count": 0,
                "probe_gradient_tensor_count": 2,
            },
        )

    def test_deit_last_block_is_pre_norm_nchw_192_by_14_by_14(self) -> None:
        model = create_student(num_classes=37, drop_path_rate=0.1).eval()
        model.requires_grad_(False)
        with torch.inference_mode():
            _, intermediates = model.forward_intermediates(
                torch.randn(1, 3, 224, 224),
                indices=[11],
                norm=False,
                output_fmt="NCHW",
                intermediates_only=False,
            )
        self.assertEqual(len(intermediates), 1)
        self.assertEqual(tuple(intermediates[0].shape), (1, 192, 14, 14))
        self.assertFalse(intermediates[0].requires_grad)


class Phase1ProbeSmokeConfigTest(unittest.TestCase):
    def test_smoke_config_is_non_scientific_and_matches_locked_protocol(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
        _validate_smoke_config(protocol, smoke, PROTOCOL_PATH)
        self.assertFalse(smoke["scientific_result"])
        self.assertFalse(smoke["official_test_accessed"])
        self.assertEqual(smoke["data"]["test_samples"], 0)
        self.assertEqual(smoke["task_count"]["lr_candidates"], 18)

    def test_variant_names_are_unambiguous(self) -> None:
        self.assertEqual(_variant("vanilla", None), "vanilla")
        self.assertEqual(_variant("ibkd", 0.25), "ibkd_lambda_0.25")
        self.assertEqual(_variant("ibkd", 0.5), "ibkd_lambda_0.5")


if __name__ == "__main__":
    unittest.main()
