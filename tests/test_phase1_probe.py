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
    evaluate_probe_both_resolutions,
    module_state_sha256,
    train_candidate,
)
from ibkd_seg.phase1.probe_data import PetRecord, load_targets
from ibkd_seg.phase1.run_probe_smoke import (
    _smoke_policy_gates,
    _validate_smoke_config,
    _variant,
)
from ibkd_seg.phase1.run_probe_full import (
    EXPECTED_ENCODER_SEEDS,
    EXPECTED_PROBE_SEEDS,
    EXPECTED_VARIANTS,
    METRIC_PATHS,
    _aggregate,
    _select_candidate,
    _validate_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_phase1_v1.json"
SMOKE_B64_PATH = (
    REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_probe_smoke_b64_v1.json"
)
SMOKE_B128_PATH = (
    REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_probe_smoke_b128_v1.json"
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

    def test_grid_input_metrics_and_capture_share_one_probe_pass(self) -> None:
        class CountingProbe(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return torch.stack((features[:, 0], features[:, 1]), dim=1)

        probe = CountingProbe()
        features = torch.zeros(3, 192, 2, 2)
        features[:, 1] = 1.0
        grid_targets = torch.ones(3, 2, 2, dtype=torch.uint8)
        input_targets = torch.ones(3, 4, 4, dtype=torch.uint8)
        metrics, captured = evaluate_probe_both_resolutions(
            probe,
            features,
            grid_targets,
            input_targets,
            batch_size=2,
            device=torch.device("cpu"),
            input_size=4,
            capture_indices={1},
        )
        self.assertEqual(probe.calls, 2)
        self.assertEqual(metrics["grid_14x14"]["mean_iou"], 1.0)
        self.assertEqual(metrics["input_224"]["mean_iou"], 1.0)
        self.assertEqual(set(captured), {1})
        self.assertEqual(tuple(captured[1].shape), (4, 4))


class Phase1ProbeFullReportingTest(unittest.TestCase):
    def test_full_runner_accepts_the_committed_locked_protocol(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        _validate_protocol(protocol, PROTOCOL_PATH)

    def test_lr_tie_selects_lower_learning_rate(self) -> None:
        state = {"weight": torch.tensor([1.0])}
        selected_state, selected = _select_candidate(
            [
                (
                    state,
                    {
                        "learning_rate": 0.01,
                        "best_validation_grid_mean_iou": 0.75,
                    },
                ),
                (
                    {"weight": torch.tensor([2.0])},
                    {
                        "learning_rate": 0.03,
                        "best_validation_grid_mean_iou": 0.75,
                    },
                ),
            ]
        )
        self.assertIs(selected_state, state)
        self.assertEqual(selected["learning_rate"], 0.01)

    def test_aggregate_uses_probe_means_then_encoder_seed_statistics(self) -> None:
        results = []
        for variant_index, variant in enumerate(EXPECTED_VARIANTS):
            for encoder_seed in EXPECTED_ENCODER_SEEDS:
                for probe_seed in EXPECTED_PROBE_SEEDS:
                    value = variant_index / 10 + encoder_seed / 100 + probe_seed / 1000
                    split_metrics = {
                        resolution: {
                            metric: value
                            for candidate_resolution, metric in METRIC_PATHS.values()
                            if candidate_resolution == resolution
                        }
                        for resolution in {path[0] for path in METRIC_PATHS.values()}
                    }
                    results.append(
                        {
                            "variant": variant,
                            "encoder_seed": encoder_seed,
                            "probe_seed": probe_seed,
                            "validation": split_metrics,
                            "test": split_metrics,
                        }
                    )
        report = _aggregate(results)
        vanilla = report["variants"]["vanilla"]
        encoder_one = vanilla["by_encoder_seed"]["1"]["test"][
            "input_224_mean_iou"
        ]
        self.assertAlmostEqual(encoder_one["mean"], 0.013)
        across = vanilla["across_encoder_seed_means"]["test"][
            "input_224_mean_iou"
        ]
        self.assertAlmostEqual(across["mean"], 0.023)
        contrast = report["paired_primary_contrasts"][
            "ibkd_lambda_0.25_minus_alg"
        ]["difference_summary"]
        self.assertAlmostEqual(contrast["mean"], 0.1)
        self.assertEqual(contrast["count"], 3)


class Phase1ProbeSmokeConfigTest(unittest.TestCase):
    def test_smoke_config_is_non_scientific_and_matches_locked_protocol(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        for smoke_path, expected_batch_size in (
            (SMOKE_B64_PATH, 64),
            (SMOKE_B128_PATH, 128),
        ):
            with self.subTest(batch_size=expected_batch_size):
                smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
                _validate_smoke_config(protocol, smoke, PROTOCOL_PATH)
                self.assertEqual(
                    smoke["classification_input"]["batch_size"],
                    expected_batch_size,
                )
                self.assertFalse(smoke["scientific_result"])
                self.assertFalse(smoke["official_test_accessed"])
                self.assertEqual(smoke["data"]["test_samples"], 0)
                self.assertEqual(smoke["task_count"]["lr_candidates"], 18)
                policy_gates = _smoke_policy_gates(smoke)
                self.assertEqual(
                    policy_gates,
                    {
                        "official_test_not_accessed": True,
                        "smoke_metrics_for_scientific_selection_forbidden": True,
                        "smoke_marked_non_scientific": True,
                    },
                )
                self.assertTrue(all(policy_gates.values()))

    def test_variant_names_are_unambiguous(self) -> None:
        self.assertEqual(_variant("vanilla", None), "vanilla")
        self.assertEqual(_variant("ibkd", 0.25), "ibkd_lambda_0.25")
        self.assertEqual(_variant("ibkd", 0.5), "ibkd_lambda_0.5")


if __name__ == "__main__":
    unittest.main()
