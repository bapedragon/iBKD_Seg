from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

import torch

from ibkd_seg.phase1.controllers import GuidanceController
from ibkd_seg.phase1.data import build_stratified_split
from ibkd_seg.phase1.full_matrix import build_full_tasks
from ibkd_seg.phase1.models import IBKD, LocalityGuidance, ResNet56
from ibkd_seg.phase1.run_full import aggregate_students
from ibkd_seg.phase1.timing_matrix import build_tasks


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_phase1_v1.json"


class Phase1TimingMatrixTest(unittest.TestCase):
    def test_exact_six_by_two_student_matrix(self) -> None:
        tasks = build_tasks(Path("/tmp/results"))
        self.assertEqual(len(tasks), 12)
        observed = {
            (task.method, task.fusion_ratio, task.batch_size) for task in tasks
        }
        expected = {
            (method, fusion_ratio, batch)
            for method, fusion_ratio in (
                ("vanilla", None),
                ("kd", None),
                ("lg", None),
                ("alg", None),
                ("ibkd", 0.25),
                ("ibkd", 0.5),
            )
            for batch in (64, 128)
        }
        self.assertEqual(observed, expected)
        self.assertEqual(len({task.run_name for task in tasks}), 12)

    def test_config_marks_timing_as_non_scientific(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertTrue(config["status"].startswith("locked_full_classification_"))
        self.assertEqual(
            config["classification"]["student"]["candidate_train_batch_sizes"],
            [64, 128],
        )
        self.assertEqual(
            config["classification"]["methods"]["ibkd"][
                "candidate_fusion_ratio_lambdas"
            ],
            [0.25, 0.5],
        )
        smoke = config["smoke"]
        self.assertFalse(smoke["scientific_result"])
        self.assertTrue(smoke["selection_from_smoke_metrics_forbidden"])
        self.assertFalse(smoke["official_test_accessed"])
        self.assertEqual(smoke["student_task_count"], 12)
        self.assertEqual(smoke["samples"], {"train": 2940, "validation": 740, "test": 0})
        self.assertEqual(smoke["status"], "complete_13_of_13_no_oom")


class Phase1FullMatrixTest(unittest.TestCase):
    def test_each_batch_issue_has_six_variants_by_three_seeds(self) -> None:
        for batch_size in (64, 128):
            tasks = build_full_tasks(Path("/tmp/results"), batch_size=batch_size)
            self.assertEqual(len(tasks), 18)
            observed = {
                (task.method, task.fusion_ratio, task.batch_size, task.seed)
                for task in tasks
            }
            expected = {
                (method, fusion_ratio, batch_size, seed)
                for method, fusion_ratio in (
                    ("vanilla", None),
                    ("kd", None),
                    ("lg", None),
                    ("alg", None),
                    ("ibkd", 0.25),
                    ("ibkd", 0.5),
                )
                for seed in (1, 2, 3)
            }
            self.assertEqual(observed, expected)
            self.assertEqual(len({task.run_name for task in tasks}), 18)

    def test_aggregate_reports_raw_mean_and_sample_sd(self) -> None:
        rows = [
            {
                "kind": "student",
                "method": "ibkd",
                "fusion_ratio_lambda": 0.25,
                "batch_size": 64,
                "seed": seed,
                "status": "complete",
                "test_macro_top1": value,
                "test_overall_top1": value + 1.0,
                "test_top5": value + 2.0,
            }
            for seed, value in ((1, 70.0), (2, 71.0), (3, 72.0))
        ]
        aggregate = aggregate_students(rows)[0]
        self.assertTrue(aggregate["complete"])
        self.assertEqual(aggregate["completed_seeds"], [1, 2, 3])
        self.assertEqual(aggregate["test_macro_top1"]["mean"], 71.0)
        self.assertEqual(
            aggregate["test_macro_top1"]["sample_standard_deviation"], 1.0
        )
        self.assertEqual(
            aggregate["test_macro_top1"]["raw_by_seed"],
            {"1": 70.0, "2": 71.0, "3": 72.0},
        )


class Phase1SplitTest(unittest.TestCase):
    def test_stratified_split_is_deterministic_and_balanced(self) -> None:
        # Oxford trainval totals 3,680; this synthetic class-size distribution
        # matches the total while exercising unequal source class counts.
        sizes = [100] * 17 + [99] * 20
        image_ids: list[str] = []
        labels: list[int] = []
        for label, size in enumerate(sizes):
            image_ids.extend(f"breed{label}_{index}" for index in range(size))
            labels.extend([label] * size)
        first = build_stratified_split(image_ids, labels)
        second = build_stratified_split(image_ids, labels)
        self.assertEqual(first, second)
        train_indices, validation_indices, manifest = first
        self.assertEqual((len(train_indices), len(validation_indices)), (2940, 740))
        validation_counts = Counter(labels[index] for index in validation_indices)
        self.assertEqual(validation_counts, Counter({label: 20 for label in range(37)}))
        self.assertEqual(len(manifest["validation_image_ids_sha256"]), 64)


class Phase1ModelContractTest(unittest.TestCase):
    def test_resnet56_feature_contract(self) -> None:
        model = ResNet56(num_classes=37).eval()
        with torch.no_grad():
            features = model.forward_features(torch.randn(2, 3, 32, 32))
            logits = model(torch.randn(2, 3, 32, 32))
        self.assertEqual([tuple(item.shape) for item in features], [
            (2, 16, 32, 32),
            (2, 32, 16, 16),
            (2, 64, 8, 8),
        ])
        self.assertEqual(tuple(logits.shape), (2, 37))

    def test_spatial_losses_are_finite_and_differentiable(self) -> None:
        students = [torch.randn(1, 192, 4, 4, requires_grad=True) for _ in range(12)]
        teachers = [
            torch.randn(1, 16, 4, 4),
            torch.randn(1, 32, 2, 2),
            torch.randn(1, 64, 1, 1),
        ]
        lg = LocalityGuidance()
        lg_loss = lg(students, teachers)
        self.assertTrue(bool(torch.isfinite(lg_loss)))
        ibkd = IBKD()
        alignment, fusion = ibkd(students, teachers)
        combined = 0.25 * fusion + 0.75 * alignment
        self.assertTrue(bool(torch.isfinite(combined)))
        combined.backward()
        self.assertIsNotNone(students[0].grad)

    def test_controller_boundaries_match_alg_and_ibkd(self) -> None:
        alg = GuidanceController(kind="alg", warmup_epochs=0)
        self.assertEqual(alg.beta_for_epoch(1), 2.5)
        alg.observe(1, 1.0, beta_used=2.5)
        self.assertEqual(alg.beta_for_epoch(2), 2.5)
        alg.observe(2, 1.0, beta_used=2.5)
        self.assertEqual(alg.stop_epoch, 2)

        ibkd = GuidanceController(kind="ibkd", warmup_epochs=20)
        for epoch in range(1, 3):
            self.assertEqual(ibkd.beta_for_epoch(epoch), 2.5)
            ibkd.observe(epoch, 1.0, beta_used=2.5)
        self.assertIsNone(ibkd.stop_epoch)


if __name__ == "__main__":
    unittest.main()
