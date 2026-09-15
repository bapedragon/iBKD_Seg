from __future__ import annotations

import argparse
import inspect
import json
import unittest
from pathlib import Path

import torch

from ibkd_seg.phase1.models import (
    IBKD,
    IBKD_AGGREGATION_MODES,
    STUDENT_BLOCKS,
    TransformerAggregationPooling,
)
from ibkd_seg.phase1.run_cub_ibkd_connection_full import (
    AGGREGATION_VARIANTS,
    EXPECTED_EXECUTION_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    _classification_flat_rows,
    _completion_counts,
    _evaluate_official_test,
    _probe_aggregates,
    _validate_configs,
    run,
)
from ibkd_seg.phase1.train_full import validate_args as validate_full_args
from ibkd_seg.phase1.train_timing import _ibkd_aggregation_audit, file_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "phase1/phase1_cub/mechanism_analysis"
PROTOCOL_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_full_v1.json"
)
EXECUTION_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_full_execution_v1.json"
)
FULL_SCRIPT = EXPERIMENT / "scripts/run_main_l0_ibkd_connection_full_b128.sh"
AUDIT = (
    EXPERIMENT
    / "reports/main_l0_aggregation_checkpoint_audit_v1/"
    "aggregation_checkpoint_audit.json"
)
REMOVED_SMOKE_PATHS = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_smoke_v1.json",
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_connection_smoke_b128_seed1.sh",
    ROOT / "src/ibkd_seg/phase1/run_cub_ibkd_connection_smoke.py",
)


def _full_args(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dataset": "cub",
        "kind": "student",
        "method": "ibkd",
        "teacher_architecture": "resnet50_224_scratch",
        "scientific_cub_r50_teacher": True,
        "seed_extension_full": False,
        "loader_pilot_full": False,
        "loader_followup_full": False,
        "mechanism_ablation_full": True,
        "ibkd_aggregation_mode": "fixed_uniform_all",
        "defer_official_test": True,
        "batch_size": 128,
        "fusion_ratio": 0.25,
        "teacher_checkpoint": Path("teacher.pt"),
        "protocol_config": PROTOCOL_CONFIG,
        "batch_profile_role": "posthoc_main_l0_mechanism_ablation",
        "cub_loader_profile": "l0_current_strong",
        "eval_batch_size": 200,
        "num_workers": 4,
        "seed": 1,
        "alg_controller_warmup_epochs": 0,
        "posthoc_diagnostic_id": None,
    }
    values.update(changes)
    return argparse.Namespace(**values)


class Phase1CubMechanismAnalysisTest(unittest.TestCase):
    def test_canonical_aggregation_state_dict_contract_is_unchanged(self) -> None:
        aggregation = TransformerAggregationPooling()
        self.assertEqual(aggregation.mode, "learned_all")
        self.assertEqual(set(aggregation.state_dict()), {"weights"})
        self.assertEqual(tuple(aggregation.weights.shape), (3, STUDENT_BLOCKS))
        self.assertTrue(
            torch.equal(aggregation.weights, torch.zeros(3, STUDENT_BLOCKS))
        )
        self.assertIn("aggregation.weights", IBKD().state_dict())

    def test_fixed_aggregation_matrices_have_exact_intended_connections(self) -> None:
        features = [
            torch.full((1, 1, 1, 1), float(block))
            for block in range(STUDENT_BLOCKS)
        ]
        expected = {
            "fixed_uniform_all": [5.5, 5.5, 5.5],
            "fixed_stage_match": [0.0, 6.0, 11.0],
            "fixed_last": [11.0, 11.0, 11.0],
        }
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                aggregation = TransformerAggregationPooling(mode)
                self.assertEqual(aggregation.state_dict(), {})
                weights = aggregation.normalized_weights()
                self.assertTrue(torch.equal(weights.sum(dim=1), torch.ones(3)))
                outputs = aggregation(features).flatten()
                self.assertTrue(
                    torch.allclose(outputs, torch.tensor(values), atol=0.0, rtol=0.0)
                )

    def test_aggregation_audit_is_executable_for_every_locked_mode(self) -> None:
        expected_entropy = {
            "learned_all": 1.0,
            "fixed_uniform_all": 1.0,
            "fixed_stage_match": 0.0,
            "fixed_last": 0.0,
        }
        self.assertEqual(tuple(IBKD_AGGREGATION_MODES), AGGREGATION_VARIANTS)
        for mode in AGGREGATION_VARIANTS:
            with self.subTest(mode=mode):
                audit = _ibkd_aggregation_audit(IBKD(aggregation_mode=mode))
                self.assertEqual(audit["mode"], mode)
                self.assertEqual(len(audit["probabilities"]), 3)
                for value in audit["normalized_entropy_by_teacher_stage"]:
                    self.assertAlmostEqual(value, expected_entropy[mode], places=6)

    def test_locked_protocol_and_released_execution_contract_validate(self) -> None:
        protocol = json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
        execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(PROTOCOL_CONFIG), EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(file_sha256(EXECUTION_CONFIG), EXPECTED_EXECUTION_SHA256)
        for seed in (1, 2, 3):
            base, base_path = _validate_configs(
                protocol,
                PROTOCOL_CONFIG,
                execution,
                EXECUTION_CONFIG,
                seed,
            )
            self.assertTrue(base_path.is_file())
            self.assertIn("frozen_probe", base)
        self.assertTrue(
            execution["release_decision"][
                "all_encoder_seeds_run_regardless_of_earlier_results"
            ]
        )
        self.assertFalse(execution["release_decision"]["retain_smoke_outputs"])

    def test_full_training_path_allows_only_the_locked_mechanism_scope(self) -> None:
        for seed in (1, 2, 3):
            for mode in AGGREGATION_VARIANTS:
                validate_full_args(
                    _full_args(seed=seed, ibkd_aggregation_mode=mode)
                )

        with self.assertRaisesRegex(ValueError, "must defer test"):
            validate_full_args(_full_args(defer_official_test=False))
        with self.assertRaisesRegex(ValueError, "Mechanism ablation requires"):
            validate_full_args(_full_args(fusion_ratio=0.5))
        with self.assertRaisesRegex(ValueError, "locked L0"):
            validate_full_args(_full_args(cub_loader_profile="l2_conservative_spatial"))
        with self.assertRaisesRegex(ValueError, "Mechanism ablation requires"):
            validate_full_args(_full_args(method="lg", fusion_ratio=None))
        with self.assertRaises(ValueError):
            validate_full_args(_full_args(mechanism_ablation_full=False))

    def test_per_seed_completion_counts_match_released_gate(self) -> None:
        execution = json.loads(EXECUTION_CONFIG.read_text(encoding="utf-8"))
        classification_rows = [
            {"official_test": {"macro_top1": 1.0}}
            for _ in AGGREGATION_VARIANTS
        ]
        probe_rows = [
            {"candidates": [{}, {}, {}], "official_test_evaluations": 1}
            for _ in range(20)
        ]
        self.assertEqual(
            _completion_counts(classification_rows, probe_rows),
            execution["per_shard_completion_gate"],
        )

    def test_reports_paired_differences_from_learned_all(self) -> None:
        classification = []
        for index, mode in enumerate(AGGREGATION_VARIANTS):
            classification.append(
                {
                    "aggregation_mode": mode,
                    "encoder_seed": 1,
                    "checkpoint_path": f"{mode}.pt",
                    "official_test": {
                        "macro_top1": 50.0 - index,
                        "overall_top1": 50.0 - index,
                        "top5": 80.0,
                    },
                    "summary": {
                        "selected_epoch": 10,
                        "selected_validation": {"macro_top1": 40.0 - index},
                        "controller_final": {"stop_epoch": 100},
                        "ibkd_aggregation": {},
                        "checkpoint_sha256": str(index) * 64,
                    },
                }
            )
        classification_flat = _classification_flat_rows(classification)
        self.assertEqual(
            [row["test_macro_top1_delta_from_learned_all"] for row in classification_flat],
            [0.0, -1.0, -2.0, -3.0],
        )

        probes = []
        for index, mode in enumerate(AGGREGATION_VARIANTS):
            for probe_seed in range(1, 6):
                probes.append(
                    {
                        "aggregation_mode": mode,
                        "probe_seed": probe_seed,
                        "test_input_224_mean_iou": 0.5 - index * 0.01,
                    }
                )
        probe_aggregates = _probe_aggregates(probes)
        self.assertAlmostEqual(
            probe_aggregates[1]["mean_paired_difference_from_learned_all"],
            -0.01,
        )

    def test_official_test_is_reached_only_after_all_validation_selections(self) -> None:
        run_source = inspect.getsource(run)
        classifier_index = run_source.index("classification_rows = _train_classifiers")
        probe_index = run_source.index("probe_rows = _train_probes")
        test_index = run_source.index("_evaluate_official_test")
        self.assertLess(classifier_index, probe_index)
        self.assertLess(probe_index, test_index)

        test_source = inspect.getsource(_evaluate_official_test)
        marker_index = test_source.index(
            "all_validation_selections_complete_before_test.json"
        )
        classification_test_index = test_source.index("build_official_test_loader")
        probe_test_index = test_source.index("load_official_test_records")
        self.assertLess(marker_index, classification_test_index)
        self.assertLess(marker_index, probe_test_index)

    def test_h200_entry_is_seed_sharded_and_retains_only_full_artifacts(self) -> None:
        script = FULL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('encoder_seed="${1:-}"', script)
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_ibkd_connection_full", script)
        self.assertIn('--encoder-seed "${encoder_seed}"', script)
        self.assertIn("full_execution_v1.json", script)
        self.assertIn('tee "${output_root}/run.log"', script)
        self.assertTrue(FULL_SCRIPT.stat().st_mode & 0o111)
        for path in REMOVED_SMOKE_PATHS:
            self.assertFalse(path.exists(), str(path))

    def test_existing_checkpoint_observation_stays_separate(self) -> None:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["input_lineage_id"], "main_l0_v3")
        self.assertEqual(audit["checkpoints_audited"], 6)
        self.assertFalse(audit["causal_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
