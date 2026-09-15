from __future__ import annotations

import argparse
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
from ibkd_seg.phase1.run_cub_ibkd_connection_smoke import (
    AGGREGATION_VARIANTS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_FULL_CONFIG_SHA256,
    _validate_config,
)
from ibkd_seg.phase1.train_timing import (
    _ibkd_aggregation_audit,
    file_sha256,
    validate_args,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "phase1/phase1_cub/mechanism_analysis"
SMOKE_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_smoke_v1.json"
)
FULL_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_full_v1.json"
)
SMOKE_SCRIPT = (
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_connection_smoke_b128_seed1.sh"
)
AUDIT = (
    EXPERIMENT
    / "reports/main_l0_aggregation_checkpoint_audit_v1/"
    "aggregation_checkpoint_audit.json"
)


def _timing_args(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dataset": "cub",
        "kind": "student",
        "method": "ibkd",
        "teacher_architecture": "resnet50_224_scratch",
        "access_official_test": False,
        "batch_size": 128,
        "fusion_ratio": 0.25,
        "teacher_checkpoint": Path("teacher.pt"),
        "scientific_cub_r50_teacher": True,
        "seed_extension_smoke": False,
        "eval_batch_size": 200,
        "num_workers": 4,
        "seed": 1,
        "alg_controller_warmup_epochs": 0,
        "save_student_checkpoint": True,
        "cub_loader_profile": "l0_current_strong",
        "loader_pilot_smoke": False,
        "loader_followup_smoke": False,
        "mechanism_ablation_smoke": True,
        "ibkd_aggregation_mode": "fixed_uniform_all",
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
        ibkd = IBKD()
        self.assertIn("aggregation.weights", ibkd.state_dict())

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

    def test_unknown_aggregation_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown iBKD aggregation mode"):
            TransformerAggregationPooling("unknown")
        self.assertEqual(tuple(IBKD_AGGREGATION_MODES), AGGREGATION_VARIANTS)

    def test_post_training_aggregation_audit_is_executable(self) -> None:
        expected_entropy = {
            "learned_all": 1.0,
            "fixed_uniform_all": 1.0,
            "fixed_stage_match": 0.0,
            "fixed_last": 0.0,
        }
        for mode in AGGREGATION_VARIANTS:
            with self.subTest(mode=mode):
                audit = _ibkd_aggregation_audit(IBKD(aggregation_mode=mode))
                self.assertEqual(audit["mode"], mode)
                self.assertEqual(len(audit["probabilities"]), 3)
                self.assertEqual(
                    len(audit["normalized_entropy_by_teacher_stage"]), 3
                )
                for value in audit["normalized_entropy_by_teacher_stage"]:
                    self.assertAlmostEqual(
                        value,
                        expected_entropy[mode],
                        places=6,
                    )

    def test_locked_smoke_and_full_configs_validate(self) -> None:
        smoke = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
        full_path = _validate_config(smoke, SMOKE_CONFIG)
        self.assertEqual(full_path, FULL_CONFIG)
        self.assertEqual(file_sha256(SMOKE_CONFIG), EXPECTED_CONFIG_SHA256)
        self.assertEqual(file_sha256(FULL_CONFIG), EXPECTED_FULL_CONFIG_SHA256)
        full = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            [row["id"] for row in full["classification"]["variants"]],
            list(AGGREGATION_VARIANTS),
        )
        self.assertEqual(
            full["execution_gate"]["current_state"],
            "full_blocked_until_smoke_passes",
        )
        self.assertFalse(full["completed_main_result_replaced"])

    def test_timing_path_allows_only_the_locked_mechanism_scope(self) -> None:
        for mode in AGGREGATION_VARIANTS:
            validate_args(_timing_args(ibkd_aggregation_mode=mode))

        with self.assertRaisesRegex(ValueError, "requires --mechanism-ablation-smoke"):
            validate_args(_timing_args(mechanism_ablation_smoke=False))
        with self.assertRaisesRegex(ValueError, "requires validation-only main-L0"):
            validate_args(_timing_args(fusion_ratio=0.5))
        with self.assertRaisesRegex(ValueError, "requires validation-only main-L0"):
            validate_args(_timing_args(cub_loader_profile="l2_conservative_spatial"))
        with self.assertRaisesRegex(ValueError, "requires validation-only main-L0"):
            validate_args(
                _timing_args(
                    method="lg",
                    fusion_ratio=None,
                    ibkd_aggregation_mode="fixed_uniform_all",
                )
            )

    def test_h200_entry_and_existing_checkpoint_audit_are_separate(self) -> None:
        script = SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("run_cub_ibkd_connection_smoke", script)
        self.assertIn("main_l0_ibkd_connection_smoke_v1", script)
        self.assertIn('tee "${output_root}/run.log"', script)
        self.assertTrue(SMOKE_SCRIPT.stat().st_mode & 0o111)

        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["input_lineage_id"], "main_l0_v3")
        self.assertEqual(audit["checkpoints_audited"], 6)
        self.assertFalse(audit["causal_claim_allowed"])
        entropies = [
            row["normalized_entropy"]
            for row in audit["three_seed_aggregates"]
        ]
        self.assertGreaterEqual(min(entropies), 0.99)


if __name__ == "__main__":
    unittest.main()
