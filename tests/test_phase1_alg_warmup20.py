from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from ibkd_seg.phase1.controllers import GuidanceController
from ibkd_seg.phase1.run_alg_warmup20_full import (
    EXPERIMENT_ID,
    _validate_full_config,
    _write_status,
)
from ibkd_seg.phase1.run_alg_warmup20_smoke import (
    DIAGNOSTIC_ID,
    _select_candidate,
    _validate_config,
)
from ibkd_seg.phase1.train_full import (
    ALG_WARMUP20_DIAGNOSTIC_ID,
    validate_args as validate_full_args,
)
from ibkd_seg.phase1.train_timing import validate_args as validate_timing_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY_ROOT / "phase1/configs/oxford_iiit_pet_phase1_v1.json"
DIAGNOSTIC_PATH = (
    REPOSITORY_ROOT
    / "phase1/configs/oxford_iiit_pet_alg_warmup20_diagnostic_v1.json"
)
SCRIPT_PATH = REPOSITORY_ROOT / "phase1/scripts/run_alg_warmup20_smoke_b128.sh"
FULL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "phase1/configs/oxford_iiit_pet_alg_warmup20_full_v1.json"
)
FULL_SCRIPT_PATH = (
    REPOSITORY_ROOT / "phase1/scripts/run_alg_warmup20_full_b128.sh"
)
RELEASE_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "phase1/reports/classification/batch128/checkpoint_release.json"
)


def _timing_args(**updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "kind": "student",
        "method": "alg",
        "batch_size": 128,
        "fusion_ratio": None,
        "teacher_checkpoint": Path("teacher.pt"),
        "eval_batch_size": 200,
        "num_workers": 4,
        "seed": 1,
        "alg_controller_warmup_epochs": 20,
        "save_student_checkpoint": True,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _full_args(**updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "kind": "student",
        "method": "alg",
        "batch_size": 128,
        "fusion_ratio": None,
        "teacher_checkpoint": Path("teacher.pt"),
        "eval_batch_size": 200,
        "num_workers": 4,
        "seed": 1,
        "alg_controller_warmup_epochs": 20,
        "posthoc_diagnostic_id": ALG_WARMUP20_DIAGNOSTIC_ID,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class Phase1AlgWarmup20DiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
        cls.full_config = json.loads(
            FULL_CONFIG_PATH.read_text(encoding="utf-8")
        )

    def test_committed_config_is_a_non_replacing_single_field_diagnostic(self) -> None:
        _validate_config(self.diagnostic, self.protocol, PROTOCOL_PATH)
        self.assertEqual(self.diagnostic["diagnostic_id"], DIAGNOSTIC_ID)
        self.assertTrue(
            self.diagnostic["interpretation"]["canonical_alg_result_is_retained"]
        )
        self.assertTrue(
            self.diagnostic["interpretation"][
                "replacement_of_locked_phase1_result_forbidden"
            ]
        )
        self.assertEqual(
            self.diagnostic["interpretation"][
                "change_exactly_one_alg_controller_field"
            ],
            "controller_warmup_epochs_0_to_20",
        )
        self.assertEqual(
            self.diagnostic["classification"]["optimizer_lr_warmup_epochs"],
            20,
        )
        self.assertEqual(
            self.diagnostic["classification"]["controller"]["warmup_epochs"],
            20,
        )
        self.assertFalse(self.diagnostic["smoke"]["scientific_result"])
        self.assertFalse(self.diagnostic["smoke"]["official_test_accessed"])

    def test_config_validation_rejects_any_other_controller_warmup(self) -> None:
        changed = copy.deepcopy(self.diagnostic)
        changed["classification"]["controller"]["warmup_epochs"] = 19
        with self.assertRaisesRegex(RuntimeError, "controller_warmup_20"):
            _validate_config(changed, self.protocol, PROTOCOL_PATH)

    def test_timing_checkpoint_export_is_restricted_to_alg_warmup20(self) -> None:
        validate_timing_args(_timing_args())
        with self.assertRaisesRegex(ValueError, "Only ALG accepts"):
            validate_timing_args(_timing_args(method="lg"))
        with self.assertRaisesRegex(ValueError, "reserved for ALG warm-up-20"):
            validate_timing_args(
                _timing_args(
                    alg_controller_warmup_epochs=0,
                    save_student_checkpoint=True,
                )
            )

    def test_alg_warmup20_defers_only_the_stop_decision(self) -> None:
        controller = GuidanceController(kind="alg", warmup_epochs=20)
        for epoch in (1, 2):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, 1.0, beta_used=2.5)
        state = controller.state_dict()
        self.assertTrue(state["active"])
        self.assertIsNone(state["stop_epoch"])
        self.assertEqual(state["beta_history"], [2.5, 2.5])
        self.assertEqual(state["smoothed_derivative_history"], [None, None])
        self.assertEqual(state["stop_comparison"], "greater_or_equal")

    def test_probe_lr_tie_keeps_the_first_locked_candidate(self) -> None:
        first_state = {"weight": object()}
        selected_state, selected = _select_candidate(
            [
                (
                    first_state,
                    {
                        "learning_rate": 0.01,
                        "best_validation_grid_mean_iou": 0.5,
                    },
                ),
                (
                    {"weight": object()},
                    {
                        "learning_rate": 0.03,
                        "best_validation_grid_mean_iou": 0.5,
                    },
                ),
            ]
        )
        self.assertIs(selected_state, first_state)
        self.assertEqual(selected["learning_rate"], 0.01)

    def test_full_training_entry_is_restricted_to_labeled_diagnostic(self) -> None:
        validate_full_args(_full_args())
        with self.assertRaisesRegex(ValueError, "fixed post-hoc diagnostic id"):
            validate_full_args(_full_args(posthoc_diagnostic_id="wrong"))
        with self.assertRaisesRegex(ValueError, "fixed to batch 128"):
            validate_full_args(_full_args(batch_size=64))
        with self.assertRaisesRegex(ValueError, "Only ALG accepts"):
            validate_full_args(_full_args(method="lg"))

    def test_full_config_locks_three_classifiers_and_fifteen_probes(self) -> None:
        _validate_full_config(
            self.full_config,
            self.diagnostic,
            self.protocol,
            diagnostic_path=DIAGNOSTIC_PATH,
            protocol_path=PROTOCOL_PATH,
            release_manifest_path=RELEASE_MANIFEST_PATH,
        )
        self.assertEqual(self.full_config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(
            self.full_config["expected_output"],
            {
                "classification_students": 3,
                "probe_lr_candidates": 45,
                "selected_probes": 15,
                "classification_test_evaluations": 3,
                "probe_test_evaluations": 15,
            },
        )
        self.assertFalse(self.full_config["confirmatory_main_result"])
        self.assertFalse(self.full_config["canonical_phase1_result_replaced"])

    def test_full_sequence_status_marks_test_access_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _write_status(
                output_dir,
                status="running",
                phase="probe_official_test",
                classification_complete=3,
                probe_selections_complete=15,
                probe_test_evaluations_complete=0,
                active={"stage": "probe_official_test_preparation"},
                probe_official_test_accessed=True,
            )
            status = json.loads(
                (output_dir / "sequence_status.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertTrue(status["probe_official_test_accessed"])
        self.assertEqual(status["probe_test_evaluations_complete"], 0)

    def test_h200_entry_script_runs_one_combined_smoke(self) -> None:
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.run_alg_warmup20_smoke", script)
        self.assertIn("--smoke", script)
        self.assertIn("--device cuda", script)
        self.assertTrue(SCRIPT_PATH.stat().st_mode & 0o111)

    def test_h200_full_entry_runs_classification_then_probe(self) -> None:
        script = FULL_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.run_alg_warmup20_full", script)
        self.assertIn("--full-diagnostic", script)
        self.assertIn(
            "phase1/reports/classification/batch128/checkpoint_release.json",
            script,
        )
        self.assertIn("--device cuda", script)
        self.assertTrue(FULL_SCRIPT_PATH.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
