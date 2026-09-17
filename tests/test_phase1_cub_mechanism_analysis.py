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
from ibkd_seg.phase1.controllers import GuidanceController
from ibkd_seg.phase1.run_cub_ibkd_connection_matched_smoke import (
    EXPECTED_CONFIG_SHA256 as EXPECTED_MATCHED_SMOKE_CONFIG_SHA256,
    EXPECTED_FULL_CONFIG_SHA256 as EXPECTED_MATCHED_FULL_CONFIG_SHA256,
    _probability_contract,
    validate_config as validate_matched_smoke_config,
)
from ibkd_seg.phase1.run_cub_ibkd_connection_full import (
    AGGREGATION_VARIANTS,
    EXPECTED_CLASSIFICATION_EXECUTION_SHA256,
    EXPECTED_CLASSIFICATION_PROTOCOL_SHA256,
    EXPECTED_EXECUTION_SHA256,
    EXPECTED_MATCHED_PROTOCOL_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    _classification_flat_rows,
    _completion_counts,
    _evaluate_official_test,
    _probe_aggregates,
    _validate_configs,
    run,
)
from ibkd_seg.phase1.run_cub_ibkd_connection_replay_smoke import (
    EXPECTED_CONFIG_SHA256 as EXPECTED_REPLAY_SMOKE_CONFIG_SHA256,
    _validate_summary as validate_replay_smoke_summary,
    validate_config as validate_replay_smoke_config,
)
from ibkd_seg.phase1.run_cub_ibkd_connection_replay_full import (
    EXPECTED_CONFIG_SHA256 as EXPECTED_REPLAY_FULL_CONFIG_SHA256,
    validate_config as validate_replay_full_config,
)
from ibkd_seg.phase1.train_full import validate_args as validate_full_args
from ibkd_seg.phase1.train_timing import (
    _ibkd_aggregation_audit,
    file_sha256,
    validate_args as validate_timing_args,
)


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
REPLAY_SMOKE_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_learned_all_replay_smoke_v1.json"
)
REPLAY_SMOKE_SCRIPT = (
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_learned_all_replay_smoke_b128_seed1.sh"
)
REPLAY_FULL_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_learned_all_replay_full_v1.json"
)
REPLAY_FULL_SCRIPT = (
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_learned_all_replay_full_b128_seed1.sh"
)
MATCHED_SMOKE_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_matched_smoke_v2.json"
)
MATCHED_FULL_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_matched_full_v2.json"
)
MATCHED_SMOKE_SCRIPT = (
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_connection_matched_smoke_b128_seed1.sh"
)
CLASSIFICATION_FULL_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_classification_full_v3.json"
)
CLASSIFICATION_FULL_EXECUTION_CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_connection_classification_full_execution_v3.json"
)
CLASSIFICATION_FULL_SCRIPT = (
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_connection_classification_full_b128.sh"
)
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
        "controlled_aa_full": False,
        "mechanism_replay_full": False,
        "ibkd_aggregation_mode": "fixed_uniform_all",
        "ibkd_fixed_guidance_epochs": None,
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


def _replay_timing_args(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dataset": "cub",
        "kind": "student",
        "teacher_architecture": "resnet50_224_scratch",
        "access_official_test": False,
        "method": "ibkd",
        "batch_size": 128,
        "fusion_ratio": 0.25,
        "teacher_checkpoint": Path("teacher.pt"),
        "scientific_cub_r50_teacher": True,
        "seed_extension_smoke": False,
        "eval_batch_size": 200,
        "num_workers": 0,
        "seed": 1,
        "alg_controller_warmup_epochs": 0,
        "save_student_checkpoint": True,
        "cub_loader_profile": "l0_current_strong",
        "loader_pilot_smoke": False,
        "loader_followup_smoke": False,
        "controlled_aa_smoke": False,
        "mechanism_replay_smoke": True,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def _matched_timing_args(mode: str, **changes: object) -> argparse.Namespace:
    values = vars(_replay_timing_args(mechanism_replay_smoke=False))
    values.update(
        {
            "mechanism_ablation_smoke": True,
            "ibkd_aggregation_mode": mode,
            "ibkd_fixed_guidance_epochs": 123,
        }
    )
    values.update(changes)
    return argparse.Namespace(**values)


def _replay_full_args(**changes: object) -> argparse.Namespace:
    values = vars(
        _full_args(
            mechanism_ablation_full=False,
            mechanism_replay_full=True,
            ibkd_aggregation_mode="learned_all",
            defer_official_test=False,
            protocol_config=REPLAY_FULL_CONFIG,
            batch_profile_role="posthoc_issue760_learned_all_replay",
            num_workers=0,
        )
    )
    values.update(changes)
    return argparse.Namespace(**values)


class Phase1CubMechanismAnalysisTest(unittest.TestCase):
    def test_classification_only_full_reuses_issue776_smoke_without_probe(self) -> None:
        self.assertEqual(
            file_sha256(CLASSIFICATION_FULL_CONFIG),
            EXPECTED_CLASSIFICATION_PROTOCOL_SHA256,
        )
        self.assertEqual(
            file_sha256(CLASSIFICATION_FULL_EXECUTION_CONFIG),
            EXPECTED_CLASSIFICATION_EXECUTION_SHA256,
        )
        protocol = json.loads(
            CLASSIFICATION_FULL_CONFIG.read_text(encoding="utf-8")
        )
        execution = json.loads(
            CLASSIFICATION_FULL_EXECUTION_CONFIG.read_text(encoding="utf-8")
        )
        self.assertEqual(
            protocol["scope"],
            {
                "classification_only": True,
                "frozen_probe_runs": 0,
                "segmentation_metrics": False,
            },
        )
        self.assertNotIn("frozen_probe", protocol)
        self.assertEqual(protocol["smoke_gate"]["source_h200_issue"], 776)
        self.assertTrue(
            execution["release_decision"]["frozen_probe_removed_from_full_scope"]
        )
        for seed in (1, 2, 3):
            _validate_configs(
                protocol,
                CLASSIFICATION_FULL_CONFIG,
                execution,
                CLASSIFICATION_FULL_EXECUTION_CONFIG,
                seed,
            )
            for mode in AGGREGATION_VARIANTS:
                validate_full_args(
                    _full_args(
                        seed=seed,
                        ibkd_aggregation_mode=mode,
                        ibkd_fixed_guidance_epochs=123,
                        protocol_config=CLASSIFICATION_FULL_CONFIG,
                        num_workers=0,
                    )
                )
        script = CLASSIFICATION_FULL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("connection_classification_full_v3.json", script)
        self.assertIn("--num-workers 0", script)
        self.assertNotIn("probe", script.lower())
        self.assertTrue(CLASSIFICATION_FULL_SCRIPT.stat().st_mode & 0o111)
        run_source = inspect.getsource(run)
        self.assertLess(
            run_source.index("if classification_only:"),
            run_source.index("probe_rows = _train_probes"),
        )
        self.assertIn('"segmentation_annotations_loaded": False', run_source)

    def test_matched_duration_smoke_is_locked_and_executable(self) -> None:
        self.assertEqual(
            file_sha256(MATCHED_SMOKE_CONFIG),
            EXPECTED_MATCHED_SMOKE_CONFIG_SHA256,
        )
        self.assertEqual(
            file_sha256(MATCHED_FULL_CONFIG),
            EXPECTED_MATCHED_FULL_CONFIG_SHA256,
        )
        config = validate_matched_smoke_config(MATCHED_SMOKE_CONFIG)
        self.assertEqual(config["student"]["fixed_guidance_epochs"], 123)
        self.assertEqual(
            config["scope"]["aggregation_variants"],
            list(AGGREGATION_VARIANTS),
        )
        for mode in AGGREGATION_VARIANTS:
            validate_timing_args(_matched_timing_args(mode))
        with self.assertRaisesRegex(ValueError, "fixed guidance epoch 123"):
            validate_timing_args(
                _matched_timing_args("learned_all", ibkd_fixed_guidance_epochs=103)
            )
        script = MATCHED_SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_cub_ibkd_connection_matched_smoke", script)
        self.assertNotIn("--access-official-test", script)
        self.assertTrue(MATCHED_SMOKE_SCRIPT.stat().st_mode & 0o111)

    def test_fixed_guidance_horizon_is_inclusive(self) -> None:
        controller = GuidanceController(
            kind="ibkd", warmup_epochs=20, fixed_stop_epoch=123
        )
        for epoch in range(1, 124):
            self.assertEqual(controller.beta_for_epoch(epoch), 2.5)
            controller.observe(epoch, 1.0, beta_used=2.5)
        self.assertEqual(controller.stop_epoch, 123)
        self.assertFalse(controller.active)
        self.assertEqual(controller.beta_for_epoch(124), 0.0)
        state = controller.state_dict()
        self.assertEqual(state["stop_policy"], "fixed_epoch")
        self.assertEqual(state["fixed_stop_epoch"], 123)

    def test_fixed_connection_probability_contracts(self) -> None:
        for mode in AGGREGATION_VARIANTS:
            audit = _ibkd_aggregation_audit(IBKD(aggregation_mode=mode))
            self.assertTrue(_probability_contract(mode, audit), mode)

    def test_single_learned_all_replay_full_is_released_after_smoke(self) -> None:
        self.assertEqual(
            file_sha256(REPLAY_FULL_CONFIG),
            EXPECTED_REPLAY_FULL_CONFIG_SHA256,
        )
        config = validate_replay_full_config(REPLAY_FULL_CONFIG)
        self.assertEqual(config["smoke_gate"]["source_h200_issue"], 767)
        self.assertEqual(config["smoke_gate"]["execution_gates"], "11/11")
        self.assertEqual(config["scope"]["variants"], ["learned_all"])
        self.assertFalse(config["scope"]["frozen_probe"])
        self.assertEqual(config["official_test_policy"]["total_evaluations"], 1)

        validate_full_args(_replay_full_args())
        invalid = (
            {"num_workers": 4},
            {"defer_official_test": True},
            {"fusion_ratio": 0.5},
            {"batch_size": 64},
            {"seed": 2},
            {"ibkd_aggregation_mode": "fixed_uniform_all"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_full_args(_replay_full_args(**changes))

        script = REPLAY_FULL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_cub_ibkd_connection_replay_full", script)
        self.assertNotIn("run_cub_ibkd_connection_full ", script)
        self.assertTrue(REPLAY_FULL_SCRIPT.stat().st_mode & 0o111)

    def test_single_learned_all_replay_smoke_is_locked_and_test_sealed(self) -> None:
        self.assertEqual(
            file_sha256(REPLAY_SMOKE_CONFIG),
            EXPECTED_REPLAY_SMOKE_CONFIG_SHA256,
        )
        config = validate_replay_smoke_config(REPLAY_SMOKE_CONFIG)
        self.assertEqual(config["scope"]["aggregation_variants"], ["learned_all"])
        self.assertFalse(config["scope"]["official_test_accessed"])
        self.assertFalse(config["scope"]["frozen_probe"])

        validate_timing_args(_replay_timing_args())
        for changes in (
            {"num_workers": 4},
            {"access_official_test": True},
            {"fusion_ratio": 0.5},
            {"batch_size": 64},
            {"seed": 2},
            {"method": "lg", "fusion_ratio": None},
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "Mechanism replay"):
                    validate_timing_args(_replay_timing_args(**changes))

        script = REPLAY_SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_cub_ibkd_connection_replay_smoke", script)
        self.assertNotIn("--access-official-test", script)
        self.assertTrue(REPLAY_SMOKE_SCRIPT.stat().st_mode & 0o111)

    def test_replay_smoke_summary_requires_complete_hash_trace(self) -> None:
        epoch = {
            "input_stream_sha256": "input",
            "rng_state_sha256_after_epoch": "rng",
            "student_state_sha256_after_epoch": "student",
            "guidance_state_sha256_after_epoch": "guidance",
            "peak_cuda_memory_bytes": 1,
            "peak_cuda_memory_reserved_bytes": 2,
        }
        summary = {
            "status": "complete",
            "method": "ibkd",
            "fusion_ratio_lambda": 0.25,
            "batch_size": 128,
            "seed": 1,
            "cub_loader_profile": "l0_current_strong",
            "mechanism_replay_smoke": True,
            "controlled_aa_smoke": False,
            "ibkd_aggregation": {"mode": "learned_all"},
            "actual_epochs": 2,
            "planned_epochs": 300,
            "official_test_accessed": False,
            "official_test": None,
            "split_manifest": {
                "validation_image_ids_sha256": (
                    "263d2f165326262706101e52af3763c6d183be709dafe3578a9aca0f546bf854"
                )
            },
            "teacher_checkpoint_sha256": (
                "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3"
            ),
            "teacher_model_state_sha256": (
                "96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7"
            ),
            "initial_student_state_sha256": "initial-student",
            "initial_guidance_state_sha256": "initial-guidance",
            "checkpoint_sha256": "checkpoint",
            "student_state_sha256": "student",
            "guidance_state_sha256": "guidance",
            "controlled_reproducibility": {
                "enabled": True,
                "formal_bitwise_determinism": False,
                "scientific_path_preserved": True,
            },
            "epochs": [epoch, dict(epoch)],
        }
        checks = validate_replay_smoke_summary(summary)
        self.assertTrue(all(checks.values()), checks)
        summary["epochs"][1]["input_stream_sha256"] = None
        self.assertFalse(
            validate_replay_smoke_summary(summary)["epoch_trace_hashes"]
        )

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
