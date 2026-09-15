from __future__ import annotations

import argparse
import copy
import inspect
import unittest
from pathlib import Path

from ibkd_seg.phase1.run_cub_ibkd_controlled_aa_full import (
    CONTROLLED_ENVIRONMENT,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_NONDETERMINISTIC_KERNELS,
    compare_summaries,
    validate_config,
)
from ibkd_seg.phase1.train_full import (
    run_student,
    validate_args as validate_full_args,
)
from ibkd_seg.phase1.train_timing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "phase1/phase1_cub/reproducibility"
CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_controlled_aa_full_v1.json"
)
BASE_CONFIG = (
    ROOT / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_b64_guided_seed1_full_v4.json"
)
SCRIPT = (
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_controlled_aa_full_b128_seed1.sh"
)


def _full_args(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dataset": "cub",
        "kind": "student",
        "method": "ibkd",
        "batch_size": 128,
        "fusion_ratio": 0.25,
        "data_dir": Path("data"),
        "output_dir": Path("output"),
        "run_name": "run_A",
        "teacher_checkpoint": Path("teacher.pt"),
        "teacher_architecture": "resnet50_224_scratch",
        "scientific_cub_r50_teacher": True,
        "seed_extension_full": False,
        "loader_pilot_full": False,
        "loader_followup_full": False,
        "mechanism_ablation_full": False,
        "controlled_aa_full": True,
        "ibkd_aggregation_mode": "learned_all",
        "defer_official_test": False,
        "cub_loader_profile": "l0_current_strong",
        "protocol_config": BASE_CONFIG,
        "batch_profile_role": "locked_v3_partial_cell",
        "eval_batch_size": 200,
        "num_workers": 0,
        "seed": 1,
        "alg_controller_warmup_epochs": 0,
        "posthoc_diagnostic_id": None,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def _controlled_contract() -> dict[str, object]:
    return {
        "enabled": True,
        "formal_bitwise_determinism": False,
        "observed_nondeterministic_operations": [
            {"kernel": kernel} for kernel in EXPECTED_NONDETERMINISTIC_KERNELS
        ],
    }


def _summary() -> dict[str, object]:
    history = [
        {
            "epoch": epoch,
            "lr": 0.0005,
            "beta": 2.5 if epoch <= 100 else 0.0,
            "train_loss": 1.0,
            "train_ce": 0.5,
            "train_guidance": 0.2,
            "train_alignment": 0.1,
            "train_fusion": 0.1,
            "train_top1": 20.0,
            "validation": {
                "macro_top1": 21.0,
                "overall_top1": 21.0,
                "top5": 50.0,
            },
            "seconds_including_validation": 40.0,
            "peak_cuda_memory_bytes": 12,
            "peak_cuda_memory_reserved_bytes": 17,
            "input_stream_sha256": f"input-{epoch}",
            "student_state_sha256_after_epoch": f"student-{epoch}",
            "guidance_state_sha256_after_epoch": f"guidance-{epoch}",
            "rng_state_sha256_after_epoch": f"rng-{epoch}",
        }
        for epoch in range(1, 301)
    ]
    return {
        "status": "complete",
        "scientific_result": False,
        "confirmatory_main_result": False,
        "posthoc_reproducibility_audit": True,
        "controlled_aa_full": True,
        "eligible_locked_v3_matrix_cell": False,
        "dataset": "CUB-200-2011",
        "num_classes": 200,
        "kind": "student",
        "method": "ibkd",
        "fusion_ratio_lambda": 0.25,
        "batch_size": 128,
        "epochs": 300,
        "seed": 1,
        "initial_student_state_sha256": "initial-student",
        "initial_guidance_state_sha256": "initial-guidance",
        "teacher_checkpoint_sha256": "teacher-file",
        "teacher_model_state_sha256": "teacher-state",
        "teacher_architecture": "resnet50_224_scratch",
        "protocol_config_sha256": "base-protocol",
        "batch_profile_role": "locked_v3_partial_cell",
        "cub_loader_profile": "l0_current_strong",
        "ibkd_aggregation_mode": "learned_all",
        "guidance_controller_warmup_epochs": 20,
        "optimizer_contract": "adamw",
        "split_manifest": {"validation_image_ids_sha256": "validation"},
        "controlled_reproducibility": _controlled_contract(),
        "selected_epoch": 200,
        "selected_validation": {
            "macro_top1": 22.0,
            "overall_top1": 22.0,
            "top5": 55.0,
        },
        "official_test": {
            "macro_top1": 23.0,
            "overall_top1": 23.0,
            "top5": 56.0,
        },
        "official_test_evaluations": 1,
        "official_test_accessed": True,
        "official_test_used_for_training_or_selection": False,
        "official_test_policy": "once_after_validation_selection",
        "selected_checkpoint_strict_reloaded": True,
        "checkpoint": "/output/checkpoint.pt",
        "checkpoint_sha256": "checkpoint",
        "student_state_sha256": "student-selected",
        "guidance_state_sha256": "guidance-selected",
        "controller_final": {"stop_epoch": 100},
        "ibkd_aggregation": {"mode": "learned_all"},
        "history": history,
        "runtime": {
            "python": "3.10",
            "platform": "linux",
            "torch": "2.11",
            "torchvision": "0.26",
            "timm": "1.0.27",
            "device": "cuda",
            "gpu_name": "H200",
            "cuda": "13",
            "git_commit": "a" * 40,
        },
    }


class Phase1CubReproducibilityFullTest(unittest.TestCase):
    def test_locked_full_config_validates(self) -> None:
        self.assertEqual(file_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        config = validate_config(CONFIG)
        self.assertFalse(config["scientific_result"])
        self.assertEqual(config["official_test_policy"]["total_evaluations"], 2)

    def test_full_flag_is_restricted_to_exact_scope(self) -> None:
        validate_full_args(_full_args())
        invalid = (
            {"num_workers": 4},
            {"seed": 2},
            {"fusion_ratio": 0.5},
            {"batch_size": 64, "batch_profile_role": "batch64_sensitivity"},
            {"cub_loader_profile": "l2_conservative_spatial"},
            {"defer_official_test": True},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "Controlled A/A"):
                    validate_full_args(_full_args(**changes))

    def test_comparison_gates_controls_but_reports_numerical_spread(self) -> None:
        run_a = _summary()
        run_b = copy.deepcopy(run_a)
        run_b["history"][10]["seconds_including_validation"] = 99.0
        run_b["history"][10]["train_loss"] = 1.0001
        run_b["selected_epoch"] = 201
        run_b["official_test"]["macro_top1"] = 22.5
        result = compare_summaries(run_a, run_b)
        self.assertTrue(result["execution_control_gates_passed"])
        observation = result["long_horizon_numerical_observation"]
        self.assertEqual(observation["selected_epoch"]["absolute_difference"], 1)
        self.assertAlmostEqual(
            observation["official_test"]["macro_top1"]["absolute_difference"],
            0.5,
        )
        self.assertEqual(
            observation["history"]["first_epoch_with_numerical_difference"], 11
        )

        run_b["history"][20]["input_stream_sha256"] = "different"
        result = compare_summaries(run_a, run_b)
        self.assertFalse(result["execution_control_gates_passed"])
        self.assertIn("epoch_input_rng_lr_control", result["failures"])

    def test_entry_script_uses_one_mig_and_two_process_orchestrator(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        for key, value in CONTROLLED_ENVIRONMENT.items():
            self.assertIn(f"export {key}={value}", script)
        self.assertIn("ibkd_seg.phase1.run_cub_ibkd_controlled_aa_full", script)
        self.assertNotIn("--defer-official-test", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_full_trainer_records_per_epoch_control_hashes(self) -> None:
        source = inspect.getsource(run_student)
        for field in (
            "input_stream_sha256",
            "student_state_sha256_after_epoch",
            "guidance_state_sha256_after_epoch",
            "rng_state_sha256_after_epoch",
        ):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
