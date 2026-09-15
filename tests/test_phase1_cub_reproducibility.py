from __future__ import annotations

import argparse
import copy
import inspect
import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.run_cub_ibkd_controlled_aa_smoke import (
    CONTROLLED_ENVIRONMENT,
    EXPECTED_CONFIG_SHA256,
    compare_summaries,
    validate_config,
)
from ibkd_seg.phase1.train_timing import (
    configure_controlled_reproducibility,
    file_sha256,
    validate_args as validate_timing_args,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "phase1/phase1_cub/reproducibility"
CONFIG = (
    EXPERIMENT
    / "configs/cub200_r50_224_b128_main_l0_ibkd_controlled_aa_smoke_v2.json"
)
SCRIPT = (
    EXPERIMENT
    / "scripts/run_main_l0_ibkd_controlled_aa_smoke_b128_seed1.sh"
)


def _timing_args(**changes: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dataset": "cub",
        "kind": "student",
        "teacher_architecture": "resnet50_224_scratch",
        "access_official_test": False,
        "method": "ibkd",
        "batch_size": 128,
        "fusion_ratio": 0.25,
        "data_dir": Path("data"),
        "output_dir": Path("output"),
        "run_name": "run_A",
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
        "controlled_aa_smoke": True,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def _summary() -> dict[str, object]:
    epoch = {
        "epoch": 1,
        "lr": 0.0001,
        "beta": 2.5,
        "train_loss": 1.0,
        "train_ce": 0.5,
        "train_guidance": 0.2,
        "train_alignment": 0.1,
        "train_fusion": 0.1,
        "train_top1": 1.0,
        "validation": {"macro_top1": 2.0},
        "seconds_including_validation": 10.0,
        "peak_cuda_memory_bytes": 100,
        "peak_cuda_memory_reserved_bytes": 200,
        "input_stream_sha256": "input",
        "student_state_sha256_after_epoch": "student-epoch",
        "guidance_state_sha256_after_epoch": "guidance-epoch",
        "rng_state_sha256_after_epoch": "rng",
    }
    runtime = {
        "python": "3.10",
        "platform": "linux",
        "torch": "2.11",
        "torchvision": "0.26",
        "timm": "1.0.27",
        "device": "cuda",
        "gpu_name": "H200",
        "cuda": "13",
        "git_commit": "a" * 40,
    }
    return {
        "status": "complete",
        "scientific_result": False,
        "official_test_accessed": False,
        "official_test": None,
        "controlled_aa_smoke": True,
        "dataset": "CUB-200-2011",
        "kind": "student",
        "method": "ibkd",
        "batch_size": 128,
        "seed": 1,
        "fusion_ratio_lambda": 0.25,
        "actual_epochs": 2,
        "planned_epochs": 300,
        "initial_student_state_sha256": "initial-student",
        "initial_guidance_state_sha256": "initial-guidance",
        "teacher_checkpoint_sha256": "teacher-file",
        "teacher_model_state_sha256": "teacher-state",
        "teacher_checkpoint_kind": "cub_r50_v3_scientific",
        "controller": {"stop_epoch": None},
        "guidance_controller_warmup_epochs": 20,
        "ibkd_aggregation": {"mode": "learned_all"},
        "student_state_sha256": "student-final",
        "guidance_state_sha256": "guidance-final",
        "optimizer_contract": "adamw",
        "split_manifest": {"validation_image_ids_sha256": "validation"},
        "controlled_reproducibility": {
            "enabled": True,
            "formal_bitwise_determinism": False,
            "known_nondeterministic_operation": {"kernel": "compute_grad_input"},
        },
        "checkpoint_sha256": "checkpoint-container",
        "epochs": [epoch, {**epoch, "epoch": 2}],
        "runtime": runtime,
    }


class Phase1CubReproducibilityTest(unittest.TestCase):
    def test_locked_smoke_config_validates(self) -> None:
        self.assertEqual(file_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        config = validate_config(CONFIG)
        self.assertFalse(config["scientific_result"])
        self.assertFalse(config["official_test_policy"]["accessed"])
        self.assertEqual(
            config["controlled_reproducibility"]["independent_fresh_processes"], 2
        )

    def test_timing_flag_is_restricted_to_exact_aa_scope(self) -> None:
        validate_timing_args(_timing_args())
        invalid = (
            {"num_workers": 4},
            {"access_official_test": True},
            {"fusion_ratio": 0.5},
            {"batch_size": 64},
            {"seed": 2},
            {"cub_loader_profile": "l2_conservative_spatial"},
            {"method": "lg", "fusion_ratio": None},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "Controlled A/A"):
                    validate_timing_args(_timing_args(**changes))

    def test_comparison_ignores_timing_but_not_scientific_trace(self) -> None:
        run_a = _summary()
        run_b = copy.deepcopy(run_a)
        run_b["epochs"][0]["seconds_including_validation"] = 99.0
        run_b["epochs"][0]["peak_cuda_memory_bytes"] = 999
        self.assertTrue(
            compare_summaries(run_a, run_b)["execution_control_gates_passed"]
        )

        run_b["epochs"][1]["train_loss"] = 1.0001
        result = compare_summaries(run_a, run_b)
        self.assertTrue(result["execution_control_gates_passed"])
        self.assertFalse(
            result["numerical_observation_not_a_smoke_gate"]["epoch_metrics"][
                "all_reported_values_exact"
            ]
        )

        run_b["epochs"][1]["input_stream_sha256"] = "different-input"
        result = compare_summaries(run_a, run_b)
        self.assertFalse(result["execution_control_gates_passed"])
        self.assertIn("epoch_input_rng_control", result["failures"])

    def test_h200_entry_sets_environment_and_runs_two_process_orchestrator(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        for key, value in CONTROLLED_ENVIRONMENT.items():
            self.assertIn(f"export {key}={value}", script)
        self.assertIn(
            "ibkd_seg.phase1.run_cub_ibkd_controlled_aa_smoke", script
        )
        self.assertNotIn("--access-official-test", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_warn_only_limitation_is_explicit(self) -> None:
        source = inspect.getsource(configure_controlled_reproducibility)
        self.assertIn(
            "torch.use_deterministic_algorithms(True, warn_only=True)", source
        )
        self.assertIn('"formal_bitwise_determinism": False', source)
        self.assertIn('"kernel": "compute_grad_input"', source)
        self.assertIn('torch.set_float32_matmul_precision("highest")', source)

    def test_config_exact_gate_covers_inputs_rng_and_model_states(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        fields = set(config["comparison_gate"]["exact_match_required"])
        self.assertTrue(
            {
                "epoch_input_stream_sha256",
                "epoch_rng_state_sha256",
                "runtime_contract",
                "known_nondeterministic_operation_disclosure",
            }.issubset(fields)
        )


if __name__ == "__main__":
    unittest.main()
