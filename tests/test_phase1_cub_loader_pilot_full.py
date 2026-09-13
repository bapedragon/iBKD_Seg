from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from ibkd_seg.phase1.cub_loader_profiles import LOADER_PROFILE_ORDER
from ibkd_seg.phase1.run_cub_loader_pilot_full import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_VARIANTS,
    _classification_command,
    _validate_config,
)
from ibkd_seg.phase1.train_full import validate_args
from ibkd_seg.phase1.train_timing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "phase1/phase1_cub/image_loader_experiment/configs/"
    "cub200_r50_224_b128_seed1_loader_pilot_full_v1.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/image_loader_experiment/scripts/"
    "run_r50_224_loader_pilot_full_b128_seed1.sh"
)


def _student_args(profile: str, variant: str) -> argparse.Namespace:
    variants = {
        "lg": ("lg", None, 0),
        "alg_warmup20": ("alg", None, 20),
        "ibkd_lambda_0.25": ("ibkd", 0.25, 0),
        "ibkd_lambda_0.5": ("ibkd", 0.5, 0),
    }
    method, fusion_ratio, alg_warmup = variants[variant]
    return argparse.Namespace(
        dataset="cub",
        kind="student",
        method=method,
        batch_size=128,
        fusion_ratio=fusion_ratio,
        teacher_checkpoint=Path("teacher.pt"),
        teacher_architecture="resnet50_224_scratch",
        scientific_cub_r50_teacher=True,
        seed_extension_full=False,
        loader_pilot_full=True,
        cub_loader_profile=profile,
        protocol_config=CONFIG,
        batch_profile_role="exploratory_loader_pilot",
        eval_batch_size=200,
        num_workers=4,
        seed=1,
        alg_controller_warmup_epochs=alg_warmup,
        posthoc_diagnostic_id=None,
    )


class Phase1CubLoaderPilotFullTest(unittest.TestCase):
    def test_locked_config_and_three_shard_gate(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        _validate_config(config, CONFIG)
        self.assertEqual(file_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        self.assertEqual(tuple(config["scope"]["loader_profiles"]), LOADER_PROFILE_ORDER)
        self.assertEqual(tuple(config["scope"]["variants"]), EXPECTED_VARIANTS)
        self.assertTrue(config["scope"]["one_loader_profile_per_h200_job"])
        self.assertTrue(
            config["scope"]["all_three_shards_required_before_loader_selection"]
        )
        self.assertEqual(
            config["completion_gate_per_shard"]["official_test_evaluations"], 0
        )
        self.assertTrue(
            config["execution"]["single_all_profile_job_forbidden_by_10h_limit"]
        )

    def test_all_twelve_classification_cells_validate_and_stay_test_closed(self) -> None:
        commands = []
        for profile in LOADER_PROFILE_ORDER:
            for variant in EXPECTED_VARIANTS:
                args = _student_args(profile, variant)
                validate_args(args)
                runner_args = argparse.Namespace(
                    profile=profile,
                    config=CONFIG,
                    data_dir=Path("/tmp/data"),
                    teacher_checkpoint=Path("/tmp/teacher.pt"),
                    eval_batch_size=200,
                    num_workers=4,
                )
                command, _name = _classification_command(
                    runner_args,
                    variant=variant,
                    student_root=Path("/tmp/output"),
                )
                commands.append(command)
                self.assertIn("--loader-pilot-full", command)
                self.assertIn("exploratory_loader_pilot", command)
                self.assertIn(profile, command)
                self.assertNotIn("--access-official-test", command)
        self.assertEqual(len(commands), 12)

    def test_nondefault_loader_cannot_escape_pilot(self) -> None:
        args = _student_args("l1_matched_weak", "lg")
        args.loader_pilot_full = False
        args.batch_profile_role = "locked_v3_partial_cell"
        with self.assertRaisesRegex(ValueError, "Non-pilot CUB"):
            validate_args(args)

    def test_loader_pilot_rejects_seed_or_batch_drift(self) -> None:
        args = _student_args("l0_current_strong", "lg")
        args.seed = 2
        with self.assertRaisesRegex(ValueError, "fixed to batch 128 and seed 1"):
            validate_args(args)
        args = _student_args("l0_current_strong", "lg")
        args.batch_size = 64
        with self.assertRaisesRegex(ValueError, "fixed to batch 128 and seed 1"):
            validate_args(args)

    def test_h200_entry_accepts_exactly_one_profile(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('profile="${1:?', script)
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_loader_pilot_full", script)
        self.assertIn("--profile", script)
        self.assertIn("loader_pilot_full_v1.json", script)
        self.assertNotIn("run_cub_r50_teacher_full", script)
        self.assertIn('tee "${output_root}/run.log"', script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_runner_source_never_imports_official_test_loader(self) -> None:
        source = (
            ROOT / "src/ibkd_seg/phase1/run_cub_loader_pilot_full.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("load_official_test_records", source)
        self.assertNotIn("build_official_test_loader", source)
        self.assertIn('"official_test_evaluations": 0', source)


if __name__ == "__main__":
    unittest.main()
