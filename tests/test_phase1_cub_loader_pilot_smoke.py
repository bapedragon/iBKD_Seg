from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

import torch

from ibkd_seg.phase1.cub_loader_profiles import LOADER_PROFILE_ORDER
from ibkd_seg.phase1.probe import train_candidate
from ibkd_seg.phase1.run_cub_loader_pilot_smoke import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_VARIANTS,
    VARIANT_ARGUMENTS,
    _classification_command,
    _completion_gate_for_profiles,
    _normalize_profiles,
    _validate_config,
)
from ibkd_seg.phase1.train_timing import file_sha256, validate_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_loader_pilot_smoke_v2.json"
)
FAILED_V1_CONFIG = CONFIG.with_name(
    "cub200_r50_224_b128_loader_pilot_smoke_v1.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/scripts/"
    "run_r50_224_loader_pilot_smoke_b128_seed1.sh"
)
L1_L2_SCRIPT = SCRIPT.with_name(
    "run_r50_224_loader_pilot_smoke_l1_l2_b128_seed1.sh"
)
AUDIT_SUMMARY = (
    ROOT
    / "phase1/phase1_cub/reports/loader_pilot/damage_audit_v1/summary.json"
)
AUDIT_SOURCE = AUDIT_SUMMARY.with_name("source_manifest.json")
FAILED_V1_SUMMARY = (
    ROOT
    / "phase1/phase1_cub/reports/loader_pilot/"
    "failed_smoke_v1_issue746/failure_summary.json"
)


def _timing_args(profile: str, variant: str) -> argparse.Namespace:
    method, fusion_ratio, _warmup = VARIANT_ARGUMENTS[variant]
    return argparse.Namespace(
        dataset="cub",
        kind="student",
        method=method,
        teacher_architecture="resnet50_224_scratch",
        access_official_test=False,
        scientific_cub_r50_teacher=True,
        seed_extension_smoke=False,
        loader_pilot_smoke=True,
        cub_loader_profile=profile,
        batch_size=128,
        fusion_ratio=fusion_ratio,
        teacher_checkpoint=Path("teacher.pt"),
        eval_batch_size=200,
        num_workers=4,
        seed=1,
        alg_controller_warmup_epochs=20 if method == "alg" else 0,
        save_student_checkpoint=True,
    )


class Phase1CubLoaderPilotSmokeTest(unittest.TestCase):
    def test_damage_audit_result_is_traceable_and_diagnostic_only(self) -> None:
        summary = json.loads(AUDIT_SUMMARY.read_text(encoding="utf-8"))
        source = json.loads(AUDIT_SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["source_h200_issue"], 745)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["official_test_accessed"])
        self.assertTrue(
            summary["profiles"]["l2_conservative_spatial"][
                "visible_part_retention"
            ]
            > summary["profiles"]["l0_current_strong"][
                "visible_part_retention"
            ]
        )
        self.assertTrue(
            summary["contracts"]["loader_selection_from_audit_forbidden"]
        )
        self.assertEqual(source["source_h200_issue"], 745)
        self.assertEqual(len(source["source_sha256"]), 64)

    def test_locked_config_provenance_and_matrix(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        _validate_config(config, CONFIG)
        self.assertEqual(file_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        self.assertEqual(tuple(config["loader_profiles"]), LOADER_PROFILE_ORDER)
        self.assertEqual(
            tuple(config["classification"]["variants"]), EXPECTED_VARIANTS
        )
        self.assertEqual(
            config["completion_gate"]["classification_students"], 12
        )
        self.assertEqual(
            config["completion_gate"]["official_test_evaluations"], 0
        )
        self.assertEqual(
            file_sha256(FAILED_V1_CONFIG),
            config["failed_predecessor"]["config_sha256"],
        )
        probe = config["frozen_probe"]["probe"]
        self.assertEqual(probe["parameter_count"], 386)
        self.assertEqual(probe["initialization"]["weight_std"], 0.01)
        self.assertEqual(probe["optimizer"]["name"], "sgd")
        self.assertEqual(probe["scheduler"]["name"], "cosine")

    def test_issue746_failure_is_preserved_as_non_scientific(self) -> None:
        failure = json.loads(FAILED_V1_SUMMARY.read_text(encoding="utf-8"))
        self.assertFalse(failure["scientific_result"])
        self.assertEqual(failure["source_h200_issue"], 746)
        self.assertEqual(failure["completed"]["classification_students"], 12)
        self.assertEqual(
            failure["completed"]["segmentation_probe_candidates"], 0
        )
        self.assertEqual(failure["completed"]["official_test_evaluations"], 0)
        self.assertEqual(
            failure["replacement"]["sha256"], EXPECTED_CONFIG_SHA256
        )

    def test_all_twelve_timing_commands_are_validation_only(self) -> None:
        commands = []
        for profile in LOADER_PROFILE_ORDER:
            for variant in EXPECTED_VARIANTS:
                with self.subTest(profile=profile, variant=variant):
                    validate_args(_timing_args(profile, variant))
                    command, _name, _directory = _classification_command(
                        data_dir=Path("/tmp/data"),
                        output_dir=Path("/tmp/output"),
                        teacher_checkpoint=Path("/tmp/teacher.pt"),
                        profile=profile,
                        variant=variant,
                        eval_batch_size=200,
                        num_workers=4,
                    )
                    commands.append(command)
                    self.assertIn("--loader-pilot-smoke", command)
                    self.assertIn(profile, command)
                    self.assertNotIn("--access-official-test", command)
        self.assertEqual(len(commands), 12)

    def test_l1_l2_operational_subset_uses_dynamic_gate(self) -> None:
        profiles = _normalize_profiles(
            ["l1_matched_weak", "l2_conservative_spatial"]
        )
        gate = _completion_gate_for_profiles(profiles)
        self.assertEqual(
            profiles, ("l1_matched_weak", "l2_conservative_spatial")
        )
        self.assertEqual(gate["loader_profiles"], 2)
        self.assertEqual(gate["classification_students"], 8)
        self.assertEqual(gate["segmentation_probe_lr_candidates"], 24)
        self.assertEqual(gate["part_probe_validation_selections"], 8)
        self.assertEqual(gate["spatial_cka_values"], 96)
        self.assertEqual(gate["attention_qualitative_pngs"], 32)
        self.assertEqual(gate["official_test_evaluations"], 0)
        with self.assertRaisesRegex(ValueError, "locked L0 -> L1 -> L2 order"):
            _normalize_profiles(
                ["l2_conservative_spatial", "l1_matched_weak"]
            )

    def test_frozen_probe_runtime_schema_executes_one_candidate(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        probe_config = config["frozen_probe"]["probe"]
        generator = torch.Generator().manual_seed(746)
        features = torch.randn(2, 192, 14, 14, generator=generator)
        targets = torch.zeros(2, 14, 14, dtype=torch.long)
        targets[:, :, 7:] = 1
        state, result = train_candidate(
            features,
            targets,
            features.clone(),
            targets.clone(),
            probe_config=probe_config,
            learning_rate=0.01,
            seed=1,
            device=torch.device("cpu"),
            epochs=1,
        )
        self.assertEqual(set(state), {"weight", "bias"})
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(
            result["gradient_contract"]["probe_gradient_tensor_count"], 2
        )

    def test_non_default_profile_cannot_escape_pilot(self) -> None:
        args = _timing_args("l1_matched_weak", "lg")
        args.loader_pilot_smoke = False
        with self.assertRaisesRegex(ValueError, "require --loader-pilot-smoke"):
            validate_args(args)

    def test_pilot_cannot_open_official_test(self) -> None:
        args = _timing_args("l0_current_strong", "lg")
        args.access_official_test = True
        with self.assertRaisesRegex(ValueError, "validation-only"):
            validate_args(args)

    def test_h200_script_reuses_audited_teacher(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_loader_pilot_smoke", script)
        self.assertIn("loader_pilot_smoke_v2.json", script)
        self.assertNotIn("loader_pilot_smoke_v1.json", script)
        self.assertIn("teacher_best_validation.pt", script)
        self.assertIn('tee "${output_root}/run.log"', script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_l1_l2_h200_script_is_validation_only_subset(self) -> None:
        script = L1_L2_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_loader_pilot_smoke", script)
        self.assertIn(
            "--profiles l1_matched_weak l2_conservative_spatial", script
        )
        self.assertNotIn("--access-official-test", script)
        self.assertIn('tee "${output_root}/run.log"', script)
        self.assertTrue(L1_L2_SCRIPT.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
