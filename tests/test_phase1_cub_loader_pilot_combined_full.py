from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ibkd_seg.phase1.run_cub_loader_pilot_full import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_TEACHER_SHA256,
    EXPECTED_TEACHER_STATE_SHA256,
    EXPECTED_VARIANTS,
)
from ibkd_seg.phase1.summarize_cub_loader_pilot_l1_l2 import (
    JOB_ID,
    PROFILES,
    _validate_job_config,
    run as summarize,
)


ROOT = Path(__file__).resolve().parents[1]
JOB_CONFIG = (
    ROOT
    / "phase1/phase1_cub/configs/"
    "cub200_r50_224_b128_seed1_loader_pilot_l1_l2_combined_job_v1.json"
)
SCRIPT = (
    ROOT
    / "phase1/phase1_cub/scripts/"
    "run_r50_224_loader_pilot_full_l1_l2_b128_seed1.sh"
)
PER_PROFILE_GATE = {
    "loader_profiles": 1,
    "classification_students": 4,
    "classification_checkpoints": 4,
    "segmentation_probe_lr_candidates": 60,
    "segmentation_probe_validation_selections": 20,
    "part_probe_lr_candidates": 60,
    "part_probe_validation_selections": 20,
    "spatial_cka_values": 48,
    "attention_metric_rows": 4,
    "attention_qualitative_pngs": 16,
    "new_checkpoints": 44,
    "official_test_evaluations": 0,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("value\n" + "1\n" * count, encoding="utf-8")


def _synthetic_profile(root: Path, profile: str) -> None:
    classifications = [
        {
            "profile": profile,
            "variant": variant,
            "validation_macro_top1": 70.0 + index,
            "official_test_evaluations": 0,
        }
        for index, variant in enumerate(EXPECTED_VARIANTS)
    ]
    segmentation = []
    part = []
    for index, variant in enumerate(EXPECTED_VARIANTS):
        for probe_seed in range(1, 6):
            segmentation.append(
                {
                    "profile": profile,
                    "variant": variant,
                    "probe_seed": probe_seed,
                    "official_test_evaluations": 0,
                    "validation": {
                        "input_224": {"mean_iou": 0.50 + index / 100}
                    },
                }
            )
            part.append(
                {
                    "profile": profile,
                    "variant": variant,
                    "probe_seed": probe_seed,
                    "official_test_evaluations": 0,
                    "validation": {"micro_pck_at_0.1": 0.20 + index / 100},
                }
            )
    cka = [
        {
            "variant": variant,
            "student_block": block,
            "centered_linear_cka": 0.10 + block / 100,
        }
        for variant in EXPECTED_VARIANTS
        for block in range(12)
    ]
    attention = [
        {
            "variant": variant,
            "global_micro_patch_average_precision": 0.3,
            "pointing_game_peak_inside_mask": 0.4,
        }
        for variant in EXPECTED_VARIANTS
    ]
    part_score = sum(0.20 + index / 100 for index in range(4)) / 4
    seg_score = sum(0.50 + index / 100 for index in range(4)) / 4
    _write_json(
        root / "summary.json",
        {
            "status": "complete",
            "protocol_id": "cub200_phase1_r50_224_b128_seed1_loader_pilot_full_v1",
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "loader_profile": profile,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "confirmatory_main_result": False,
            "all_three_shards_required_before_loader_selection": True,
            "official_test_accessed": False,
            "official_test_evaluations": 0,
            "this_shard_alone_must_not_select_loader": True,
            "classification": classifications,
            "loader_selection_scores": {
                "primary_mean_validation_part_micro_pck_at_0.1": part_score,
                "tie_break_1_mean_validation_frozen_segmentation_input224_miou": seg_score,
            },
            "completion_gate": PER_PROFILE_GATE,
            "runtime": {"elapsed_seconds": 100.0},
        },
    )
    _write_json(
        root / "segmentation_probe/results.json",
        {
            "status": "complete",
            "profile": profile,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "selections": segmentation,
            "official_test_evaluations": 0,
        },
    )
    _write_json(
        root / "part_probe/results.json",
        {
            "status": "complete",
            "profile": profile,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "selections": part,
            "official_test_evaluations": 0,
        },
    )
    _write_json(
        root / "spatial_cka/results.json",
        {
            "status": "complete",
            "profile": profile,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "rows": cka,
            "official_test_used": False,
        },
    )
    _write_json(
        root / "attention_gt/results.json",
        {
            "status": "complete",
            "profile": profile,
            "scientific_result": True,
            "exploratory_loader_pilot": True,
            "rows": attention,
            "official_test_evaluations": 0,
        },
    )
    _write_json(
        root / "checkpoint_audit.json",
        {
            "status": "pass",
            "teacher": {
                "checkpoint_sha256": EXPECTED_TEACHER_SHA256,
                "model_state_sha256": EXPECTED_TEACHER_STATE_SHA256,
            },
        },
    )
    manifest_entries = []
    kinds = (
        ["external_shared_teacher"]
        + ["classification_encoder"] * 4
        + ["selected_segmentation_probe"] * 20
        + ["selected_part_probe"] * 20
    )
    for index, kind in enumerate(kinds):
        checkpoint = root / "synthetic_checkpoints" / f"{index}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{profile}:{index}".encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest_entries.append(
            {"kind": kind, "path": str(checkpoint), "sha256": digest}
        )
    _write_json(
        root / "checkpoint_manifest.json",
        {
            "count": 45,
            "external_shared_teacher_count": 1,
            "new_checkpoint_count": 44,
            "entries": manifest_entries,
        },
    )
    _write_csv(root / "segmentation_probe/candidates.csv", 60)
    _write_csv(root / "part_probe/candidates.csv", 60)
    qualitative = root / "attention_gt/qualitative"
    qualitative.mkdir(parents=True, exist_ok=True)
    for index in range(16):
        (qualitative / f"{index}.png").write_bytes(b"png")


class Phase1CubLoaderPilotCombinedFullTest(unittest.TestCase):
    def test_job_config_keeps_scientific_protocol_and_test_sealed(self) -> None:
        config = json.loads(JOB_CONFIG.read_text(encoding="utf-8"))
        _validate_job_config(config, JOB_CONFIG)
        self.assertEqual(config["job_id"], JOB_ID)
        self.assertEqual(tuple(config["profiles"]), PROFILES)
        self.assertFalse(config["policy"]["official_test_accessed"])
        self.assertFalse(config["policy"]["loader_selection_in_this_job"])
        self.assertEqual(config["completion_gate"]["new_checkpoints"], 88)

    def test_h200_runner_is_sequential_and_keeps_profile_outputs_separate(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertLess(
            script.index("run_profile l1_matched_weak"),
            script.index("run_profile l2_conservative_spatial"),
        )
        self.assertIn('profile_output="${combined_root}/${profile}"', script)
        self.assertIn('profile_cache="${cache_root}/${profile}"', script)
        self.assertIn("ibkd_seg.phase1.release_asset", script)
        self.assertIn("ibkd_seg.phase1.run_cub_loader_pilot_full", script)
        self.assertIn("summarize_cub_loader_pilot_l1_l2", script)
        self.assertNotIn("--access-official-test", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)

    def test_combined_audit_lists_eight_results_and_exact_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in PROFILES:
                _synthetic_profile(root / profile, profile)
            output = StringIO()
            args = argparse.Namespace(combined_root=root, job_config=JOB_CONFIG)
            with redirect_stdout(output):
                summary = summarize(args)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(summary["results"]), 8)
            self.assertEqual(summary["completion_gate"]["new_checkpoints"], 88)
            self.assertEqual(summary["official_test_evaluations"], 0)
            self.assertIn("new_checkpoints=88/88", output.getvalue())
            self.assertTrue((root / "combined_summary.json").is_file())

    def test_combiner_never_imports_official_test_loader(self) -> None:
        source = (
            ROOT / "src/ibkd_seg/phase1/summarize_cub_loader_pilot_l1_l2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("load_official_test_records", source)
        self.assertNotIn("build_official_test_loader", source)


if __name__ == "__main__":
    unittest.main()
