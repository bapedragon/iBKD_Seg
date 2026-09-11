from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R50_REPORT = (
    ROOT
    / "phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3"
)
R56_REPORT = ROOT / "phase1/phase1_cub/reports/legacy_resnet56_v2_guided"
R50_V4_REPORT = (
    ROOT
    / "phase1/phase1_cub/reports/frozen_probe/"
    "resnet50_224_b128_b64_guided_seed1_v4"
)
R50_V5_REPORT = (
    ROOT
    / "phase1/phase1_cub/reports/frozen_probe/"
    "resnet50_224_b128_guided_3seed_v5"
)
DIRECT_V2_REPORT = (
    ROOT
    / "phase1/phase1_cub/reports/direct_spatial/"
    "resnet50_224_b128_seed1_v2"
)
DIRECT_V2_3SEED_REPORT = (
    ROOT
    / "phase1/phase1_cub/reports/direct_spatial/"
    "resnet50_224_b128_guided_3seed_v2"
)


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


class Phase1CubResultTest(unittest.TestCase):
    def test_r50_teacher_is_the_audited_shared_v3_artifact(self) -> None:
        summary = _json(R50_REPORT / "summary.json")
        audit = _json(R50_REPORT / "checkpoint_audit.json")
        source = _json(R50_REPORT / "source_manifest.json")

        self.assertEqual(summary["status"], "complete_audited")
        self.assertEqual(
            summary["protocol_id"],
            "cub200_phase1_resnet50_224_scratch_b128_full_v3",
        )
        self.assertEqual(summary["architecture"], "torchvision_resnet50")
        self.assertEqual(summary["initialization"], "scratch")
        self.assertFalse(summary["external_pretraining"])
        self.assertEqual(summary["train_validation_test_counts"], [5394, 600, 5794])
        self.assertEqual(summary["selected_epoch"], 165)
        self.assertTrue(
            math.isclose(
                summary["official_test"]["macro_top1"],
                41.17768406867981,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(
            summary["checkpoint_sha256"],
            "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3",
        )
        self.assertEqual(summary["checkpoint_sha256"], audit["checkpoint"]["checkpoint_sha256"])
        self.assertTrue(audit["all_checks_passed"])
        self.assertEqual(audit["official_test"]["evaluations"], 1)
        self.assertFalse(audit["official_test"]["used_for_training_or_selection"])
        self.assertEqual(source["h200_issue_id"], "722")
        self.assertEqual(source["checkpoint_count"], 1)
        self.assertTrue(source["source_archive"]["all_member_crc_verified"])

    def test_superseded_r56_guided_result_remains_separate_and_complete(self) -> None:
        classification = _json(R56_REPORT / "classification_summary.json")
        probe = _json(R56_REPORT / "probe_summary.json")
        audit = _json(R56_REPORT / "checkpoint_audit.json")
        source = _json(R56_REPORT / "source_manifest.json")

        self.assertEqual(classification["status"], "complete_audited_superseded_v2")
        self.assertEqual(probe["status"], "complete_audited_superseded_v2")
        self.assertEqual(classification["independent_unit"], "encoder_seed")
        self.assertEqual(classification["independent_n"], 3)
        self.assertEqual(probe["independent_unit"], "encoder_seed")
        self.assertEqual(probe["independent_n"], 3)
        self.assertEqual(probe["probe_seeds_per_encoder"], 5)
        self.assertEqual(probe["official_test_evaluations"], 45)
        self.assertTrue(probe["all_45_selections_completed_before_official_test"])

        by_variant = {item["variant"]: item for item in probe["aggregates"]}
        self.assertGreater(
            by_variant["alg_warmup20"]["mean"],
            by_variant["ibkd_lambda_0.25"]["mean"],
        )
        self.assertGreater(
            by_variant["alg_warmup20"]["mean"],
            by_variant["ibkd_lambda_0.5"]["mean"],
        )
        self.assertEqual(audit["checkpoint_count"], 55)
        self.assertTrue(audit["all_file_hashes_match"])
        self.assertTrue(audit["all_strict_loads_passed"])
        self.assertTrue(audit["all_floating_tensors_finite"])
        self.assertEqual(source["h200_issue_id"], "716")
        self.assertEqual(source["checkpoint_count"], 55)
        self.assertTrue(source["source_archive"]["all_member_crc_verified"])

    def test_binary_checkpoints_are_not_tracked_inside_report_directories(self) -> None:
        self.assertEqual(list(R50_REPORT.rglob("*.pt")), [])
        self.assertEqual(list(R56_REPORT.rglob("*.pt")), [])
        self.assertEqual(list(R50_V4_REPORT.rglob("*.pt")), [])
        self.assertEqual(list(R50_V5_REPORT.rglob("*.pt")), [])
        self.assertEqual(list(DIRECT_V2_REPORT.rglob("*.pt")), [])
        self.assertEqual(list(DIRECT_V2_3SEED_REPORT.rglob("*.pt")), [])

    def test_r50_v4_guided_seed1_profiles_are_audited_but_partial(self) -> None:
        classification = _json(R50_V4_REPORT / "classification_summary.json")
        probe = _json(R50_V4_REPORT / "probe_summary.json")
        audit = _json(R50_V4_REPORT / "checkpoint_audit.json")
        source = _json(R50_V4_REPORT / "source_manifest.json")

        self.assertEqual(classification["status"], "complete_audited_partial_v4")
        self.assertEqual(probe["status"], "complete_audited_partial_v4")
        self.assertEqual(classification["independent_encoder_n_per_batch"], 1)
        self.assertFalse(classification["encoder_seed_standard_deviation_estimable"])
        self.assertTrue(probe["probe_seed_is_not_independent_replication"])
        self.assertEqual(probe["probe_seeds_per_encoder"], 5)
        self.assertFalse(probe["final_six_method_three_seed_matrix_complete"])
        self.assertEqual(probe["official_test_evaluations"], 40)

        cells = {
            (cell["batch_size"], cell["variant"]): cell for cell in probe["cells"]
        }
        self.assertTrue(
            math.isclose(
                cells[(128, "lg")]["test_input_224_mean_iou_probe_seed_mean"],
                0.736954398657864,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                cells[(128, "ibkd_lambda_0.25")][
                    "test_input_224_mean_iou_probe_seed_mean"
                ],
                0.7197203787525142,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertGreater(
            cells[(128, "lg")]["test_input_224_mean_iou_probe_seed_mean"],
            cells[(128, "ibkd_lambda_0.25")][
                "test_input_224_mean_iou_probe_seed_mean"
            ],
        )
        self.assertGreater(
            cells[(64, "alg_warmup20")]["test_input_224_mean_iou_probe_seed_mean"],
            cells[(64, "ibkd_lambda_0.25")][
                "test_input_224_mean_iou_probe_seed_mean"
            ],
        )

        self.assertEqual(audit["new_checkpoint_count"], 48)
        self.assertEqual(audit["classification_encoder_checkpoint_count"], 8)
        self.assertEqual(audit["selected_probe_checkpoint_count"], 40)
        self.assertTrue(audit["all_file_hashes_match"])
        self.assertTrue(audit["all_strict_loads_passed"])
        self.assertTrue(audit["all_floating_tensors_finite"])
        self.assertEqual(source["h200_issue_id"], "727")
        self.assertEqual(source["checkpoint_count"], 48)
        self.assertEqual(
            source["source_archive"]["canonical_filename"],
            "phase1_cub_resnet50_224_b128_b64_guided_seed1_v4_issue727.zip",
        )
        self.assertTrue(source["source_archive"]["all_member_crc_verified"])

    def test_r50_v5_guided_batch128_is_complete_for_three_encoder_seeds(self) -> None:
        classification = _json(R50_V5_REPORT / "classification_summary.json")
        probe = _json(R50_V5_REPORT / "probe_summary.json")
        audit = _json(R50_V5_REPORT / "checkpoint_audit.json")
        source = _json(R50_V5_REPORT / "source_manifest.json")

        self.assertEqual(classification["status"], "complete_audited_guided_3seed_v5")
        self.assertEqual(probe["status"], "complete_audited_guided_3seed_v5")
        self.assertEqual(classification["encoder_seeds"], [1, 2, 3])
        self.assertEqual(classification["independent_n"], 3)
        self.assertEqual(probe["independent_n"], 3)
        self.assertEqual(probe["probe_seeds_per_encoder"], 5)
        self.assertTrue(probe["probe_seed_is_not_independent_replication"])
        self.assertFalse(probe["final_six_method_three_seed_matrix_complete"])
        self.assertEqual(probe["official_test_evaluations"], 60)

        by_variant = {row["variant"]: row for row in probe["aggregates"]}
        self.assertTrue(
            math.isclose(
                by_variant["lg"]["mean"],
                0.7347006323996028,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                by_variant["ibkd_lambda_0.25"]["mean"],
                0.7031841065184272,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertGreater(by_variant["lg"]["mean"], by_variant["alg_warmup20"]["mean"])
        self.assertGreater(by_variant["alg_warmup20"]["mean"], by_variant["ibkd_lambda_0.25"]["mean"])
        self.assertEqual(audit["issue730_new_checkpoint_count"], 48)
        self.assertTrue(audit["all_file_hashes_match"])
        self.assertTrue(audit["all_strict_loads_passed"])
        self.assertEqual(source["h200_issue_id"], "730")
        self.assertEqual(source["checkpoint_count"], 48)

    def test_direct_spatial_v2_seed1_is_complete_audited_and_not_final(self) -> None:
        summary = _json(DIRECT_V2_REPORT / "direct_spatial_summary.json")
        audit = _json(DIRECT_V2_REPORT / "checkpoint_audit.json")
        source = _json(DIRECT_V2_REPORT / "source_manifest.json")

        self.assertEqual(summary["status"], "complete_audited_seed1_only")
        self.assertEqual(summary["encoder_seed"], 1)
        self.assertEqual(summary["independent_encoder_n"], 1)
        self.assertFalse(summary["final_encoder_seed_inference"])
        self.assertEqual(summary["official_test_evaluations"], 24)
        pck = {row["variant"]: row for row in summary["part_pck_primary"]}
        cka = {row["variant"]: row for row in summary["spatial_cka_block11_secondary"]}
        attention = {row["variant"]: row for row in summary["attention_gt_secondary"]}
        self.assertGreater(
            pck["lg"]["test_micro_pck_at_0.1_probe_seed_mean"],
            pck["ibkd_lambda_0.25"]["test_micro_pck_at_0.1_probe_seed_mean"],
        )
        self.assertGreater(
            cka["lg"]["centered_linear_cka"],
            cka["ibkd_lambda_0.25"]["centered_linear_cka"],
        )
        self.assertGreater(
            attention["ibkd_lambda_0.25"]["global_micro_patch_average_precision"],
            attention["lg"]["global_micro_patch_average_precision"],
        )
        self.assertEqual(audit["part_probes"]["selected_part_probe_checkpoint_count"], 20)
        self.assertTrue(audit["part_probes"]["all_strict_loads_passed"])
        self.assertTrue(audit["reused_encoders"]["hashes_match_audited_issue727_batch128_seed1"])
        self.assertEqual(source["h200_issue_id"], "737")
        self.assertEqual(source["checkpoint_count"], 20)
        self.assertEqual(len(list((DIRECT_V2_REPORT / "attention_qualitative").glob("*.png"))), 32)

    def test_direct_spatial_v2_is_complete_for_three_encoder_seeds(self) -> None:
        summary = _json(DIRECT_V2_3SEED_REPORT / "direct_spatial_summary.json")
        audit = _json(DIRECT_V2_3SEED_REPORT / "checkpoint_audit.json")
        source = _json(DIRECT_V2_3SEED_REPORT / "source_manifest.json")

        self.assertEqual(summary["status"], "complete_audited_guided_3seed_v2")
        self.assertEqual(summary["encoder_seeds"], [1, 2, 3])
        self.assertEqual(summary["independent_encoder_n"], 3)
        self.assertTrue(summary["probe_seed_is_not_independent_replication"])
        self.assertTrue(summary["final_encoder_seed_inference"])
        self.assertTrue(summary["guided_four_method_direct_spatial_block_complete"])
        self.assertEqual(summary["official_test_evaluations"]["total"], 72)
        self.assertFalse(summary["official_test_used_for_selection"])

        pck = {
            row["variant"]: row
            for row in summary["part_pck_primary"]["aggregates"]
        }
        cka = {
            row["variant"]: row
            for row in summary["spatial_cka_block11_secondary"]["aggregates"]
        }
        attention = {
            row["variant"]: row
            for row in summary["attention_gt_secondary"]["patch_ap_aggregates"]
        }
        self.assertTrue(
            math.isclose(
                pck["lg"]["mean"],
                0.27510137175394705,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertGreater(pck["lg"]["mean"], pck["alg_warmup20"]["mean"])
        self.assertGreater(
            pck["alg_warmup20"]["mean"],
            pck["ibkd_lambda_0.25"]["mean"],
        )
        self.assertGreater(cka["lg"]["mean"], cka["ibkd_lambda_0.25"]["mean"])
        self.assertGreater(
            attention["ibkd_lambda_0.25"]["mean"],
            attention["lg"]["mean"],
        )
        self.assertEqual(audit["issue739_new_checkpoint_count"], 40)
        self.assertTrue(audit["reused_encoders"]["hashes_match_audited_issue730"])
        self.assertTrue(audit["part_probes"]["all_strict_loads_passed"])
        self.assertEqual(source["h200_issue_id"], "739")
        self.assertEqual(source["checkpoint_count"], 40)
        self.assertEqual(
            len(
                list(
                    (
                        DIRECT_V2_3SEED_REPORT
                        / "attention_qualitative_seed2_3"
                    ).glob("*.png")
                )
            ),
            64,
        )

    def test_release_manifests_bind_the_remote_assets_and_checkpoint_roles(self) -> None:
        teacher = _json(R50_REPORT / "checkpoint_release.json")
        legacy = _json(R56_REPORT / "artifact_release.json")
        v4 = _json(R50_V4_REPORT / "artifact_release.json")
        v5 = _json(R50_V5_REPORT / "artifact_release.json")
        direct = _json(DIRECT_V2_REPORT / "artifact_release.json")

        self.assertEqual(teacher["size_bytes"], 89065279)
        self.assertEqual(
            teacher["sha256"],
            "d7ca69d23a99e20ac3950bb8c0b954ce77dd1213245a7d707447d1997565b0b5",
        )
        self.assertEqual(teacher["contents"]["teacher_checkpoints"], 1)
        self.assertEqual(
            teacher["teacher_checkpoint"]["checkpoint_sha256"],
            "ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3",
        )
        self.assertTrue(teacher["remote_asset_digest_verified"])

        self.assertEqual(legacy["size_bytes"], 192296835)
        self.assertEqual(
            legacy["sha256"],
            "8ce5d2e9407a0522babfd4d25a238af6d7ce3d72b2aea6d508cedf5764e54064",
        )
        self.assertEqual(legacy["contents"]["total_checkpoints"], 55)
        self.assertFalse(legacy["canonical_phase1_cub_v3_result"])
        self.assertTrue(legacy["remote_asset_digest_verified"])

        self.assertEqual(v4["size_bytes"], 524076632)
        self.assertEqual(
            v4["sha256"],
            "9d9f7b4554b588bed69abf5e435149222af4f70d9a0fd8159d47d37319fdbc96",
        )
        self.assertEqual(v4["contents"]["classification_student_checkpoints"], 8)
        self.assertEqual(v4["contents"]["selected_probe_checkpoints"], 40)
        self.assertEqual(v4["contents"]["total_checkpoints"], 48)
        self.assertTrue(v4["canonical_phase1_cub_v3_partial_result"])
        self.assertFalse(v4["final_six_method_three_seed_matrix_complete"])
        self.assertTrue(v4["remote_asset_digest_verified"])

        self.assertEqual(v5["size_bytes"], 524004112)
        self.assertEqual(
            v5["sha256"],
            "9f1aeabf002cf1ba728b105b53601cbc59158e1fe11fa8a61672612fcaff6212",
        )
        self.assertEqual(v5["contents"]["classification_student_checkpoints"], 8)
        self.assertEqual(v5["contents"]["selected_probe_checkpoints"], 40)
        self.assertEqual(v5["contents"]["total_checkpoints"], 48)
        self.assertTrue(v5["canonical_phase1_cub_v3_guided_block_complete"])
        self.assertFalse(v5["final_six_method_three_seed_matrix_complete"])
        self.assertTrue(v5["remote_asset_digest_verified"])

        self.assertEqual(direct["size_bytes"], 5959912)
        self.assertEqual(
            direct["sha256"],
            "534c863ea7bddd3db706ea44f108d89fdd24428370f6cfdac887e24f844630a2",
        )
        self.assertEqual(direct["contents"]["part_probe_checkpoints"], 20)
        self.assertEqual(direct["contents"]["qualitative_pngs"], 32)
        self.assertEqual(direct["contents"]["total_checkpoints"], 20)
        self.assertFalse(direct["final_encoder_seed_inference"])
        self.assertTrue(direct["remote_asset_digest_verified"])


if __name__ == "__main__":
    unittest.main()
