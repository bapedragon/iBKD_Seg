import contextlib
import copy
import io
import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ibkd_seg.cityscapes.tiny_repeat import (HASHES, compare_repeats, fixed_calibration,
                                           gradient_hash, rng_hash, warning_summary)
from ibkd_seg.cityscapes.tiny_smoke import REPEAT_CONFIG, load_config, main


class TinyRepeatTests(unittest.TestCase):
    def fixture(self):
        config = load_config(REPEAT_CONFIG)
        rows = []
        for plan in config["runs"]:
            row = {"run_id": plan["id"], "method": plan["method"], "status": "passed",
                   "completed_steps": 25, "losses": []}
            for key in ("student_initial_state_sha256", "guidance_initial_state_sha256",
                        "teacher_state_sha256", "input_sha256", "source_sha256", "config_sha256",
                        "calibration_input_sha256"):
                row[key] = key
            for step in range(1, 26):
                row["losses"].append({"step": step, "beta": config["diagnostic_fixed_beta"],
                                      "loss": 2., "ce": 1.5, "guidance": 3., "grad_norm_unclipped": 4.,
                                      "trace": {key: key + str(step) for key in HASHES}})
            rows.append(row)
        return config, rows

    def test_common_protocol_preserved_and_beta_not_reestimated(self):
        old = load_config(REPEAT_CONFIG.with_name("smoke25_v1.json"))
        new = load_config(REPEAT_CONFIG)
        for key, value in old.items():
            if key not in {"protocol_id", "run_kind", "runs", "beta_status"}:
                self.assertEqual(new[key], value, key)
        beta = new["diagnostic_fixed_beta"]
        self.assertEqual(beta, 0.018658411532808426)
        for rows in ([{"ce": 3., "guidance": 1.}], [{"ce": 5., "guidance": 100.}]):
            c = fixed_calibration(rows, beta)
            self.assertEqual(c["pilot_beta"], beta)
            self.assertEqual(c["beta_candidates"], [])

    def test_all_pairs_reported_and_first_mismatch_located(self):
        c, rows = self.fixture()
        self.assertEqual(compare_repeats(rows, c)["status"], "passed")
        rows[1]["losses"][3]["guidance"] = 3.1
        rows[1]["losses"][0]["trace"]["student_gradient_sha256"] = "different"
        report = compare_repeats(rows, c)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["pairs"]), 3)
        pair = report["pairs"]["lg_repeat"]
        self.assertEqual(pair["first_scalar_mismatch"]["step"], 4)
        self.assertEqual(pair["exact_hash_comparison"]["student_gradient_sha256"]["first_different_step"], 1)
        self.assertTrue(report["pairs"]["lg_a_vs_alg"]["passed"])
        self.assertEqual(report["interpretation"], "same_LG_repeat_varies_not_evidence_of_ALG_controller_effect")

    def test_cross_method_only_mismatch_has_distinct_interpretation(self):
        c, rows = self.fixture()
        rows[2]["losses"][5]["loss"] += .1
        report = compare_repeats(rows, c)
        self.assertTrue(report["pairs"]["lg_repeat"]["passed"])
        self.assertEqual(report["interpretation"], "LG_repeat_matches_but_cross_method_differs_requires_investigation")

    def test_mismatched_beta_input_rng_and_missing_trace_rejected(self):
        c, original = self.fixture()
        for key in ("beta", "input_sha256", "rng_before", "missing"):
            rows = copy.deepcopy(original)
            record = rows[1]["losses"][0]
            if key == "beta":
                record[key] += 1e-12
            elif key == "missing":
                del record["trace"]["student_gradient_sha256"]
            else:
                record["trace"][key] = "different"
            report = compare_repeats(rows, c)
            self.assertEqual(report["status"], "failed", key)

    def test_missing_failed_and_partial_runs_do_not_look_successful(self):
        c, rows = self.fixture()
        self.assertEqual(compare_repeats(rows[:1], c)["status"], "failed")
        rows[0]["status"] = "failed"
        rows[0]["losses"] = rows[0]["losses"][:5]
        result = compare_repeats(rows, c)
        self.assertFalse(result["all_runs_completed"])
        self.assertEqual(len(result["pairs"]), 3)
        self.assertTrue(result["pairs"]["lg_b_vs_alg"]["passed"])

    def test_hashes_do_not_change_rng_and_detect_gradient_changes(self):
        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        before = rng_hash()
        self.assertEqual(before, rng_hash())
        torch.rand(1)
        self.assertNotEqual(before, rng_hash())
        model = torch.nn.Linear(2, 2)
        model(torch.ones(1, 2)).sum().backward()
        before = gradient_hash(model)
        self.assertEqual(before, gradient_hash(model))
        model.weight.grad[0, 0] += .01
        self.assertNotEqual(before, gradient_hash(model))

    def test_warning_text_count_and_failure_final_json_survive(self):
        def failing_measure(*args):
            for _ in range(2):
                warnings.warn("test operation has no deterministic implementation", UserWarning)
            raise RuntimeError("synthetic replay failure")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ["tiny_smoke", "--config", str(REPEAT_CONFIG), "--run-id", "lg_a",
                    "--cache-root", str(root / "cache"), "--data-dir", str(root / "data"),
                    "--manifest", str(root / "manifest.json"), "--output-dir", str(root / "out")]
            stdout = io.StringIO()
            with patch("sys.argv", argv), patch("ibkd_seg.cityscapes.official_api.bootstrap"), \
                 patch("ibkd_seg.cityscapes.official_assets.verify", return_value={"weights": {}}), \
                 patch("ibkd_seg.cityscapes.tiny_smoke.measure", side_effect=failing_measure), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 1)
            final = json.loads(stdout.getvalue().strip().splitlines()[-1].split("] ", 1)[1])
            self.assertEqual(final["status"], "failed")
            self.assertEqual(final["deterministic_warning_count"], 2)
            self.assertEqual(final["warning_summary"][0]["count"], 2)
            self.assertIn("no deterministic implementation", final["warning_summary"][0]["message"])
            self.assertTrue((root / "out/warnings.json").exists())


if __name__ == "__main__":
    unittest.main()
