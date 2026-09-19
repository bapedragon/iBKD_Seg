import unittest

from ibkd_seg.cityscapes.reproducibility_audit import (
    canonical_trajectory,
    compare_runs,
    compare_update_paths,
    diagnose_observational_probe,
)


class CityscapesReproducibilityTests(unittest.TestCase):
    @staticmethod
    def summary(final="student", guide="guide"):
        return {
            "completed_steps": 2,
            "input_stream_sha256": "inputs",
            "student_initial_state_sha256": "initial",
            "teacher_state_sha256": "teacher",
            "student_final_state_sha256": final,
            "guidance_final_state_sha256": guide,
            "optimizer_final_state_sha256": "optimizer",
            "diagnostic_validation": {"miou": 0.1},
        }

    @staticmethod
    def steps(
        second_loss=2.0,
        seconds=1.0,
        second_input="input-2",
        second_gradient="gradient-2",
    ):
        return [
            {
                "step": 1,
                "epoch": 1,
                "loss": 3.0,
                "ce": 2.5,
                "guidance": 10.0,
                "beta": 0.05,
                "lr": 0.01,
                "grad_norm_unclipped": 4.0,
                "input_sha256": "input-1",
                "gradient_sha256": "gradient-1",
                "seconds": seconds,
            },
            {
                "step": 2,
                "epoch": 1,
                "loss": second_loss,
                "ce": 1.8,
                "guidance": 4.0,
                "beta": 0.05,
                "lr": 0.009,
                "grad_norm_unclipped": 2.0,
                "input_sha256": second_input,
                "gradient_sha256": second_gradient,
                "seconds": seconds,
            },
        ]

    def test_wall_clock_is_excluded_from_trajectory(self):
        self.assertEqual(
            canonical_trajectory(self.steps(seconds=1.0)),
            canonical_trajectory(self.steps(seconds=9.0)),
        )

    def test_exact_runs_pass(self):
        result = compare_runs(
            self.summary(), self.summary(), self.steps(seconds=1), self.steps(seconds=9)
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["mismatched_steps"], [])

    def test_numerical_or_final_state_difference_fails(self):
        result = compare_runs(
            self.summary(),
            self.summary(final="different"),
            self.steps(),
            self.steps(second_loss=2.001),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["trajectory_exact"])
        self.assertFalse(result["checks"]["student_final_state_exact"])
        self.assertEqual(result["mismatched_steps"], [2])

    def test_empty_failed_runs_never_pass(self):
        left = self.summary()
        right = self.summary()
        left["completed_steps"] = right["completed_steps"] = 0
        result = compare_runs(left, right, [], [], expected_steps=25)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["nonempty_trajectories"])
        self.assertFalse(result["checks"]["expected_steps_completed"])

    def test_per_step_input_mismatch_is_reported(self):
        result = compare_runs(
            self.summary(),
            self.summary(),
            self.steps(),
            self.steps(second_input="different"),
            expected_steps=2,
            require_step_input_hashes=True,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_input_mismatch_step"], 2)
        self.assertEqual(result["input_mismatched_step_count"], 1)

    def test_observational_diagnosis_separates_repeat_from_method_path(self):
        exact = compare_runs(
            self.summary(), self.summary(), self.steps(), self.steps(),
            expected_steps=2, require_step_input_hashes=True,
        )
        changed = compare_runs(
            self.summary(), self.summary(final="different"),
            self.steps(), self.steps(second_loss=2.001),
            expected_steps=2, require_step_input_hashes=True,
        )
        diagnosis = diagnose_observational_probe({
            "lg_repeat": exact,
            "lg_vs_alg_before_controller_action": changed,
        })
        self.assertEqual(diagnosis["code"], "lg_alg_execution_path_difference")
        self.assertTrue(diagnosis["lg_repeat_exact"])
        self.assertFalse(diagnosis["lg_vs_alg_pre_action_exact"])

    def test_per_step_gradient_mismatch_is_reported(self):
        result = compare_runs(
            self.summary(),
            self.summary(),
            self.steps(),
            self.steps(second_gradient="different"),
            expected_steps=2,
            require_step_input_hashes=True,
            require_step_gradient_hashes=True,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_gradient_mismatch_step"], 2)
        self.assertEqual(result["gradient_mismatched_step_count"], 1)

    def test_update_path_allows_logged_scalar_drift_when_updates_are_exact(self):
        result = compare_update_paths(
            self.summary(),
            self.summary(),
            self.steps(),
            self.steps(second_loss=2.000000476837158),
            expected_steps=2,
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["scalar_values_exact"])
        self.assertFalse(result["exact_all_fields_passed"])
        self.assertEqual(result["scalar_fields"]["loss"]["first_mismatch_step"], 2)

    def test_update_path_requires_gradient_and_optimizer_state_hashes(self):
        right_summary = self.summary()
        right_summary["optimizer_final_state_sha256"] = "different"
        result = compare_update_paths(
            self.summary(),
            right_summary,
            self.steps(),
            self.steps(second_gradient="different"),
            expected_steps=2,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["per_step_gradient_hashes_exact"])
        self.assertFalse(result["checks"]["optimizer_final_state_exact"])


if __name__ == "__main__":
    unittest.main()
