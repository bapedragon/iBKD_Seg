import unittest

from ibkd_seg.cityscapes.reproducibility_audit import (
    canonical_trajectory,
    compare_runs,
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
            "diagnostic_validation": {"miou": 0.1},
        }

    @staticmethod
    def steps(second_loss=2.0, seconds=1.0):
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


if __name__ == "__main__":
    unittest.main()
