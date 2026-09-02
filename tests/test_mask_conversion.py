from __future__ import annotations

import unittest

import numpy as np

from ibkd_seg.phase0.masks import binary_foreground_mask, recover_foreground_alpha


class MaskConversionTest(unittest.TestCase):
    def test_recovers_exact_binary_composite(self) -> None:
        original = np.asarray(
            [[[255, 0, 0], [0, 255, 0]], [[255, 255, 0], [255, 0, 255]]],
            dtype=np.uint8,
        )
        foreground = np.asarray([[True, False], [False, True]])
        composite = np.where(
            foreground[..., None],
            original,
            np.asarray([0, 0, 255], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            binary_foreground_mask(original, composite),
            foreground,
        )

    def test_recovers_soft_alpha(self) -> None:
        original = np.full((1, 2, 3), [240, 60, 30], dtype=np.float32)
        background = np.asarray([0, 0, 255], dtype=np.float32)
        expected = np.asarray([[0.25, 0.75]], dtype=np.float32)
        composite = expected[..., None] * original + (1.0 - expected[..., None]) * background
        actual = recover_foreground_alpha(original, composite)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_array_equal(
            binary_foreground_mask(original, composite),
            np.asarray([[False, True]]),
        )

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            recover_foreground_alpha(
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((2, 3, 3), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
