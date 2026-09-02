from __future__ import annotations

import unittest

from ibkd_seg.phase0.download import plan_ranges


class DownloadRangesTest(unittest.TestCase):
    def test_ranges_cover_remaining_bytes_once(self) -> None:
        ranges = plan_ranges(start=3, total_size=14, connections=4)
        self.assertEqual(ranges, [(3, 5), (6, 8), (9, 11), (12, 13)])
        covered = [value for start, end in ranges for value in range(start, end + 1)]
        self.assertEqual(covered, list(range(3, 14)))

    def test_no_ranges_when_complete(self) -> None:
        self.assertEqual(plan_ranges(start=10, total_size=10, connections=8), [])

    def test_rejects_invalid_connections(self) -> None:
        with self.assertRaises(ValueError):
            plan_ranges(start=0, total_size=10, connections=0)


if __name__ == "__main__":
    unittest.main()
