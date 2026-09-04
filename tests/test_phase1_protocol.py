from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ibkd_seg.phase1.config import (
    ProtocolError,
    effective_protocol,
    load_protocol,
    protocol_digest,
    validate_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "phase1/configs/flowers102_phase1a_v1.json"


class Phase1ProtocolTest(unittest.TestCase):
    def test_locked_protocol_is_valid_and_hashable(self) -> None:
        config = load_protocol(CONFIG_PATH)
        self.assertEqual(config["probe"]["parameter_count"], 386)
        self.assertEqual(len(protocol_digest(config)), 64)

    def test_smoke_override_does_not_mutate_locked_config(self) -> None:
        config = load_protocol(CONFIG_PATH)
        original = copy.deepcopy(config)
        smoke = effective_protocol(config, smoke=True)
        self.assertEqual(config, original)
        self.assertEqual(smoke["execution_mode"], "smoke")
        self.assertEqual(smoke["probe"]["epochs"], config["smoke"]["epochs"])
        self.assertTrue(smoke["non_comparable_smoke_override"])

    def test_posthoc_exclusion_is_rejected(self) -> None:
        config = load_protocol(CONFIG_PATH)
        config["dataset"]["mask"]["posthoc_exclusion"] = "remove_empty_masks"
        with self.assertRaises(ProtocolError):
            validate_protocol(config)


if __name__ == "__main__":
    unittest.main()
