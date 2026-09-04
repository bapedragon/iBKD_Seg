from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ibkd_seg.phase05.config import (
    ProtocolError,
    effective_protocol,
    load_protocol,
    protocol_digest,
    validate_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "phase0.5/configs/flowers102_phase05_v1.json"
EXPECTED_DIGEST = "2cf353e0e9e6dcc0b0b01b75eb17039dde2a20ac5af07f1c1683ec672c3afae2"


class Phase05ProtocolTest(unittest.TestCase):
    def test_locked_protocol_is_valid_and_hashable(self) -> None:
        config = load_protocol(CONFIG_PATH)
        self.assertEqual(config["probe"]["parameter_count"], 386)
        self.assertEqual(protocol_digest(config), EXPECTED_DIGEST)

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
