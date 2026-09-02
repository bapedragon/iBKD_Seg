from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ibkd_seg.phase0.checkpoints import load_manifest, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


class CheckpointManifestTest(unittest.TestCase):
    def test_manifest_ids_and_hashes_are_well_formed(self) -> None:
        manifest = load_manifest(REPO_ROOT / "manifests/checkpoints.json")
        ids = [entry["id"] for entry in manifest["checkpoints"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 3)
        for entry in manifest["checkpoints"]:
            self.assertEqual(len(entry["sha256"]), 64)
            int(entry["sha256"], 16)
            self.assertGreater(entry["size_bytes"], 0)

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.bin"
            path.write_bytes(b"ibkd-seg")
            self.assertEqual(
                sha256_file(path),
                "b90a7e2b0e940361aa59df26451de0a6801247323196fffd77d1f150d853477c",
            )


if __name__ == "__main__":
    unittest.main()
