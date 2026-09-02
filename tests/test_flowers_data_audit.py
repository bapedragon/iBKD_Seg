from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from ibkd_seg.phase0.flowers_data import _asset_records, audit_dataset


class FlowersDataAuditTest(unittest.TestCase):
    def test_asset_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "asset.bin").write_bytes(b"official bytes")

            records = _asset_records(
                root,
                [
                    {
                        "name": "asset.bin",
                        "size_bytes": len(b"official bytes"),
                        "sha256": "0" * 64,
                    }
                ],
            )

            self.assertEqual(records[0]["status"], "fail")
            self.assertEqual(records[0]["expected_sha256"], "0" * 64)
            self.assertNotEqual(records[0]["sha256"], records[0]["expected_sha256"])

    def test_synthetic_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "jpg"
            mask_root = root / "segmim"
            image_root.mkdir()
            mask_root.mkdir()
            for image_id in range(1, 5):
                source = np.full((8, 10, 3), [200, 50 + image_id, 20], dtype=np.uint8)
                composite = source.copy()
                composite[:, :4] = [0, 0, 255]
                Image.fromarray(source).save(image_root / f"image_{image_id:05d}.jpg")
                Image.fromarray(composite).save(mask_root / f"segmim_{image_id:05d}.jpg")

            savemat(root / "imagelabels.mat", {"labels": np.asarray([[1, 1, 2, 2]])})
            savemat(
                root / "setid.mat",
                {
                    "trnid": np.asarray([[1, 2]]),
                    "valid": np.asarray([[3]]),
                    "tstid": np.asarray([[4]]),
                },
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "assets": [],
                        "expected": {
                            "images": 4,
                            "masks": 4,
                            "classes": 2,
                            "train_ids": 2,
                            "val_ids": 1,
                            "test_ids": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = audit_dataset(root, manifest_path, mask_samples=2)

            self.assertEqual(report["status"], "pass")
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["counts"]["images"], 4)
            self.assertEqual(report["mask_profile"]["sample_count"], 2)
            self.assertFalse(report["mask_quality"]["empty_ids"])


if __name__ == "__main__":
    unittest.main()
