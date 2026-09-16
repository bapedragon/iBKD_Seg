from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from ibkd_seg.cityscapes.data import Cityscapes, LABEL_IDS, audit
from ibkd_seg.cityscapes.public_smoke import teacher_input
from ibkd_seg.cityscapes.real_smoke import validate_manifest
from ibkd_seg.cityscapes.smoke import make_fixture


def manifest_fixture():
    splits = {}
    for split, count in (("train", 2975), ("val", 500)):
        splits[split] = [{"id": f"{split}_{i:06d}", "size_wh": [2048, 1024],
                          "image": f"leftImg8bit/{split}/city/{split}_{i:06d}_leftImg8bit.png",
                          "mask": f"gtFine/{split}/city/{split}_{i:06d}_gtFine_labelIds.png"}
                         for i in range(count)]
    return {"dataset": "Cityscapes", "synthetic": False, "label_ids": list(LABEL_IDS), "splits": splits}


class RealCityscapesSmokeContract(unittest.TestCase):
    def test_shared_raw_geometry_preserves_teacher_normalization(self):
        config = {"scale_range": [1.0, 1.0], "crop_size": [32, 64],
                  "crop_attempts": 3, "max_category_ratio": 0.75}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            manifest = audit(root, synthetic=True)
            for training, split in ((True, "train"), (False, "val")):
                rows = manifest["splits"][split]
                raw = Cityscapes(root, rows, config, training=training, seed=1, normalize=False)
                normalized = Cityscapes(root, rows, config, training=training, seed=1)
                rgb, target, sample_id = raw[0]
                imagenet, normalized_target, normalized_id = normalized[0]
                torch.testing.assert_close(teacher_input(rgb[None])[0], imagenet, rtol=0, atol=0)
                self.assertTrue(torch.equal(target, normalized_target))
                self.assertEqual(sample_id, normalized_id)
                self.assertTrue(bool((rgb >= 0).all() & (rgb <= 1).all()))

    def test_manifest_rejects_synthetic_subset_test_overlap_and_unsafe_paths(self):
        original = manifest_fixture()
        validate_manifest(original)
        invalid = []
        value = copy.deepcopy(original)
        value["synthetic"] = True
        invalid.append(value)
        value = copy.deepcopy(original)
        value["splits"]["train"].pop()
        invalid.append(value)
        value = copy.deepcopy(original)
        value["splits"]["test"] = []
        invalid.append(value)
        value = copy.deepcopy(original)
        value["splits"]["val"][0]["id"] = value["splits"]["train"][0]["id"]
        invalid.append(value)
        value = copy.deepcopy(original)
        value["splits"]["val"][0]["image"] = "../outside.png"
        invalid.append(value)
        value = copy.deepcopy(original)
        value["splits"]["val"][0]["size_wh"] = [64, 32]
        invalid.append(value)
        for index, manifest in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
