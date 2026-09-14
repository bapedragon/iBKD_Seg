from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ibkd_seg.cityscapes.prepare import archive_plan, extract_checked


class CityscapesZipPreparation(unittest.TestCase):
    def make_zip(self, path: Path, *, wrapped: bool = True, extra: str | None = None) -> str:
        prefix = "leftImg8bit_trainvaltest/" if wrapped else ""
        image = "leftImg8bit/train/city/city_000000_000001_leftImg8bit.png"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(prefix + image, b"training-image-bytes")
            archive.writestr(prefix + image.replace("/train/", "/val/"), b"validation-image-bytes")
            archive.writestr(prefix + image.replace("/train/", "/test/"), b"test-image-bytes")
            archive.writestr("__MACOSX/._metadata", b"apple metadata")
            archive.writestr(prefix + "license.txt", b"source license")
            if extra:
                archive.writestr(extra, b"unexpected")
        return image

    @patch("ibkd_seg.cityscapes.prepare.EXPECTED", {"train": 1, "val": 1})
    def test_normalizes_both_layouts_and_retains_license_without_test_or_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for wrapped in (False, True):
                path = root / f"{wrapped}.zip"
                image = self.make_zip(path, wrapped=wrapped)
                target = root / f"prepared-{wrapped}"
                with zipfile.ZipFile(path) as archive:
                    plan = archive_plan(archive, "leftImg8bit_trainvaltest.zip")
                    first = extract_checked(archive, plan, target)
                    second = extract_checked(archive, plan, target)
                self.assertEqual(first, second)
                self.assertEqual(first["crc_checked_members"], 5)
                self.assertEqual(len(first["extracted_files"]), 3)
                self.assertEqual((target / image).read_bytes(), b"training-image-bytes")
                self.assertFalse((target / "leftImg8bit/test").exists())
                self.assertFalse((target / "__MACOSX").exists())

    @patch("ibkd_seg.cityscapes.prepare.EXPECTED", {"train": 1, "val": 1})
    def test_rejects_path_traversal_and_duplicate_normalized_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.zip"
            for extra in ("../escape", "leftImg8bit/train/city/city_000000_000001_leftImg8bit.png"):
                self.make_zip(path, extra=extra)
                with zipfile.ZipFile(path) as archive, self.assertRaises(ValueError):
                    archive_plan(archive, "leftImg8bit_trainvaltest.zip")

    @patch("ibkd_seg.cityscapes.prepare.EXPECTED", {"train": 1, "val": 1})
    def test_detects_existing_data_conflict_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.zip"
            image = self.make_zip(path)
            target = root / "prepared" / image
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different-existing-data")
            with zipfile.ZipFile(path) as archive:
                plan = archive_plan(archive, "leftImg8bit_trainvaltest.zip")
                with self.assertRaises(ValueError):
                    extract_checked(archive, plan, root / "prepared")
            self.assertEqual(target.read_bytes(), b"different-existing-data")
            self.assertFalse(list(root.rglob(".prepare-*")))

    @patch("ibkd_seg.cityscapes.prepare.EXPECTED", {"train": 1, "val": 1})
    def test_checks_crc_even_for_unextracted_test_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.zip"
            self.make_zip(path)
            contents = bytearray(path.read_bytes())
            contents[contents.index(b"test-image-bytes")] ^= 1
            path.write_bytes(contents)
            with zipfile.ZipFile(path) as archive:
                plan = archive_plan(archive, "leftImg8bit_trainvaltest.zip")
                with self.assertRaises(zipfile.BadZipFile):
                    extract_checked(archive, plan, root / "prepared")


if __name__ == "__main__":
    unittest.main()
