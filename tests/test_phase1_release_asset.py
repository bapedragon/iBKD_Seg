from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from ibkd_seg.phase1.release_asset import download_and_extract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase1ReleaseAssetTest(unittest.TestCase):
    def test_download_verifies_and_installs_all_nineteen_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "classification_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "batch_size": 64,
                        "completed_tasks": 19,
                        "failed_tasks": 0,
                    }
                ),
                encoding="utf-8",
            )
            for seed in range(18):
                path = source / "students" / f"student_{seed}"
                path.mkdir(parents=True)
                (path / "student_best_validation.pt").write_bytes(b"student")
            teacher = source / "teacher" / "teacher_1"
            teacher.mkdir(parents=True)
            (teacher / "teacher_best_validation.pt").write_bytes(b"teacher")

            archive_path = root / "asset.tar.gz"
            with tarfile.open(archive_path, mode="w:gz") as archive:
                for path in sorted(source.rglob("*")):
                    archive.add(path, arcname=path.relative_to(source))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "download_url": archive_path.as_uri(),
                        "asset_name": archive_path.name,
                        "size_bytes": archive_path.stat().st_size,
                        "sha256": _sha256(archive_path),
                        "source": {"total_checkpoints": 19},
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "installed"
            download_and_extract(manifest_path, destination, root / "downloads")
            self.assertTrue((destination / "classification_summary.json").is_file())
            self.assertEqual(len(list(destination.rglob("*_best_validation.pt"))), 19)
            # A second call validates and reuses the installed result.
            download_and_extract(manifest_path, destination, root / "downloads")

    def test_path_traversal_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "malicious.tar.gz"
            with tarfile.open(archive_path, mode="w:gz") as archive:
                member = tarfile.TarInfo("../outside.txt")
                payload = b"blocked"
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "download_url": archive_path.as_uri(),
                        "asset_name": archive_path.name,
                        "size_bytes": archive_path.stat().st_size,
                        "sha256": _sha256(archive_path),
                        "source": {"total_checkpoints": 19},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "path traversal"):
                download_and_extract(
                    manifest_path,
                    root / "installed",
                    root / "downloads",
                )
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
