#!/usr/bin/env python3
"""Download and safely extract an audited Phase 1 checkpoint release asset."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(message, flush=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "download_url",
        "size_bytes",
        "sha256",
        "asset_name",
        "source",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"invalid checkpoint release manifest: {path}")
    if len(str(payload["sha256"])) != 64:
        raise RuntimeError("release asset SHA-256 is malformed")
    return payload


def _download(
    url: str,
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    retries: int = 3,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        digest = hashlib.sha256()
        downloaded = 0
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "iBKD-Seg-Phase1-checkpoint-fetcher/1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, path.open(
                "wb"
            ) as output:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if downloaded % (64 * 1024 * 1024) < len(chunk):
                        log(
                            "[CHECKPOINT_DOWNLOAD] "
                            f"bytes={downloaded}/{expected_bytes} attempt={attempt}"
                        )
            if downloaded != expected_bytes:
                raise RuntimeError(
                    f"release asset byte mismatch: {downloaded} != {expected_bytes}"
                )
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "release asset SHA-256 mismatch: "
                    f"{actual_sha256} != {expected_sha256}"
                )
            return
        except Exception as error:  # retry network and integrity failures alike
            last_error = error
            path.unlink(missing_ok=True)
            if attempt < retries:
                log(
                    f"[CHECKPOINT_DOWNLOAD_RETRY] attempt={attempt} "
                    f"error={type(error).__name__}: {error}"
                )
                time.sleep(2 * attempt)
    assert last_error is not None
    raise RuntimeError(f"checkpoint asset download failed after {retries} attempts") from last_error


def _validate_members(archive: tarfile.TarFile, extraction_root: Path) -> None:
    resolved_root = extraction_root.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise RuntimeError(f"release archive contains a link: {member.name}")
        target = (extraction_root / member.name).resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(
                f"release archive contains path traversal: {member.name}"
            ) from error


def _validate_extracted(root: Path, manifest: dict[str, Any]) -> None:
    source = manifest["source"]
    batch_size = int(source["classification_batch_size"])
    if batch_size not in (64, 128):
        raise RuntimeError(
            f"unsupported classification batch size in release manifest: {batch_size}"
        )
    summary_path = root / "classification_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("release asset does not contain classification_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("batch_size") != batch_size
        or summary.get("completed_tasks") != 19
        or summary.get("failed_tasks") != 0
    ):
        raise RuntimeError("extracted classification suite summary is not complete")
    checkpoint_paths = sorted(root.rglob("*_best_validation.pt"))
    expected_count = int(source["total_checkpoints"])
    if len(checkpoint_paths) != expected_count:
        raise RuntimeError(
            f"release contains {len(checkpoint_paths)} checkpoints; "
            f"expected {expected_count}"
        )
    student_count = sum(path.name == "student_best_validation.pt" for path in checkpoint_paths)
    teacher_count = sum(path.name == "teacher_best_validation.pt" for path in checkpoint_paths)
    if student_count != 18 or teacher_count != 1:
        raise RuntimeError(
            "release checkpoint roles are incomplete: "
            f"student={student_count} teacher={teacher_count}"
        )


def download_and_extract(
    manifest_path: Path,
    destination: Path,
    download_dir: Path,
) -> dict[str, Any]:
    """Fetch a release asset, verify it, and atomically install its contents."""

    manifest = _load_manifest(manifest_path)
    batch_size = int(manifest["source"]["classification_batch_size"])
    if (destination / "classification_summary.json").is_file():
        _validate_extracted(destination, manifest)
        log(f"[CHECKPOINT_RELEASE] existing audited input: {destination}")
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty incomplete destination: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"phase1_pet_b{batch_size}_",
        suffix=".tar.gz",
        dir=download_dir,
        delete=False,
    ) as handle:
        archive_path = Path(handle.name)
    extraction_root = Path(
        tempfile.mkdtemp(
            prefix=f".phase1_pet_b{batch_size}_extract_",
            dir=destination.parent,
        )
    )
    try:
        log(f"[CHECKPOINT_DOWNLOAD] url={manifest['download_url']}")
        _download(
            str(manifest["download_url"]),
            archive_path,
            expected_bytes=int(manifest["size_bytes"]),
            expected_sha256=str(manifest["sha256"]),
        )
        log("[CHECKPOINT_DOWNLOAD] byte_size_and_sha256=pass")
        with tarfile.open(archive_path, mode="r:gz") as archive:
            _validate_members(archive, extraction_root)
            if "filter" in inspect.signature(archive.extractall).parameters:
                archive.extractall(extraction_root, filter="data")
            else:  # Python 3.10 reference environment; members were checked above.
                archive.extractall(extraction_root)
        _validate_extracted(extraction_root, manifest)
        if destination.exists():
            destination.rmdir()
        extraction_root.replace(destination)
        log(
            "[CHECKPOINT_RELEASE_DONE] "
            f"destination={destination} checkpoints=19 status=pass"
        )
        return manifest
    finally:
        archive_path.unlink(missing_ok=True)
        if extraction_root.exists():
            shutil.rmtree(extraction_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_extract(args.manifest, args.destination, args.download_dir)


if __name__ == "__main__":
    main()
