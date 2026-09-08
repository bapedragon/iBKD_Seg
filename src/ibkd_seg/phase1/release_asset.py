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


CUB_R50_TEACHER_ASSET_TYPE = "phase1_cub_resnet50_224_scratch_teacher_v3"


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
        if not member.isdir() and not member.isfile():
            raise RuntimeError(
                f"release archive contains a special file: {member.name}"
            )
        target = (extraction_root / member.name).resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(
                f"release archive contains path traversal: {member.name}"
            ) from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_kind(manifest: dict[str, Any]) -> str:
    asset_type = manifest.get("asset_type")
    if asset_type == CUB_R50_TEACHER_ASSET_TYPE:
        return "cub_r50_teacher_v3"
    if asset_type is None and "classification_batch_size" in manifest["source"]:
        return "pet_classification"
    raise RuntimeError(f"unsupported checkpoint release asset type: {asset_type!r}")


def _validate_pet_classification(root: Path, manifest: dict[str, Any]) -> int:
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
    return expected_count


def _validate_cub_r50_teacher(root: Path, manifest: dict[str, Any]) -> int:
    import torch

    from .run_cub_r50_teacher_full import (
        EXPECTED_CONFIG_SHA256,
        load_scientific_teacher,
    )

    source = manifest["source"]
    checkpoint_contract = manifest.get("teacher_checkpoint", {})
    if (
        source.get("h200_job_id") != 722
        or source.get("protocol_id")
        != "cub200_phase1_resnet50_224_scratch_b128_full_v3"
        or source.get("protocol_config_sha256") != EXPECTED_CONFIG_SHA256
    ):
        raise RuntimeError("CUB v3 teacher release source contract mismatch")
    summary_path = root / "teacher_summary.json"
    checkpoint_path = root / "teacher_best_validation.pt"
    artifact_manifest_path = root / "artifact_manifest.json"
    if not all(
        path.is_file()
        for path in (summary_path, checkpoint_path, artifact_manifest_path)
    ):
        raise RuntimeError("CUB v3 teacher release is missing required artifacts")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    if (
        summary.get("status") != "complete"
        or summary.get("scientific_result") is not True
        or summary.get("official_test_evaluations") != 1
        or summary.get("official_test_used_for_training_or_selection") is not False
        or artifact_manifest.get("status") != "pass"
        or artifact_manifest.get("experiment_kind") != "resnet50-v3-teacher"
    ):
        raise RuntimeError("CUB v3 teacher release completion contract failed")
    checkpoint_paths = sorted(root.rglob("*.pt"))
    if checkpoint_paths != [checkpoint_path]:
        raise RuntimeError(
            "CUB v3 teacher release must contain exactly teacher_best_validation.pt"
        )
    checkpoint_hash = _file_sha256(checkpoint_path)
    if (
        checkpoint_path.stat().st_size != checkpoint_contract.get("size_bytes")
        or checkpoint_hash != checkpoint_contract.get("checkpoint_sha256")
        or checkpoint_hash != summary.get("checkpoint_sha256")
    ):
        raise RuntimeError("CUB v3 teacher checkpoint byte contract mismatch")
    model, _, loaded_hash, state_hash = load_scientific_teacher(
        checkpoint_path, device=torch.device("cpu")
    )
    del model
    if (
        loaded_hash != checkpoint_hash
        or state_hash != checkpoint_contract.get("model_state_sha256")
        or state_hash != summary.get("model_state_sha256")
    ):
        raise RuntimeError("CUB v3 teacher checkpoint model-state contract mismatch")
    return 1


def _validate_extracted(root: Path, manifest: dict[str, Any]) -> int:
    kind = _asset_kind(manifest)
    if kind == "cub_r50_teacher_v3":
        return _validate_cub_r50_teacher(root, manifest)
    return _validate_pet_classification(root, manifest)


def _existing_marker(destination: Path, kind: str) -> Path:
    if kind == "cub_r50_teacher_v3":
        return destination / "teacher_best_validation.pt"
    return destination / "classification_summary.json"


def download_and_extract(
    manifest_path: Path,
    destination: Path,
    download_dir: Path,
) -> dict[str, Any]:
    """Fetch a release asset, verify it, and atomically install its contents."""

    manifest = _load_manifest(manifest_path)
    kind = _asset_kind(manifest)
    batch_size = manifest["source"].get("classification_batch_size")
    if _existing_marker(destination, kind).is_file():
        checkpoint_count = _validate_extracted(destination, manifest)
        log(f"[CHECKPOINT_RELEASE] existing audited input: {destination}")
        return manifest
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty incomplete destination: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        "phase1_cub_r50_teacher_v3_"
        if kind == "cub_r50_teacher_v3"
        else f"phase1_pet_b{int(batch_size)}_"
    )
    with tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=".tar.gz",
        dir=download_dir,
        delete=False,
    ) as handle:
        archive_path = Path(handle.name)
    extraction_root = Path(
        tempfile.mkdtemp(
            prefix=f".{prefix}extract_",
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
        checkpoint_count = _validate_extracted(extraction_root, manifest)
        if destination.exists():
            destination.rmdir()
        extraction_root.replace(destination)
        log(
            "[CHECKPOINT_RELEASE_DONE] "
            f"destination={destination} checkpoints={checkpoint_count} status=pass"
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
