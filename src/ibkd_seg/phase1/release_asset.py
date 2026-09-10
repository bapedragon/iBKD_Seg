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
CUB_R50_GUIDED_SEED1_ASSET_TYPE = (
    "phase1_cub_resnet50_224_guided_seed1_batch_profiles_v4"
)
CUB_R50_GUIDED_SEED23_ASSET_TYPE = (
    "phase1_cub_resnet50_224_guided_seeds2_3_v5"
)
CUB_R50_GUIDED_SEED1_AUDIT_SHA256 = (
    "82264ed949643c55124981fc8008f8dc368ad257673781d2510d01ccf1cc0516"
)
CUB_R50_GUIDED_SEED23_AUDIT_SHA256 = (
    "ef8179527840a67a007936f1ba7d2cef0adff5fef405c6c15b87d6dd1b60c716"
)


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


def _is_macos_metadata(member_name: str) -> bool:
    components = [
        component
        for component in member_name.split("/")
        if component not in {"", "."}
    ]
    return any(
        component == "__MACOSX" or component.startswith("._")
        for component in components
    )


def _validate_members(
    archive: tarfile.TarFile, extraction_root: Path
) -> list[tarfile.TarInfo]:
    resolved_root = extraction_root.resolve()
    payload_members: list[tarfile.TarInfo] = []
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
        if not _is_macos_metadata(member.name):
            payload_members.append(member)
    return payload_members


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
    if asset_type == CUB_R50_GUIDED_SEED1_ASSET_TYPE:
        return "cub_r50_guided_seed1_v4"
    if asset_type == CUB_R50_GUIDED_SEED23_ASSET_TYPE:
        return "cub_r50_guided_seed23_v5"
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
    checkpoint_path_resolved = checkpoint_path.resolve(strict=True)
    unexpected_checkpoints = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.pt")
        if path.is_file() and path.resolve(strict=True) != checkpoint_path_resolved
    )
    if unexpected_checkpoints:
        raise RuntimeError(
            "CUB v3 teacher release contains unexpected checkpoints: "
            + ", ".join(unexpected_checkpoints)
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


def _validate_cub_r50_guided_seed1(root: Path, manifest: dict[str, Any]) -> int:
    """Audit all issue-727 checkpoint bytes and strict-load its eight encoders."""

    import torch

    from .models import create_student
    from .train_timing import state_dict_sha256

    repository_root = Path(__file__).resolve().parents[3]
    audit_path = (
        repository_root
        / "phase1/phase1_cub/reports/frozen_probe/"
        "resnet50_224_b128_b64_guided_seed1_v4/checkpoint_audit.json"
    )
    if (
        not audit_path.is_file()
        or _file_sha256(audit_path) != CUB_R50_GUIDED_SEED1_AUDIT_SHA256
    ):
        raise RuntimeError("CUB v4 committed checkpoint audit is missing or changed")

    source = manifest.get("source", {})
    contents = manifest.get("contents", {})
    if (
        source.get("h200_job_id") != 727
        or source.get("protocol_id")
        != "cub200_phase1_r50_224_guided_b128_b64_seed1_full_v4"
        or source.get("protocol_config_sha256")
        != "bbecaa8b48e43325e8b4eb342e6dfbfa146ffee0e7b8b31d641e654a90925633"
        or contents.get("classification_student_checkpoints") != 8
        or contents.get("selected_probe_checkpoints") != 40
        or contents.get("total_checkpoints") != 48
        or manifest.get("remote_asset_digest_verified") is not True
    ):
        raise RuntimeError("CUB v4 guided seed-1 release source contract mismatch")

    combined_path = root / "combined_full_summary.json"
    if not combined_path.is_file():
        raise RuntimeError("CUB v4 guided release is missing combined_full_summary.json")
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    if (
        combined.get("status") != "complete"
        or combined.get("scientific_result") is not True
        or combined.get("counts", {}).get("new_checkpoints") != 48
        or combined.get("counts", {}).get("classification_students") != 8
        or combined.get("counts", {}).get("selected_probes") != 40
    ):
        raise RuntimeError("CUB v4 guided release completion contract failed")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    entries = audit.get("entries", [])
    if (
        audit.get("status") != "pass"
        or audit.get("new_checkpoint_count") != 48
        or len(entries) != 48
    ):
        raise RuntimeError("CUB v4 committed checkpoint audit is incomplete")

    expected_paths: set[Path] = set()
    encoder_count = 0
    for entry in entries:
        relative = Path(str(entry.get("path_under_ignored_raw_root", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe CUB v4 checkpoint path: {relative}")
        checkpoint_path = root / relative
        expected_paths.add(checkpoint_path.resolve())
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != entry.get("bytes")
            or _file_sha256(checkpoint_path) != entry.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"CUB v4 checkpoint byte contract mismatch: {relative}")
        if entry.get("kind") != "classification_encoder":
            continue

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = create_student(num_classes=200, drop_path_rate=0.1)
        incompatible = model.load_state_dict(payload["student"], strict=True)
        state_hash = state_dict_sha256(model)
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
            or state_hash != entry.get("model_state_sha256")
        ):
            raise RuntimeError(f"CUB v4 encoder strict-load contract failed: {relative}")
        encoder_count += 1
        del model, payload

    actual_paths = {
        path.resolve() for path in root.rglob("*.pt") if path.is_file()
    }
    if actual_paths != expected_paths or encoder_count != 8:
        raise RuntimeError(
            "CUB v4 release checkpoint inventory mismatch: "
            f"expected={len(expected_paths)} actual={len(actual_paths)} "
            f"encoders={encoder_count}"
        )
    return 48


def _validate_cub_r50_guided_seed23(root: Path, manifest: dict[str, Any]) -> int:
    """Audit issue-730 bytes and strict-load its eight seed-2/3 encoders."""

    import torch

    from .models import create_student
    from .train_timing import state_dict_sha256

    repository_root = Path(__file__).resolve().parents[3]
    audit_path = (
        repository_root
        / "phase1/phase1_cub/reports/frozen_probe/"
        "resnet50_224_b128_guided_3seed_v5/checkpoint_audit.json"
    )
    if (
        not audit_path.is_file()
        or _file_sha256(audit_path) != CUB_R50_GUIDED_SEED23_AUDIT_SHA256
    ):
        raise RuntimeError("CUB v5 committed checkpoint audit is missing or changed")

    source = manifest.get("source", {})
    contents = manifest.get("contents", {})
    if (
        source.get("h200_job_id") != 730
        or source.get("protocol_id")
        != "cub200_phase1_r50_224_guided_b128_s23_b64_s2_full_v5"
        or source.get("protocol_config_sha256")
        != "f3531c648f65e6f51e48bbeda7ad38b1fc5931d88e04b01c97c6ff71aad437b9"
        or contents.get("classification_student_checkpoints") != 8
        or contents.get("selected_probe_checkpoints") != 40
        or contents.get("total_checkpoints") != 48
        or manifest.get("remote_asset_digest_verified") is not True
    ):
        raise RuntimeError("CUB v5 guided seed-2/3 release source contract mismatch")

    combined_path = root / "combined_full_summary.json"
    artifact_manifest_path = root / "artifact_manifest.json"
    if not combined_path.is_file() or not artifact_manifest_path.is_file():
        raise RuntimeError("CUB v5 guided release is missing completion evidence")
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    if (
        combined.get("status") != "complete"
        or combined.get("scientific_result") is not True
        or combined.get("encoder_seeds") != [2, 3]
        or combined.get("counts", {}).get("new_checkpoints") != 48
        or combined.get("counts", {}).get("classification_students") != 8
        or combined.get("counts", {}).get("selected_probes") != 40
        or artifact_manifest.get("status") != "pass"
        or artifact_manifest.get("experiment_kind")
        != "resnet50-v5-guided-seeds2-3"
    ):
        raise RuntimeError("CUB v5 guided release completion contract failed")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    entries = audit.get("entries", [])
    if (
        audit.get("status") != "pass"
        or audit.get("issue730_new_checkpoint_count") != 48
        or len(entries) != 48
    ):
        raise RuntimeError("CUB v5 committed checkpoint audit is incomplete")

    expected_paths: set[Path] = set()
    encoder_count = 0
    encoder_seeds: set[int] = set()
    for entry in entries:
        relative = Path(str(entry.get("path_under_ignored_raw_root", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe CUB v5 checkpoint path: {relative}")
        checkpoint_path = root / relative
        expected_paths.add(checkpoint_path.resolve())
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != entry.get("bytes")
            or _file_sha256(checkpoint_path) != entry.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"CUB v5 checkpoint byte contract mismatch: {relative}")
        if entry.get("kind") != "classification_encoder":
            continue

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = create_student(num_classes=200, drop_path_rate=0.1)
        incompatible = model.load_state_dict(payload["student"], strict=True)
        state_hash = state_dict_sha256(model)
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
            or state_hash != entry.get("model_state_sha256")
        ):
            raise RuntimeError(f"CUB v5 encoder strict-load contract failed: {relative}")
        encoder_count += 1
        encoder_seeds.add(int(entry["encoder_seed"]))
        del model, payload

    actual_paths = {path.resolve() for path in root.rglob("*.pt") if path.is_file()}
    if actual_paths != expected_paths or encoder_count != 8 or encoder_seeds != {2, 3}:
        raise RuntimeError(
            "CUB v5 release checkpoint inventory mismatch: "
            f"expected={len(expected_paths)} actual={len(actual_paths)} "
            f"encoders={encoder_count} seeds={sorted(encoder_seeds)}"
        )
    return 48


def _validate_extracted(root: Path, manifest: dict[str, Any]) -> int:
    kind = _asset_kind(manifest)
    if kind == "cub_r50_teacher_v3":
        return _validate_cub_r50_teacher(root, manifest)
    if kind == "cub_r50_guided_seed1_v4":
        return _validate_cub_r50_guided_seed1(root, manifest)
    if kind == "cub_r50_guided_seed23_v5":
        return _validate_cub_r50_guided_seed23(root, manifest)
    return _validate_pet_classification(root, manifest)


def _existing_marker(destination: Path, kind: str) -> Path:
    if kind == "cub_r50_teacher_v3":
        return destination / "teacher_best_validation.pt"
    if kind == "cub_r50_guided_seed1_v4":
        return destination / "combined_full_summary.json"
    if kind == "cub_r50_guided_seed23_v5":
        return destination / "combined_full_summary.json"
    return destination / "classification_summary.json"


def _resolve_payload_root(extraction_root: Path, kind: str) -> Path:
    """Locate one release payload and normalize an optional wrapper directory."""

    marker_name = _existing_marker(Path("."), kind).name
    marker_candidates = sorted(
        path
        for path in extraction_root.rglob(marker_name)
        if path.is_file()
    )
    if len(marker_candidates) != 1:
        raise RuntimeError(
            "release archive must contain exactly one payload marker "
            f"{marker_name}; found {len(marker_candidates)}"
        )

    payload_root = marker_candidates[0].parent
    files_outside_payload = sorted(
        path.relative_to(extraction_root).as_posix()
        for path in extraction_root.rglob("*")
        if path.is_file() and not path.is_relative_to(payload_root)
    )
    if files_outside_payload:
        raise RuntimeError(
            "release archive contains files outside its detected payload root: "
            + ", ".join(files_outside_payload)
        )
    return payload_root


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
    if kind == "cub_r50_teacher_v3":
        prefix = "phase1_cub_r50_teacher_v3_"
    elif kind == "cub_r50_guided_seed1_v4":
        prefix = "phase1_cub_r50_guided_seed1_v4_"
    elif kind == "cub_r50_guided_seed23_v5":
        prefix = "phase1_cub_r50_guided_seed23_v5_"
    else:
        prefix = f"phase1_pet_b{int(batch_size)}_"
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
            payload_members = _validate_members(archive, extraction_root)
            ignored_members = len(archive.getmembers()) - len(payload_members)
            if ignored_members:
                log(
                    "[CHECKPOINT_RELEASE] "
                    f"ignored_macos_metadata_members={ignored_members}"
                )
            if "filter" in inspect.signature(archive.extractall).parameters:
                archive.extractall(
                    extraction_root,
                    members=payload_members,
                    filter="data",
                )
            else:  # Python 3.10 reference environment; members were checked above.
                archive.extractall(extraction_root, members=payload_members)
        payload_root = _resolve_payload_root(extraction_root, kind)
        if payload_root != extraction_root:
            relative_payload_root = payload_root.relative_to(extraction_root)
            log(
                "[CHECKPOINT_RELEASE] "
                f"normalized_payload_root={relative_payload_root.as_posix()}"
            )
        checkpoint_count = _validate_extracted(payload_root, manifest)
        if destination.exists():
            destination.rmdir()
        payload_root.replace(destination)
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
