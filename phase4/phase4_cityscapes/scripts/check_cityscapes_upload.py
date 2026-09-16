#!/usr/bin/env python3
"""Read-only Cityscapes upload check, using Python's standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED = {
    "leftImg8bit_trainvaltest.zip": {
        "bytes": 11598567370,
        "sha256": "1927eb6450e29ebde0d611d30b874abe96747f5d092f6ec0cbbb6a3e1f6ce09d",
        "component": "leftImg8bit", "suffix": "_leftImg8bit.png",
    },
    "gtFine_trainvaltest.zip": {
        "bytes": 263041307,
        "sha256": "dbacde05ea136f3036aa24bfc87b5272f0d999e34380e10cbb919f7368448332",
        "component": "gtFine", "suffix": "_gtFine_labelIds.png",
    },
}
SPLITS = {"train": 2975, "val": 500, "test": 1525}


def locate(root: Path, max_depth: int = 4) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {name: [] for name in EXPECTED}
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset mount is missing or not a directory: {root}")

    def onerror(error: OSError) -> None:
        raise error

    visited = 0
    for parent, directories, files in os.walk(root, followlinks=False, onerror=onerror):
        visited += len(files) + len(directories)
        if visited > 100000:
            raise ValueError("Search exceeded 100000 entries; specify the exact ZIP directory")
        depth = len(Path(parent).relative_to(root).parts)
        if depth >= max_depth:
            directories[:] = []
        else:
            directories[:] = sorted(d for d in directories if not d.startswith("."))
        for name in EXPECTED:
            if name in files:
                found[name].append(Path(parent) / name)
    return found


def check_archive(path: Path, expected: dict) -> tuple[dict, dict[str, set]]:
    before = path.stat()
    result = {"path": str(path.resolve()), "bytes": before.st_size,
              "expected_bytes": expected["bytes"], "expected_sha256": expected["sha256"]}
    digest = hashlib.sha256()
    last = time.monotonic()
    done = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
            done += len(block)
            if time.monotonic() - last >= 10:
                print(f"[SHA256_PROGRESS] file={path.name} bytes={done}/{before.st_size}", flush=True)
                last = time.monotonic()
    result["sha256"] = digest.hexdigest()
    result["bytes_match"] = before.st_size == expected["bytes"]
    result["sha256_match"] = result["sha256"] == expected["sha256"]
    print("[FILE_HASH] " + json.dumps(result, ensure_ascii=False), flush=True)
    ids: dict[str, set] = {split: set() for split in SPLITS}
    duplicates = 0
    last = time.monotonic()
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for index, member in enumerate(members, 1):
            # Reading to EOF checks CRC without extracting or altering the ZIP.
            with archive.open(member) as source:
                for _ in iter(lambda: source.read(1024 * 1024), b""):
                    pass
            parts = PurePosixPath(member.filename).parts
            if parts and parts[0] == path.stem:
                parts = parts[1:]
            if (len(parts) == 4 and parts[0] == expected["component"] and
                    parts[1] in SPLITS and parts[-1].endswith(expected["suffix"])):
                key = (parts[2], parts[-1].removesuffix(expected["suffix"]))
                duplicates += key in ids[parts[1]]
                ids[parts[1]].add(key)
            if time.monotonic() - last >= 10 or index == len(members):
                print(f"[CRC_PROGRESS] file={path.name} members={index}/{len(members)}", flush=True)
                last = time.monotonic()
    after = path.stat()
    result.update({
        "crc": "passed", "crc_checked_members": len(members),
        "counts": {split: len(values) for split, values in ids.items()},
        "duplicate_sample_ids": duplicates,
        "unchanged_during_check": (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                                  (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    })
    result["status"] = "passed" if (
        result["bytes_match"] and result["sha256_match"] and
        result["counts"] == SPLITS and not duplicates and result["unchanged_during_check"]
    ) else "failed"
    return result, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, default=Path("/app/data/chaoyang"))
    parser.add_argument("--output", type=Path,
                        default=Path("/app/output/cityscapes_upload_check_v1/summary.json"))
    args = parser.parse_args()
    started = time.monotonic()
    report = {"status": "failed", "search_root": str(args.search_root),
              "source_reference": "User ZIPs fully audited locally on 2026-09-14",
              "data_extracted": False, "training_started": False,
              "test_used_for_evaluation": False, "files": [], "errors": []}
    pairs = {}
    try:
        found = locate(args.search_root)
        print("[ZIP_LOCATIONS] " + json.dumps({k: list(map(str, v)) for k, v in found.items()}), flush=True)
        for name, expected in EXPECTED.items():
            try:
                if len(found[name]) != 1:
                    raise ValueError(f"{name}: expected one file, found {len(found[name])}")
                result, ids = check_archive(found[name][0], expected)
                report["files"].append(result)
                pairs[name] = ids
                print("[ARCHIVE_RESULT] " + json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as error:
                report["errors"].append(f"{name}: {type(error).__name__}: {error}")
        report["image_label_pairs_match"] = (len(pairs) == 2 and
                                             pairs[next(iter(EXPECTED))] == pairs["gtFine_trainvaltest.zip"])
        if (len(report["files"]) == 2 and not report["errors"] and
                report["image_label_pairs_match"] and all(r["status"] == "passed" for r in report["files"])):
            report["status"] = "passed"
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[CITYSCAPES_UPLOAD_REPORT] " + json.dumps(report, ensure_ascii=False), flush=True)
    print(f"[CITYSCAPES_UPLOAD_CHECK_DONE] status={report['status']} summary={args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
