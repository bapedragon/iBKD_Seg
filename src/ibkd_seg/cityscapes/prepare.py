"""Prepare the two user-downloaded Cityscapes ZIPs; no network or credentials."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .data import audit, save_json, sha256

PACKAGES = {
    "leftImg8bit_trainvaltest.zip": "leftImg8bit",
    "gtFine_trainvaltest.zip": "gtFine",
}
EXPECTED = {"train": 2975, "val": 500}


def archive_plan(archive: zipfile.ZipFile, package: str) -> dict[str, Path]:
    """Accept official roots or one enclosing package folder; skip test/metadata."""
    component = PACKAGES[package]
    wrapper = Path(package).stem
    destinations: dict[str, Path] = {}
    seen = set()
    counts: Counter = Counter()
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts
        if (member.filename.startswith("/") or ".." in parts or
                "\\" in member.filename or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError(f"Unsafe ZIP entry: {member.filename}")
        if member.is_dir() or not parts or parts[0] == "__MACOSX":
            continue
        if parts[0] == wrapper:
            parts = parts[1:]
        if parts in (("README",), ("license.txt",)):
            relative = Path("source_info", wrapper, *parts)
        elif (len(parts) == 4 and parts[0] == component and
              parts[1] in EXPECTED and not parts[-1].startswith(".")):
            relative = Path(*parts)
            suffix = "_leftImg8bit.png" if component == "leftImg8bit" else "_gtFine_labelIds.png"
            if parts[-1].endswith(suffix):
                counts[parts[1]] += 1
        else:
            continue
        if relative in seen or member.filename in destinations:
            raise ValueError(f"Duplicate ZIP destination: {relative}")
        seen.add(relative)
        destinations[member.filename] = relative
    if dict(counts) != EXPECTED:
        raise ValueError(f"{package}: expected {EXPECTED}, found {dict(counts)}")
    return destinations


def extract_checked(archive: zipfile.ZipFile, plan: dict[str, Path], root: Path) -> dict:
    """Read every member to EOF (CRC), keeping only planned train/val files."""
    members = [m for m in archive.infolist() if not m.is_dir()]
    records = []
    last_progress = time.monotonic()
    for index, member in enumerate(members, 1):
        relative = plan.get(member.filename)
        if relative is None:
            with archive.open(member) as source:
                for _ in iter(lambda: source.read(1024 * 1024), b""):
                    pass
        else:
            target = root / relative
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Destination escapes data root: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".prepare-", delete=False) as output:
                    temporary = Path(output.name)
                    with archive.open(member) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                            output.write(block)
                value = digest.hexdigest()
                if target.exists():
                    if target.stat().st_size != member.file_size or sha256(target) != value:
                        raise ValueError(f"Existing data differs; choose a new data directory: {target}")
                else:
                    temporary.replace(target)
                records.append({"path": relative.as_posix(), "bytes": member.file_size, "sha256": value})
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if time.monotonic() - last_progress >= 10 or index == len(members):
            print(f"[ZIP] {Path(archive.filename).name}: {index}/{len(members)} CRC checked", flush=True)
            last_progress = time.monotonic()
    return {"crc_checked_members": len(members), "extracted_files": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Validate BOTH inventories before writing any dataset files.
    plans = {}
    needed = 0
    for package in PACKAGES:
        path = args.zip_dir / package
        with zipfile.ZipFile(path) as archive:
            plans[package] = archive_plan(archive, package)
            needed += sum(m.file_size for m in archive.infolist() if m.filename in plans[package])
    if shutil.disk_usage(root).free < needed + 1024 ** 3:
        raise OSError(f"Need at least {needed + 1024 ** 3} free bytes")
    provenance = {
        "dataset": "Cityscapes", "source": "user-provided downloads",
        "official_download_page": "https://www.cityscapes-dataset.com/downloads/",
        "official_archive_digest_verified": False,
        "note": "Local ZIP byte hashes identify supplied archives; not an official MD5 comparison.",
        "extracted_splits": list(EXPECTED), "test_used_for_training_or_evaluation": False,
        "archives": [],
    }
    for package in PACKAGES:
        path = args.zip_dir / package
        print(f"[SHA256] {package}", flush=True)
        record = {"name": package, "bytes": path.stat().st_size, "sha256": sha256(path)}
        with zipfile.ZipFile(path) as archive:
            record.update(extract_checked(archive, plans[package], root))
        provenance["archives"].append(record)
    print("[AUDIT] Decode every train/val image and mask; verify dimensions, labels and hashes", flush=True)

    def progress(split: str, count: int, total: int) -> None:
        if count % 100 == 0 or count == total:
            print(f"[AUDIT] {split}: {count}/{total}", flush=True)

    manifest = audit(root, progress=progress)
    save_json(root / "manifest.json", manifest)
    provenance.update({
        "status": "passed", "counts": {s: len(rows) for s, rows in manifest["splits"].items()},
        "manifest_sha256": sha256(root / "manifest.json"),
        "extracted_bytes": sum(r["bytes"] for a in provenance["archives"] for r in a["extracted_files"]),
    })
    save_json(root / "preparation.json", provenance)
    print(f"[CITYSCAPES_DATA_READY] train=2975 val=500 data_dir={root}", flush=True)


if __name__ == "__main__":
    main()
