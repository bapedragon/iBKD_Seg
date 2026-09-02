"""느린 공식 데이터 서버를 위한 재개 가능한 multi-range downloader."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import time
from pathlib import Path


def plan_ranges(start: int, total_size: int, connections: int) -> list[tuple[int, int]]:
    if start < 0 or total_size <= 0 or start > total_size:
        raise ValueError("invalid byte range bounds")
    if connections <= 0:
        raise ValueError("connections must be positive")
    if start == total_size:
        return []
    remaining = total_size - start
    chunk_size = (remaining + connections - 1) // connections
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < total_size:
        end = min(total_size - 1, cursor + chunk_size - 1)
        ranges.append((cursor, end))
        cursor = end + 1
    return ranges


def _download_range(
    url: str,
    destination: Path,
    start: int,
    end: int,
    retries: int,
    timeout: float,
) -> None:
    expected_size = end - start + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > expected_size:
        raise ValueError(f"oversized partial file: {destination}")
    if existing_size == expected_size:
        return

    for attempt in range(1, retries + 1):
        existing_size = destination.stat().st_size if destination.exists() else 0
        request_start = start + existing_size
        size_before_attempt = existing_size
        command = [
            "curl",
            "--silent",
            "--show-error",
            "--location",
            "--fail",
            "--connect-timeout",
            str(timeout),
            "--speed-limit",
            "1024",
            "--speed-time",
            "90",
            "--user-agent",
            "ibkd-seg-phase0/0.1",
            "--range",
            f"{request_start}-{end}",
            url,
        ]
        try:
            with destination.open("ab") as handle:
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    text=False,
                )
        except OSError as error:
            if attempt == retries:
                raise RuntimeError(
                    f"failed range {start}-{end} after {retries} attempts"
                ) from error
            time.sleep(min(2**attempt, 10))
            continue

        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            return
        if actual_size > expected_size:
            # A server that ignores Range may return the entire object. Keep the
            # previously verified prefix and fail instead of accepting bad bytes.
            with destination.open("r+b") as handle:
                handle.truncate(size_before_attempt)
            raise RuntimeError(
                f"range {start}-{end} wrote {actual_size} bytes, expected {expected_size}"
            )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            if attempt == retries:
                raise RuntimeError(
                    f"failed range {start}-{end} after {retries} attempts: "
                    f"curl exited with status {result.returncode}: {message}"
                )
            time.sleep(min(2**attempt, 10))
            continue
        if attempt < retries:
            time.sleep(min(2**attempt, 10))

    raise RuntimeError(f"range {start}-{end} did not complete")


def download_file(
    url: str,
    output: Path,
    expected_size: int,
    connections: int = 8,
    retries: int = 4,
    timeout: float = 60.0,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix_size = output.stat().st_size if output.exists() else 0
    if prefix_size > expected_size:
        raise ValueError(
            f"existing file is larger than expected: {prefix_size} > {expected_size}"
        )
    if prefix_size == expected_size:
        print(f"파일 크기 검증 완료, 다운로드 생략: {output}")
        return

    ranges = plan_ranges(prefix_size, expected_size, connections)
    parts_root = output.parent / f".{output.name}.parts-{prefix_size}"
    parts_root.mkdir(parents=True, exist_ok=True)

    def fetch(byte_range: tuple[int, int]) -> tuple[int, int]:
        start, end = byte_range
        part_path = parts_root / f"{start:012d}-{end:012d}.part"
        _download_range(url, part_path, start, end, retries, timeout)
        print(f"{output.name}: byte {start}-{end} 다운로드 완료")
        return byte_range

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        completed = list(executor.map(fetch, ranges))

    assembling = output.parent / f".{output.name}.assembling"
    with assembling.open("wb") as destination:
        if output.exists():
            with output.open("rb") as prefix:
                shutil.copyfileobj(prefix, destination, length=1024 * 1024)
        for start, end in sorted(completed):
            part_path = parts_root / f"{start:012d}-{end:012d}.part"
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())

    if assembling.stat().st_size != expected_size:
        raise RuntimeError(
            f"assembled size mismatch: {assembling.stat().st_size} != {expected_size}"
        )
    os.replace(assembling, output)
    for part_path in parts_root.iterdir():
        part_path.unlink()
    parts_root.rmdir()
    print(f"다운로드 완료: {output} ({expected_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    download_file(
        url=args.url,
        output=args.output,
        expected_size=args.size,
        connections=args.connections,
        retries=args.retries,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
