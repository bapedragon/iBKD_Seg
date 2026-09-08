"""CUB-200-2011 classification data contract for Phase 1.

The official training split is deterministically divided into train and
validation partitions.  The official test split is intentionally never
instantiated by the smoke path.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.transforms import InterpolationMode

from .data import IMAGENET_MEAN, IMAGENET_STD, evaluation_transform, train_transform


DATASET_NAME = "CUB-200-2011"
DATASET_DIRECTORY = "CUB_200_2011"
ARCHIVE_NAME = "CUB_200_2011.tgz"
ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
ARCHIVE_BYTES = 1_150_585_339
ARCHIVE_SHA256 = "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"
DOWNLOAD_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz?download=1"
)
NUM_CLASSES = 200
OFFICIAL_TRAIN_COUNT = 5_994
OFFICIAL_TEST_COUNT = 5_794
OFFICIAL_TOTAL_COUNT = 11_788
SPLIT_SEED = 2027
VALIDATION_PER_CLASS = 3
DERIVED_TRAIN_COUNT = 5_394
DERIVED_VALIDATION_COUNT = 600
RESNET50_IMAGE_SIZE = 224
RESNET50_EVAL_RESIZE_SIZE = 256
DOWNLOAD_RETRIES = 8
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_PROGRESS_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class CubRecord:
    image_id: int
    relative_path: str
    label: int
    is_train: bool


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_content_range(
    value: str | None,
    *,
    expected_start: int,
    expected_bytes: int,
) -> None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value or "")
    if match is None:
        raise RuntimeError(f"missing or malformed Content-Range: {value!r}")
    start, end, total = (int(field) for field in match.groups())
    if (
        start != expected_start
        or end < start
        or end >= expected_bytes
        or total != expected_bytes
    ):
        raise RuntimeError(
            "unexpected Content-Range: "
            f"received={value!r} expected_start={expected_start} "
            f"expected_total={expected_bytes}"
        )


def _download(
    url: str,
    destination: Path,
    *,
    expected_md5: str,
    expected_bytes: int,
    retries: int = DOWNLOAD_RETRIES,
) -> None:
    if expected_bytes <= 0:
        raise ValueError("expected_bytes must be positive")
    if retries <= 0:
        raise ValueError("retries must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_file() and partial.stat().st_size > expected_bytes:
        print(
            f"[CUB_DOWNLOAD_RESET] file={destination.name} "
            f"reason=oversized_partial bytes={partial.stat().st_size}",
            flush=True,
        )
        partial.unlink()

    if partial.is_file() and partial.stat().st_size == expected_bytes:
        actual_md5 = file_digest(partial, "md5")
        if actual_md5 == expected_md5:
            partial.replace(destination)
            print(
                f"[CUB_DOWNLOAD_COMPLETE] file={destination.name} "
                f"bytes={expected_bytes} source=existing_partial",
                flush=True,
            )
            return
        print(
            f"[CUB_DOWNLOAD_RESET] file={destination.name} "
            "reason=full_partial_md5_mismatch",
            flush=True,
        )
        partial.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {
            "User-Agent": "Mozilla/5.0 (iBKD-Seg Phase1 CUB downloader)",
            "Accept-Encoding": "identity",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        print(
            f"[CUB_DOWNLOAD_ATTEMPT] file={destination.name} attempt={attempt}/{retries} "
            f"resume_from={offset} expected_bytes={expected_bytes}",
            flush=True,
        )

        try:
            with urllib.request.urlopen(
                request, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                status = response.getcode()
                if offset:
                    if status != 206:
                        raise RuntimeError(
                            "server ignored resume Range request: "
                            f"status={status} offset={offset}"
                        )
                    _validate_content_range(
                        response.headers.get("Content-Range"),
                        expected_start=offset,
                        expected_bytes=expected_bytes,
                    )
                elif status == 206:
                    _validate_content_range(
                        response.headers.get("Content-Range"),
                        expected_start=0,
                        expected_bytes=expected_bytes,
                    )
                elif status != 200:
                    raise RuntimeError(f"unexpected HTTP status: {status}")

                downloaded = offset
                next_progress = (
                    (downloaded // DOWNLOAD_PROGRESS_BYTES) + 1
                ) * DOWNLOAD_PROGRESS_BYTES
                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > expected_bytes:
                            raise RuntimeError(
                                "download exceeded expected byte size: "
                                f"actual>{expected_bytes}"
                            )
                        if downloaded >= next_progress:
                            print(
                                f"[CUB_DOWNLOAD_PROGRESS] file={destination.name} "
                                f"bytes={downloaded}/{expected_bytes}",
                                flush=True,
                            )
                            next_progress += DOWNLOAD_PROGRESS_BYTES

            actual_bytes = partial.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    "download ended before expected byte size: "
                    f"actual={actual_bytes} expected={expected_bytes}"
                )

            actual_md5 = file_digest(partial, "md5")
            if actual_md5 != expected_md5:
                partial.unlink()
                raise RuntimeError(
                    f"archive MD5 mismatch for {destination.name}: "
                    f"expected={expected_md5} actual={actual_md5}; "
                    "discarded_full_partial=true"
                )
            partial.replace(destination)
            print(
                f"[CUB_DOWNLOAD_COMPLETE] file={destination.name} "
                f"bytes={expected_bytes} attempts={attempt}",
                flush=True,
            )
            return
        except Exception as error:
            last_error = error
            preserved_bytes = partial.stat().st_size if partial.is_file() else 0
            if preserved_bytes > expected_bytes:
                partial.unlink()
                preserved_bytes = 0
            if attempt < retries:
                print(
                    f"[CUB_DOWNLOAD_RETRY] file={destination.name} "
                    f"attempt={attempt}/{retries} preserved_bytes={preserved_bytes} "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(min(2 * attempt, 10))

    assert last_error is not None
    preserved_bytes = partial.stat().st_size if partial.is_file() else 0
    raise RuntimeError(
        f"download failed after {retries} attempts for {destination.name}; "
        f"preserved_bytes={preserved_bytes} partial={partial}"
    ) from last_error


def _read_indexed_values(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"required CUB metadata is missing: {path}")
    values: dict[int, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = raw_line.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed CUB metadata at {path}:{line_number}")
        image_id = int(fields[0])
        if image_id in values:
            raise RuntimeError(f"duplicate image id {image_id} in {path}")
        values[image_id] = fields[1]
    return values


def resolve_dataset_root(root: Path) -> Path:
    root = root.expanduser()
    candidates = (
        root,
        root / DATASET_DIRECTORY,
        root / "cub200" / DATASET_DIRECTORY,
        root / "CUB200" / DATASET_DIRECTORY,
    )
    for candidate in candidates:
        if (candidate / "images.txt").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "CUB-200-2011 is not extracted; expected images.txt under: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def read_records(dataset_root: Path) -> list[CubRecord]:
    paths = _read_indexed_values(dataset_root / "images.txt")
    labels = _read_indexed_values(dataset_root / "image_class_labels.txt")
    splits = _read_indexed_values(dataset_root / "train_test_split.txt")
    if not (paths.keys() == labels.keys() == splits.keys()):
        raise RuntimeError("CUB metadata files contain different image-id sets")

    records: list[CubRecord] = []
    for image_id in sorted(paths):
        relative_path = paths[image_id]
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe CUB relative image path: {relative_path!r}")
        label = int(labels[image_id]) - 1
        split = int(splits[image_id])
        if not 0 <= label < NUM_CLASSES:
            raise RuntimeError(f"invalid CUB label for image {image_id}: {label + 1}")
        if split not in (0, 1):
            raise RuntimeError(f"invalid CUB split flag for image {image_id}: {split}")
        records.append(
            CubRecord(
                image_id=image_id,
                relative_path=relative_path,
                label=label,
                is_train=bool(split),
            )
        )
    return records


def validate_official_layout(dataset_root: Path) -> list[CubRecord]:
    records = read_records(dataset_root)
    train_count = sum(record.is_train for record in records)
    test_count = len(records) - train_count
    classes = {record.label for record in records}
    if len(records) != OFFICIAL_TOTAL_COUNT:
        raise RuntimeError(
            f"unexpected CUB total count {len(records)}; expected {OFFICIAL_TOTAL_COUNT}"
        )
    if (train_count, test_count) != (OFFICIAL_TRAIN_COUNT, OFFICIAL_TEST_COUNT):
        raise RuntimeError(
            "unexpected CUB official split: "
            f"train={train_count} test={test_count}"
        )
    if classes != set(range(NUM_CLASSES)):
        raise RuntimeError(
            f"unexpected CUB classes {len(classes)}; expected {NUM_CLASSES}"
        )
    for record in records:
        image_path = dataset_root / "images" / record.relative_path
        if not image_path.is_file():
            raise RuntimeError(f"missing CUB image: {image_path}")
    return records


def ensure_cub200(root: Path, *, download: bool = True) -> Path:
    try:
        dataset_root = resolve_dataset_root(root)
    except FileNotFoundError:
        if not download:
            raise
        root = root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        archive = root / ARCHIVE_NAME
        if not archive.is_file() or file_digest(archive, "md5") != ARCHIVE_MD5:
            archive.unlink(missing_ok=True)
            _download(
                DOWNLOAD_URL,
                archive,
                expected_md5=ARCHIVE_MD5,
                expected_bytes=ARCHIVE_BYTES,
            )
        if archive.stat().st_size != ARCHIVE_BYTES:
            raise RuntimeError(
                f"CUB archive byte-size mismatch: expected={ARCHIVE_BYTES} "
                f"actual={archive.stat().st_size}"
            )
        actual_sha256 = file_digest(archive)
        if actual_sha256 != ARCHIVE_SHA256:
            raise RuntimeError(
                "CUB archive SHA-256 mismatch: "
                f"expected={ARCHIVE_SHA256} actual={actual_sha256}"
            )
        print(f"[CUB_EXTRACT] archive={archive}", flush=True)
        extract_archive(str(archive), str(root))
        dataset_root = resolve_dataset_root(root)
    validate_official_layout(dataset_root)
    return dataset_root


def _digest_ids(values: Sequence[int]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def build_stratified_split(
    records: Sequence[CubRecord],
    *,
    validation_per_class: int = VALIDATION_PER_CLASS,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Split official-train records without consulting official-test samples."""

    records = list(records)
    if not records or any(not record.is_train for record in records):
        raise ValueError("CUB split builder accepts official-train records only")
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")
    if len({record.image_id for record in records}) != len(records):
        raise ValueError("CUB official-train image ids must be unique")

    groups: dict[int, list[tuple[int, CubRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.label].append((index, record))
    if sorted(groups) != list(range(NUM_CLASSES)):
        raise ValueError(
            f"expected zero-based CUB labels 0..{NUM_CLASSES - 1}"
        )

    train_indices: list[int] = []
    validation_indices: list[int] = []
    class_counts: dict[str, dict[str, int]] = {}
    for label in range(NUM_CLASSES):
        ranked = sorted(
            groups[label],
            key=lambda item: (
                hashlib.sha256(
                    f"{split_seed}:{item[1].image_id}".encode("utf-8")
                ).hexdigest(),
                item[1].image_id,
            ),
        )
        if len(ranked) <= validation_per_class:
            raise ValueError(
                f"CUB class {label} has only {len(ranked)} official-train samples"
            )
        validation_items = ranked[:validation_per_class]
        train_items = ranked[validation_per_class:]
        validation_indices.extend(index for index, _ in validation_items)
        train_indices.extend(index for index, _ in train_items)
        class_counts[str(label)] = {
            "official_train": len(ranked),
            "train": len(train_items),
            "validation": len(validation_items),
        }

    train_indices.sort()
    validation_indices.sort()
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("CUB train and validation indices overlap")
    if sorted(train_indices + validation_indices) != list(range(len(records))):
        raise RuntimeError("CUB train and validation do not partition official train")
    train_ids = [records[index].image_id for index in train_indices]
    validation_ids = [records[index].image_id for index in validation_indices]
    validation_ids_sorted = sorted(validation_ids)
    manifest = {
        "dataset": DATASET_NAME,
        "source_split": "official_train",
        "split_seed": split_seed,
        "ranking_key": "sha256(f'{split_seed}:{image_id}')_then_image_id",
        "validation_per_class": validation_per_class,
        "official_train_samples": len(records),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "train_image_ids_sha256": _digest_ids(train_ids),
        "validation_image_ids_sha256": _digest_ids(validation_ids_sorted),
        "train_image_ids": train_ids,
        "validation_image_ids": validation_ids_sorted,
        "class_counts": class_counts,
    }
    return train_indices, validation_indices, manifest


class CUBClassificationDataset(Dataset[tuple[Any, int]]):
    def __init__(
        self,
        dataset_root: Path,
        records: Sequence[CubRecord],
        *,
        transform: Callable[[Image.Image], Any],
    ) -> None:
        self.dataset_root = dataset_root
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        image_path = self.dataset_root / "images" / record.relative_path
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        return self.transform(image), record.label


def resnet50_train_transform() -> transforms.Compose:
    """Conventional scratch ResNet-50 CUB training view fixed for v3."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                RESNET50_IMAGE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def resnet50_evaluation_transform() -> transforms.Compose:
    """Resize-short-side then center-crop view for the ResNet-50 teacher."""

    return transforms.Compose(
        [
            transforms.Resize(
                RESNET50_EVAL_RESIZE_SIZE,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(RESNET50_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_validation_loaders(
    data_dir: Path,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], dict[str, Any]]:
    """Build the fixed 5,394/600 loaders without opening official test."""

    dataset_root = ensure_cub200(data_dir, download=True)
    official_train = [
        record for record in read_records(dataset_root) if record.is_train
    ]
    if len(official_train) != OFFICIAL_TRAIN_COUNT:
        raise RuntimeError("CUB official-train count changed")
    train_indices, validation_indices, manifest = build_stratified_split(
        official_train
    )
    if (len(train_indices), len(validation_indices)) != (
        DERIVED_TRAIN_COUNT,
        DERIVED_VALIDATION_COUNT,
    ):
        raise RuntimeError("CUB derived train/validation counts changed")
    train_records = [official_train[index] for index in train_indices]
    validation_records = [official_train[index] for index in validation_indices]
    train_dataset = CUBClassificationDataset(
        dataset_root,
        train_records,
        transform=train_transform(),
    )
    validation_dataset = CUBClassificationDataset(
        dataset_root,
        validation_records,
        transform=evaluation_transform(),
    )

    def seed_worker(worker_id: int) -> None:
        del worker_id
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator().manual_seed(seed)
    shared = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **shared,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **shared,
    )
    return train_loader, validation_loader, manifest


def build_resnet50_train_validation_loaders(
    data_dir: Path,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], dict[str, Any]]:
    """Build the fixed split with the locked ResNet-50/224 transforms."""

    dataset_root = ensure_cub200(data_dir, download=True)
    official_train = [
        record for record in read_records(dataset_root) if record.is_train
    ]
    train_indices, validation_indices, manifest = build_stratified_split(
        official_train
    )
    train_records = [official_train[index] for index in train_indices]
    validation_records = [official_train[index] for index in validation_indices]
    train_dataset = CUBClassificationDataset(
        dataset_root,
        train_records,
        transform=resnet50_train_transform(),
    )
    validation_dataset = CUBClassificationDataset(
        dataset_root,
        validation_records,
        transform=resnet50_evaluation_transform(),
    )

    def seed_worker(worker_id: int) -> None:
        del worker_id
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator().manual_seed(seed)
    shared = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **shared,
        ),
        DataLoader(
            validation_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            **shared,
        ),
        manifest,
    )


def build_official_test_loader(
    data_dir: Path,
    *,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    """Instantiate the untouched official test split after selection."""

    dataset_root = ensure_cub200(data_dir, download=True)
    test_records = [
        record for record in read_records(dataset_root) if not record.is_train
    ]
    if len(test_records) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("CUB official-test count changed")
    test_dataset = CUBClassificationDataset(
        dataset_root,
        test_records,
        transform=evaluation_transform(),
    )
    return DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def build_resnet50_official_test_loader(
    data_dir: Path,
    *,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    """Instantiate official test with the ResNet-50 native evaluation view."""

    dataset_root = ensure_cub200(data_dir, download=True)
    test_records = [
        record for record in read_records(dataset_root) if not record.is_train
    ]
    if len(test_records) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("CUB official-test count changed")
    dataset = CUBClassificationDataset(
        dataset_root,
        test_records,
        transform=resnet50_evaluation_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
