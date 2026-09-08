"""CUB-200-2011 RGB/binary-mask contract for the Phase 1 probe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .cub_data import (
    ARCHIVE_BYTES,
    ARCHIVE_MD5,
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    DATASET_NAME,
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
    OFFICIAL_TEST_COUNT,
    OFFICIAL_TRAIN_COUNT,
    CubRecord,
    _download,
    build_stratified_split,
    ensure_cub200,
    file_digest,
    read_records,
)
from .data import IMAGENET_MEAN, IMAGENET_STD


SEGMENTATION_ARCHIVE_NAME = "segmentations.tgz"
SEGMENTATION_ARCHIVE_MD5 = "4d47ba1228eae64f2fa547c47bc65255"
SEGMENTATION_ARCHIVE_BYTES = 39_272_883
SEGMENTATION_ARCHIVE_SHA256 = (
    "dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f"
)
SEGMENTATION_DOWNLOAD_URL = (
    "https://data.caltech.edu/records/w9d68-gec53/files/"
    "segmentations.tgz?download=1"
)


@dataclass(frozen=True)
class CubProbeRecord:
    image_id: int
    label: int
    relative_path: str
    image_path: Path
    mask_path: Path


def ids_sha256(records: Sequence[CubProbeRecord]) -> str:
    payload = "".join(f"{record.image_id}\n" for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_segmentation_root(data_dir: Path, dataset_root: Path) -> Path:
    candidates = (
        dataset_root / "segmentations",
        data_dir / "segmentations",
        data_dir / "CUB_200_2011" / "segmentations",
        data_dir / "cub200" / "CUB_200_2011" / "segmentations",
        data_dir / "CUB200" / "CUB_200_2011" / "segmentations",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "CUB segmentation masks are not extracted; expected segmentations under: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def ensure_cub_segmentations(
    data_dir: Path,
    dataset_root: Path,
    *,
    download: bool = True,
) -> Path:
    try:
        return resolve_segmentation_root(data_dir, dataset_root)
    except FileNotFoundError:
        if not download:
            raise
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / SEGMENTATION_ARCHIVE_NAME
        if (
            not archive.is_file()
            or file_digest(archive, "md5") != SEGMENTATION_ARCHIVE_MD5
        ):
            archive.unlink(missing_ok=True)
            _download(
                SEGMENTATION_DOWNLOAD_URL,
                archive,
                expected_md5=SEGMENTATION_ARCHIVE_MD5,
                expected_bytes=SEGMENTATION_ARCHIVE_BYTES,
            )
        if archive.stat().st_size != SEGMENTATION_ARCHIVE_BYTES:
            raise RuntimeError(
                "CUB segmentation archive byte-size mismatch: "
                f"expected={SEGMENTATION_ARCHIVE_BYTES} actual={archive.stat().st_size}"
            )
        actual_sha256 = file_digest(archive)
        if actual_sha256 != SEGMENTATION_ARCHIVE_SHA256:
            raise RuntimeError(
                "CUB segmentation archive SHA-256 mismatch: "
                f"expected={SEGMENTATION_ARCHIVE_SHA256} actual={actual_sha256}"
            )
        print(f"[CUB_MASK_EXTRACT] archive={archive}", flush=True)
        extract_archive(str(archive), str(data_dir))
        return resolve_segmentation_root(data_dir, dataset_root)


def _to_probe_record(
    record: CubRecord,
    *,
    dataset_root: Path,
    segmentation_root: Path,
) -> CubProbeRecord:
    relative = Path(record.relative_path)
    image_path = dataset_root / "images" / relative
    mask_path = segmentation_root / relative.with_suffix(".png")
    if not image_path.is_file() or not mask_path.is_file():
        raise RuntimeError(
            f"missing CUB image/mask pair for image_id={record.image_id}: "
            f"image={image_path} mask={mask_path}"
        )
    return CubProbeRecord(
        image_id=record.image_id,
        label=record.label,
        relative_path=record.relative_path,
        image_path=image_path,
        mask_path=mask_path,
    )


def load_train_validation_records(
    data_dir: Path,
    *,
    download: bool = True,
) -> tuple[dict[str, list[CubProbeRecord]], dict[str, Any], dict[str, Any]]:
    """Load masks only for the fixed official-train-derived partitions."""

    dataset_root = ensure_cub200(data_dir, download=download)
    segmentation_root = ensure_cub_segmentations(
        data_dir,
        dataset_root,
        download=download,
    )
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

    def make(indices: Sequence[int]) -> list[CubProbeRecord]:
        return [
            _to_probe_record(
                official_train[index],
                dataset_root=dataset_root,
                segmentation_root=segmentation_root,
            )
            for index in indices
        ]

    source = {
        "dataset": DATASET_NAME,
        "image_archive_bytes": ARCHIVE_BYTES,
        "image_archive_md5": ARCHIVE_MD5,
        "image_archive_sha256": ARCHIVE_SHA256,
        "segmentation_archive_bytes": SEGMENTATION_ARCHIVE_BYTES,
        "segmentation_archive_md5": SEGMENTATION_ARCHIVE_MD5,
        "segmentation_archive_sha256": SEGMENTATION_ARCHIVE_SHA256,
        "dataset_root": str(dataset_root),
        "segmentation_root": str(segmentation_root),
    }
    return {
        "train": make(train_indices),
        "validation": make(validation_indices),
    }, manifest, source


def load_official_test_records(
    data_dir: Path,
    *,
    download: bool = True,
) -> tuple[list[CubProbeRecord], dict[str, Any]]:
    """Resolve official-test RGB/mask pairs only after validation selection."""

    dataset_root = ensure_cub200(data_dir, download=download)
    segmentation_root = ensure_cub_segmentations(
        data_dir,
        dataset_root,
        download=download,
    )
    official_test = [
        record for record in read_records(dataset_root) if not record.is_train
    ]
    if len(official_test) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("CUB official-test count changed")
    records = [
        _to_probe_record(
            record,
            dataset_root=dataset_root,
            segmentation_root=segmentation_root,
        )
        for record in official_test
    ]
    source = {
        "dataset": DATASET_NAME,
        "image_archive_name": ARCHIVE_NAME,
        "image_archive_bytes": ARCHIVE_BYTES,
        "image_archive_md5": ARCHIVE_MD5,
        "image_archive_sha256": ARCHIVE_SHA256,
        "segmentation_archive_name": SEGMENTATION_ARCHIVE_NAME,
        "segmentation_archive_bytes": SEGMENTATION_ARCHIVE_BYTES,
        "segmentation_archive_md5": SEGMENTATION_ARCHIVE_MD5,
        "segmentation_archive_sha256": SEGMENTATION_ARCHIVE_SHA256,
        "dataset_root": str(dataset_root),
        "segmentation_root": str(segmentation_root),
    }
    return records, source


class CubImageDataset(Dataset[tuple[torch.Tensor, str]]):
    """Deterministic RGB-only view used after the student is frozen."""

    def __init__(
        self,
        records: Sequence[CubProbeRecord],
        *,
        input_size: int = 224,
    ) -> None:
        self.records = list(records)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
            image = TF.resize(
                image,
                [self.input_size, self.input_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, str(record.image_id)


def load_targets(
    record: CubProbeRecord,
    *,
    input_size: int = 224,
    grid_size: tuple[int, int] = (14, 14),
    occupancy_threshold: float = 0.5,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold the official grayscale mask at ``> 0`` and pool to 14x14."""

    del ignore_index  # CUB's binary masks have no ambiguous/boundary class.
    grid_height, grid_width = grid_size
    if input_size % grid_height or input_size % grid_width:
        raise ValueError("input size must be divisible by both grid dimensions")
    with Image.open(record.image_path) as image_handle:
        image_size = image_handle.size
    with Image.open(record.mask_path) as mask_handle:
        if mask_handle.size != image_size:
            raise RuntimeError(
                f"CUB image/mask size mismatch for image_id={record.image_id}"
            )
        mask = TF.resize(
            mask_handle.convert("L"),
            [input_size, input_size],
            interpolation=InterpolationMode.NEAREST,
        )
        raw = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy())

    mapped = raw.gt(0).to(torch.uint8)
    patch_height = input_size // grid_height
    patch_width = input_size // grid_width
    foreground_blocks = (
        mapped.to(torch.bool)
        .reshape(grid_height, patch_height, grid_width, patch_width)
        .permute(0, 2, 1, 3)
        .reshape(grid_height, grid_width, -1)
    )
    ratios = foreground_blocks.float().mean(dim=-1)
    grid_target = ratios.ge(occupancy_threshold).to(torch.uint8)
    return mapped, grid_target
