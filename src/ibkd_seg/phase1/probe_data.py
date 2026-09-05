"""Oxford-IIIT Pet RGB/trimap input contract for the Phase 1 probe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OFFICIAL_TEST_COUNT,
    OFFICIAL_TRAINVAL_COUNT,
    build_stratified_split,
)


@dataclass(frozen=True)
class PetRecord:
    image_id: str
    label: int
    image_path: Path
    trimap_path: Path


def ids_sha256(records: Sequence[PetRecord]) -> str:
    payload = "".join(f"{record.image_id}\n" for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_train_validation_records(
    data_dir: Path,
    *,
    download: bool = True,
) -> tuple[dict[str, list[PetRecord]], dict[str, Any]]:
    """Load only official trainval, then reproduce the locked 2,940/740 split."""

    dataset = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        download=download,
    )
    if len(dataset) != OFFICIAL_TRAINVAL_COUNT:
        raise RuntimeError(
            f"unexpected official trainval count {len(dataset)}; "
            f"expected {OFFICIAL_TRAINVAL_COUNT}"
        )
    images = getattr(dataset, "_images", None)
    labels = getattr(dataset, "_labels", None)
    if images is None or labels is None:
        raise RuntimeError("torchvision OxfordIIITPet internals changed")
    image_ids = [Path(path).stem for path in images]
    integer_labels = [int(label) for label in labels]
    train_indices, validation_indices, manifest = build_stratified_split(
        image_ids,
        integer_labels,
    )
    if (len(train_indices), len(validation_indices)) != (2940, 740):
        raise RuntimeError("derived Phase 1 split is not 2,940/740")

    base = data_dir / "oxford-iiit-pet"

    def records(indices: Sequence[int]) -> list[PetRecord]:
        result: list[PetRecord] = []
        for index in indices:
            image_id = image_ids[index]
            record = PetRecord(
                image_id=image_id,
                label=integer_labels[index],
                image_path=base / "images" / f"{image_id}.jpg",
                trimap_path=base / "annotations" / "trimaps" / f"{image_id}.png",
            )
            if not record.image_path.is_file() or not record.trimap_path.is_file():
                raise RuntimeError(f"missing image/trimap pair for {image_id}")
            result.append(record)
        return result

    return {
        "train": records(train_indices),
        "validation": records(validation_indices),
    }, manifest


def load_official_test_records(data_dir: Path) -> list[PetRecord]:
    """Instantiate the untouched official test split after model selection."""

    dataset = OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        download=False,
    )
    if len(dataset) != OFFICIAL_TEST_COUNT:
        raise RuntimeError(
            f"unexpected official test count {len(dataset)}; expected {OFFICIAL_TEST_COUNT}"
        )
    images = getattr(dataset, "_images", None)
    labels = getattr(dataset, "_labels", None)
    if images is None or labels is None:
        raise RuntimeError("torchvision OxfordIIITPet internals changed")
    base = data_dir / "oxford-iiit-pet"
    records: list[PetRecord] = []
    for image_path, label in zip(images, labels, strict=True):
        image_id = Path(image_path).stem
        record = PetRecord(
            image_id=image_id,
            label=int(label),
            image_path=base / "images" / f"{image_id}.jpg",
            trimap_path=base / "annotations" / "trimaps" / f"{image_id}.png",
        )
        if not record.image_path.is_file() or not record.trimap_path.is_file():
            raise RuntimeError(f"missing official test image/trimap pair for {image_id}")
        records.append(record)
    if len({record.image_id for record in records}) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("official test contains duplicate image ids")
    return records


class PetImageDataset(Dataset[tuple[torch.Tensor, str]]):
    """Deterministic RGB-only view used while the encoder is frozen."""

    def __init__(self, records: Sequence[PetRecord], *, input_size: int = 224) -> None:
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
        return tensor, record.image_id


def load_targets(
    record: PetRecord,
    *,
    input_size: int = 224,
    grid_size: tuple[int, int] = (14, 14),
    occupancy_threshold: float = 0.5,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map Pet trimap values and derive valid-pixel occupancy grid targets."""

    grid_height, grid_width = grid_size
    if input_size % grid_height or input_size % grid_width:
        raise ValueError("input size must be divisible by both grid dimensions")
    with Image.open(record.image_path) as image_handle:
        image_size = image_handle.size
    with Image.open(record.trimap_path) as trimap_handle:
        if trimap_handle.size != image_size:
            raise RuntimeError(f"image/trimap size mismatch for {record.image_id}")
        trimap = TF.resize(
            trimap_handle,
            [input_size, input_size],
            interpolation=InterpolationMode.NEAREST,
        )
        raw = torch.from_numpy(np.asarray(trimap, dtype=np.uint8).copy())

    raw_values = set(int(value) for value in torch.unique(raw))
    if not raw_values.issubset({1, 2, 3}):
        raise RuntimeError(
            f"invalid trimap values for {record.image_id}: {sorted(raw_values)}"
        )
    mapped = torch.full((input_size, input_size), ignore_index, dtype=torch.uint8)
    mapped[raw.eq(1)] = 1
    mapped[raw.eq(2)] = 0

    patch_height = input_size // grid_height
    patch_width = input_size // grid_width
    valid = mapped.ne(ignore_index)
    foreground = mapped.eq(1)
    valid_blocks = (
        valid.reshape(grid_height, patch_height, grid_width, patch_width)
        .permute(0, 2, 1, 3)
        .reshape(grid_height, grid_width, -1)
    )
    foreground_blocks = (
        foreground.reshape(grid_height, patch_height, grid_width, patch_width)
        .permute(0, 2, 1, 3)
        .reshape(grid_height, grid_width, -1)
    )
    valid_counts = valid_blocks.sum(dim=-1)
    foreground_counts = foreground_blocks.sum(dim=-1)
    grid_target = torch.full(
        (grid_height, grid_width),
        ignore_index,
        dtype=torch.uint8,
    )
    usable = valid_counts.gt(0)
    ratios = foreground_counts.float() / valid_counts.clamp_min(1).float()
    grid_target[usable] = ratios[usable].ge(occupancy_threshold).to(torch.uint8)
    return mapped, grid_target


def evenly_spaced_subset(
    records: Sequence[PetRecord],
    count: int,
) -> list[PetRecord]:
    """Deterministic helper retained for unit tests or explicit debug runs."""

    if count <= 0 or count > len(records):
        raise ValueError("subset count must be in 1..len(records)")
    if count == len(records):
        return list(records)
    if count == 1:
        return [records[0]]
    indices = [index * (len(records) - 1) // (count - 1) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("evenly spaced subset produced duplicate indices")
    return [records[index] for index in indices]
