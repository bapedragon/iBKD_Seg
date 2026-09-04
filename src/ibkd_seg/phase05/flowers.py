"""Paired Flowers-102 image and pseudo-mask inputs for Phase 0.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ibkd_seg.phase0.masks import binary_foreground_mask


@dataclass(frozen=True)
class FlowersRecord:
    image_id: int
    image_path: Path
    mask_path: Path


def _only_match(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern}, found {len(matches)}")
    return matches[0]


def _split_ids(setid_path: Path) -> dict[str, list[int]]:
    values = loadmat(setid_path)
    return {
        "train": sorted(int(value) for value in np.asarray(values["trnid"]).reshape(-1)),
        "validation": sorted(int(value) for value in np.asarray(values["valid"]).reshape(-1)),
        "test": sorted(int(value) for value in np.asarray(values["tstid"]).reshape(-1)),
    }


def load_flowers_records(data_root: Path) -> dict[str, list[FlowersRecord]]:
    extracted_root = data_root / "extracted"
    if not extracted_root.is_dir():
        extracted_root = data_root
    image_root = _only_match(extracted_root, "jpg")
    mask_root = _only_match(extracted_root, "segmim")
    splits = _split_ids(_only_match(data_root, "setid.mat"))
    records: dict[str, list[FlowersRecord]] = {}
    for split, image_ids in splits.items():
        split_records = [
            FlowersRecord(
                image_id=image_id,
                image_path=image_root / f"image_{image_id:05d}.jpg",
                mask_path=mask_root / f"segmim_{image_id:05d}.jpg",
            )
            for image_id in image_ids
        ]
        missing = [
            record.image_id
            for record in split_records
            if not record.image_path.is_file() or not record.mask_path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{split} contains missing image/mask IDs: {missing[:10]}")
        records[split] = split_records
    return records


def evenly_spaced_subset(records: list[FlowersRecord], count: int) -> list[FlowersRecord]:
    if count <= 0:
        raise ValueError("subset count must be positive")
    if count >= len(records):
        return list(records)
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    return [records[int(index)] for index in indices]


class FlowersImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[FlowersRecord], input_config: dict[str, Any]) -> None:
        self.records = records
        self.size = int(input_config["size"])
        self.mean = [float(value) for value in input_config["normalization_mean"]]
        self.std = [float(value) for value in input_config["normalization_std"]]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
            image = TF.resize(
                image,
                [self.size, self.size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return tensor, record.image_id


def load_targets(
    record: FlowersRecord,
    input_size: int,
    grid_size: tuple[int, int],
    alpha_threshold: float,
    occupancy_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uint8 targets at input and patch-grid resolution."""

    with Image.open(record.image_path) as image_handle, Image.open(record.mask_path) as mask_handle:
        original = np.asarray(image_handle.convert("RGB"), dtype=np.uint8)
        composite = np.asarray(mask_handle.convert("RGB"), dtype=np.uint8)
    native = binary_foreground_mask(original, composite, threshold=alpha_threshold)
    native_tensor = torch.from_numpy(native.astype(np.float32))[None, None]
    input_target = F.interpolate(
        native_tensor,
        size=(input_size, input_size),
        mode="nearest",
    )[0, 0]
    occupancy = F.interpolate(
        input_target[None, None],
        size=grid_size,
        mode="area",
    )[0, 0]
    grid_target = occupancy >= occupancy_threshold
    return input_target.to(torch.uint8), grid_target.to(torch.uint8)
