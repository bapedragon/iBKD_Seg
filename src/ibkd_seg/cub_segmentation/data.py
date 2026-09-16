"""Fixed CUB RGB/mask batches for the direct-segmentation smoke run."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ibkd_seg.cityscapes.data import json_hash
from ibkd_seg.phase1.cub_data import (
    DERIVED_TRAIN_COUNT,
    DERIVED_VALIDATION_COUNT,
)
from ibkd_seg.phase1.cub_probe_data import (
    CubProbeRecord,
    ids_sha256,
    load_train_validation_records,
)
from ibkd_seg.phase1.data import IMAGENET_MEAN, IMAGENET_STD


def _flip_for(record: CubProbeRecord, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{record.image_id}".encode()).digest()
    return bool(digest[0] & 1)


def load_pair(
    record: CubProbeRecord,
    *,
    input_size: int,
    horizontal_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one official RGB/mask pair and apply exactly paired geometry."""

    with Image.open(record.image_path) as image_handle:
        image = image_handle.convert("RGB")
        original_size = image.size
        image = TF.resize(
            image,
            [input_size, input_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    with Image.open(record.mask_path) as mask_handle:
        if mask_handle.size != original_size:
            raise RuntimeError(
                f"CUB image/mask size mismatch for image_id={record.image_id}"
            )
        mask = TF.resize(
            mask_handle.convert("L"),
            [input_size, input_size],
            interpolation=InterpolationMode.NEAREST,
        )
    if horizontal_flip:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    image_tensor = TF.normalize(
        TF.to_tensor(image),
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    mask_tensor = torch.from_numpy(
        np.asarray(mask, dtype=np.uint8).copy()
    ).gt(0).long()
    if not torch.isfinite(image_tensor).all():
        raise RuntimeError(f"non-finite CUB image tensor: {record.image_id}")
    labels = set(int(value) for value in torch.unique(mask_tensor))
    if not labels.issubset({0, 1}) or 1 not in labels:
        raise RuntimeError(
            f"invalid/empty CUB foreground mask for image_id={record.image_id}: {labels}"
        )
    return image_tensor, mask_tensor


def _batches(samples, batch_size: int):
    return [
        (
            torch.stack([sample[0] for sample in samples[index:index + batch_size]]),
            torch.stack([sample[1] for sample in samples[index:index + batch_size]]),
        )
        for index in range(0, len(samples), batch_size)
    ]


def prepare_fixed_batches(data_dir: Path, config: dict, *, download: bool = True):
    """Use the same pre-result-selected images and tensors for every method."""

    records, split_manifest, source = load_train_validation_records(
        data_dir,
        download=download,
    )
    if {key: len(value) for key, value in records.items()} != {
        "train": DERIVED_TRAIN_COUNT,
        "validation": DERIVED_VALIDATION_COUNT,
    }:
        raise RuntimeError("CUB fixed train/validation split count changed")
    train_ids = {record.image_id for record in records["train"]}
    validation_ids = {record.image_id for record in records["validation"]}
    if train_ids & validation_ids:
        raise RuntimeError("CUB train/validation image IDs overlap")
    if config["train_samples"] != config["steps"] * config["batch_size"]:
        raise ValueError("train_samples must equal steps * batch_size")

    selected = {
        "train": sorted(records["train"], key=lambda record: record.image_id)[
            : config["train_samples"]
        ],
        "validation": sorted(
            records["validation"], key=lambda record: record.image_id
        )[: config["validation_samples"]],
    }
    if any(not values for values in selected.values()):
        raise RuntimeError("CUB smoke selection is empty")

    digest = hashlib.sha256()
    tensors = {}
    flips = {}
    for split, split_records in selected.items():
        samples = []
        flips[split] = {}
        for record in split_records:
            flip = split == "train" and _flip_for(record, config["data_seed"])
            image, mask = load_pair(
                record,
                input_size=config["input_size"],
                horizontal_flip=flip,
            )
            digest.update(f"{split}:{record.image_id}:flip={int(flip)}".encode())
            for tensor in (image, mask):
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
            samples.append((image, mask))
            flips[split][str(record.image_id)] = flip
        tensors[split] = samples

    identity = {
        "dataset": "CUB-200-2011 official RGB and segmentation masks",
        "source_split": "official_train_only",
        "split_manifest_sha256": json_hash(split_manifest),
        "train_image_ids_sha256": ids_sha256(records["train"]),
        "validation_image_ids_sha256": ids_sha256(records["validation"]),
        "selected_ids": {
            split: [record.image_id for record in split_records]
            for split, split_records in selected.items()
        },
        "selected_horizontal_flips": flips,
        "input_tensor_sha256": digest.hexdigest(),
        "input_size": config["input_size"],
        "mask_mapping": "official grayscale value > 0 becomes bird=1; else background=0",
        "official_test_model_input": False,
        "official_test_metrics": False,
    }
    audit = {
        "status": "passed",
        "dataset": identity["dataset"],
        "source": source,
        "split": {
            "source": "official train (5,994 images)",
            "train": len(records["train"]),
            "validation": len(records["validation"]),
            "overlap": 0,
            "split_seed": split_manifest["split_seed"],
            "validation_per_class": split_manifest["validation_per_class"],
            "split_manifest_sha256": identity["split_manifest_sha256"],
        },
        "pair_inventory": {
            "train": len(records["train"]),
            "validation": len(records["validation"]),
            "all_paths_exist": True,
        },
        "smoke_selection": identity,
        "note": (
            "The loader resolves only official-train-derived train/validation records. "
            "The official test split is not instantiated for model input or metrics."
        ),
    }
    train_batches = _batches(tensors["train"], config["batch_size"])
    validation_batches = _batches(
        tensors["validation"], config["validation_batch_size"]
    )
    return train_batches, validation_batches, identity, audit
