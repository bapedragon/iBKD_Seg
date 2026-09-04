"""Oxford-IIIT Pet train/validation data contract for Phase 1."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode


NUM_CLASSES = 37
OFFICIAL_TRAINVAL_COUNT = 3680
OFFICIAL_TEST_COUNT = 3669
SPLIT_SEED = 2027
VALIDATION_PER_CLASS = 20
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _digest_lines(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_stratified_split(
    image_ids: Sequence[str],
    labels: Sequence[int],
    *,
    validation_per_class: int = VALIDATION_PER_CLASS,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Select a deterministic, balanced validation set from official trainval.

    Samples within each class are ranked by SHA-256 of
    ``f"{split_seed}:{image_id}"`` and then by image id.  Returned indices keep
    official file order, which makes manifests easy to audit.
    """

    if len(image_ids) != len(labels):
        raise ValueError("image_ids and labels must have the same length")
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Oxford-IIIT Pet image ids must be unique")

    groups: dict[int, list[str]] = defaultdict(list)
    for image_id, label in zip(image_ids, labels, strict=True):
        groups[int(label)].append(str(image_id))
    if sorted(groups) != list(range(NUM_CLASSES)):
        raise ValueError(
            f"Expected zero-based labels 0..{NUM_CLASSES - 1}, got {sorted(groups)}"
        )

    validation_ids: set[str] = set()
    class_counts: dict[str, dict[str, int]] = {}
    for label in range(NUM_CLASSES):
        members = groups[label]
        if len(members) <= validation_per_class:
            raise ValueError(
                f"Class {label} has {len(members)} samples; cannot reserve "
                f"{validation_per_class} for validation"
            )
        ranked = sorted(
            members,
            key=lambda image_id: (
                hashlib.sha256(f"{split_seed}:{image_id}".encode()).hexdigest(),
                image_id,
            ),
        )
        selected = ranked[:validation_per_class]
        validation_ids.update(selected)
        class_counts[str(label)] = {
            "official_trainval": len(members),
            "train": len(members) - validation_per_class,
            "validation": validation_per_class,
        }

    train_indices = [
        index for index, image_id in enumerate(image_ids) if image_id not in validation_ids
    ]
    validation_indices = [
        index for index, image_id in enumerate(image_ids) if image_id in validation_ids
    ]
    train_ids = [str(image_ids[index]) for index in train_indices]
    validation_ids_ordered = [str(image_ids[index]) for index in validation_indices]
    manifest = {
        "dataset": "Oxford-IIIT Pet",
        "source_split": "official_trainval",
        "split_seed": split_seed,
        "ranking_key": "sha256(f'{split_seed}:{image_id}')_then_image_id",
        "validation_per_class": validation_per_class,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "train_image_ids_sha256": _digest_lines(train_ids),
        "validation_image_ids_sha256": _digest_lines(validation_ids_ordered),
        "train_image_ids": train_ids,
        "validation_image_ids": validation_ids_ordered,
        "class_counts": class_counts,
    }
    return train_indices, validation_indices, manifest


def train_transform() -> Any:
    """Build the submitted LG/DeiT-style training transform."""

    import timm

    return timm.data.create_transform(
        input_size=(3, 224, 224),
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _image_ids(dataset: OxfordIIITPet) -> list[str]:
    images = getattr(dataset, "_images", None)
    if images is None:
        raise RuntimeError("torchvision OxfordIIITPet no longer exposes _images")
    return [Path(path).stem for path in images]


def _labels(dataset: OxfordIIITPet) -> list[int]:
    labels = getattr(dataset, "_labels", None)
    if labels is None:
        raise RuntimeError("torchvision OxfordIIITPet no longer exposes _labels")
    return [int(label) for label in labels]


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def build_train_validation_loaders(
    data_dir: Path,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], dict[str, Any]]:
    """Download only trainval and build the fixed 2,940/740 loaders.

    The official test split is deliberately never instantiated by timing code.
    """

    train_base = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        transform=train_transform(),
        download=True,
    )
    validation_base = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        transform=evaluation_transform(),
        download=False,
    )
    if len(train_base) != OFFICIAL_TRAINVAL_COUNT:
        raise RuntimeError(
            f"Unexpected official trainval count {len(train_base)}; "
            f"expected {OFFICIAL_TRAINVAL_COUNT}"
        )
    image_ids = _image_ids(train_base)
    labels = _labels(train_base)
    if image_ids != _image_ids(validation_base) or labels != _labels(validation_base):
        raise RuntimeError("Train and validation views do not share identical samples")
    train_indices, validation_indices, manifest = build_stratified_split(
        image_ids,
        labels,
    )
    if (len(train_indices), len(validation_indices)) != (2940, 740):
        raise RuntimeError(
            "Unexpected derived split counts: "
            f"train={len(train_indices)} validation={len(validation_indices)}"
        )

    def seed_worker(worker_id: int) -> None:
        del worker_id
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)

    common = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(train_base, train_indices),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        Subset(validation_base, validation_indices),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, manifest
