"""Oxford-IIIT Pet train/validation data contract for Phase 1."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
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
ARCHIVE_CONTRACTS = {
    "images.tar.gz": {
        "bytes": 791918971,
        "md5": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        "sha256": None,
    },
    "annotations.tar.gz": {
        "bytes": 19173078,
        "md5": "95a8c909bbe2e81eed6a22bccdf3f68f",
        "sha256": "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
    },
}
ANNOTATION_HASHES = {
    "list.txt": "6a54ab256e22f7a33c6f17a7669e58ea5f6f9c7a080ec2622c205aefd4b354da",
    "trainval.txt": "408f3f609481b939c94634169e6413414b733a3faeba440cbdcc5c02142eebdc",
    "test.txt": "a5454003774ffe01f4f322756d3ba5495bae21cb30bb217ab285dbfa2bef245c",
}


def _digest_lines(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_annotation_split(path: Path) -> list[tuple[str, int]]:
    records: list[tuple[str, int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError(f"Malformed {path.name}:{line_number}")
        image_id, label, _, _ = fields
        zero_based = int(label) - 1
        if not 0 <= zero_based < NUM_CLASSES:
            raise RuntimeError(f"Invalid breed label in {path.name}:{line_number}")
        records.append((image_id, zero_based))
    return records


def prepare_and_audit_dataset(data_dir: Path) -> dict[str, Any]:
    """Download and integrity-audit all official files before a full run.

    Test files are decoded only for dataset integrity here.  No test target or
    image statistic is exposed to model fitting, checkpoint selection, or
    hyperparameter decisions.
    """

    OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        download=True,
    )
    base = data_dir / "oxford-iiit-pet"
    archives: dict[str, Any] = {}
    for filename, expected in ARCHIVE_CONTRACTS.items():
        path = base / filename
        if not path.is_file():
            raise RuntimeError(f"Downloaded archive is missing: {path}")
        size = path.stat().st_size
        md5 = file_digest(path, "md5")
        sha256 = file_digest(path, "sha256")
        if size != expected["bytes"] or md5 != expected["md5"]:
            raise RuntimeError(
                f"Archive contract mismatch for {filename}: bytes={size} md5={md5}"
            )
        if expected["sha256"] is not None and sha256 != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {filename}")
        archives[filename] = {"bytes": size, "md5": md5, "sha256": sha256}

    annotation_dir = base / "annotations"
    annotation_hashes: dict[str, str] = {}
    for filename, expected_sha256 in ANNOTATION_HASHES.items():
        actual = file_digest(annotation_dir / filename)
        if actual != expected_sha256:
            raise RuntimeError(f"Official annotation hash mismatch for {filename}")
        annotation_hashes[filename] = actual

    split_records = {
        split: _parse_annotation_split(annotation_dir / f"{split}.txt")
        for split in ("trainval", "test")
    }
    if len(split_records["trainval"]) != OFFICIAL_TRAINVAL_COUNT:
        raise RuntimeError("Official trainval count mismatch")
    if len(split_records["test"]) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("Official test count mismatch")
    trainval_ids = {record[0] for record in split_records["trainval"]}
    test_ids = {record[0] for record in split_records["test"]}
    if len(trainval_ids) != OFFICIAL_TRAINVAL_COUNT or len(test_ids) != OFFICIAL_TEST_COUNT:
        raise RuntimeError("Duplicate image id in an official split")
    if trainval_ids & test_ids:
        raise RuntimeError("Official trainval and test splits overlap")

    class_counts: dict[str, dict[str, int]] = {}
    decoded = 0
    for split, records in split_records.items():
        counts = [0] * NUM_CLASSES
        for image_id, label in records:
            counts[label] += 1
            image_path = base / "images" / f"{image_id}.jpg"
            trimap_path = annotation_dir / "trimaps" / f"{image_id}.png"
            if not image_path.is_file() or not trimap_path.is_file():
                raise RuntimeError(f"Missing image/trimap pair for {image_id}")
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                image_size = rgb.size
            with Image.open(trimap_path) as trimap:
                values = np.asarray(trimap)
                trimap_size = trimap.size
            if image_size != trimap_size:
                raise RuntimeError(f"Image/trimap size mismatch for {image_id}")
            unique_values = set(int(value) for value in np.unique(values))
            if not unique_values.issubset({1, 2, 3}):
                raise RuntimeError(
                    f"Invalid trimap values for {image_id}: {sorted(unique_values)}"
                )
            decoded += 1
            if decoded % 1000 == 0:
                print(f"[DATA_AUDIT_PROGRESS] decoded={decoded}/7349", flush=True)
        class_counts[split] = {str(label): count for label, count in enumerate(counts)}

    return {
        "status": "pass",
        "dataset": "Oxford-IIIT Pet",
        "archives": archives,
        "annotation_hashes": annotation_hashes,
        "split_counts": {
            "trainval": len(split_records["trainval"]),
            "test": len(split_records["test"]),
            "total": decoded,
        },
        "class_counts": class_counts,
        "unique_ids": True,
        "disjoint_official_splits": True,
        "image_label_trimap_one_to_one": True,
        "rgb_and_trimap_decode": True,
        "trimap_values": [1, 2, 3],
        "test_content_access": "integrity_audit_only",
        "official_test_used_for_training_or_selection": False,
    }


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
        np.random.seed(worker_seed)

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


def build_official_test_loader(
    data_dir: Path,
    *,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    """Instantiate official test only after validation selection is complete."""

    test_dataset = OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        transform=evaluation_transform(),
        download=False,
    )
    if len(test_dataset) != OFFICIAL_TEST_COUNT:
        raise RuntimeError(
            f"Unexpected official test count {len(test_dataset)}; "
            f"expected {OFFICIAL_TEST_COUNT}"
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
