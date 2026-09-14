"""Official labelIds, paired geometry, and content-addressed dataset audit."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# Official cityscapesScripts/helpers/labels.py evaluation subset, in trainId order.
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
IGNORE = 255
MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
LOOKUP = np.full(256, IGNORE, dtype=np.uint8)
LOOKUP[list(LABEL_IDS)] = np.arange(19, dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def encode_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Expected a single-channel uint8 *_gtFine_labelIds.png")
    if not np.isin(mask, list(range(34)) + [255]).all():
        raise ValueError("Unexpected official labelId")
    return LOOKUP[mask]


def discover(root: Path, split: str) -> list[dict]:
    if split not in {"train", "val"}:
        raise ValueError("This pilot uses train/val only; test labels must not be used")
    rows = []
    for image in sorted((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png")):
        stem = image.name.removesuffix("_leftImg8bit.png")
        mask = root / "gtFine" / split / image.parent.name / f"{stem}_gtFine_labelIds.png"
        if not mask.is_file():
            raise FileNotFoundError(mask)
        rows.append({"id": stem, "image": str(image.relative_to(root)), "mask": str(mask.relative_to(root))})
    if not rows:
        raise FileNotFoundError(f"No Cityscapes {split} images under {root / 'leftImg8bit'}")
    return rows


def audit(root: Path, *, synthetic: bool = False,
          progress: Callable[[str, int, int], None] | None = None) -> dict:
    splits = {}
    for split, expected in (("train", 2975), ("val", 500)):
        rows = discover(root, split)
        if not synthetic and len(rows) != expected:
            raise ValueError(f"{split}: expected {expected} images, found {len(rows)}")
        for index, row in enumerate(rows, 1):
            for kind in ("image", "mask"):
                path = root / row[kind]
                row[kind + "_bytes"] = path.stat().st_size
                row[kind + "_sha256"] = sha256(path)
            with Image.open(root / row["image"]) as image, Image.open(root / row["mask"]) as mask:
                image.load()  # Decode the whole PNG, not just its dimensions.
                if image.size != mask.size or (not synthetic and image.size != (2048, 1024)):
                    raise ValueError(f"Unexpected image/mask dimensions: {row['id']}")
                labels = encode_mask(np.asarray(mask))
                if not (labels != IGNORE).any():
                    raise ValueError(f"No valid target pixels: {row['id']}")
                row["size_wh"] = list(image.size)
            if progress is not None:
                progress(split, index, len(rows))
        splits[split] = rows
    if {r["id"] for r in splits["train"]} & {r["id"] for r in splits["val"]}:
        raise ValueError("Train/val overlap")
    return {"dataset": "Cityscapes", "synthetic": synthetic, "label_ids": list(LABEL_IDS), "splits": splits}


def verify_manifest(root: Path, manifest: dict) -> None:
    if manifest["dataset"] != "Cityscapes" or manifest["label_ids"] != list(LABEL_IDS):
        raise ValueError("Incompatible dataset manifest")
    for split, rows in manifest["splits"].items():
        found = discover(root, split)
        if [(r["id"], r["image"], r["mask"]) for r in rows] != [(r["id"], r["image"], r["mask"]) for r in found]:
            raise ValueError(f"{split} file list changed after audit")
        for row in rows:
            for kind in ("image", "mask"):
                path = root / row[kind]
                if path.stat().st_size != row[kind + "_bytes"] or sha256(path) != row[kind + "_sha256"]:
                    raise ValueError(f"Dataset changed after audit: {path}")


class Cityscapes(Dataset):
    def __init__(self, root: Path, rows: list[dict], config: dict, *, training: bool, seed: int):
        self.root, self.rows, self.config = root, rows, config
        self.training, self.seed, self.epoch = training, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            image = source.convert("RGB")
        with Image.open(self.root / row["mask"]) as source:
            mask = Image.fromarray(encode_mask(np.asarray(source)))
        if self.training:
            # Geometry is independent of worker count, method, and resume boundaries.
            rng = random.Random(f"cityscapes:{self.seed}:{self.epoch}:{row['id']}")
            scale = rng.uniform(*self.config["scale_range"])
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)
            height, width = self.config["crop_size"]
            padding = (0, 0, max(0, width - image.width), max(0, height - image.height))
            image = ImageOps.expand(image, padding, fill=(124, 116, 104))
            mask = ImageOps.expand(mask, padding, fill=IGNORE)
            for _ in range(self.config["crop_attempts"]):
                left = rng.randint(0, image.width - width)
                top = rng.randint(0, image.height - height)
                box = (left, top, left + width, top + height)
                candidate = mask.crop(box)
                values = np.asarray(candidate)
                counts = np.bincount(values[values != IGNORE], minlength=19)
                if counts.sum() and counts.max() / counts.sum() < self.config["max_category_ratio"]:
                    break
            # Always retain a sample after the bounded crop search; no exclusion.
            image, mask = image.crop(box), candidate
            if rng.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        target = torch.from_numpy(np.array(mask, dtype=np.int64, copy=True))
        return (pixels - MEAN) / STD, target, row["id"]
