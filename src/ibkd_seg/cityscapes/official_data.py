"""Use the author's CityscapesDataset and complete MMSeg augmentation pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import encode_mask, sha256
from .real_smoke import validate_manifest
from .runtime import seed_all


def batches(root, manifest_path, config, *, cpu_small=False):
    from segm.data.cityscapes import CityscapesDataset
    if root.name != "cityscapes":
        raise ValueError("Original data loader requires a directory named cityscapes")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    chosen = {split: sorted(manifest["splits"][split], key=lambda row: row["id"])[:count]
              for split, count in (("train", config["steps"] * config["batch_size"]), ("val", 2))}
    label_provenance = {}
    for split, rows in chosen.items():
        for row in rows:
            for kind in ("image", "mask"):
                path = root / row[kind]
                if path.stat().st_size != row[kind + "_bytes"] or sha256(path) != row[kind + "_sha256"]:
                    raise RuntimeError(f"Changed data: {path}")
            path = root / row["mask"]
            target = path.with_name(path.name.replace("_labelIds.png", "_labelTrainIds.png"))
            with Image.open(path) as im:
                converted = encode_mask(np.asarray(im))
            if target.exists():
                with Image.open(target) as im:
                    if not np.array_equal(np.asarray(im), converted):
                        raise RuntimeError(f"Conflicting converted label: {target}")
            else:
                Image.fromarray(converted).save(target)
            label_provenance[str(target.relative_to(root))] = sha256(target)
    os.environ["DATASET"] = str(root.parent)
    digest = hashlib.sha256()
    datasets, output = {}, {}
    for split, rows in chosen.items():
        # CPU test only reduces spatial dimensions and batch, retaining the
        # full L/16 model, decoder, official weights and all transform classes.
        dataset = CityscapesDataset(image_size=64 if cpu_small else config.get("image_size", 1024),
                                    crop_size=32 if cpu_small else config["crop_size"],
                                    split=split, normalization="vit")
        datasets[split] = dataset
        by_id = {Path(row["filename"]).name.removesuffix("_leftImg8bit.png"): index
                 for index, row in enumerate(dataset.dataset.img_infos)}
        items = []
        for index, row in enumerate(rows):
            seed_all(config["seed"] + 100 + index + (10000 if split == "val" else 0))
            item = dataset[by_id[row["id"]]]
            if split == "train":
                image, target = item["im"], item["segmentation"].long()
                if image.shape != (3, config["crop_size"], config["crop_size"]):
                    raise RuntimeError("Upstream transform output size mismatch")
                if not torch.isfinite(image).all() or not (target != 255).any():
                    raise RuntimeError("Invalid/all-void upstream training sample")
                items.append((image, target))
                tensors = [image, target]
            else:
                # Keep the same source ground truth at native resolution.
                with Image.open(root / row["mask"]) as im:
                    target = torch.from_numpy(encode_mask(np.asarray(im)).astype(np.int64))
                if cpu_small:
                    from torch.nn import functional as F
                    target = F.interpolate(target[None, None].float(), size=(64, 128), mode="nearest")[0, 0].long()
                ims = [image[None] for image in item["im"]]
                items.append((ims, item["im_metas"], target))
                tensors = [*ims, target]
            digest.update(f"{split}:{row['id']}".encode())
            for tensor in tensors:
                digest.update(str((tensor.shape, tensor.dtype)).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
        if split == "train":
            size = config["batch_size"]
            output[split] = [(torch.stack([x[0] for x in items[i:i+size]]),
                              torch.stack([x[1] for x in items[i:i+size]]))
                             for i in range(0, len(items), size)]
        else:
            output[split] = items
    return output, {"manifest_sha256": sha256(manifest_path), "input_sha256": digest.hexdigest(),
                    "selected_ids": {s: [r["id"] for r in rows] for s, rows in chosen.items()},
                    "converted_label_hashes": label_provenance, "test_accessed": False,
                    "cpu_reduced_resolution": cpu_small,
                    "upstream_train_pipeline": datasets["train"].config.data.train.pipeline}
