"""Full fine train/val using upstream transforms and restartable sampling."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .data import encode_mask, sha256, json_hash, save_json, verify_manifest
from .real_smoke import validate_manifest


def sample_seed(seed, epoch, sample_id):
    text = f"cityscapes-full-v1:{seed}:{epoch}:{sample_id}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def epoch_order(count, seed, epoch):
    generator = torch.Generator().manual_seed(sample_seed(seed, epoch, "permutation"))
    return torch.randperm(count, generator=generator).tolist()


def prepare_labels(root, manifest_path, selected, output):
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    verify_manifest(root, manifest)
    label_hashes = {}
    for split in ("train", "val"):
        rows = sorted(manifest["splits"][split], key=lambda r: r["id"])[:selected[split]]
        for index, row in enumerate(rows, 1):
            source = root / row["mask"]
            target = source.with_name(source.name.replace("_labelIds.png", "_labelTrainIds.png"))
            with Image.open(source) as im:
                converted = encode_mask(np.asarray(im))
            if target.exists():
                with Image.open(target) as im:
                    if not np.array_equal(np.asarray(im), converted):
                        raise ValueError(f"Converted label differs: {target}")
            else:
                temporary = target.with_suffix(".png.tmp")
                Image.fromarray(converted).save(temporary, format="PNG")
                temporary.replace(target)
            label_hashes[str(target.relative_to(root))] = sha256(target)
            if index % 500 == 0 or index == len(rows):
                print(f"[L16_LABELS] {split}={index}/{len(rows)}", flush=True)
    report = {"manifest_sha256": sha256(manifest_path), "label_hashes": label_hashes,
              "test_used": False, "counts": selected}
    save_json(output / "labels.json", report)
    return manifest, json_hash(report)


class FullDataset(Dataset):
    def __init__(self, root, manifest, config, split):
        from segm.data.cityscapes import CityscapesDataset
        root = Path(root).resolve()
        if root.name != "cityscapes":
            raise ValueError("Original loader requires a directory named cityscapes")
        os.environ["DATASET"] = str(root.parent)
        self.root, self.config, self.split = root, config, split
        self.epoch = 1
        self.rows = sorted(manifest["splits"][split], key=lambda r: r["id"])[:config[split + "_samples"]]
        self.base = CityscapesDataset(image_size=config["image_size"], crop_size=config["crop_size"],
                                     split=split, normalization="vit")
        mapping = {Path(row["filename"]).name.removesuffix("_leftImg8bit.png"): i
                   for i, row in enumerate(self.base.dataset.img_infos)}
        self.indices = [mapping[row["id"]] for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Upstream MMSeg augmentations draw from numpy.random. Isolating this
        # stream makes workers/prefetch/resume independent of model dropout.
        rng_np, rng_py = np.random.get_state(), random.getstate()
        seed = sample_seed(self.config["seed"], self.epoch if self.split == "train" else 0, row["id"])
        try:
            np.random.seed(seed)
            random.seed(seed)
            item = self.base[self.indices[index]]
        finally:
            np.random.set_state(rng_np)
            random.setstate(rng_py)
        if self.split == "train":
            return item["im"], item["segmentation"].long(), row["id"]
        with Image.open(self.root / row["mask"]) as im:
            target = torch.from_numpy(encode_mask(np.asarray(im)).astype(np.int64))
        if self.config["run_kind"] == "cpu_verification_only":
            from torch.nn.functional import interpolate
            target = interpolate(target[None, None].float(), size=(64, 128), mode="nearest")[0, 0].long()
        return [im[None] for im in item["im"]], item["im_metas"], target, row["id"]


def train_loader(dataset, epoch, next_batch, device):
    config = dataset.config
    dataset.epoch = epoch
    order = epoch_order(len(dataset), config["seed"], epoch)
    generator = torch.Generator().manual_seed(sample_seed(config["seed"], epoch, "loader"))
    kwargs = {}
    if config["data_workers"]:
        # These workers use CPU tensors/OpenCV only. The original legacy data
        # packages are inherited, so no CUDA or model construction in workers.
        kwargs.update(multiprocessing_context="fork", prefetch_factor=2)
    return DataLoader(dataset, batch_size=config["batch_size"],
                      sampler=order[next_batch * config["batch_size"]:], drop_last=False,
                      generator=generator, num_workers=config["data_workers"],
                      pin_memory=device.type == "cuda", **kwargs)


def batch_hash(images, target, ids):
    digest = hashlib.sha256(json.dumps(list(ids)).encode())
    for tensor in (images, target):
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()
