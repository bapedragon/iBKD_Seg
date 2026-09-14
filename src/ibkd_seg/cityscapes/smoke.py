"""Real model/optimizer smoke over unmistakably synthetic Cityscapes-format PNGs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import LABEL_IDS, audit, save_json, verify_manifest
from .runtime import DEFAULT_CONFIG, METHODS, evaluate_checkpoint, load_config, train


def make_fixture(root: Path):
    rng = np.random.default_rng(7)
    for split, count in (("train", 2), ("val", 1)):
        for index in range(count):
            stem = f"synthetic{split}_000000_{index:06d}"
            image_path = root / "leftImg8bit" / split / "synthetic" / f"{stem}_leftImg8bit.png"
            mask_path = root / "gtFine" / split / "synthetic" / f"{stem}_gtFine_labelIds.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
            labels = np.array(LABEL_IDS, dtype=np.uint8)[np.indices((64, 96))[1] % 19]
            labels[:2] = 255
            Image.fromarray(pixels).save(image_path)
            Image.fromarray(labels).save(mask_path)


def run_smoke(output: Path, device: torch.device):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Smoke output must be new/empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(DEFAULT_CONFIG)
    config.update({
        "protocol_id": "cityscapes_synthetic_pixel_accuracy_smoke_v2", "status": "synthetic_smoke_not_scientific",
        "crop_size": [32, 64], "eval_stride": [32, 32], "scale_range": [1.0, 1.0],
        "decoder_channels": 8, "attention_query_chunk": 7, "epochs": 1,
        "warmup_epochs": 0, "validation_every": 1, "precision": "fp32",
        "num_workers": 0, "batch_size": 2, "student_seeds": [1],
    })
    save_json(output / "synthetic_config.json", config)
    root = output / "data"
    make_fixture(root)
    manifest = audit(root, synthetic=True)
    verify_manifest(root, manifest)
    save_json(output / "synthetic_manifest.json", manifest)
    teacher_output = output / "teacher"
    summaries = [train(config=config, data_root=root, manifest=manifest, output=teacher_output,
                       kind="teacher", method=None, seed=1, device=device)]
    initial = None
    for method in METHODS:
        result = train(config=config, data_root=root, manifest=manifest, output=output / method,
                       kind="student", method=method, seed=1, device=device,
                       teacher_checkpoint=None if method == "vanilla" else teacher_output / "best.pt")
        current = result["identity"]["initial_model_sha256"]
        if initial is not None and current != initial:
            raise RuntimeError("Initial student/decoder weights differed across methods")
        initial = current
        summaries.append(result)
    verified = evaluate_checkpoint(output / "ibkd" / "best.pt", root, manifest, device)
    for metric in ("pixel_accuracy", "miou"):
        expected = summaries[-1]["selected_val_" + metric]
        if abs(verified["validation"][metric] - expected) > 1e-10:
            raise RuntimeError(f"Reloaded checkpoint evaluation did not reproduce {metric}")
    save_json(output / "smoke_report.json", {
        "status": "passed", "synthetic": True, "scientific_result": False,
        "checked": ["labelIds audit", "real ResNet50/DeiT forward/backward", "all four methods",
                    "identical student initialization", "native sliding-window evaluation", "strict checkpoint reload"],
        "device": str(device), "crop_size": config["crop_size"], "runs": summaries,
        "limitations": ["Synthetic PNGs only", "No Cityscapes accuracy measured", "No 512x512 GPU memory measurement"],
    })
    print(json.dumps({"status": "passed", "synthetic": True, "report": str(output / "smoke_report.json")}), flush=True)
