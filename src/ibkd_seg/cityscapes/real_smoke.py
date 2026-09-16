"""Bounded Cityscapes data-connection smoke; random weights, no full training."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import Cityscapes, LABEL_IDS, json_hash, save_json, sha256, verify_manifest
from .public_smoke import measure
from .runtime import REPO

CONFIG = REPO / "phase4/phase4_cityscapes/configs/deeplabv3_segmenter_real_smoke_v1.json"


def validate_manifest(manifest):
    if (manifest.get("dataset") != "Cityscapes" or manifest.get("synthetic") is not False or
            manifest.get("label_ids") != list(LABEL_IDS)):
        raise ValueError("Real smoke requires an audited, nonsynthetic Cityscapes manifest")
    if {s: len(rows) for s, rows in manifest["splits"].items()} != {"train": 2975, "val": 500}:
        raise ValueError("Real smoke requires the complete train/val inventory; no test split")
    ids = {s: [r["id"] for r in rows] for s, rows in manifest["splits"].items()}
    if any(len(values) != len(set(values)) for values in ids.values()) or set(ids["train"]) & set(ids["val"]):
        raise ValueError("Duplicate or overlapping sample IDs")
    for split, rows in manifest["splits"].items():
        for row in rows:
            if row["size_wh"] != [2048, 1024]:
                raise ValueError("Unexpected Cityscapes dimensions")
            for kind, component in (("image", "leftImg8bit"), ("mask", "gtFine")):
                path = Path(row[kind])
                if path.is_absolute() or ".." in path.parts or path.parts[:2] != (component, split):
                    raise ValueError("Manifest path is outside the intended train/val split")


def prepare_batches(root, manifest_path, config, *, cpu_small=False):
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if config["train_samples"] != config["steps"] * config["batch_size"]:
        raise ValueError("One fixed batch is required per training step")
    selected = {s: sorted(manifest["splits"][s], key=lambda r: r["id"])[:config[s + "_samples"]]
                for s in ("train", "val")}
    # Every child rechecks the exact files it consumes. The parent checks all files.
    for rows in selected.values():
        for row in rows:
            for kind in ("image", "mask"):
                path = root / row[kind]
                if path.stat().st_size != row[kind + "_bytes"] or sha256(path) != row[kind + "_sha256"]:
                    raise ValueError(f"Selected sample changed after audit: {path}")
    batches = {}
    digest = hashlib.sha256()
    for split, rows in selected.items():
        dataset = Cityscapes(root, rows, config, training=split == "train", seed=config["seed"], normalize=False)
        dataset.epoch = 1
        samples = []
        for index in range(len(dataset)):
            rgb, target, sample_id = dataset[index]
            if split == "val" and cpu_small:
                # Explicit CPU-only resize; GPU validation keeps original resolution.
                size = config["native_eval_size"]
                rgb = F.interpolate(rgb[None], size=size, mode="bilinear", align_corners=False)[0]
                target = F.interpolate(target[None, None].float(), size=size, mode="nearest")[0, 0].long()
            if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError(f"Invalid raw RGB input: {sample_id}")
            if not ((target == 255) | ((target >= 0) & (target < 19))).all() or not (target != 255).any():
                raise ValueError(f"Invalid/all-void target: {sample_id}")
            digest.update(f"{split}:{sample_id}".encode())
            for tensor in (rgb, target):
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
            samples.append((rgb, target))
        batch_size = config["batch_size"] if split == "train" else 1
        batches[split] = [(torch.stack([x[0] for x in samples[i:i + batch_size]]),
                           torch.stack([x[1] for x in samples[i:i + batch_size]]))
                          for i in range(0, len(samples), batch_size)]
    identity = {"manifest_sha256": sha256(manifest_path), "input_sha256": digest.hexdigest(),
                "selected_ids": {s: [r["id"] for r in rows] for s, rows in selected.items()},
                "cpu_reduced_resolution": cpu_small, "test_accessed": False}
    return batches["train"], batches["val"], identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-small", action="store_true")
    parser.add_argument("--method", choices=("vanilla", "lg", "alg", "ibkd"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.cpu_small:
        if args.device != "cpu":
            parser.error("--cpu-small requires --device cpu")
        config.update(protocol_id=config["protocol_id"] + "_cpu_small", crop_size=[32, 64],
                      eval_stride=[32, 48], native_eval_size=[48, 96], attention_query_chunk=7, precision="fp32")
    elif args.device != "cuda":
        parser.error("CPU requires --cpu-small; GPU crop/batch must not be silently reduced")
    torch.set_num_threads(2 if args.device == "cpu" else 4)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    report = {"status": "running", "synthetic": False, "scientific_result": False,
              "pretrained_weights_used": False, "config": config, "runs": [],
              "limitations": ["Random weights; only data/model/gradient/resume connection is checked",
                              "Only six training and two validation images; no full validation or ranking",
                              "Full-training pretrained checkpoints and recipe remain separate work"]}
    try:
        if args.method:
            train_batches, eval_batches, identity = prepare_batches(args.data_dir, args.manifest, config,
                                                                    cpu_small=args.cpu_small)
            save_json(output / "data_identity.json", identity)
            measure(args.method, config, torch.device(args.device), output, train_batches=train_batches,
                    eval_batches=eval_batches, data_identity=identity)
            return
        manifest = json.loads(args.manifest.read_text())
        validate_manifest(manifest)
        print("[CITYSCAPES_REAL_MANIFEST] checking all train/val file hashes", flush=True)
        verify_manifest(args.data_dir, manifest)
        report["manifest_sha256"] = sha256(args.manifest)
        save_json(output / "smoke_summary.json", report)
        for method in config["methods"]:
            print(f"[CITYSCAPES_REAL_SMOKE_START] method={method} random_weights=true", flush=True)
            command = [sys.executable, "-m", "ibkd_seg.cityscapes.real_smoke", "--device", args.device,
                       "--data-dir", str(args.data_dir.resolve()), "--manifest", str(args.manifest.resolve()),
                       "--output-dir", str(output / method), "--method", method]
            if args.cpu_small:
                command.append("--cpu-small")
            subprocess.run(command, check=True)
            row = json.loads((output / method / "summary.json").read_text())
            report["runs"].append(row)
            print(f"[CITYSCAPES_REAL_METHOD_DONE] method={method} status={row['status']} "
                  f"diagnostic_pixel_accuracy={row['diagnostic_pixel_accuracy']:.6f} "
                  f"diagnostic_miou={row['diagnostic_miou']:.6f} "
                  f"peak_cuda_bytes={row['train_peak_allocated_bytes']}", flush=True)
            save_json(output / "smoke_summary.json", report)
        if len({r["student_initial_state_sha256"] for r in report["runs"]}) != 1:
            raise RuntimeError("Student initialization differs across methods")
        if len({r["teacher_state_sha256"] for r in report["runs"] if r["method"] != "vanilla"}) != 1:
            raise RuntimeError("Teacher initialization differs across guided methods")
        if len({json_hash(r["data_identity"]) for r in report["runs"]}) != 1:
            raise RuntimeError("Samples/crops/masks differ across methods")
        if any(r["data_identity"]["manifest_sha256"] != report["manifest_sha256"] for r in report["runs"]):
            raise RuntimeError("Manifest changed during the smoke run")
        report["status"] = "passed"
        save_json(output / "smoke_summary.json", report)
        print(f"[CITYSCAPES_REAL_SMOKE_DONE] status=passed methods=4/4 scientific_result=false "
              f"summary={output / 'smoke_summary.json'}", flush=True)
    except Exception as error:
        report.update(status="failed", error=repr(error), failed_method=args.method or locals().get("method"))
        save_json(output / ("failure.json" if args.method else "smoke_summary.json"), report)
        print(f"[CITYSCAPES_REAL_SMOKE_DONE] status=failed error={error!r}", flush=True)
        raise


if __name__ == "__main__":
    main()
