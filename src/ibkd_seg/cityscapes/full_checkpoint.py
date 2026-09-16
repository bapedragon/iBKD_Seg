"""Atomic rolling checkpoints with a previous generation and portable pointers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import torch

from .data import sha256, save_json


def artifact_path(root, record):
    relative = Path(record["file"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "checkpoints":
        raise ValueError("Unsafe checkpoint relative path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Checkpoint escapes its run directory")
    return path


def verify_artifact(root, record):
    path = artifact_path(root, record)
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return path


def atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def record(root, path):
    return {"file": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def save_best(root, model, signature, epoch, scores):
    path = root / "checkpoints" / f"best_epoch_{epoch:04d}.pt"
    atomic_torch_save(path, {"model": model.state_dict(), "signature": signature,
                             "epoch": epoch, "metrics": scores})
    return dict(record(root, path), epoch=epoch, metrics=scores)


def save_checkpoint(root, payload):
    pointer = root / "resume.json"
    old = json.loads(pointer.read_text()) if pointer.exists() else {}
    step = payload["progress"]["global_step"]
    path = root / "checkpoints" / f"resume_{step:09d}_{uuid.uuid4().hex[:12]}.pt"
    atomic_torch_save(path, payload)
    current = dict(record(root, path), best=payload["progress"]["best"], global_step=step)
    manifest = {"format": 1, "current": current, "previous": old.get("current")}
    save_json(pointer, manifest)
    # Keep complete generations, including the best student referenced by each.
    keep = set()
    for item in (manifest["current"], manifest["previous"]):
        if item:
            keep.add(item["file"])
            if item["best"]:
                keep.add(item["best"]["file"])
    for pattern in ("resume_*.pt", "best_epoch_*.pt"):
        for candidate in (root / "checkpoints").glob(pattern):
            if str(candidate.relative_to(root)) not in keep:
                candidate.unlink()
    return manifest


def load_checkpoint(pointer, signature, destination):
    manifest = json.loads(pointer.read_text())
    if manifest.get("format") != 1:
        raise ValueError("Unknown resume manifest format")
    root = pointer.parent
    errors = []
    for name in ("current", "previous"):
        item = manifest.get(name)
        if item is None:
            continue
        try:
            path = verify_artifact(root, item)
            if item["best"]:
                verify_artifact(root, item["best"])
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["signature"] != signature:
            raise ValueError("Resume rejected: method/seed/config/code/data/runtime identity differs")
        if payload["progress"]["best"] != item["best"]:
            raise ValueError("Resume pointer and checkpoint disagree about best student")
        if item["best"] and root.resolve() != destination.resolve():
            source = verify_artifact(root, item["best"])
            target = artifact_path(destination, item["best"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            verify_artifact(destination, item["best"])
        return payload, {"generation": name, "global_step": item["global_step"], "fallback_errors": errors}
    raise RuntimeError("No intact resume generation: " + "; ".join(errors))


def tree_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            value = item.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()
