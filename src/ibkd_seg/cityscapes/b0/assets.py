"""Pinned public sources/assets; no random or partial-checkpoint fallback."""
from __future__ import annotations

import argparse
import ast
import json
import sys
import types
import urllib.request
from pathlib import Path

from ..data import save_json, sha256

REPO = Path(__file__).resolve().parents[4]
SPEC = REPO / "phase4/Cityscapes_SegFormer-B0/configs"


def download(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        if record.get("gdrive_id"):
            import gdown
            if not gdown.download(id=record["gdrive_id"], output=str(tmp), quiet=True):
                raise RuntimeError(f"Download failed: {path.name}")
        else:
            with urllib.request.urlopen(record["url"], timeout=120) as source, tmp.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
        if sha256(tmp) != record["sha256"]:
            raise ValueError(f"Downloaded SHA-256 mismatch: {path.name}")
        tmp.replace(path)
    if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Cached asset differs: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "url": record["url"]}


def prepare(cache):
    verify_protocols()
    spec = json.loads((SPEC / "b0_asset_sources_v1.json").read_text())
    result = {"sources": {}, "weights": {}}
    for relative, record in spec["sources"].items():
        result["sources"][relative] = download(record, cache / "cirkd" / relative)
    for name, record in spec["weights"].items():
        result["weights"][name] = download(record, cache / "weights" / name)
    save_json(cache / "asset_manifest.json", result)
    return result


def verify_protocols():
    lock=json.loads((SPEC/"b0_smoke_v2.json").read_text())
    for name,expected in lock["protocol_sha256"].items():
        if sha256(SPEC/name)!=expected:
            raise ValueError(f"Protocol changed after smoke specification was frozen: {name}")


def load_module(name, path, *, strip_legacy=False):
    """Only strip unused tkinter/pip and MMCV's unused auto-loader import."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    tree = ast.parse(path.read_text(), filename=str(path))
    if strip_legacy:
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.ImportFrom) and n.module in {"tkinter.tix", "pip", "mmcv.runner"})]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def modules(cache):
    root = cache / "cirkd"
    spec = json.loads((SPEC / "b0_asset_sources_v1.json").read_text())
    for relative, record in spec["sources"].items():
        if sha256(root / relative) != record["sha256"]:
            raise ValueError(f"Upstream source changed: {relative}")
    for name in ("b0_cirkd", "b0_cirkd.base_models"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    load_module("b0_cirkd.base_models.resnet", root / "models/base_models/resnet.py")
    load_module("b0_cirkd.segbase", root / "models/segbase.py")
    teacher = load_module("b0_cirkd.deeplabv3", root / "models/deeplabv3.py")
    student = load_module("b0_cirkd.segformer", root / "models/segformer.py", strip_legacy=True)
    data = load_module("b0_cirkd.data", root / "dataset/cityscapes.py")
    return student, teacher, data


def hf_to_cirkd(state):
    """Inverse of the upstream HF SegFormer key rename and K/V split, no arithmetic."""
    import re
    import torch
    result, used = {}, set()
    for key, value in state.items():
        if key.startswith("classifier."):
            used.add(key)  # ImageNet classifier is intentionally discarded.
            continue
        if not key.startswith("segformer.encoder."):
            raise ValueError(f"Unexpected NVIDIA checkpoint key: {key}")
        name = key.removeprefix("segformer.encoder.")
        if ".attention.self.value." in name:
            continue
        if ".attention.self.key." in name:
            other = key.replace(".attention.self.key.", ".attention.self.value.")
            value = torch.cat([value, state[other]], dim=0)
            used.add(other)
            name = name.replace(".attention.self.key.", ".attn.kv.")
        name = re.sub(r"^patch_embeddings\.(\d+)\.", lambda m: f"patch_embed{int(m[1])+1}.", name)
        name = re.sub(r"^layer_norm\.(\d+)\.", lambda m: f"norm{int(m[1])+1}.", name)
        name = re.sub(r"^block\.(\d+)\.", lambda m: f"block{int(m[1])+1}.", name)
        for old, new in (("layer_norm_1", "norm1"), ("layer_norm_2", "norm2"),
                         ("layer_norm", "norm"), ("attention.self.query", "attn.q"),
                         ("attention.output.dense", "attn.proj"), ("attention.self", "attn"),
                         ("dense1", "fc1"), ("dense2", "fc2")):
            name = name.replace(old, new)
        if name in result:
            raise ValueError(f"Duplicate converted key: {name}")
        result[name] = value
        used.add(key)
    if used != set(state):
        raise ValueError(f"Unconverted checkpoint keys: {set(state)-used}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.cache.resolve())
    print(json.dumps({"status": "passed", "stage": "assets", "assets": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
