"""C2VKD* transfer to Segmenter Tiny, with an explicitly substituted CLIP pool."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import torch
from torch import nn

from .b0.losses import AttentionPool, C2VKD, pdd, resize
from .official_assets import sha

POOL_ASSET = {
    "url": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "bytes": 291791292,
    "sha256": "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599",
}
COEFFICIENTS = {"ce": 0.0, "pdd": 1.0, "global": 0.1, "patch": 0.1, "linguistic": 0.5}


def verify_pool_asset(path):
    if path.stat().st_size != POOL_ASSET["bytes"] or sha(path) != POOL_ASSET["sha256"]:
        raise RuntimeError(f"CLIP RN101 byte/SHA256 mismatch: {path}")
    return dict(POOL_ASSET, path=str(path), status="passed", used_part="visual.attnpool only")


def prepare_pool_asset(root):
    path = root / "weights/clip_rn101.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".pt.part")
        with urllib.request.urlopen(POOL_ASSET["url"], timeout=120) as response, temporary.open("wb") as out:
            for chunk in iter(lambda: response.read(8 << 20), b""):
                out.write(chunk)
        verify_pool_asset(temporary)
        temporary.replace(path)
    return verify_pool_asset(path)


def pixel_pdd(student, teacher, labels):
    # Same-size interpolation inside the shared implementation is identity.
    # Resize here with Segmenter's align_corners=False and translate void IDs.
    return pdd(resize(student, labels.shape[-2:]), resize(teacher, labels.shape[-2:]),
               labels.masked_fill(labels == 255, -1))


class FinalFeatureCapture:
    """Read the actual final encoder LayerNorm output; never alter the forward."""
    def __init__(self, model, *, student_channels=192):
        self.student_channels = student_channels
        self.tokens = None
        self.handle = model.encoder.norm.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.tokens = output

    def forward(self, model, images):
        self.tokens = None
        logits = model(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16
        tokens = self.tokens
        self.tokens = None
        if tokens is None or tokens.shape[1:] != (h * w + 1, self.student_channels):
            raise RuntimeError(f"Expected one CLS + patch tokens, width{self.student_channels}")
        return logits, tokens[:, 1:].transpose(1, 2).reshape(images.shape[0], self.student_channels, h, w)

    def close(self):
        self.handle.remove()


class TinyC2VKD(C2VKD):
    def __init__(self, pool_path, *, student_channels=192):
        nn.Module.__init__(self)
        self.asset = verify_pool_asset(Path(pool_path))
        self.visual = nn.Conv2d(student_channels, 2048, 1, bias=False)
        self.linguistic = nn.Conv2d(student_channels, 512, 1, bias=False)
        self.pool = AttentionPool(pool_path)

    def forward(self, feature, teacher, slogits, tlogits, labels):
        # Reuse the audited B0 C2VKD* reductions/geometry. Only the feature source,
        # input channel count and common Segmenter logit/void handling differ.
        return super().forward([feature] * 4, teacher,
                               resize(slogits, labels.shape[-2:]),
                               resize(tlogits, labels.shape[-2:]),
                               labels.masked_fill(labels == 255, -1))


def weighted_components(raw):
    return {name: value * COEFFICIENTS[name] for name, value in raw.items()}


def training_loss(ce, guided):
    """PDD already includes labels. Standalone CE is diagnostic, not added twice."""
    return COEFFICIENTS["ce"] * ce + guided


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    args = parser.parse_args()
    print("[TI16_C2VKD_POOL_ASSET] " + json.dumps(prepare_pool_asset(args.cache_root.resolve())), flush=True)
