"""Verify checkpoint bytes, metadata, strict loading, and feature-grid shape."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint manifest schema")
    if not isinstance(manifest.get("checkpoints"), list):
        raise ValueError("checkpoint manifest must contain a checkpoints list")
    return manifest


def _compare(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if actual == expected else "fail",
            "actual": actual,
            "expected": expected,
        }
    )


def _deep_audit(entry: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    import timm
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checks: list[dict[str, Any]] = []
    for key in entry["required_keys"]:
        _compare(checks, f"required_key:{key}", key in checkpoint, True)

    expected = entry["expected"]
    _compare(checks, "method", checkpoint.get("method"), entry["method"])
    for key in ("dataset", "student", "timm_model", "num_classes"):
        _compare(checks, key, checkpoint.get(key), expected[key])
    _compare(checks, "seed", checkpoint.get("args", {}).get("seed"), expected["seed"])
    _compare(
        checks,
        "batch_size",
        checkpoint.get("args", {}).get("batch_size"),
        expected["batch_size"],
    )
    _compare(
        checks,
        "best_accuracy",
        checkpoint.get("best_accuracy"),
        expected["best_accuracy"],
    )
    _compare(
        checks,
        "teacher_sha256",
        checkpoint.get("teacher", {}).get("sha256"),
        expected["teacher_sha256"],
    )

    model = timm.create_model(
        expected["timm_model"],
        pretrained=False,
        num_classes=expected["num_classes"],
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    _compare(checks, "strict_load_missing_keys", list(incompatible.missing_keys), [])
    _compare(checks, "strict_load_unexpected_keys", list(incompatible.unexpected_keys), [])

    model.requires_grad_(False)
    model.eval()
    _compare(
        checks,
        "encoder_trainable_parameter_count",
        sum(parameter.requires_grad for parameter in model.parameters()),
        0,
    )
    with torch.no_grad():
        final_tokens, features = model.forward_intermediates(
            torch.zeros(1, 3, 224, 224),
            indices=list(range(12)),
            norm=False,
            output_fmt="NCHW",
        )
    feature_shapes = [list(feature.shape) for feature in features]
    _compare(checks, "intermediate_feature_count", len(feature_shapes), 12)
    _compare(checks, "final_feature_shape", feature_shapes[-1], [1, 192, 14, 14])

    # Exercise the exact Phase 1 gradient boundary: the frozen feature is fed
    # into a trainable 1x1 probe, and only the probe may receive gradients.
    probe = torch.nn.Conv2d(192, 2, kernel_size=1)
    probe_logits = probe(features[-1].detach())
    probe_logits.square().mean().backward()
    _compare(
        checks,
        "probe_parameter_count",
        sum(parameter.numel() for parameter in probe.parameters()),
        386,
    )
    _compare(
        checks,
        "encoder_gradient_tensor_count_after_probe_backward",
        sum(parameter.grad is not None for parameter in model.parameters()),
        0,
    )
    _compare(
        checks,
        "probe_gradient_tensor_count_after_backward",
        sum(parameter.grad is not None for parameter in probe.parameters()),
        2,
    )
    return {
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
        "feature_shapes": feature_shapes,
        "final_token_shape": list(final_tokens.shape),
        "model_state_entries": len(checkpoint["model"]),
    }


def audit_checkpoint(entry: dict[str, Any], source_root: Path, deep: bool) -> dict[str, Any]:
    relative_path = Path(entry["relative_path"])
    checkpoint_path = source_root / relative_path
    checks: list[dict[str, Any]] = []
    _compare(checks, "exists", checkpoint_path.is_file(), True)
    if checkpoint_path.is_file():
        _compare(checks, "size_bytes", checkpoint_path.stat().st_size, entry["size_bytes"])
        _compare(checks, "sha256", sha256_file(checkpoint_path), entry["sha256"])

    result: dict[str, Any] = {
        "id": entry["id"],
        "role": entry["role"],
        "method": entry["method"],
        "protocol_family": entry["protocol_family"],
        "relative_path": entry["relative_path"],
        "checks": checks,
    }
    if deep and all(item["status"] == "pass" for item in checks):
        result["deep_audit"] = _deep_audit(entry, checkpoint_path)
    result["status"] = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    if result.get("deep_audit", {}).get("status") == "fail":
        result["status"] = "fail"
    return result


def audit_manifest(manifest_path: Path, source_root: Path, deep: bool = True) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    results = [audit_checkpoint(entry, source_root, deep) for entry in manifest["checkpoints"]]
    try:
        import timm
        import torch
        import torchvision

        dependencies: dict[str, str | None] = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
        }
    except ImportError:
        dependencies = {"torch": None, "torchvision": None, "timm": None}
    return {
        "schema_version": 1,
        "audit": "flowers102_checkpoint_contract",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dependencies": dependencies,
        "deep_audit": deep,
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "checkpoints": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = audit_manifest(args.manifest, args.source_root, deep=not args.hash_only)
    except ImportError as error:
        print(f"Missing dependency for deep audit: {error}", file=sys.stderr)
        print("Install requirements or rerun with --hash-only.", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
