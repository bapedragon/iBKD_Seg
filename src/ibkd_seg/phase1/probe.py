"""Frozen 1x1 segmentation probe primitives for Oxford-IIIT Pet Phase 1."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Confusion:
    """Global binary confusion counts after ignored pixels are removed."""

    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    ignored: int = 0

    def add(self, other: "Confusion") -> "Confusion":
        return Confusion(
            true_positive=self.true_positive + other.true_positive,
            true_negative=self.true_negative + other.true_negative,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
            ignored=self.ignored + other.ignored,
        )

    def metrics(self) -> dict[str, int | float]:
        def safe(numerator: int, denominator: int) -> float:
            return 1.0 if denominator == 0 else numerator / denominator

        tp, tn, fp, fn = (
            self.true_positive,
            self.true_negative,
            self.false_positive,
            self.false_negative,
        )
        foreground_iou = safe(tp, tp + fp + fn)
        background_iou = safe(tn, tn + fp + fn)
        valid = tp + tn + fp + fn
        return {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "ignored": self.ignored,
            "valid_pixels": valid,
            "foreground_iou": foreground_iou,
            "background_iou": background_iou,
            "mean_iou": (foreground_iou + background_iou) / 2,
            "foreground_dice": safe(2 * tp, 2 * tp + fp + fn),
            "pixel_accuracy": safe(tp + tn, valid),
        }


def confusion_from_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> Confusion:
    """Build binary confusion counts while excluding boundary target pixels."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    valid = target.ne(ignore_index)
    valid_targets = target[valid]
    valid_predictions = prediction[valid]
    if valid_targets.numel():
        target_values = set(int(value) for value in torch.unique(valid_targets))
        prediction_values = set(int(value) for value in torch.unique(valid_predictions))
        if not target_values.issubset({0, 1}):
            raise ValueError(f"invalid target values: {sorted(target_values)}")
        if not prediction_values.issubset({0, 1}):
            raise ValueError(f"invalid prediction values: {sorted(prediction_values)}")
    predicted = valid_predictions.to(torch.bool)
    expected = valid_targets.to(torch.bool)
    return Confusion(
        true_positive=int((predicted & expected).sum().item()),
        true_negative=int((~predicted & ~expected).sum().item()),
        false_positive=int((predicted & ~expected).sum().item()),
        false_negative=int((~predicted & expected).sum().item()),
        ignored=int((~valid).sum().item()),
    )


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in a module state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_probe(
    probe_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> torch.nn.Conv2d:
    """Create the locked 386-parameter linear spatial probe."""

    torch.manual_seed(seed)
    probe = torch.nn.Conv2d(192, 2, kernel_size=1, bias=True)
    torch.nn.init.normal_(
        probe.weight,
        mean=0.0,
        std=float(probe_config["initialization"]["weight_std"]),
    )
    torch.nn.init.constant_(
        probe.bias,
        float(probe_config["initialization"]["bias"]),
    )
    parameter_count = sum(parameter.numel() for parameter in probe.parameters())
    if parameter_count != int(probe_config["parameter_count"]):
        raise RuntimeError(
            f"probe parameter count violates protocol: {parameter_count}"
        )
    return probe.to(device)


@torch.inference_mode()
def evaluate_probe(
    probe: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    ignore_index: int = 255,
    output_size: tuple[int, int] | None = None,
) -> dict[str, int | float]:
    """Evaluate by one global confusion matrix, not an image-wise average."""

    if len(features) != len(targets):
        raise ValueError("feature/target sample counts differ")
    probe.eval()
    confusion = Confusion()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        logits = probe(features[start:end].to(device, non_blocking=True))
        if output_size is not None:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        prediction = logits.argmax(dim=1).cpu()
        confusion = confusion.add(
            confusion_from_tensors(
                prediction,
                targets[start:end],
                ignore_index=ignore_index,
            )
        )
    metrics = confusion.metrics()
    if int(metrics["valid_pixels"]) == 0:
        raise RuntimeError("probe evaluation contains no valid pixels")
    return metrics


def train_candidate(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    probe_config: dict[str, Any],
    learning_rate: float,
    seed: int,
    device: torch.device,
    epochs: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Train one LR candidate and retain its best validation-grid epoch."""

    if train_features.requires_grad or validation_features.requires_grad:
        raise RuntimeError("cached encoder features must not require gradients")
    probe = build_probe(probe_config, seed, device)
    initial_probe_state_sha256 = module_state_sha256(probe)
    optimizer_config = probe_config["optimizer"]
    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=learning_rate,
        momentum=float(optimizer_config["momentum"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        nesterov=bool(optimizer_config["nesterov"]),
    )
    actual_epochs = int(probe_config["epochs"] if epochs is None else epochs)
    if actual_epochs <= 0:
        raise ValueError("probe epochs must be positive")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=actual_epochs,
        eta_min=float(probe_config["scheduler"]["minimum_learning_rate"]),
    )
    batch_size = int(probe_config["batch_size"])
    ignore_index = int(probe_config["loss"]["ignore_index"])
    generator = torch.Generator().manual_seed(seed)
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    gradient_contract: dict[str, int] | None = None
    batch_order_sha256_by_epoch: list[str] = []

    for epoch in range(1, actual_epochs + 1):
        probe.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        batch_order_sha256_by_epoch.append(
            hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
        )
        loss_sum = 0.0
        valid_pixel_count = 0
        for start in range(0, len(permutation), batch_size):
            indexes = permutation[start : start + batch_size]
            inputs = train_features[indexes].to(device, non_blocking=True)
            targets = train_targets[indexes].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            batch_valid_pixels = int(targets.ne(ignore_index).sum().item())
            if batch_valid_pixels == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = probe(inputs)
            loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite probe loss")
            loss.backward()
            if gradient_contract is None:
                gradient_contract = {
                    "cached_feature_gradient_tensor_count": int(inputs.grad is not None),
                    "probe_gradient_tensor_count": sum(
                        parameter.grad is not None for parameter in probe.parameters()
                    ),
                }
            optimizer.step()
            loss_sum += float(loss.item()) * batch_valid_pixels
            valid_pixel_count += batch_valid_pixels

        if valid_pixel_count == 0:
            raise RuntimeError("probe training epoch contains no valid pixels")
        validation_metrics = evaluate_probe(
            probe,
            validation_features,
            validation_targets,
            batch_size=batch_size,
            device=device,
            ignore_index=ignore_index,
        )
        metric = float(validation_metrics["mean_iou"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / valid_pixel_count,
                "validation_grid_mean_iou": metric,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        # Strict greater-than preserves the earlier epoch on an exact tie.
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        scheduler.step()

    if best_state is None or gradient_contract is None:
        raise RuntimeError("probe candidate failed to produce a valid checkpoint")
    return best_state, {
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "best_validation_grid_mean_iou": best_metric,
        "initial_probe_state_sha256": initial_probe_state_sha256,
        "batch_order_sha256_by_epoch": batch_order_sha256_by_epoch,
        "gradient_contract": gradient_contract,
        "history": history,
    }


def probe_from_state(
    probe_config: dict[str, Any],
    seed: int,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Conv2d:
    probe = build_probe(probe_config, seed, device)
    incompatible = probe.load_state_dict(copy.deepcopy(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict probe reload returned incompatible keys")
    return probe
