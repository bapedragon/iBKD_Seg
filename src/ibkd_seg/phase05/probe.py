"""Train and evaluate a 1x1 probe on cached frozen feature grids."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Confusion:
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(self, other: "Confusion") -> "Confusion":
        return Confusion(
            self.true_positive + other.true_positive,
            self.true_negative + other.true_negative,
            self.false_positive + other.false_positive,
            self.false_negative + other.false_negative,
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
        total = tp + tn + fp + fn
        return {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "foreground_iou": foreground_iou,
            "background_iou": background_iou,
            "mean_iou": (foreground_iou + background_iou) / 2,
            "foreground_dice": safe(2 * tp, 2 * tp + fp + fn),
            "pixel_accuracy": safe(tp + tn, total),
        }


def confusion_from_tensors(prediction: torch.Tensor, target: torch.Tensor) -> Confusion:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    predicted = prediction.to(torch.bool)
    expected = target.to(torch.bool)
    return Confusion(
        true_positive=int((predicted & expected).sum().item()),
        true_negative=int((~predicted & ~expected).sum().item()),
        false_positive=int((predicted & ~expected).sum().item()),
        false_negative=int((~predicted & expected).sum().item()),
    )


def build_probe(config: dict[str, Any], seed: int, device: torch.device) -> torch.nn.Conv2d:
    torch.manual_seed(seed)
    probe = torch.nn.Conv2d(192, 2, kernel_size=1, bias=True)
    torch.nn.init.normal_(
        probe.weight,
        mean=0.0,
        std=float(config["initialization"]["weight_std"]),
    )
    torch.nn.init.constant_(probe.bias, float(config["initialization"]["bias"]))
    if sum(parameter.numel() for parameter in probe.parameters()) != config["parameter_count"]:
        raise RuntimeError("probe parameter count violates protocol")
    return probe.to(device)


@torch.inference_mode()
def evaluate_probe(
    probe: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
    output_size: tuple[int, int] | None = None,
) -> dict[str, int | float]:
    probe.eval()
    confusion = Confusion()
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        logits = probe(features[start:end].to(device))
        if output_size is not None:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        prediction = logits.argmax(dim=1).cpu()
        confusion = confusion.add(confusion_from_tensors(prediction, targets[start:end]))
    return confusion.metrics()


def train_candidate(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    probe_config: dict[str, Any],
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    probe = build_probe(probe_config, seed, device)
    optimizer_config = probe_config["optimizer"]
    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=learning_rate,
        momentum=float(optimizer_config["momentum"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        nesterov=bool(optimizer_config["nesterov"]),
    )
    epochs = int(probe_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(probe_config["scheduler"]["eta_min"]),
    )
    batch_size = int(probe_config["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    best_metric = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    gradient_contract: dict[str, int] | None = None

    for epoch in range(1, epochs + 1):
        probe.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, len(permutation), batch_size):
            indexes = permutation[start : start + batch_size]
            inputs = train_features[indexes].to(device)
            targets = train_targets[indexes].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(inputs)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
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
            loss_sum += float(loss.item()) * len(indexes)
            sample_count += len(indexes)

        validation_metrics = evaluate_probe(
            probe,
            validation_features,
            validation_targets,
            batch_size=batch_size,
            device=device,
        )
        metric = float(validation_metrics["mean_iou"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                "validation_grid_mean_iou": metric,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        scheduler.step()

    assert best_state is not None and gradient_contract is not None
    return best_state, {
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "best_validation_grid_mean_iou": best_metric,
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
    probe.load_state_dict(copy.deepcopy(state), strict=True)
    return probe
