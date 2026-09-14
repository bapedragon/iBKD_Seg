"""Native-resolution sliding-window evaluation and global 19-class IoU."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .data import CLASS_NAMES, IGNORE


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("bf16 training/evaluation requires CUDA; use fp32 for CPU smoke")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def confusion_update(matrix, prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction/target dimensions differ")
    valid = target != IGNORE
    truth, predicted = target[valid].long(), prediction[valid].long()
    if truth.numel() and (truth.min() < 0 or truth.max() >= 19 or predicted.min() < 0 or predicted.max() >= 19):
        raise ValueError("Out-of-range trainId")
    matrix += torch.bincount(19 * truth + predicted, minlength=19**2).reshape(19, 19).to(matrix.device)


def metrics(matrix):
    matrix = matrix.double().cpu()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    present = union > 0
    if not present.any():
        raise ValueError("No valid evaluation pixels")
    iou = intersection / union.clamp_min(1)
    return {
        "miou": float(iou[present].mean()),
        "class_iou": {name: float(iou[i]) if present[i] else None for i, name in enumerate(CLASS_NAMES)},
        "evaluated_classes": int(present.sum()),
        "valid_pixels": int(matrix.sum()),
        "pixel_accuracy": float(intersection.sum() / matrix.sum()),
        "confusion_matrix": matrix.long().tolist(),
    }


def starts(length: int, crop: int, stride: int) -> list[int]:
    if min(length, crop, stride) <= 0 or stride > crop:
        raise ValueError("Invalid sliding-window geometry")
    return sorted(set(list(range(0, max(1, length - crop + 1), stride)) + [max(0, length - crop)]))


@torch.inference_mode()
def sliding_logits(model, images, config, device):
    if images.shape[0] != 1:
        raise ValueError("Native-resolution evaluation uses batch size 1")
    height, width = images.shape[-2:]
    crop_h, crop_w = config["crop_size"]
    stride_h, stride_w = config["eval_stride"]
    total = torch.zeros((1, 19, height, width), dtype=torch.float32, device=device)
    count = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    for top in starts(height, crop_h, stride_h):
        for left in starts(width, crop_w, stride_w):
            window = images[..., top:top + crop_h, left:left + crop_w].to(device)
            h, w = window.shape[-2:]
            window = F.pad(window, (0, crop_w - w, 0, crop_h - h))
            with autocast(device, config["precision"]):
                logits = model(window)
            logits = F.interpolate(logits.float(), size=(crop_h, crop_w), mode="bilinear", align_corners=False)
            total[..., top:top + h, left:left + w] += logits[..., :h, :w]
            count[..., top:top + h, left:left + w] += 1
    if (count == 0).any():
        raise RuntimeError("Sliding-window evaluation left uncovered pixels")
    return total / count


@torch.inference_mode()
def evaluate(model, loader, config, device):
    model.eval()
    matrix = torch.zeros((19, 19), dtype=torch.int64)
    for images, target, _ in loader:
        prediction = sliding_logits(model, images, config, device).argmax(1).cpu()
        confusion_update(matrix, prediction, target)
    return metrics(matrix)
