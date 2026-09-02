"""Dependency-light binary segmentation metrics with explicit edge handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class BinarySegmentationMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    flower_iou: float
    background_iou: float
    mean_iou: float
    flower_dice: float
    pixel_accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    # An absent class that is also never predicted is treated as perfect for
    # that class. Dataset-level aggregation should normally avoid this branch.
    return 1.0 if denominator == 0 else numerator / denominator


def binary_segmentation_metrics(
    prediction: Iterable[int | bool],
    target: Iterable[int | bool],
) -> BinarySegmentationMetrics:
    predicted_values = [bool(value) for value in prediction]
    target_values = [bool(value) for value in target]
    if len(predicted_values) != len(target_values):
        raise ValueError("prediction and target must contain the same number of pixels")
    if not predicted_values:
        raise ValueError("prediction and target must not be empty")

    tp = tn = fp = fn = 0
    for predicted, expected in zip(predicted_values, target_values, strict=True):
        if predicted and expected:
            tp += 1
        elif not predicted and not expected:
            tn += 1
        elif predicted:
            fp += 1
        else:
            fn += 1

    flower_iou = _safe_ratio(tp, tp + fp + fn)
    background_iou = _safe_ratio(tn, tn + fp + fn)
    flower_dice = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    pixel_accuracy = (tp + tn) / len(predicted_values)
    return BinarySegmentationMetrics(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        flower_iou=flower_iou,
        background_iou=background_iou,
        mean_iou=(flower_iou + background_iou) / 2,
        flower_dice=flower_dice,
        pixel_accuracy=pixel_accuracy,
    )
