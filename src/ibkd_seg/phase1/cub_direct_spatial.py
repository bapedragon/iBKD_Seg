"""Direct spatial diagnostics for frozen CUB classification encoders.

This module keeps the three diagnostics independent of the H200 orchestration:
visible-part heatmap probing, spatial-token linear CKA, and transformer attention
rollout against the official CUB foreground masks.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .cub_probe_data import CubProbeRecord


PART_COUNT = 15
GRID_SIZE = 14
INPUT_SIZE = 224


@dataclass(frozen=True)
class CubSpatialAnnotation:
    """One CUB image's original-coordinate box and fifteen part landmarks."""

    image_id: int
    bounding_box: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]
    visible: tuple[bool, ...]


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required CUB spatial annotation is missing: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_spatial_annotations(
    dataset_root: Path,
) -> tuple[tuple[str, ...], dict[int, CubSpatialAnnotation]]:
    """Parse and fully audit official CUB part and bounding-box metadata."""

    names: dict[int, str] = {}
    for line_number, line in enumerate(_read_lines(dataset_root / "parts/parts.txt"), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RuntimeError(f"malformed parts.txt line {line_number}")
        part_id = int(fields[0])
        if part_id in names:
            raise RuntimeError(f"duplicate CUB part id: {part_id}")
        names[part_id] = fields[1]
    if sorted(names) != list(range(1, PART_COUNT + 1)):
        raise RuntimeError("CUB part names must contain exactly ids 1..15")

    boxes: dict[int, tuple[float, float, float, float]] = {}
    for line_number, line in enumerate(_read_lines(dataset_root / "bounding_boxes.txt"), 1):
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"malformed bounding_boxes.txt line {line_number}")
        image_id = int(fields[0])
        box = tuple(float(value) for value in fields[1:])
        if image_id in boxes or box[2] <= 0.0 or box[3] <= 0.0:
            raise RuntimeError(f"invalid CUB bounding box for image {image_id}")
        boxes[image_id] = box  # type: ignore[assignment]

    points: dict[int, dict[int, tuple[float, float, bool]]] = {}
    for line_number, line in enumerate(_read_lines(dataset_root / "parts/part_locs.txt"), 1):
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"malformed part_locs.txt line {line_number}")
        image_id, part_id = int(fields[0]), int(fields[1])
        x, y = float(fields[2]), float(fields[3])
        visible_value = int(fields[4])
        if part_id not in names or visible_value not in (0, 1):
            raise RuntimeError(f"invalid CUB part row at line {line_number}")
        image_points = points.setdefault(image_id, {})
        if part_id in image_points:
            raise RuntimeError(f"duplicate CUB image/part row: {image_id}/{part_id}")
        image_points[part_id] = (x, y, bool(visible_value))

    if set(boxes) != set(points):
        raise RuntimeError("CUB bounding-box and part annotation image-id sets differ")
    annotations: dict[int, CubSpatialAnnotation] = {}
    for image_id in sorted(boxes):
        image_points = points[image_id]
        if sorted(image_points) != list(range(1, PART_COUNT + 1)):
            raise RuntimeError(f"image {image_id} does not contain all 15 CUB parts")
        annotations[image_id] = CubSpatialAnnotation(
            image_id=image_id,
            bounding_box=boxes[image_id],
            points=tuple((image_points[index][0], image_points[index][1]) for index in names),
            visible=tuple(image_points[index][2] for index in names),
        )
    return tuple(names[index] for index in names), annotations


def select_lowest_image_id_per_class(
    records: Sequence[CubProbeRecord], *, num_classes: int = 200
) -> list[CubProbeRecord]:
    """Build a deterministic, result-independent one-image-per-class smoke subset."""

    selected: dict[int, CubProbeRecord] = {}
    for record in sorted(records, key=lambda item: item.image_id):
        selected.setdefault(record.label, record)
    if sorted(selected) != list(range(num_classes)):
        raise RuntimeError("smoke subset cannot provide one record for every CUB class")
    return [selected[label] for label in range(num_classes)]


def gaussian_part_targets(
    *,
    image_size: tuple[int, int],
    annotation: CubSpatialAnnotation,
    grid_size: int = GRID_SIZE,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fixed-sigma heatmaps while retaining original-coordinate geometry.

    CUB contains a small number of annotations that are marked visible even though
    their coordinates lie outside the image.  An image encoder cannot localize an
    out-of-frame point, so those landmarks receive zero weight just like an
    invisible landmark.  The original coordinates are retained for auditing and
    are never clipped into the image.
    """

    width, height = image_size
    if width <= 0 or height <= 0 or grid_size <= 0 or sigma <= 0.0:
        raise ValueError("image/grid dimensions and Gaussian sigma must be positive")
    points = torch.tensor(annotation.points, dtype=torch.float32)
    official_visible = torch.tensor(annotation.visible, dtype=torch.bool)
    in_image = (
        points[:, 0].ge(0.0)
        & points[:, 0].le(float(width))
        & points[:, 1].ge(0.0)
        & points[:, 1].le(float(height))
    )
    valid = official_visible & in_image

    x_center = points[:, 0] * grid_size / float(width) - 0.5
    y_center = points[:, 1] * grid_size / float(height) - 0.5
    axis = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    squared_distance = (
        (xx.unsqueeze(0) - x_center[:, None, None]).square()
        + (yy.unsqueeze(0) - y_center[:, None, None]).square()
    )
    heatmaps = torch.exp(-squared_distance / (2.0 * sigma * sigma))
    heatmaps[~valid] = 0.0
    image_size_tensor = torch.tensor([width, height], dtype=torch.float32)
    box = torch.tensor(annotation.bounding_box, dtype=torch.float32)
    return heatmaps, valid, points, image_size_tensor, box


def load_part_supervision(
    records: Sequence[CubProbeRecord],
    annotations: dict[int, CubSpatialAnnotation],
    *,
    grid_size: int = GRID_SIZE,
    sigma: float = 1.0,
) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = {
        "heatmaps": [],
        "visible": [],
        "official_visible": [],
        "excluded_out_of_frame_visible": [],
        "points": [],
        "image_sizes": [],
        "boxes": [],
    }
    for record in records:
        annotation = annotations.get(record.image_id)
        if annotation is None:
            raise RuntimeError(f"missing spatial annotation for image {record.image_id}")
        with Image.open(record.image_path) as image:
            image_size = image.size
        heatmaps, valid, points, image_size_tensor, box = gaussian_part_targets(
            image_size=image_size,
            annotation=annotation,
            grid_size=grid_size,
            sigma=sigma,
        )
        official_visible = torch.tensor(annotation.visible, dtype=torch.bool)
        values["heatmaps"].append(heatmaps)
        values["visible"].append(valid)
        values["official_visible"].append(official_visible)
        values["excluded_out_of_frame_visible"].append(
            official_visible & ~valid
        )
        values["points"].append(points)
        values["image_sizes"].append(image_size_tensor)
        values["boxes"].append(box)
    return {key: torch.stack(items) for key, items in values.items()}


def summarize_part_supervision(
    records: Sequence[CubProbeRecord], supervision: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Return a JSON-safe audit of visible landmarks excluded as out of frame."""

    sample_count = len(records)
    expected_shape = (sample_count, PART_COUNT)
    valid = supervision.get("visible")
    official_visible = supervision.get("official_visible")
    excluded = supervision.get("excluded_out_of_frame_visible")
    points = supervision.get("points")
    image_sizes = supervision.get("image_sizes")
    if (
        valid is None
        or official_visible is None
        or excluded is None
        or points is None
        or image_sizes is None
        or valid.shape != expected_shape
        or official_visible.shape != expected_shape
        or excluded.shape != expected_shape
        or points.shape != (sample_count, PART_COUNT, 2)
        or image_sizes.shape != (sample_count, 2)
    ):
        raise ValueError("part supervision audit tensors have unexpected shapes")
    if not torch.equal(official_visible, valid | excluded):
        raise RuntimeError("part supervision validity masks are inconsistent")
    if torch.logical_and(valid, excluded).any():
        raise RuntimeError("valid and excluded CUB part masks overlap")

    affected: list[dict[str, Any]] = []
    for sample_index in torch.nonzero(excluded.any(dim=1), as_tuple=False).flatten():
        index = int(sample_index.item())
        part_indices = torch.nonzero(excluded[index], as_tuple=False).flatten()
        affected.append(
            {
                "image_id": int(records[index].image_id),
                "part_ids": [int(value.item()) + 1 for value in part_indices],
                "coordinates": [
                    [float(value) for value in points[index, part_index].tolist()]
                    for part_index in part_indices
                ],
                "image_size": [float(value) for value in image_sizes[index].tolist()],
            }
        )
    no_valid = [
        int(records[index].image_id)
        for index in torch.nonzero(~valid.any(dim=1), as_tuple=False).flatten().tolist()
    ]
    return {
        "records": sample_count,
        "official_visible_keypoints": int(official_visible.sum().item()),
        "valid_visible_keypoints": int(valid.sum().item()),
        "excluded_out_of_frame_visible_keypoints": int(excluded.sum().item()),
        "affected_image_count": len(affected),
        "affected_image_ids": [row["image_id"] for row in affected],
        "excluded_landmarks": affected,
        "images_without_valid_keypoints": no_valid,
    }


class PartHeatmapProbe(nn.Conv2d):
    def __init__(self, channels: int = 192, parts: int = PART_COUNT) -> None:
        super().__init__(channels, parts, kernel_size=1, bias=True)


def build_part_probe(
    seed: int, *, channels: int = 192, parts: int = PART_COUNT
) -> PartHeatmapProbe:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        probe = PartHeatmapProbe(channels=channels, parts=parts)
        nn.init.normal_(probe.weight, mean=0.0, std=0.01)
        nn.init.zeros_(probe.bias)
    return probe


def masked_heatmap_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or visible.shape != prediction.shape[:2]:
        raise ValueError("part heatmap prediction/target/visibility shapes differ")
    weights = visible.to(dtype=prediction.dtype).unsqueeze(-1).unsqueeze(-1)
    denominator = weights.sum() * prediction.shape[-2] * prediction.shape[-1]
    if denominator.item() <= 0:
        raise RuntimeError("part heatmap batch has no visible landmarks")
    return ((prediction - target).square() * weights).sum() / denominator


def part_localization_metrics(
    logits: torch.Tensor,
    *,
    visible: torch.Tensor,
    points: torch.Tensor,
    image_sizes: torch.Tensor,
    boxes: torch.Tensor,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Decode patch centers and compute visible-keypoint PCK in the original frame."""

    if logits.ndim != 4 or logits.shape[1:] != (PART_COUNT, GRID_SIZE, GRID_SIZE):
        raise ValueError("expected part logits shaped [N,15,14,14]")
    sample_count = logits.shape[0]
    if (
        visible.shape != (sample_count, PART_COUNT)
        or points.shape != (sample_count, PART_COUNT, 2)
        or image_sizes.shape != (sample_count, 2)
        or boxes.shape != (sample_count, 4)
    ):
        raise ValueError("part geometry tensors have unexpected shapes")
    indices = logits.flatten(2).argmax(dim=-1)
    rows = torch.div(indices, GRID_SIZE, rounding_mode="floor")
    columns = indices.remainder(GRID_SIZE)
    predicted_x = (columns.to(torch.float32) + 0.5) * image_sizes[:, 0:1] / GRID_SIZE
    predicted_y = (rows.to(torch.float32) + 0.5) * image_sizes[:, 1:2] / GRID_SIZE
    predicted = torch.stack((predicted_x, predicted_y), dim=-1)
    distances = torch.linalg.vector_norm(predicted - points, dim=-1)
    normalizers = torch.maximum(boxes[:, 2], boxes[:, 3]).unsqueeze(1)
    normalized_error = distances / normalizers
    correct = normalized_error.le(threshold) & visible
    visible_count = int(visible.sum().item())
    if visible_count == 0:
        raise RuntimeError("part localization evaluation has no visible landmarks")
    part_counts = visible.sum(dim=0)
    part_correct = correct.sum(dim=0)
    per_part = [
        (
            float(part_correct[index].item() / part_counts[index].item())
            if part_counts[index]
            else None
        )
        for index in range(PART_COUNT)
    ]
    return {
        "visible_keypoints": visible_count,
        "correct_keypoints": int(correct.sum().item()),
        "micro_pck_at_0.1": float(correct.sum().item() / visible_count),
        "mean_normalized_localization_error": float(normalized_error[visible].mean().item()),
        "per_part_pck_at_0.1": per_part,
        "per_part_visible_counts": [int(value) for value in part_counts.tolist()],
    }


@torch.no_grad()
def evaluate_part_probe(
    probe: nn.Module,
    features: torch.Tensor,
    supervision: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    chunks: list[torch.Tensor] = []
    probe.eval()
    for start in range(0, len(features), batch_size):
        chunks.append(probe(features[start : start + batch_size].to(device)).cpu())
    logits = torch.cat(chunks)
    return part_localization_metrics(
        logits,
        visible=supervision["visible"],
        points=supervision["points"],
        image_sizes=supervision["image_sizes"],
        boxes=supervision["boxes"],
    )


def _state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def train_part_candidate(
    *,
    train_features: torch.Tensor,
    train_supervision: dict[str, torch.Tensor],
    validation_features: torch.Tensor,
    validation_supervision: dict[str, torch.Tensor],
    learning_rate: float,
    epochs: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train one deterministic frozen-feature linear heatmap candidate."""

    if learning_rate <= 0.0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("part probe LR, epochs, and batch size must be positive")
    probe = build_part_probe(seed).to(device)
    initial_state_sha256 = _state_sha256(probe)
    optimizer = torch.optim.SGD(
        probe.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )
    generator = torch.Generator().manual_seed(seed)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        probe.train()
        order = torch.randperm(len(train_features), generator=generator)
        running_loss = 0.0
        visible_total = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            feature = train_features[indices].to(device)
            target = train_supervision["heatmaps"][indices].to(device)
            visible = train_supervision["visible"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(feature)
            loss = masked_heatmap_mse(prediction, target, visible)
            loss.backward()
            optimizer.step()
            count = int(visible.sum().item())
            running_loss += float(loss.detach().item()) * count
            visible_total += count
        metrics = evaluate_part_probe(
            probe,
            validation_features,
            validation_supervision,
            device=device,
            batch_size=batch_size,
        )
        row = {
            "epoch": epoch,
            "train_visible_weighted_loss": running_loss / visible_total,
            "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
            "validation": metrics,
        }
        history.append(row)
        score = metrics["micro_pck_at_0.1"]
        if best is None or score > best["validation"]["micro_pck_at_0.1"]:
            best = {
                "epoch": epoch,
                "validation": metrics,
                "probe_state": {
                    key: value.detach().cpu().clone()
                    for key, value in probe.state_dict().items()
                },
            }
        scheduler.step()
    assert best is not None
    return {
        "learning_rate": learning_rate,
        "seed": seed,
        "epochs": epochs,
        "initial_probe_state_sha256": initial_state_sha256,
        "best_epoch": best["epoch"],
        "best_validation": best["validation"],
        "probe_state": best["probe_state"],
        "history": history,
    }


class LinearCKAAccumulator:
    """Exact centered linear CKA using streaming sufficient statistics."""

    def __init__(
        self,
        x_channels: int,
        y_channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.count = 0
        self.sum_x = torch.zeros(x_channels, dtype=dtype, device=device)
        self.sum_y = torch.zeros(y_channels, dtype=dtype, device=device)
        self.xtx = torch.zeros(x_channels, x_channels, dtype=dtype, device=device)
        self.yty = torch.zeros(y_channels, y_channels, dtype=dtype, device=device)
        self.xty = torch.zeros(x_channels, y_channels, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("CKA updates require paired 2-D representation matrices")
        if x.shape[1] != self.sum_x.numel() or y.shape[1] != self.sum_y.numel():
            raise ValueError("CKA update channel count changed")
        x = x.to(device=self.sum_x.device, dtype=self.sum_x.dtype)
        y = y.to(device=self.sum_y.device, dtype=self.sum_y.dtype)
        self.count += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.xtx += x.T @ x
        self.yty += y.T @ y
        self.xty += x.T @ y

    @torch.no_grad()
    def compute(self) -> float:
        if self.count < 2:
            raise RuntimeError("CKA requires at least two observations")
        scale = 1.0 / self.count
        centered_xtx = self.xtx - torch.outer(self.sum_x, self.sum_x) * scale
        centered_yty = self.yty - torch.outer(self.sum_y, self.sum_y) * scale
        centered_xty = self.xty - torch.outer(self.sum_x, self.sum_y) * scale
        numerator = centered_xty.square().sum()
        denominator = torch.sqrt(
            centered_xtx.square().sum() * centered_yty.square().sum()
        )
        if not torch.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("CKA denominator is zero or non-finite")
        value = numerator / denominator
        if not torch.isfinite(value):
            raise RuntimeError("CKA result is non-finite")
        scalar = float(value.item())
        if -1e-12 <= scalar <= 1.0 + 1e-12:
            return min(1.0, max(0.0, scalar))
        return scalar


@torch.no_grad()
def attention_rollout(student: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return normalized CLS-to-patch rollout maps for a standard DeiT encoder."""

    blocks = list(student.blocks)
    if len(blocks) != 12 or getattr(student, "num_prefix_tokens", None) != 1:
        raise RuntimeError("attention rollout expects 12-block DeiT with one CLS token")
    captured: list[torch.Tensor | None] = [None] * len(blocks)
    handles = []
    fused_values: list[bool | None] = []

    def make_hook(index: int):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[index] = output.detach()

        return hook

    try:
        for index, block in enumerate(blocks):
            fused = getattr(block.attn, "fused_attn", None)
            fused_values.append(fused)
            if fused is not None:
                block.attn.fused_attn = False
            handles.append(block.attn.attn_drop.register_forward_hook(make_hook(index)))
        student(images)
    finally:
        for handle in handles:
            handle.remove()
        for block, fused in zip(blocks, fused_values, strict=True):
            if fused is not None:
                block.attn.fused_attn = fused

    if any(value is None for value in captured):
        raise RuntimeError("failed to capture all twelve DeiT attention tensors")
    attention = [value for value in captured if value is not None]
    token_count = attention[0].shape[-1]
    identity = torch.eye(token_count, dtype=attention[0].dtype, device=images.device)
    joint = identity.unsqueeze(0).expand(images.shape[0], -1, -1)
    for value in attention:
        if (
            value.ndim != 4
            or value.shape[0] != images.shape[0]
            or value.shape[-2:] != (token_count, token_count)
        ):
            raise RuntimeError("captured DeiT attention shape changed across layers")
        augmented = value.mean(dim=1) + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        joint = augmented @ joint
    rollout = joint[:, 0, 1:].reshape(images.shape[0], GRID_SIZE, GRID_SIZE)
    return rollout / rollout.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)


def load_mask_views(
    record: CubProbeRecord,
    *,
    input_size: int = INPUT_SIZE,
    grid_size: int = GRID_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(record.image_path) as image:
        image_size = image.size
    with Image.open(record.mask_path) as mask_image:
        if mask_image.size != image_size:
            raise RuntimeError(f"CUB image/mask size mismatch for image {record.image_id}")
        resized = TF.resize(
            mask_image.convert("L"),
            [input_size, input_size],
            interpolation=InterpolationMode.NEAREST,
        )
        mask = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).gt(0)
    if input_size % grid_size:
        raise ValueError("attention mask input size must be divisible by patch grid")
    patch = input_size // grid_size
    occupancy = (
        mask.reshape(grid_size, patch, grid_size, patch)
        .permute(0, 2, 1, 3)
        .reshape(grid_size, grid_size, -1)
        .float()
        .mean(dim=-1)
    )
    return mask, occupancy


def binary_average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Threshold-grouped, non-interpolated binary average precision."""

    scores = scores.detach().flatten().to(torch.float64).cpu()
    labels = labels.detach().flatten().to(torch.bool).cpu()
    if scores.numel() != labels.numel() or scores.numel() == 0:
        raise ValueError("AP scores and labels must be non-empty and equally sized")
    positives = int(labels.sum().item())
    if positives == 0:
        raise RuntimeError("average precision is undefined without positives")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order].to(torch.float64)
    true_positive = torch.cumsum(sorted_labels, dim=0)
    false_positive = torch.cumsum(1.0 - sorted_labels, dim=0)
    distinct_ends = torch.ones_like(sorted_scores, dtype=torch.bool)
    distinct_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    true_positive = true_positive[distinct_ends]
    false_positive = false_positive[distinct_ends]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positives
    previous_recall = torch.cat((torch.zeros(1, dtype=recall.dtype), recall[:-1]))
    return float(((recall - previous_recall) * precision).sum().item())


def attention_gt_metrics(
    rollout: torch.Tensor,
    masks_224: torch.Tensor,
    occupancies: torch.Tensor,
) -> dict[str, Any]:
    if rollout.ndim != 3 or rollout.shape[1:] != (GRID_SIZE, GRID_SIZE):
        raise ValueError("rollout must have shape [N,14,14]")
    if masks_224.shape != (len(rollout), INPUT_SIZE, INPUT_SIZE):
        raise ValueError("attention masks must have shape [N,224,224]")
    if occupancies.shape != rollout.shape:
        raise ValueError("attention occupancy shape differs from rollout")
    patch_labels = occupancies.ge(0.5)
    micro_ap = binary_average_precision(rollout, patch_labels)
    foreground_mass = (rollout * occupancies).sum(dim=(1, 2))
    upsampled = F.interpolate(
        rollout.unsqueeze(1),
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    peak = upsampled.flatten(1).argmax(dim=1)
    peak_inside = masks_224.flatten(1).gather(1, peak[:, None]).squeeze(1)
    return {
        "images": len(rollout),
        "patches": int(rollout.numel()),
        "positive_patches": int(patch_labels.sum().item()),
        "global_micro_patch_average_precision": micro_ap,
        "pointing_game_peak_inside_mask": float(peak_inside.float().mean().item()),
        "foreground_attention_mass_mean": float(foreground_mass.mean().item()),
    }


def save_attention_triptych(
    *,
    record: CubProbeRecord,
    rollout: torch.Tensor,
    destination: Path,
) -> None:
    """Save original, GT-overlay, and rollout-overlay panels without extra deps."""

    with Image.open(record.image_path) as handle:
        original = handle.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    mask, _ = load_mask_views(record)
    original_array = np.asarray(original, dtype=np.float32)
    mask_array = mask.numpy()[..., None].astype(np.float32)
    gt_array = original_array * (1.0 - 0.35 * mask_array)
    gt_array[..., 1] += 255.0 * 0.35 * mask_array[..., 0]

    attention = F.interpolate(
        rollout[None, None].float(),
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    attention = attention / attention.max().clamp_min(1e-12)
    alpha = (0.65 * attention.numpy())[..., None]
    heat_array = original_array * (1.0 - alpha)
    heat_array[..., 0] += 255.0 * alpha[..., 0]
    triptych = np.concatenate(
        (original_array, np.clip(gt_array, 0, 255), np.clip(heat_array, 0, 255)),
        axis=1,
    ).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(triptych, mode="RGB").save(destination)


def feature_map_to_observations(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[-2:] != (GRID_SIZE, GRID_SIZE):
        raise ValueError("spatial CKA feature must have a 14x14 grid")
    return feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])


def assert_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"{name} is not a finite probability: {value}")
