"""Versioned CUB student-loader profiles and auditable crop geometry.

The completed Phase-1 v3 runs keep using ``l0_current_strong``.  The other
profiles belong to the separately versioned post-hoc loader experiment and
must be named explicitly by a pilot or selected-loader follow-up runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data import IMAGENET_MEAN, IMAGENET_STD


L0_CURRENT_STRONG = "l0_current_strong"
L1_MATCHED_WEAK = "l1_matched_weak"
L2_CONSERVATIVE_SPATIAL = "l2_conservative_spatial"
LOADER_PROFILE_ORDER = (
    L0_CURRENT_STRONG,
    L1_MATCHED_WEAK,
    L2_CONSERVATIVE_SPATIAL,
)


@dataclass(frozen=True)
class CropGeometry:
    scale: tuple[float, float]
    ratio: tuple[float, float]

    @property
    def identity(self) -> str:
        return (
            f"rrc_scale_{self.scale[0]:g}_{self.scale[1]:g}_"
            f"ratio_{self.ratio[0]:g}_{self.ratio[1]:g}"
        )


_GEOMETRY = {
    L0_CURRENT_STRONG: CropGeometry((0.08, 1.0), (3.0 / 4.0, 4.0 / 3.0)),
    L1_MATCHED_WEAK: CropGeometry((0.08, 1.0), (3.0 / 4.0, 4.0 / 3.0)),
    L2_CONSERVATIVE_SPATIAL: CropGeometry((0.5, 1.0), (3.0 / 4.0, 4.0 / 3.0)),
}


def validate_loader_profile(profile: str) -> str:
    if profile not in LOADER_PROFILE_ORDER:
        raise ValueError(
            f"unknown CUB loader profile {profile!r}; expected one of "
            f"{LOADER_PROFILE_ORDER}"
        )
    return profile


def crop_geometry(profile: str) -> CropGeometry:
    return _GEOMETRY[validate_loader_profile(profile)]


def loader_profile_contract(profile: str) -> dict[str, Any]:
    profile = validate_loader_profile(profile)
    geometry = crop_geometry(profile)
    strong = profile == L0_CURRENT_STRONG
    return {
        "profile": profile,
        "input_size": 224,
        "geometry_id": geometry.identity,
        "random_resized_crop": {
            "scale": list(geometry.scale),
            "ratio": list(geometry.ratio),
            "interpolation": "bicubic",
        },
        "horizontal_flip_probability": 0.5,
        "color_jitter_argument": 0.4 if strong else None,
        "auto_augment": "rand-m9-mstd0.5-inc1" if strong else None,
        "random_erasing_probability": 0.25 if strong else 0.0,
        "normalization": "imagenet",
        "teacher_student_random_geometry_shared": True,
        "classification_annotations_used": ["rgb_image", "species_label"],
        "spatial_annotations_used_for_training": False,
    }


def build_cub_student_train_transform(profile: str) -> Any:
    """Build one explicit CUB student transform without changing the v3 default."""

    profile = validate_loader_profile(profile)
    if profile == L0_CURRENT_STRONG:
        import timm

        return timm.data.create_transform(
            input_size=(3, 224, 224),
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation="bicubic",
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    geometry = crop_geometry(profile)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=geometry.scale,
                ratio=geometry.ratio,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def sample_random_resized_crop(
    image: Image.Image,
    *,
    profile: str,
    generator: torch.Generator,
) -> tuple[int, int, int, int]:
    """Sample torchvision-compatible ``top, left, height, width`` parameters.

    An explicit generator makes the annotation-only audit independent of the
    training process RNG and lets profiles with the same geometry use exactly
    paired crop samples.
    """

    geometry = crop_geometry(profile)
    width, height = image.size
    area = height * width
    log_ratio = torch.log(torch.tensor(geometry.ratio, dtype=torch.float64))

    for _ in range(10):
        target_area = area * float(
            torch.empty((), dtype=torch.float64).uniform_(
                geometry.scale[0], geometry.scale[1], generator=generator
            )
        )
        aspect_ratio = math.exp(
            float(
                torch.empty((), dtype=torch.float64).uniform_(
                    float(log_ratio[0]), float(log_ratio[1]), generator=generator
                )
            )
        )
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(
                torch.randint(
                    0, height - crop_height + 1, (), generator=generator
                ).item()
            )
            left = int(
                torch.randint(
                    0, width - crop_width + 1, (), generator=generator
                ).item()
            )
            return top, left, crop_height, crop_width

    input_ratio = width / height
    if input_ratio < min(geometry.ratio):
        crop_width = width
        crop_height = int(round(crop_width / min(geometry.ratio)))
    elif input_ratio > max(geometry.ratio):
        crop_height = height
        crop_width = int(round(crop_height * max(geometry.ratio)))
    else:
        crop_width, crop_height = width, height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return top, left, crop_height, crop_width
