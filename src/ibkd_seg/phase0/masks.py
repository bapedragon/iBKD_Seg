"""Recover binary foreground masks from the official blue-screen composites."""

from __future__ import annotations

import numpy as np


BLUE_BACKGROUND_RGB = np.asarray((0, 0, 255), dtype=np.float32)


def _validate_rgb_pair(original: np.ndarray, composite: np.ndarray) -> None:
    if original.shape != composite.shape:
        raise ValueError(
            "original and composite must have identical shapes: "
            f"{original.shape} != {composite.shape}"
        )
    if original.ndim != 3 or original.shape[-1] != 3:
        raise ValueError("original and composite must be H x W x 3 RGB arrays")


def recover_foreground_alpha(
    original: np.ndarray,
    composite: np.ndarray,
    background_rgb: np.ndarray = BLUE_BACKGROUND_RGB,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Estimate alpha in ``composite = alpha*original + (1-alpha)*blue``.

    The released Oxford files are JPEG blue-screen composites rather than
    exact class-index masks. Least-squares alpha recovery uses both the source
    image and composite and is less brittle than exact RGB equality under JPEG
    compression. Pixels whose source color is indistinguishable from the blue
    key are conservatively assigned to background.
    """

    _validate_rgb_pair(original, composite)
    source = np.asarray(original, dtype=np.float32)
    observed = np.asarray(composite, dtype=np.float32)
    background = np.asarray(background_rgb, dtype=np.float32)
    if background.shape != (3,):
        raise ValueError("background_rgb must contain exactly three RGB values")
    direction = source - background
    denominator = np.square(direction).sum(axis=-1)
    numerator = ((observed - background) * direction).sum(axis=-1)
    alpha = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > epsilon,
    )
    return np.clip(alpha, 0.0, 1.0)


def binary_foreground_mask(
    original: np.ndarray,
    composite: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a boolean flower mask using the protocol-locked alpha threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    return recover_foreground_alpha(original, composite) >= threshold
