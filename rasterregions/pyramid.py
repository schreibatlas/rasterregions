"""Coarse-to-fine foreground occupancy pyramids."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import numpy as np
from PIL import Image

from .models import ForegroundEvidence, ForegroundPyramid, ForegroundPyramidLevel


def foreground_pyramid(
    evidence: ForegroundEvidence,
    *,
    scales: Sequence[float] = (0.25, 0.5, 1.0),
    mask_threshold: float = 0.5,
) -> ForegroundPyramid:
    """Build deterministic coarse-to-fine occupancy levels from source evidence.

    Coarse coverage represents fractional occupancy of the source foreground
    mask. The exact full-resolution level is retained as the final level and
    remains authoritative for topology.
    """
    if not isinstance(evidence, ForegroundEvidence):
        raise ValueError("evidence must be a ForegroundEvidence instance")
    if isinstance(mask_threshold, bool) or not isfinite(mask_threshold):
        raise ValueError("mask_threshold must be finite and between zero and one")
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be finite and between zero and one")
    if evidence.mask.ndim != 2 or min(evidence.mask.shape) <= 0:
        raise ValueError("evidence mask dimensions must be positive")

    try:
        requested_scales = tuple(scales)
    except TypeError as exc:
        raise ValueError("scales must be a nonempty sequence of finite values in (0, 1]") from exc
    if not requested_scales:
        raise ValueError("scales must not be empty")

    normalized_scales: list[float] = []
    for scale in requested_scales:
        if isinstance(scale, (bool, np.bool_)):
            raise ValueError("scales must be finite values in (0, 1]")
        try:
            normalized = float(scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scales must be finite values in (0, 1]") from exc
        if not isfinite(normalized) or not 0.0 < normalized <= 1.0:
            raise ValueError("scales must be finite values in (0, 1]")
        if normalized_scales and normalized <= normalized_scales[-1]:
            raise ValueError("scales must be strictly increasing")
        normalized_scales.append(normalized)
    if normalized_scales[-1] < 1.0:
        normalized_scales.append(1.0)

    source_height, source_width = evidence.mask.shape
    source = evidence.mask.astype(np.float32)
    levels: list[ForegroundPyramidLevel] = []
    seen_shapes: set[tuple[int, int]] = set()

    for scale in normalized_scales:
        width = max(1, round(source_width * scale))
        height = max(1, round(source_height * scale))
        shape = (height, width)
        if shape == (source_height, source_width) and scale < 1.0:
            continue
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)

        if scale == 1.0:
            coverage = evidence.mask.astype(np.float32, copy=True)
        else:
            resized = Image.fromarray(source).resize((width, height), resample=Image.Resampling.BOX)
            coverage = np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)

        levels.append(
            ForegroundPyramidLevel(
                index=len(levels),
                source_shape=(source_height, source_width),
                shape=shape,
                scale_x=width / source_width,
                scale_y=height / source_height,
                coverage=coverage,
                mask_threshold=mask_threshold,
            )
        )

    return ForegroundPyramid(evidence=evidence, levels=tuple(levels))
