from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from ._image import NormalizedImage, normalize_image
from .models import MaskConfig, MaskMethod, MaskMode, MaskResult


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _quantize_luminance(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _otsu_threshold_u8(values: np.ndarray) -> int:
    histogram = np.bincount(values.ravel(), minlength=256)
    if histogram.sum() == 0:
        return 0
    probability = histogram.astype(float) / histogram.sum()
    levels = np.arange(256, dtype=float)
    background_weight = np.cumsum(probability)
    background_mean = np.cumsum(probability * levels)
    total_mean = background_mean[-1]
    denominator = background_weight * (1.0 - background_weight)
    between_variance = np.divide(
        (total_mean * background_weight - background_mean) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return int(np.argmax(between_variance))


def _chroma(rgb: np.ndarray, config: MaskConfig) -> tuple[np.ndarray, np.ndarray]:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    colored = (saturation >= config.saturation_threshold) & (maximum < config.color_max_value)
    return saturation, colored


def _paper_threshold(luminance: np.ndarray, delta: float, config: MaskConfig) -> float:
    paper_level = float(np.percentile(luminance, config.paper_percentile))
    return max(0.0, min(config.threshold_max, paper_level - delta))


def _dark_mask(luminance: np.ndarray, config: MaskConfig) -> tuple[np.ndarray, float]:
    threshold = max(
        config.dark_threshold_min,
        min(config.threshold_max, _paper_threshold(luminance, config.dark_delta, config)),
    )
    return luminance <= threshold, threshold


def _neutral_dark_mask(
    luminance: np.ndarray, saturation: np.ndarray, config: MaskConfig
) -> tuple[np.ndarray, float]:
    threshold = _paper_threshold(luminance, config.neutral_dark_delta, config)
    mask = (saturation < config.saturation_threshold) & (luminance <= threshold)
    return mask, threshold


def _adaptive_mask(luminance: np.ndarray, config: MaskConfig) -> np.ndarray:
    window_size = config.adaptive_window_size
    offset = config.adaptive_offset
    height, width = luminance.shape
    radius = window_size // 2
    integral = np.pad(luminance, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    y0 = np.maximum(np.arange(height) - radius, 0)
    y1 = np.minimum(np.arange(height) + radius + 1, height)
    x0 = np.maximum(np.arange(width) - radius, 0)
    x1 = np.minimum(np.arange(width) + radius + 1, width)
    total = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    count = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    local_mean = total / count
    global_ceiling = min(
        config.adaptive_global_ceiling_max,
        float(np.percentile(luminance, config.paper_percentile)) - config.adaptive_global_delta,
    )
    threshold = np.minimum(local_mean - offset, global_ceiling)
    return luminance <= threshold


def _candidate_score(
    mask: np.ndarray,
    luminance: np.ndarray,
    saturation: np.ndarray,
    config: MaskConfig,
    *,
    neutral_fraction: float = 0.0,
) -> float:
    fraction = float(mask.mean())
    if fraction <= 0.0 or fraction > config.maximum_foreground_fraction:
        return -1.0

    foreground_luminance = float(np.median(luminance[mask]))
    background = ~mask
    if np.any(background):
        contrast = float(
            np.clip(
                (np.median(luminance[background]) - foreground_luminance) / 128.0,
                0,
                1,
            )
        )
    else:
        contrast = 0.0
    fraction_score = float(
        np.clip(1.0 - abs(fraction - config.target_foreground_fraction) / 0.25, 0, 1)
    )
    coverage_score = min(fraction / 0.12, 1.0)
    chroma_score = float(np.median(saturation[mask]))

    foreground_count = int(mask.sum())
    border_count = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    border_penalty = min(border_count / max(1, foreground_count), 1.0)
    long_rows = mask.mean(axis=1) >= config.long_structure_occupancy
    long_columns = mask.mean(axis=0) >= config.long_structure_occupancy
    long_structure = (mask & (long_rows[:, None] | long_columns[None, :])).sum() / foreground_count
    neutral_support = min(neutral_fraction / config.neutral_support_fraction, 1.0)

    return (
        0.45 * contrast
        + 0.20 * fraction_score
        + 0.20 * coverage_score
        + 0.15 * chroma_score
        - 0.30 * border_penalty
        - 0.35 * float(long_structure)
        + 0.40 * neutral_support
    )


def _select_method(
    rgb: np.ndarray,
    luminance: np.ndarray,
    saturation: np.ndarray,
    colored: np.ndarray,
    config: MaskConfig,
) -> tuple[MaskMethod, np.ndarray, float | None]:
    luminance_u8 = _quantize_luminance(luminance)
    grayscale_threshold = _otsu_threshold_u8(luminance_u8)
    dark, dark_threshold = _dark_mask(luminance, config)
    candidates: dict[MaskMethod, tuple[np.ndarray, float | None]] = {
        "grayscale": (luminance_u8 <= grayscale_threshold, float(grayscale_threshold)),
        "dark": (dark, dark_threshold),
    }

    minimum_pixels = max(
        4,
        int(rgb.shape[0] * rgb.shape[1] * config.minimum_foreground_fraction),
    )
    colored_count = int(colored.sum())
    if colored_count >= minimum_pixels:
        candidates["color"] = (colored, None)
        neutral, neutral_threshold = _neutral_dark_mask(luminance, saturation, config)
        neutral_fraction = float(neutral.mean())
        if int(neutral.sum()) >= minimum_pixels:
            candidates["hybrid"] = (colored | neutral, neutral_threshold)
    else:
        neutral_fraction = 0.0

    scored = {
        method: _candidate_score(
            mask,
            luminance,
            saturation,
            config,
            neutral_fraction=neutral_fraction if method == "hybrid" else 0.0,
        )
        for method, (mask, _) in candidates.items()
    }
    priority = {"hybrid": 3, "color": 2, "grayscale": 1, "dark": 0}
    selected = max(candidates, key=lambda method: (scored[method], priority[method]))
    mask, threshold = candidates[selected]
    return selected, mask, threshold


def _quality(
    mask: np.ndarray,
    luminance: np.ndarray,
    saturation: np.ndarray,
    method: str,
    config: MaskConfig,
) -> float:
    if not np.any(mask):
        return 0.0
    ink_luminance = float(np.median(luminance[mask]))
    background = ~mask
    if np.any(background):
        contrast = float(np.median(luminance[background])) - ink_luminance
        contrast_confidence = np.clip(contrast / config.contrast_confidence_scale, 0.0, 1.0)
    else:
        contrast_confidence = 0.0
    if method == "color":
        chroma_confidence = np.clip(
            float(np.median(saturation[mask])) / config.color_confidence_saturation_scale,
            0.0,
            1.0,
        )
        return float(0.5 * contrast_confidence + 0.5 * chroma_confidence)
    return float(contrast_confidence)


def _warnings(mask: np.ndarray, confidence: float, config: MaskConfig) -> tuple[str, ...]:
    warnings: list[str] = []
    fraction = float(mask.mean())
    if not np.any(mask):
        warnings.append("No foreground pixels were detected.")
    elif fraction > config.maximum_foreground_fraction:
        warnings.append("The configured maximum foreground fraction was exceeded.")
    elif confidence < 0.25:
        warnings.append("Foreground/background separation confidence is low.")
    return tuple(warnings)


def foreground_mask(
    image: str | Path | Image.Image | np.ndarray | NormalizedImage,
    mode: MaskMode = "auto",
    *,
    adaptive_window_size: int | None = None,
    adaptive_offset: float | None = None,
    config: MaskConfig | None = None,
) -> MaskResult:
    """Extract a boolean foreground mask using a deterministic thresholding mode.

    ``auto`` evaluates grayscale, dark-relative, color, and hybrid candidates.
    ``color`` remains color-only. ``adaptive`` uses a local mean threshold for
    images with uneven illumination.
    """
    if mode not in {"auto", "grayscale", "dark", "color", "hybrid", "adaptive"}:
        raise ValueError("mode must be auto, grayscale, dark, color, hybrid, or adaptive")
    mask_config = config or MaskConfig()
    if adaptive_window_size is not None or adaptive_offset is not None:
        mask_config = replace(
            mask_config,
            adaptive_window_size=(
                adaptive_window_size
                if adaptive_window_size is not None
                else mask_config.adaptive_window_size
            ),
            adaptive_offset=(
                adaptive_offset if adaptive_offset is not None else mask_config.adaptive_offset
            ),
        )
    rgb = image.rgb if isinstance(image, NormalizedImage) else normalize_image(image).rgb
    luminance = _luminance(rgb)
    saturation, colored = _chroma(rgb, mask_config)
    threshold: float | None = None

    if mode == "auto":
        selected, mask, threshold = _select_method(rgb, luminance, saturation, colored, mask_config)
    elif mode == "grayscale":
        luminance_u8 = _quantize_luminance(luminance)
        threshold = float(_otsu_threshold_u8(luminance_u8))
        selected = "grayscale"
        mask = luminance_u8 <= threshold
    elif mode == "dark":
        selected = "dark"
        mask, threshold = _dark_mask(luminance, mask_config)
    elif mode == "color":
        selected = "color"
        mask = colored
    elif mode == "hybrid":
        selected = "hybrid"
        neutral, threshold = _neutral_dark_mask(luminance, saturation, mask_config)
        mask = colored | neutral
    else:
        selected = "adaptive"
        mask = _adaptive_mask(luminance, mask_config)

    mask = np.asarray(mask, dtype=bool)
    confidence = _quality(mask, luminance, saturation, selected, mask_config)
    return MaskResult(
        mask,
        confidence,
        selected,
        _warnings(mask, confidence, mask_config),
        threshold=threshold,
    )
