from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class NormalizedImage:
    image: Image.Image = field(repr=False, compare=False)
    rgb: np.ndarray = field(repr=False, compare=False)
    source_path: Path | None = None


def _normalized_pixels(
    pixels: np.ndarray,
    *,
    alpha_background: tuple[int, int, int],
    source_path: Path | None,
) -> NormalizedImage:
    array = np.asarray(pixels)
    if array.ndim not in (2, 3) or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("image must be non-empty and have grayscale, RGB, or RGBA dimensions")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError("image must be grayscale, RGB, or RGBA")
    if array.dtype.kind not in "uif":
        raise ValueError("image pixels must be numeric")

    values = np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("image pixels must be finite")
    if np.issubdtype(array.dtype, np.floating) and values.max(initial=0.0) <= 1.0:
        values = values * 255.0
    values = np.clip(values, 0.0, 255.0)

    if values.ndim == 2:
        rgb = np.repeat(values[:, :, None], 3, axis=2)
    elif values.shape[2] == 4:
        alpha = values[:, :, 3:4] / 255.0
        background = np.asarray(alpha_background, dtype=np.float32).reshape(1, 1, 3)
        rgb = values[:, :, :3] * alpha + background * (1.0 - alpha)
    else:
        rgb = values

    rgb = np.array(np.clip(rgb, 0.0, 255.0), dtype=np.float32, copy=True)
    rgb.setflags(write=False)
    image = Image.fromarray(np.rint(rgb).astype(np.uint8))
    return NormalizedImage(image=image, rgb=rgb, source_path=source_path)


def _normalize_pil(
    image: Image.Image,
    *,
    alpha_background: tuple[int, int, int],
    source_path: Path | None,
) -> NormalizedImage:
    oriented = ImageOps.exif_transpose(image)
    rgba = np.asarray(oriented.convert("RGBA"), dtype=np.uint8)
    return _normalized_pixels(
        rgba,
        alpha_background=alpha_background,
        source_path=source_path,
    )


def normalize_image(
    source: str | Path | Image.Image | np.ndarray,
    *,
    alpha_background: tuple[int, int, int] = (255, 255, 255),
) -> NormalizedImage:
    """Normalize a raster source to orientation-corrected RGB pixels."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        with Image.open(path) as image:
            return _normalize_pil(
                image,
                alpha_background=alpha_background,
                source_path=path,
            )
    if isinstance(source, Image.Image):
        return _normalize_pil(
            source,
            alpha_background=alpha_background,
            source_path=None,
        )
    return _normalized_pixels(
        np.asarray(source),
        alpha_background=alpha_background,
        source_path=None,
    )
