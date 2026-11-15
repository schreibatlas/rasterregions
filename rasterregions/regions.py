from __future__ import annotations

import numpy as np
from PIL import Image

from .models import BBox


def tighten_bbox(mask: np.ndarray, bbox: BBox, *, pad: int = 0) -> BBox:
    """Tighten ``bbox`` to foreground pixels and keep padding inside that bbox.

    If the bbox contains no foreground pixels, it is returned unchanged."""
    if pad < 0:
        raise ValueError("pad must be nonnegative")
    ink = np.asarray(mask, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = ink.shape
    if bbox.x1 > width or bbox.y1 > height:
        raise ValueError("bbox exceeds mask bounds")
    local = ink[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    ys, xs = np.nonzero(local)
    if not len(xs):
        return bbox
    return BBox(
        max(bbox.x0, bbox.x0 + int(xs.min()) - pad),
        max(bbox.y0, bbox.y0 + int(ys.min()) - pad),
        min(bbox.x1, bbox.x0 + int(xs.max()) + pad + 1),
        min(bbox.y1, bbox.y0 + int(ys.max()) + pad + 1),
    )


def crop(image: Image.Image, bbox: BBox) -> Image.Image:
    """Return a copy of the image inside a half-open pixel bbox, without resizing."""
    if bbox.x1 > image.width or bbox.y1 > image.height:
        raise ValueError("bbox exceeds image bounds")
    return image.crop(bbox.as_tuple()).copy()
