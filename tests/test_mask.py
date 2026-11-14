from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import foreground_mask


def test_auto_color_mask_ignores_gray_rule() -> None:
    image = Image.new("RGB", (100, 70), "white")
    draw = ImageDraw.Draw(image)
    draw.line((4, 35, 95, 35), fill=(180, 180, 180), width=1)
    draw.line((20, 50, 58, 50), fill=(20, 70, 220), width=5)

    result = foreground_mask(image)

    assert result.method == "color"
    assert result.mask[50, 30]
    assert not result.mask[35, 30]
    assert result.confidence > 0.5


def test_auto_grayscale_mask_finds_black_foreground() -> None:
    image = Image.new("RGB", (50, 50), "white")
    ImageDraw.Draw(image).line((25, 5, 25, 44), fill="black", width=3)

    result = foreground_mask(image)

    assert result.method == "grayscale"
    assert result.mask[5:45, 25].all()
    assert result.confidence > 0.9


def test_dark_mode_handles_gray_background() -> None:
    image = np.full((40, 40), 180, dtype=np.uint8)
    image[8:32, 19:22] = 20

    result = foreground_mask(image, mode="dark")

    assert result.method == "dark"
    assert result.mask[8:32, 20].all()
    assert not result.mask[0, 0]
