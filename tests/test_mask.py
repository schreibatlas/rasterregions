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


def test_grayscale_otsu_uses_consistent_quantization() -> None:
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image[10:30, 18:22] = 15

    result = foreground_mask(image, mode="grayscale")

    assert result.mask[10:30, 18:22].all()
    assert result.threshold == 15.0
    assert not result.mask[0, 0]


def test_auto_preserves_black_and_colored_foreground_without_gray_guide() -> None:
    image = Image.new("RGB", (120, 70), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 18, 30, 48), fill="black")
    draw.rectangle((49, 18, 67, 48), fill=(20, 70, 220))
    draw.rectangle((86, 18, 104, 48), fill="black")
    draw.line((3, 60, 116, 60), fill=(180, 180, 180), width=1)

    result = foreground_mask(image, mode="auto")

    assert result.method == "hybrid"
    assert result.threshold == 175.0
    assert result.mask[30, 20]
    assert result.mask[30, 58]
    assert result.mask[30, 95]
    assert not result.mask[60, 50]


def test_rgba_mask_composites_transparent_pixels_onto_white() -> None:
    image = np.zeros((40, 40, 4), dtype=np.uint8)
    image[10:30, 18:22, 3] = 255

    result = foreground_mask(image, mode="grayscale")

    assert result.mask[10:30, 18:22].all()
    assert not result.mask[0, 0]


def test_color_mode_remains_color_only() -> None:
    image = Image.new("RGB", (80, 50), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 25, 35), fill="black")
    draw.rectangle((45, 10, 60, 35), fill=(20, 70, 220))

    result = foreground_mask(image, mode="color")

    assert result.method == "color"
    assert result.mask[20, 50]
    assert not result.mask[20, 15]


def test_adaptive_mode_handles_local_background_levels() -> None:
    gradient = np.tile(np.linspace(220, 255, 100, dtype=np.uint8), (80, 1))
    gradient[25:55, 48:52] = 30

    result = foreground_mask(gradient, mode="adaptive", adaptive_window_size=31)

    assert result.method == "adaptive"
    assert result.threshold is None
    assert result.mask[25:55, 48:52].all()
    assert not result.mask[10, 10]
    assert not result.mask[10, 90]


def test_mask_path_pil_and_array_inputs_share_normalization(tmp_path) -> None:
    image = Image.new("RGB", (80, 50), "white")
    ImageDraw.Draw(image).rectangle((20, 10, 49, 39), fill="black")
    path = tmp_path / "page.png"
    image.save(path)
    array = np.asarray(image)
    float_array = array.astype(np.float32) / 255.0

    masks = [
        foreground_mask(path).mask,
        foreground_mask(image).mask,
        foreground_mask(array).mask,
        foreground_mask(float_array).mask,
    ]

    assert all(np.array_equal(mask, masks[0]) for mask in masks[1:])


def test_auto_preserves_sparse_black_and_color_foreground() -> None:
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 80, 69, 99), fill="black")
    draw.rectangle((300, 80, 309, 99), fill=(20, 70, 220))

    result = foreground_mask(image, mode="auto")

    assert result.mask[90, 65]
    assert result.mask[90, 305]


def test_auto_sparse_mixed_ink_excludes_gray_guide() -> None:
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 80, 69, 99), fill="black")
    draw.rectangle((300, 80, 309, 99), fill=(20, 70, 220))
    draw.line((5, 150, 394, 150), fill=(180, 180, 180), width=1)

    result = foreground_mask(image, mode="auto")

    assert result.mask[90, 65]
    assert result.mask[90, 305]
    assert not result.mask[150, 100]


def test_auto_color_only_page_does_not_invent_neutral_foreground() -> None:
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).rectangle((300, 80, 309, 99), fill=(20, 70, 220))

    result = foreground_mask(image, mode="auto")

    assert result.mask[90, 305]
    assert not result.mask[0, 0]


def test_auto_sparse_black_only_page_remains_detectable() -> None:
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).rectangle((60, 80, 69, 99), fill="black")

    result = foreground_mask(image, mode="auto")

    assert result.mask[90, 65]
    assert not result.mask[0, 0]
