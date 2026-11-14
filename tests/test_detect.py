from __future__ import annotations

from PIL import Image, ImageDraw

from rasterregions import detect_regions, detect_rows


def _draw_shape(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str) -> None:
    if kind == "e":
        draw.line(
            [
                (x, y + 14),
                (x + 18, y + 14),
                (x + 20, y + 9),
                (x + 17, y + 4),
                (x + 10, y + 2),
                (x + 4, y + 5),
                (x + 1, y + 12),
                (x + 3, y + 20),
                (x + 9, y + 25),
                (x + 17, y + 24),
            ],
            fill="black",
            width=4,
        )
    elif kind == "l":
        draw.line((x + 8, y, x + 8, y + 26), fill="black", width=4)
    elif kind == "i":
        draw.line((x + 8, y + 9, x + 8, y + 26), fill="black", width=4)
        draw.ellipse((x + 6, y, x + 10, y + 4), fill="black")
    elif kind == "n":
        draw.line(
            [
                (x, y + 26),
                (x, y + 5),
                (x + 6, y + 2),
                (x + 15, y + 4),
                (x + 19, y + 10),
                (x + 19, y + 26),
            ],
            fill="black",
            width=4,
        )


def _reference_page() -> Image.Image:
    image = Image.new("RGB", (260, 170), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 259, 169), outline="black", width=2)
    for x, kind in ((20, "e"), (75, "l"), (125, "i"), (185, "n")):
        _draw_shape(draw, x, 20, kind)
    for x, kind in ((20, "n"), (80, "e"), (145, "l"), (200, "i")):
        _draw_shape(draw, x, 105, kind)
    return image


def _sample_page() -> Image.Image:
    image = Image.new("RGB", (260, 135), "white")
    draw = ImageDraw.Draw(image)
    _draw_shape(draw, 15, 35, "e")
    _draw_shape(draw, 105, 35, "i")
    _draw_shape(draw, 195, 35, "n")
    draw.line((4, 100, 255, 100), fill=(180, 180, 180), width=1)
    draw.point((250, 8), fill="black")
    return image


def test_automatic_detection_finds_rows_and_attaches_dot() -> None:
    result = detect_regions(_reference_page())

    assert result.status == "ok"
    assert len(result.regions) == 8
    assert [region.row_index for region in result.regions] == [0] * 4 + [1] * 4
    assert [region.column_index for region in result.regions] == [0, 1, 2, 3] * 2
    dotted = result.regions[2]
    assert dotted.bbox.y0 <= 20
    assert len(detect_rows(result.clean_mask)) == 2


def test_expected_count_recovers_three_regions() -> None:
    result = detect_regions(_sample_page(), expected_count=3, mode="dark")

    assert result.status == "ok"
    assert len(result.regions) == 3
    assert result.regions[1].bbox.y0 <= 35
    assert all(region.confidence >= 0.3 for region in result.regions)
    assert any("rule" in warning for warning in result.warnings)


def test_wrong_expected_count_returns_failed_result() -> None:
    result = detect_regions(_sample_page(), expected_count=4, mode="dark")

    assert result.status == "failed"
    assert result.regions == ()
    assert any("Unable to recover 4" in warning for warning in result.warnings)
