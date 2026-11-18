from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import detect_spaced_region_hypotheses, detect_spaced_regions
from rasterregions.io import spaced_hypotheses_payload

ALPHABET_ROW_COUNTS = (12, 8, 6, 11, 11, 5, 10)


def _draw_slanted_row(draw: ImageDraw.ImageDraw, count: int, y: int, width: int) -> None:
    spacing = (width - 150) / max(1, count - 1)
    for index in range(count):
        x = 30 + index * spacing
        stroke_width = 86 + (index * 7) % 12
        draw.line(
            (round(x), y + 55, round(x + stroke_width), y + 5),
            fill="black",
            width=4,
        )
        if index % 8 == 0:
            dot_x = round(x + stroke_width - 20)
            draw.ellipse((dot_x - 1, y - 3, dot_x + 1, y - 1), fill="black")


def _alphabet_page() -> Image.Image:
    image = Image.new("RGB", (1081, 1198), "white")
    draw = ImageDraw.Draw(image)
    for row_index, count in enumerate(ALPHABET_ROW_COUNTS):
        _draw_slanted_row(draw, count, 35 + row_index * 155, image.width)
    draw.point((1065, 15), fill="black")
    draw.rectangle((1058, 1110, 1059, 1111), fill="black")
    return image


def _assert_exact_ownership(result) -> None:
    owner = np.zeros(result.clean_mask.shape, dtype=bool)
    for region in result.regions:
        region_mask = np.zeros(result.clean_mask.shape, dtype=bool)
        region_mask[
            region.bbox.y0 : region.bbox.y1,
            region.bbox.x0 : region.bbox.x1,
        ] = region.mask
        assert not np.any(owner & region_mask)
        assert not np.any(region_mask & ~result.clean_mask)
        owner |= region_mask
    assert result.unassigned_mask is not None
    assert not np.any(owner & result.unassigned_mask)
    assert np.array_equal(owner | result.unassigned_mask, result.clean_mask)


def _region_signature(result) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            region.bbox,
            region.row_index,
            region.column_index,
            region.component_indices,
            region.mask.tobytes(),
        )
        for region in result.regions
    )


def test_wide_spaced_row_uses_twelve_component_anchors() -> None:
    image = Image.new("RGB", (1081, 130), "white")
    _draw_slanted_row(ImageDraw.Draw(image), 12, 35, image.width)

    result = detect_spaced_regions(image, mode="dark")

    assert result.status == "ok"
    assert len(result.regions) == 12
    row = result.diagnostics.row_segmentation[0]
    assert row.inferred_count == 12
    assert row.anchor_count == 12
    assert row.segmentation_strategy == "anchors"
    assert not result.diagnostics.search_truncated
    assert any(
        left.bbox.x1 > right.bbox.x0
        for left, right in zip(result.regions, result.regions[1:], strict=False)
    )
    _assert_exact_ownership(result)


def test_generated_seven_row_page_has_sixty_three_anchors() -> None:
    result = detect_spaced_regions(_alphabet_page(), mode="dark")

    assert [row.inferred_count for row in result.diagnostics.row_segmentation] == list(
        ALPHABET_ROW_COUNTS
    )
    assert [row.anchor_count for row in result.diagnostics.row_segmentation] == list(
        ALPHABET_ROW_COUNTS
    )
    assert len(result.regions) == 63
    assert not result.diagnostics.search_truncated
    _assert_exact_ownership(result)


def test_generated_alphabet_hypotheses_and_diagnostics_are_deterministic() -> None:
    image = _alphabet_page()
    first = detect_spaced_region_hypotheses(image, mode="dark")
    second = detect_spaced_region_hypotheses(image, mode="dark")

    assert _region_signature(first.selected) == _region_signature(second.selected)
    assert spaced_hypotheses_payload(first) == spaced_hypotheses_payload(second)
    assert [row.inferred_count for row in first.selected.diagnostics.row_segmentation] == list(
        ALPHABET_ROW_COUNTS
    )
