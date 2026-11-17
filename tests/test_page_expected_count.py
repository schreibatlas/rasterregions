from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import detect_spaced_region_hypotheses


def _two_row_page() -> Image.Image:
    image = Image.new("RGB", (180, 140), "white")
    draw = ImageDraw.Draw(image)
    for y in (15, 85):
        for x in (15, 105):
            draw.rectangle((x, y, x + 25, y + 28), fill="black")
    return image


def test_three_widely_spaced_objects_and_page_artifacts_are_detected() -> None:
    image = Image.new("RGB", (220, 80), "white")
    draw = ImageDraw.Draw(image)
    for x in (15, 80, 150):
        draw.rectangle((x, 18, x + 25, 48), fill="black")
    draw.line((0, 62, 219, 62), fill=(180, 180, 180), width=1)
    draw.point((208, 7), fill="black")

    result = detect_spaced_region_hypotheses(image, mode="dark", expected_count=3)

    assert result.selected.status in {"ok", "ambiguous"}
    assert len(result.selected.regions) == 3
    assert result.selected.diagnostics.expected_count_scope == "page"
    assert result.page_hypotheses[0].count == 3
    assert result.page_hypotheses[0].row_hypotheses[0].count == 3
    reasons = [decision.reason for decision in result.selected.component_decisions]
    assert any(reason and "rule" in reason for reason in reasons)
    assert any(reason and "tiny" in reason for reason in reasons)


def test_page_total_returns_competing_two_row_allocations_deterministically() -> None:
    image = _two_row_page()
    first = detect_spaced_region_hypotheses(image, mode="dark", expected_count=3)
    second = detect_spaced_region_hypotheses(image, mode="dark", expected_count=3)

    allocations = [
        tuple(row.count for row in page.row_hypotheses) for page in first.page_hypotheses
    ]
    second_allocations = [
        tuple(row.count for row in page.row_hypotheses) for page in second.page_hypotheses
    ]
    assert first.selected.status == "ambiguous"
    assert allocations[:2] == [(2, 1), (1, 2)]
    assert second_allocations == allocations
    assert all(page.count == 3 for page in first.page_hypotheses)
    assert [page.cost for page in first.page_hypotheses] == sorted(
        page.cost for page in first.page_hypotheses
    )
    assert np.array_equal(
        first.page_hypotheses[0].unassigned_mask,
        first.selected.unassigned_mask,
    )
