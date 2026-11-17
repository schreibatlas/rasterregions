from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from rasterregions import (
    DetectionConfig,
    SpacingConfig,
    detect_spaced_region_hypotheses,
)


def _two_row_page() -> Image.Image:
    image = Image.new("RGB", (180, 140), "white")
    draw = ImageDraw.Draw(image)
    for y in (15, 85):
        for x in (15, 105):
            draw.rectangle((x, y, x + 25, y + 28), fill="black")
    return image


def test_page_state_budget_is_reported_without_fabricating_allocation() -> None:
    config = DetectionConfig(spacing=SpacingConfig(max_page_states=1))

    result = detect_spaced_region_hypotheses(
        _two_row_page(),
        mode="dark",
        expected_count=3,
        config=config,
    )

    assert result.truncated
    assert result.page_hypotheses == ()
    assert result.failure_reason is not None
    assert result.selected.status == "failed"
    assert result.selected.diagnostics.search_truncated
    assert result.selected.diagnostics.search_limit == "page-allocation-budget"
    assert result.selected.diagnostics.expected_count == 3


def test_each_search_budget_rejects_nonpositive_values() -> None:
    for field in (
        "hypothesis_limit",
        "max_candidate_boundaries",
        "max_count_candidates",
        "max_partition_states",
        "max_page_states",
        "max_secondary_mark_branches",
    ):
        with pytest.raises(ValueError, match="positive integers"):
            SpacingConfig(**{field: 0})
