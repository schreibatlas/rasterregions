from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from rasterregions import (
    DetectionConfig,
    SpacingConfig,
    detect_spaced_region_hypotheses,
    detect_spaced_regions,
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


def test_automatic_evidence_split_is_independent_of_partition_budget() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 49, 55), fill="black")
    draw.rectangle((52, 25, 90, 55), fill="black")
    draw.rectangle((49, 39, 52, 41), fill="black")
    config = DetectionConfig(spacing=SpacingConfig(max_partition_states=1))

    result = detect_spaced_region_hypotheses(image, mode="dark", config=config)

    assert len(result.selected.regions) == 2
    assert not result.selected.diagnostics.search_truncated
    assert result.selected.diagnostics.search_limit is None
    assert result.row_hypotheses[0][0].count == 2
    row = result.selected.diagnostics.row_segmentation[0]
    assert row.anchor_count == 1
    assert row.segmentation_strategy == "hybrid"
    assert row.split_anchor_indices == (0,)
    assert row.attempted_counts == (2,)
    assert row.truncated_counts == ()


def test_exact_count_budget_failure_leaves_clean_ink_unassigned() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 49, 55), fill="black")
    draw.rectangle((52, 25, 90, 55), fill="black")
    draw.rectangle((49, 39, 52, 41), fill="black")
    config = DetectionConfig(spacing=SpacingConfig(max_partition_states=1))

    result = detect_spaced_regions(image, mode="dark", row_counts=[2], config=config)

    assert result.status == "failed"
    assert result.regions == ()
    assert result.diagnostics.search_truncated
    assert result.diagnostics.search_limit == "row-partition-budget"
    assert (result.unassigned_mask == result.clean_mask).all()
    row = result.diagnostics.row_segmentation[0]
    assert row.attempted_counts == (2,)
    assert row.truncated_counts == (2,)
    assert row.segmentation_strategy == "partition"


def test_impossible_exact_count_does_not_fabricate_regions() -> None:
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((25, 20, 54, 55), fill="black")

    result = detect_spaced_regions(image, mode="dark", row_counts=[3])

    assert result.status == "failed"
    assert result.regions == ()
    assert not result.diagnostics.search_truncated
    assert (result.unassigned_mask == result.clean_mask).all()
    row = result.diagnostics.row_segmentation[0]
    assert row.inferred_count is None
    assert row.requested_count == 3
    assert row.anchor_count == 1
    assert row.attempted_counts == (3,)
