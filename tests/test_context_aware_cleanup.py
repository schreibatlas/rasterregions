from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from rasterregions import (
    DetectionConfig,
    SpacingConfig,
    detect_spaced_region_hypotheses,
    detect_spaced_regions,
)


def _single_body_with_mark(mark_box: tuple[int, int, int, int]) -> Image.Image:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 30, 49, 59), fill="black")
    draw.rectangle(mark_box, fill="black")
    return image


def _assert_page_ownership(clean_mask: np.ndarray, page) -> None:
    owned = np.zeros(clean_mask.shape, dtype=bool)
    for region in page.regions:
        local = owned[region.bbox.y0 : region.bbox.y1, region.bbox.x0 : region.bbox.x1]
        assert not np.any(local & region.mask)
        local |= region.mask
    assert not np.any(owned & page.unassigned_mask)
    assert np.array_equal(owned | page.unassigned_mask, clean_mask)


@pytest.mark.parametrize(
    ("mark_box", "mark_pixels"),
    [((37, 20, 37, 20), ((20, 37),)), ((36, 19, 37, 20), ((19, 36), (19, 37), (20, 36), (20, 37)))],
)
def test_one_to_four_pixel_dot_above_body_is_retained_as_secondary(
    mark_box: tuple[int, int, int, int], mark_pixels: tuple[tuple[int, int], ...]
) -> None:
    result = detect_spaced_regions(_single_body_with_mark(mark_box), mode="dark")

    assert len(result.regions) == 1
    for y, x in mark_pixels:
        assert result.clean_mask[y, x]
    mark = next(
        decision for decision in result.component_decisions if decision.bbox.x0 == mark_box[0]
    )
    assert mark.action == "kept"
    assert mark.component_index in result.regions[0].component_indices

    assert mark.role == "secondary"
    body = next(
        decision
        for decision in result.component_decisions
        if decision.component_index != mark.component_index
    )
    assert body.role == "primary"
    assert mark.component_index in result.regions[0].secondary_component_indices
    assert body.component_index in result.regions[0].primary_component_indices


def test_two_tiny_dots_above_one_body_remain_distinct_secondary_components() -> None:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 30, 49, 59), fill="black")
    draw.point((32, 20), fill="black")
    draw.point((42, 20), fill="black")

    result = detect_spaced_regions(image, mode="dark")

    assert len(result.regions) == 1
    marks = [decision for decision in result.component_decisions if decision.bbox.y0 == 20]
    assert len(marks) == 2
    assert all(mark.action == "kept" for mark in marks)
    assert {mark.component_index for mark in marks}.issubset(result.regions[0].component_indices)


def test_tiny_far_away_speck_is_removed_with_stable_raw_component_id() -> None:
    image = _single_body_with_mark((3, 3, 3, 3))

    result = detect_spaced_regions(image, mode="dark")

    speck = next(decision for decision in result.component_decisions if decision.bbox.x0 == 3)
    assert speck.component_index == 0
    assert speck.action == "removed"
    assert speck.role == "micro-noise"
    assert speck.reason is not None
    assert not result.clean_mask[3, 3]


def test_tiny_suspicious_side_speck_is_retained_unassigned() -> None:
    result = detect_spaced_regions(_single_body_with_mark((52, 42, 52, 42)), mode="dark")

    assert result.clean_mask[42, 52]
    assert result.unassigned_mask[42, 52]
    mark_index = next(
        decision.component_index
        for decision in result.component_decisions
        if decision.bbox.x0 == 52
    )
    mark = next(
        decision
        for decision in result.component_decisions
        if decision.component_index == mark_index
    )
    assert mark.role == "secondary-candidate"
    assert all(mark_index not in region.component_indices for region in result.regions)


def test_tiny_ambiguous_dot_between_neighboring_bodies_stays_unassigned() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 38, 55), fill="black")
    draw.rectangle((75, 25, 98, 55), fill="black")
    draw.rectangle((52, 12, 53, 13), fill="black")
    config = DetectionConfig(spacing=SpacingConfig(mark_assignment_margin=0.99))

    result = detect_spaced_regions(image, mode="dark", config=config)

    assert result.clean_mask[12:14, 52:54].all()
    assignment = next(
        item for item in result.diagnostics.secondary_mark_assignments if item.component_index == 0
    )
    assert assignment.status == "ambiguous"
    assert assignment.target_region_index is None
    assert assignment.candidate_region_indices == (0, 1)
    mark = next(
        decision
        for decision in result.component_decisions
        if decision.component_index == assignment.component_index
    )
    assert mark.role == "ambiguous"
    assert all(
        assignment.component_index in region.ambiguous_component_indices
        for region in result.regions
    )
    assert result.unassigned_mask[12:14, 52:54].all()


def test_secondary_only_partition_cell_does_not_leave_stale_boundaries() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 38, 55), fill="black")
    draw.rectangle((75, 25, 98, 55), fill="black")
    draw.rectangle((52, 12, 54, 14), fill="black")
    config = DetectionConfig(spacing=SpacingConfig(mark_assignment_margin=0.99))

    result = detect_spaced_region_hypotheses(image, mode="dark", row_counts=[3], config=config)

    assert result.selected.status == "failed"
    assert result.selected.regions == ()
    assert np.array_equal(result.selected.unassigned_mask, result.selected.clean_mask)
    for page in result.page_hypotheses:
        _assert_page_ownership(result.selected.clean_mask, page)
        for row in page.row_hypotheses:
            assert row.count == len(row.regions)
            assert len(row.boundaries) == len(row.regions) + 1


def test_mixed_size_rows_keep_tiny_marks_without_adding_body_regions() -> None:
    image = Image.new("RGB", (180, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 18, 44, 55), fill="black")
    draw.rectangle((100, 15, 125, 55), fill="black")
    draw.rectangle((20, 85, 36, 108), fill="black")
    draw.rectangle((100, 82, 119, 108), fill="black")
    draw.point((31, 8), fill="black")
    draw.point((28, 77), fill="black")

    result = detect_spaced_regions(image, mode="dark", row_counts=[2, 2])

    assert len(result.rows) == 2
    assert len(result.regions) == 4
    assert result.clean_mask[8, 31]
    assert result.clean_mask[77, 28]
    assert sorted(region.row_index for region in result.regions) == [0, 0, 1, 1]
