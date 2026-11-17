from __future__ import annotations

import numpy as np
import pytest

from rasterregions import (
    BBox,
    PagePartitionHypothesis,
    RegionFrame,
    RegionHypothesis,
    RowPartitionHypothesis,
    SpacingConfig,
)


def _region(x0: int, x1: int, frame: RegionFrame) -> RegionHypothesis:
    mask = np.ones((3, x1 - x0), dtype=bool)
    return RegionHypothesis(
        bbox=BBox(x0, 4, x1, 7),
        mask=mask,
        component_indices=(x0,),
        frame=frame,
        cost=0.25,
        confidence=0.8,
        warnings=(),
    )


def test_region_frame_and_hypothesis_masks_are_immutable_snapshots() -> None:
    frame = RegionFrame(baseline_y=20.0, body_height=12.0, row_center_y=14.0, row_index=0)
    source = np.ones((3, 5), dtype=bool)
    region = RegionHypothesis(
        bbox=BBox(2, 4, 7, 7),
        mask=source,
        component_indices=[1, 2],
        frame=frame,
        cost=0.25,
        confidence=0.8,
        warnings=["candidate valley"],
    )

    source[0, 0] = False
    assert region.mask[0, 0]
    assert not region.mask.flags.writeable
    assert region.component_indices == (1, 2)
    assert region.warnings == ("candidate valley",)
    with pytest.raises(ValueError, match="body_height"):
        RegionFrame(baseline_y=20.0, body_height=0.0, row_center_y=14.0, row_index=0)


def test_row_and_page_hypotheses_keep_count_and_geometry_evidence() -> None:
    frame = RegionFrame(baseline_y=20.0, body_height=12.0, row_center_y=14.0, row_index=0)
    first = _region(0, 4, frame)
    second = _region(7, 12, frame)
    row = RowPartitionHypothesis(
        row_index=0,
        count=2,
        boundaries=(0, 5, 12),
        regions=(first, second),
        cost=0.5,
        unassigned_ink_fraction=0.0,
        secondary_mark_assignments=(),
    )
    page = PagePartitionHypothesis(
        row_hypotheses=(row,),
        count=2,
        cost=0.5,
        unassigned_ink_fraction=0.0,
    )

    assert page.regions == (first, second)
    assert row.regions[0].frame == frame
    with pytest.raises(ValueError, match="sum to count"):
        PagePartitionHypothesis((row,), 3, 0.5, 0.0)


def test_spacing_search_limits_are_positive_integers() -> None:
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
        with pytest.raises(ValueError, match="positive integers"):
            SpacingConfig(**{field: True})


def test_page_hypotheses_use_expected_total_and_conserve_foreground() -> None:
    from PIL import Image, ImageDraw

    from rasterregions import detect_spaced_region_hypotheses

    image = Image.new("RGB", (140, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 35, 45), fill="black")
    draw.rectangle((65, 15, 85, 45), fill="black")
    draw.rectangle((30, 85, 60, 115), fill="black")

    result = detect_spaced_region_hypotheses(image, mode="dark", expected_count=3)

    assert len(result.selected.rows) == 2
    assert len(result.selected.regions) == 3
    assert result.page_hypotheses
    assert result.page_hypotheses[0].count == 3
    assert len(result.page_hypotheses[0].regions) == len(result.selected.regions)
    for page in result.page_hypotheses:
        owner = np.zeros(result.selected.clean_mask.shape, dtype=bool)
        for region in page.regions:
            local = owner[region.bbox.y0 : region.bbox.y1, region.bbox.x0 : region.bbox.x1]
            assert not np.any(local & region.mask)
            local |= region.mask
        assert not np.any(owner & page.unassigned_mask)
        assert np.array_equal(owner | page.unassigned_mask, result.selected.clean_mask)


def test_secondary_mark_hypotheses_include_competing_owners() -> None:
    from PIL import Image, ImageDraw

    from rasterregions import detect_spaced_region_hypotheses
    from rasterregions.models import DetectionConfig

    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 38, 55), fill="black")
    draw.rectangle((75, 25, 98, 55), fill="black")
    draw.ellipse((52, 12, 55, 15), fill="black")
    config = DetectionConfig(
        spacing=SpacingConfig(mark_assignment_margin=0.99, hypothesis_limit=12)
    )

    result = detect_spaced_region_hypotheses(
        image,
        mode="dark",
        row_counts=[2],
        config=config,
    )

    assignments = [
        assignment
        for page in result.page_hypotheses
        for row in page.row_hypotheses
        for assignment in row.secondary_mark_assignments
    ]
    assert {assignment.target_region_index for assignment in assignments} >= {None, 0, 1}


def test_infeasible_expected_page_count_is_reported() -> None:
    from PIL import Image, ImageDraw

    from rasterregions import detect_spaced_region_hypotheses

    image = Image.new("RGB", (140, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 35, 45), fill="black")
    draw.rectangle((65, 15, 85, 45), fill="black")
    draw.rectangle((30, 85, 60, 115), fill="black")
    draw.rectangle((90, 85, 120, 115), fill="black")

    result = detect_spaced_region_hypotheses(
        image,
        mode="dark",
        row_counts=[2, 2],
        expected_count=3,
    )

    assert result.page_hypotheses == ()
    assert result.failure_reason is not None
    assert result.selected.status == "failed"
    assert np.array_equal(result.selected.unassigned_mask, result.selected.clean_mask)
