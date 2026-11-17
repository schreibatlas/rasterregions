from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import detect_spaced_regions
from rasterregions.models import DetectionConfig, SpacingConfig


def _assert_exact_ownership(result) -> None:
    owner = np.zeros(result.clean_mask.shape, dtype=np.uint8)
    for region in result.regions:
        region_mask = np.zeros(result.clean_mask.shape, dtype=bool)
        region_mask[
            region.bbox.y0 : region.bbox.y1,
            region.bbox.x0 : region.bbox.x1,
        ] = region.mask
        assert not np.any(owner[region_mask])
        assert not np.any(region_mask & ~result.clean_mask)
        owner[region_mask] = 1
    assert not np.any((owner > 0) & result.unassigned_mask)
    assert np.array_equal((owner > 0) | result.unassigned_mask, result.clean_mask)


def test_overlapping_slanted_bodies_are_separated_automatically() -> None:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((15, 55, 49, 18), fill="black", width=3)
    draw.line((44, 55, 70, 18), fill="black", width=3)

    result = detect_spaced_regions(image, mode="dark")

    assert result.status == "ok"
    assert len(result.regions) == 2
    assert result.components[0].bbox.x1 > result.components[1].bbox.x0
    assert result.diagnostics.row_segmentation[0].inferred_count == 2
    _assert_exact_ownership(result)


def test_width_prior_adapts_to_narrow_medium_and_wide_regions() -> None:
    image = Image.new("RGB", (180, 80), "white")
    draw = ImageDraw.Draw(image)
    for x, width in ((12, 8), (45, 20), (95, 36)):
        draw.rectangle((x, 20, x + width - 1, 50), fill="black")

    result = detect_spaced_regions(image, mode="dark")

    assert result.status == "ok"
    assert len(result.regions) == 3
    assert result.diagnostics.body_counts_per_region == (1, 1, 1)
    _assert_exact_ownership(result)


def test_detached_dot_is_assigned_once_to_the_nearest_region() -> None:
    image = Image.new("RGB", (150, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 25, 40, 55), fill="black")
    draw.ellipse((27, 12, 32, 17), fill="black")
    draw.rectangle((70, 25, 95, 55), fill="black")

    result = detect_spaced_regions(image, mode="dark")

    assert len(result.regions) == 2
    assert len(result.diagnostics.secondary_mark_assignments) == 1
    assignment = result.diagnostics.secondary_mark_assignments[0]
    assert assignment.status == "assigned"
    assert assignment.target_region_index == 0
    _assert_exact_ownership(result)


def test_detached_dot_and_crossbar_are_assigned_to_their_nearest_regions() -> None:
    image = Image.new("RGB", (150, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 25, 40, 55), fill="black")
    draw.ellipse((27, 12, 32, 17), fill="black")
    draw.rectangle((70, 25, 95, 55), fill="black")
    draw.rectangle((77, 18, 88, 20), fill="black")

    result = detect_spaced_regions(image, mode="dark")

    assert result.status == "ok"
    assert len(result.regions) == 2
    assignments = result.diagnostics.secondary_mark_assignments
    assert len(assignments) == 2
    assert [assignment.status for assignment in assignments] == ["assigned", "assigned"]
    assert [assignment.target_region_index for assignment in assignments] == [0, 1]
    for assignment in assignments:
        assert (
            assignment.component_index
            in result.regions[assignment.target_region_index].component_indices
        )
    _assert_exact_ownership(result)


def test_ambiguous_detached_mark_stays_unassigned_and_is_reported() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 38, 55), fill="black")
    draw.rectangle((75, 25, 98, 55), fill="black")
    draw.ellipse((52, 12, 55, 15), fill="black")
    config = DetectionConfig(spacing=SpacingConfig(mark_assignment_margin=0.99))

    result = detect_spaced_regions(image, mode="dark", config=config)

    assert result.status == "ambiguous"
    assignment = result.diagnostics.secondary_mark_assignments[0]
    assert assignment.status == "ambiguous"
    assert assignment.target_region_index is None
    assert result.unassigned_mask.sum() > 0
    _assert_exact_ownership(result)


def test_ascenders_and_descenders_keep_per_row_ownership() -> None:
    image = Image.new("RGB", (150, 145), "white")
    draw = ImageDraw.Draw(image)
    draw.line((22, 55, 22, 8), fill="black", width=5)
    draw.rectangle((55, 28, 78, 55), fill="black")
    draw.rectangle((22, 88, 48, 125), fill="black")
    draw.rectangle((85, 86, 110, 113), fill="black")

    result = detect_spaced_regions(image, mode="dark", row_counts=[2, 2])

    assert len(result.rows) == 2
    assert len(result.regions) == 4
    assert result.rows[0].bbox.y0 <= 8
    assert result.rows[1].bbox.y1 >= 126
    assert [row.row_index for row in result.regions] == [0, 0, 1, 1]
    _assert_exact_ownership(result)


def test_disconnected_secondary_stroke_stays_with_its_body() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 25, 40, 55), fill="black")
    draw.rectangle((27, 15, 48, 23), fill="black")
    draw.rectangle((78, 25, 103, 55), fill="black")

    result = detect_spaced_regions(image, mode="dark", row_counts=[2])

    assert len(result.regions) == 2
    assert {0, 1}.issubset(result.regions[0].component_indices)
    assert 2 in result.regions[1].component_indices
    _assert_exact_ownership(result)


def test_touching_forms_use_count_hint_or_report_ambiguity() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 49, 55), fill="black")
    draw.rectangle((52, 25, 90, 55), fill="black")
    draw.rectangle((49, 39, 52, 41), fill="black")

    inferred = detect_spaced_regions(image, mode="dark")
    exact = detect_spaced_regions(image, mode="dark", row_counts=[2])

    assert inferred.status == "ambiguous"
    assert len(inferred.regions) == 1
    assert exact.status == "ok"
    assert len(exact.regions) == 2
    assert any(
        boundary.selected and boundary.crossing_components > 0
        for boundary in exact.diagnostics.partition_boundaries
    )
    _assert_exact_ownership(inferred)
    _assert_exact_ownership(exact)


def test_failed_partition_keeps_cleaned_ink_in_unassigned_mask() -> None:
    image = Image.new("RGB", (90, 70), "white")
    ImageDraw.Draw(image).rectangle((30, 20, 55, 48), fill="black")

    result = detect_spaced_regions(image, mode="dark", row_counts=[2])

    assert result.status == "failed"
    assert result.regions == ()
    assert result.components
    assert result.clean_mask.any()
    assert np.array_equal(result.unassigned_mask, result.clean_mask)
    _assert_exact_ownership(result)


def test_spacing_padding_is_scale_aware_and_does_not_change_ownership() -> None:
    image = Image.new("RGB", (100, 70), "white")
    ImageDraw.Draw(image).rectangle((35, 20, 55, 49), fill="black")
    small_pad = detect_spaced_regions(image, mode="dark")
    large_pad = detect_spaced_regions(
        image,
        mode="dark",
        config=DetectionConfig(spacing=SpacingConfig(padding_ratio=0.2)),
    )

    assert large_pad.regions[0].bbox.width > small_pad.regions[0].bbox.width
    small_mask = np.zeros(image.size[::-1], dtype=bool)
    large_mask = np.zeros(image.size[::-1], dtype=bool)
    small_region = small_pad.regions[0]
    large_region = large_pad.regions[0]
    small_mask[
        small_region.bbox.y0 : small_region.bbox.y1,
        small_region.bbox.x0 : small_region.bbox.x1,
    ] = small_region.mask
    large_mask[
        large_region.bbox.y0 : large_region.bbox.y1,
        large_region.bbox.x0 : large_region.bbox.x1,
    ] = large_region.mask
    assert np.array_equal(small_mask, large_mask)
    _assert_exact_ownership(small_pad)
    _assert_exact_ownership(large_pad)
