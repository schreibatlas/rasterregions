from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from rasterregions import detect_regions, detect_rows
from rasterregions.models import (
    BBox,
    CleanupConfig,
    DetectionConfig,
    GroupingConfig,
    MaskConfig,
    MaskResult,
    PartitionConfig,
)


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


def _two_row_page() -> Image.Image:
    image = Image.new("RGB", (220, 130), "white")
    draw = ImageDraw.Draw(image)
    for y in (20, 85):
        for x in (25, 155):
            draw.rectangle((x, y, x + 24, y + 24), fill="black")
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


def test_expected_count_one_can_succeed() -> None:
    image = Image.new("RGB", (100, 70), "white")
    ImageDraw.Draw(image).rectangle((35, 20, 64, 49), fill="black")

    result = detect_regions(image, expected_count=1, mode="dark")

    assert result.status == "ok"
    assert len(result.regions) == 1
    assert result.regions[0].confidence >= 0.3


def test_page_border_and_long_rules_are_removed() -> None:
    border = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(border).rectangle((0, 0, 99, 79), outline="black", width=2)
    border_result = detect_regions(border, mode="dark")

    horizontal = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(horizontal).line((8, 40, 92, 40), fill="black", width=1)
    horizontal_result = detect_regions(horizontal, mode="dark")

    vertical = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(vertical).line((50, 5, 50, 75), fill="black", width=1)
    vertical_result = detect_regions(vertical, mode="dark")

    assert not border_result.clean_mask[0:2, :].any()
    assert not horizontal_result.clean_mask.any()
    assert not vertical_result.clean_mask.any()
    assert any(
        decision.action == "removed" and decision.reason is not None
        for decision in border_result.component_decisions
    )


def test_large_legitimate_component_is_not_page_artifact() -> None:
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((15, 15, 85, 65), fill="black")

    result = detect_regions(image, mode="dark")

    assert result.clean_mask[40, 50]
    assert result.component_decisions[0].action == "kept"


def test_thin_connected_multi_object_content_survives_cleanup() -> None:
    image = Image.new("RGB", (240, 100), "white")
    draw = ImageDraw.Draw(image)
    for x in (30, 100, 170):
        draw.rectangle((x, 25, x + 40, 75), outline="black", width=2)
    draw.line((70, 50, 100, 50), fill="black", width=1)
    draw.line((140, 50, 170, 50), fill="black", width=1)

    result = detect_regions(image, mode="dark")

    assert result.clean_mask[50, 85]
    assert result.clean_mask[50, 155]


def test_region_geometry_remains_inside_image_and_matches_crops() -> None:
    result = detect_regions(_reference_page())

    assert result.clean_mask.shape == result.mask_result.mask.shape
    assert [region.reading_order for region in result.regions] == sorted(
        region.reading_order for region in result.regions
    )
    for region in result.regions:
        bbox = region.bbox
        assert 0 <= bbox.x0 < bbox.x1 <= result.image.width
        assert 0 <= bbox.y0 < bbox.y1 <= result.image.height
        assert region.image.size == (bbox.width, bbox.height)
        assert region.mask.shape == (bbox.height, bbox.width)


def test_cleanup_config_can_tune_rule_removal() -> None:
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).line((10, 40, 35, 40), fill="black", width=1)
    config = DetectionConfig(cleanup=CleanupConfig(rule_min_span_ratio=0.4))

    result = detect_regions(image, mode="dark", config=config)

    assert result.clean_mask[40, 20]
    assert result.component_decisions[0].action == "kept"


def test_expected_count_defaults_to_best_row_scope() -> None:
    result = detect_regions(_two_row_page(), expected_count=2, mode="dark")

    assert result.status == "ok"
    assert len(result.regions) == 2
    assert len({region.row_index for region in result.regions}) == 1
    assert result.expected_count_scope == "row"


def test_page_scope_allocates_exact_total_across_rows() -> None:
    result = detect_regions(
        _two_row_page(), expected_count=4, expected_count_scope="page", mode="dark"
    )

    assert result.status == "ok"
    assert len(result.regions) == 4
    assert [region.row_index for region in result.regions] == [0, 0, 1, 1]
    assert [region.region_id for region in result.regions] == [
        "region-000",
        "region-001",
        "region-002",
        "region-003",
    ]
    assert [region.reading_order for region in result.regions] == [0, 1, 2, 3]
    assert result.expected_count_scope == "page"


def test_page_scope_fails_when_total_cannot_cover_all_rows() -> None:
    result = detect_regions(
        _two_row_page(), expected_count=1, expected_count_scope="page", mode="dark"
    )

    assert result.status == "failed"
    assert result.regions == ()


def test_automatic_confidence_varies_with_separation_evidence() -> None:
    image = Image.new("RGB", (140, 60), "white")
    draw = ImageDraw.Draw(image)
    for x in (20, 44, 84):
        draw.rectangle((x, 20, x + 19, 39), fill="black")

    result = detect_regions(image, mode="dark")
    confidences = [region.confidence for region in result.regions]

    assert result.status == "ok"
    assert len(confidences) == 3
    assert all(0.0 <= confidence <= 1.0 for confidence in confidences)
    assert confidences[0] < confidences[-1]


def test_detect_regions_rgba_composites_transparent_background() -> None:
    import numpy as np

    rgba = np.zeros((80, 120, 4), dtype=np.uint8)
    rgba[20:60, 40:80, 3] = 255

    result = detect_regions(rgba, mode="grayscale")

    assert result.status == "ok"
    assert not result.mask_result.mask[0, 0]
    assert result.mask_result.mask[30, 50]
    assert len(result.regions) == 1


@pytest.mark.parametrize("mode", ["auto", "grayscale", "dark", "adaptive"])
def test_float_grayscale_matches_uint8_detection(mode: str) -> None:
    import numpy as np

    float_image = np.ones((80, 120), dtype=np.float32)
    float_image[20:60, 40:80] = 0.0
    uint8_image = np.rint(float_image * 255).astype(np.uint8)

    float_result = detect_regions(float_image, mode=mode)
    uint8_result = detect_regions(uint8_image, mode=mode)

    assert float_result.status == uint8_result.status
    assert np.array_equal(float_result.mask_result.mask, uint8_result.mask_result.mask)
    assert [region.bbox for region in float_result.regions] == [
        region.bbox for region in uint8_result.regions
    ]


def test_path_pil_and_array_inputs_have_equivalent_detection(tmp_path) -> None:
    import numpy as np

    image = Image.new("RGB", (120, 80), "white")
    ImageDraw.Draw(image).rectangle((40, 20, 79, 59), fill="black")
    path = tmp_path / "page.png"
    image.save(path)
    opened = Image.open(path)
    array = np.asarray(image)
    float_array = array.astype(np.float32) / 255.0

    results = [
        detect_regions(path, mode="grayscale"),
        detect_regions(opened, mode="grayscale"),
        detect_regions(array, mode="grayscale"),
        detect_regions(float_array, mode="grayscale"),
    ]

    for result in results[1:]:
        assert result.status == results[0].status
        assert np.array_equal(result.mask_result.mask, results[0].mask_result.mask)
        assert [region.bbox for region in result.regions] == [
            region.bbox for region in results[0].regions
        ]


def test_nonfinite_and_empty_grayscale_inputs_are_rejected() -> None:
    import numpy as np
    import pytest

    nonfinite = np.ones((20, 20), dtype=np.float32)
    nonfinite[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        detect_regions(nonfinite)
    with pytest.raises(ValueError, match="empty"):
        detect_regions(np.empty((0, 20), dtype=np.uint8))


def test_path_and_pil_inputs_apply_exif_orientation(tmp_path) -> None:
    image = Image.new("RGB", (40, 20), "white")
    ImageDraw.Draw(image).rectangle((2, 3, 6, 9), fill="black")
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)

    path_result = detect_regions(path, mode="dark")
    with Image.open(path) as opened:
        pil_result = detect_regions(opened, mode="dark")

    for result in (path_result, pil_result):
        assert result.image.size == (20, 40)
        assert result.mask_result.mask.shape == (40, 20)
        assert result.mask_result.mask[4, 14]


def test_cleanup_component_ids_remain_traceable_across_results() -> None:
    image = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 199, 99), outline="black", width=2)
    for x in (30, 80, 130):
        draw.rectangle((x, 30, x + 19, 59), fill="black")

    result = detect_regions(image, mode="dark")
    kept_ids = {
        decision.component_index
        for decision in result.component_decisions
        if decision.action == "kept"
    }
    result_ids = {component.index for component in result.components}
    row_ids = {index for row in result.rows for index in row.component_indices}
    region_ids = {index for region in result.regions for index in region.component_indices}

    assert result_ids == kept_ids
    assert len(result_ids) == 3
    assert row_ids <= result_ids
    assert region_ids <= result_ids


def test_detect_regions_labels_components_only_once(monkeypatch) -> None:
    import importlib

    cleanup_module = importlib.import_module("rasterregions._cleanup")
    original_label = cleanup_module.label_components
    calls = 0

    def count_labels(mask):
        nonlocal calls
        calls += 1
        return original_label(mask)

    monkeypatch.setattr(cleanup_module, "label_components", count_labels)
    result = detect_regions(_sample_page(), mode="dark")

    assert result.status == "ok"
    assert calls == 1


def test_detection_configuration_rejects_invalid_values() -> None:
    invalid_configs = (
        lambda: CleanupConfig(tiny_area_ratio=1.1),
        lambda: GroupingConfig(row_center_gap_ratio=-1.0),
        lambda: MaskConfig(adaptive_window_size=4),
        lambda: MaskConfig(dark_delta=float("nan")),
        lambda: MaskConfig(saturation_threshold=1.1),
        lambda: PartitionConfig(crossing_cost=float("inf")),
        lambda: PartitionConfig(primary_area_ratio=-0.1),
    )
    for make_config in invalid_configs:
        with pytest.raises(ValueError):
            make_config()


def test_adaptive_mask_config_is_forwarded_through_detection(monkeypatch) -> None:
    import importlib

    detect_module = importlib.import_module("rasterregions.detect")
    original = detect_module.foreground_mask
    observed_configs = []

    def capture_config(image, mode, *, config):
        observed_configs.append(config)
        return original(image, mode=mode, config=config)

    monkeypatch.setattr(detect_module, "foreground_mask", capture_config)
    config = DetectionConfig(mask=MaskConfig(adaptive_window_size=9, adaptive_offset=3.0))
    detect_regions(_sample_page(), mode="adaptive", config=config)

    assert observed_configs == [config.mask]


def test_satellite_component_is_assigned_to_only_one_row() -> None:
    image = Image.new("RGB", (120, 110), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 15, 69, 34), fill="black")
    draw.rectangle((40, 75, 69, 94), fill="black")
    draw.rectangle((50, 48, 58, 56), fill="black")

    result = detect_regions(
        image,
        mode="dark",
        config=DetectionConfig(cleanup=CleanupConfig(tiny_area_ratio=0.005)),
    )
    satellite = next(
        component for component in result.components if component.bbox == BBox(50, 48, 59, 57)
    )
    assigned_regions = [
        region for region in result.regions if satellite.index in region.component_indices
    ]

    assert len(result.regions) == 2
    assert len(assigned_regions) == 1


def test_low_mask_confidence_does_not_gate_region_status(monkeypatch) -> None:
    import importlib

    detect_module = importlib.import_module("rasterregions.detect")
    original = detect_module.foreground_mask

    def low_confidence(image, mode, *, config):
        result = original(image, mode=mode, config=config)
        return MaskResult(
            result.mask,
            0.0,
            result.method,
            ("Mask confidence is low.",),
            threshold=result.threshold,
        )

    monkeypatch.setattr(detect_module, "foreground_mask", low_confidence)
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((35, 20, 64, 49), fill="black")
    result = detect_regions(image, mode="dark", min_confidence=0.3)

    assert result.status == "ok"
    assert result.diagnostics is not None
    assert result.diagnostics.mask_confidence == 0.0
    assert "Mask confidence is low." in result.warnings


def test_row_scope_only_computes_requested_partition_count(monkeypatch) -> None:
    import importlib

    partition_module = importlib.import_module("rasterregions._partition")
    original = partition_module._row_partition_regions
    attempted_counts = []

    def record_count(*args, **kwargs):
        attempted_counts.append(args[6])
        return original(*args, **kwargs)

    monkeypatch.setattr(partition_module, "_row_partition_regions", record_count)
    result = detect_regions(_two_row_page(), expected_count=2, mode="dark")

    assert result.status == "ok"
    assert attempted_counts == [2, 2]
