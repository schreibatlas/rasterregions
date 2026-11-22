from PIL import Image, ImageDraw

from rasterregions import (
    DetectionConfig,
    SpacingConfig,
    detect_spaced_region_hypotheses,
    detect_spaced_regions,
)


def test_h_and_m_like_bodies_keep_the_unsplit_anchor_baseline() -> None:
    image = Image.new("RGB", (220, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 35, 55), fill="black")

    draw.rectangle((65, 25, 67, 55), fill="black")
    draw.rectangle((83, 25, 85, 55), fill="black")
    draw.rectangle((65, 39, 85, 41), fill="black")

    draw.line((115, 55, 115, 25), fill="black", width=3)
    draw.line((115, 25, 125, 45), fill="black", width=3)
    draw.line((125, 45, 135, 25), fill="black", width=3)
    draw.line((135, 25, 135, 55), fill="black", width=3)
    draw.rectangle((165, 25, 185, 55), fill="black")

    result = detect_spaced_regions(image, mode="dark")
    row = result.diagnostics.row_segmentation[0]

    assert len(result.regions) == 4
    assert row.anchor_count == 4
    assert row.inferred_count == 4
    assert row.split_anchor_indices == ()
    assert all(not evidence.accepted for evidence in row.split_evidence)


def _two_touching_pair_sheet():
    image = Image.new("RGB", (280, 90), "white")
    draw = ImageDraw.Draw(image)
    for x in (10, 120, 230):
        draw.rectangle((x, 30, x + 20, 60), fill="black")
    for x in (55, 165):
        draw.rectangle((x, 30, x + 17, 60), fill="black")
        draw.rectangle((x + 23, 30, x + 40, 60), fill="black")
        draw.rectangle((x + 17, 44, x + 23, 46), fill="black")
    return image


def test_two_bound_splits_and_baseline_are_stable_across_hypothesis_budgets() -> None:
    signatures = []
    detailed = None
    for limit in (1, 5):
        result = detect_spaced_region_hypotheses(
            _two_touching_pair_sheet(),
            mode="dark",
            config=DetectionConfig(spacing=SpacingConfig(hypothesis_limit=limit)),
        )
        signatures.append(
            tuple((region.bbox, region.mask.tobytes()) for region in result.selected.regions)
        )
        assert len(result.selected.regions) == 7
        detailed = result

    assert signatures[0] == signatures[1]
    assert detailed is not None
    options = detailed.row_hypotheses[0]
    by_splits = {option.split_anchor_indices: option.count for option in options}
    assert by_splits == {(): 5, (1,): 6, (3,): 6, (1, 3): 7}
    geometries = [
        tuple((region.bbox, region.mask.tobytes()) for region in option.regions)
        for option in options
    ]
    assert len(geometries) == len(set(geometries))
