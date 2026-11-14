from __future__ import annotations

import json

from PIL import Image, ImageDraw

from rasterregions import BBox, crop, detect_regions, tighten_bbox
from rasterregions.io import detection_payload, write_detection


def test_tighten_bbox_and_crop_preserve_pixels() -> None:
    image = Image.new("RGB", (40, 30), "white")
    ImageDraw.Draw(image).rectangle((7, 5, 16, 14), fill="blue")
    mask = Image.new("1", (40, 30), 0)
    ImageDraw.Draw(mask).rectangle((7, 5, 16, 14), fill=1)

    tight = tighten_bbox(mask, BBox(3, 2, 24, 20), pad=1)
    retained = crop(image, tight)

    assert tight.as_tuple() == (6, 4, 18, 16)
    assert retained.getpixel((0, 0)) == (255, 255, 255)
    assert retained.getpixel((1, 1)) == (0, 0, 255)


def test_write_detection_emits_manifest_overlay_mask_and_crops(tmp_path) -> None:
    image = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 15, 30, 40), fill="black")
    draw.rectangle((75, 10, 95, 45), fill="black")

    result = detect_regions(image)
    manifest = write_detection(result, tmp_path)

    payload = json.loads(manifest.read_text())
    assert payload["detected_count"] == 2
    assert (tmp_path / "mask.png").is_file()
    assert (tmp_path / "regions.png").is_file()
    assert (tmp_path / "crops" / "region-000.png").is_file()


def test_detection_payload_adds_structured_reproducibility_diagnostics() -> None:
    image = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(image)
    for x in (10, 48, 86):
        draw.rectangle((x, 15, x + 18, 40), fill="black")

    result = detect_regions(image, expected_count=3, mode="dark")
    payload = detection_payload(result)

    assert result.diagnostics is not None
    assert result.diagnostics.image_size == (120, 60)
    assert result.diagnostics.raw_component_count == 3
    assert payload["rasterregions_version"]
    assert payload["image"] == {"width": 120, "height": 60}
    assert payload["parameters"]["expected_count_scope"] == "row"
    assert payload["mask"]["threshold"] is not None
    assert payload["cleanup"][0]["action"] == "kept"
    assert payload["rows"]
    assert len(payload["partition_boundaries"]) == 2
    assert payload["detected_count"] == 3
    assert payload["regions"][0]["id"] == "region-000"
    assert {
        "source",
        "status",
        "expected_count",
        "detected_count",
        "mask",
        "warnings",
        "regions",
    } <= payload.keys()
    assert {"method", "confidence", "warnings"} <= payload["mask"].keys()
    assert {
        "id",
        "row",
        "column",
        "reading_order",
        "bbox",
        "confidence",
        "component_indices",
    } <= payload["regions"][0].keys()


def test_write_detection_can_disable_individual_artifacts(tmp_path) -> None:
    image = Image.new("RGB", (40, 40), "white")
    ImageDraw.Draw(image).rectangle((12, 12, 27, 27), fill="black")
    result = detect_regions(image)

    manifest = write_detection(
        result,
        tmp_path,
        include_crops=False,
        include_overlay=False,
        include_mask=False,
    )

    assert manifest.is_file()
    assert not (tmp_path / "crops").exists()
    assert not (tmp_path / "regions.png").exists()
    assert not (tmp_path / "mask.png").exists()
