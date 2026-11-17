from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import (
    BBox,
    crop,
    detect_regions,
    detect_spaced_region_hypotheses,
    detect_spaced_regions,
    tighten_bbox,
)
from rasterregions.io import (
    detection_payload,
    spaced_hypotheses_payload,
    write_detection,
    write_spaced_hypotheses,
)


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

    assert payload["schema_version"] == 1
    assert payload["component_index_namespace"] == "raw_mask"
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

    write_detection(result, tmp_path)
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


def test_write_detection_prunes_managed_files_and_preserves_unrelated(
    tmp_path, monkeypatch
) -> None:
    import rasterregions.io as io_module

    first_image = Image.new("RGB", (120, 60), "white")
    first_draw = ImageDraw.Draw(first_image)
    first_draw.rectangle((10, 15, 30, 40), fill="black")
    first_draw.rectangle((75, 10, 95, 45), fill="black")
    first = detect_regions(first_image)

    second_image = Image.new("RGB", (50, 50), "white")
    ImageDraw.Draw(second_image).rectangle((15, 15, 34, 34), fill="black")
    second = detect_regions(second_image)
    assert len(first.regions) == 2
    assert len(second.regions) == 1

    write_detection(first, tmp_path)
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    custom_crop = tmp_path / "crops" / "custom.txt"
    custom_crop.write_text("keep", encoding="utf-8")

    replacements: list[str] = []
    original_replace = io_module.os.replace

    def record_replace(source, destination) -> None:
        replacements.append(Path(destination).name)
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", record_replace)
    manifest = write_detection(
        second,
        tmp_path,
        include_overlay=False,
        include_mask=False,
    )

    crop_names = sorted(path.name for path in (tmp_path / "crops").glob("region-*.png"))
    assert crop_names == ["region-000.png"]
    assert custom_crop.read_text(encoding="utf-8") == "keep"
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "mask.png").exists()
    assert not (tmp_path / "regions.png").exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == 1
    assert replacements[-1] == "regions.json"


def test_detection_payload_records_roi() -> None:
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((35, 20, 64, 49), fill="black")
    result = detect_regions(image, expected_count=1, mode="dark", roi=BBox(10, 10, 90, 70))

    assert detection_payload(result)["parameters"]["roi"] == [10, 10, 90, 70]


def test_tighten_bbox_padding_stays_inside_input_on_left_and_top() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:8, 5:8] = True
    bbox = BBox(5, 5, 10, 10)

    assert tighten_bbox(mask, bbox, pad=2) == bbox


def test_tighten_bbox_padding_stays_inside_input_on_right_and_bottom() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[7:10, 7:10] = True

    assert tighten_bbox(mask, BBox(0, 0, 10, 10), pad=3) == BBox(4, 4, 10, 10)


def test_tighten_bbox_empty_mask_returns_original_bbox() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    bbox = BBox(3, 4, 12, 13)

    assert tighten_bbox(mask, bbox, pad=3) == bbox


def test_spacing_diagnostics_are_serialized() -> None:
    image = Image.new("RGB", (80, 60), "white")
    ImageDraw.Draw(image).rectangle((25, 15, 49, 44), fill="black")

    result = detect_spaced_regions(image, mode="dark")
    payload = detection_payload(result)

    assert result.status == "ok"
    assert payload["spacing"]["selected_region_count"] == 1
    assert payload["spacing"]["row_count"] == 1
    assert payload["spacing"]["unassigned_ink_pixels"] == 0
    assert payload["spacing"]["rows"][0]["inferred_count"] == 1


def test_spaced_hypothesis_payload_and_overlays_are_json_serializable(tmp_path) -> None:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((15, 55, 49, 18), fill="black", width=3)
    draw.line((44, 55, 70, 18), fill="black", width=3)
    result = detect_spaced_region_hypotheses(image, mode="dark", expected_count=2)

    payload = spaced_hypotheses_payload(result)
    encoded = json.dumps(payload)
    manifest = write_spaced_hypotheses(result, tmp_path)

    assert json.loads(encoded)["search"]["truncated"] == result.truncated
    assert payload["page_hypotheses"][0]["count"] == 2
    assert payload["page_hypotheses"][0]["rows"][0]["regions"][0]["body_count"] == 1
    assert (tmp_path / "cleaned_mask.png").is_file()
    assert (tmp_path / "candidate_cuts.png").is_file()
    assert (tmp_path / "selected_partition.png").is_file()
    assert (tmp_path / "alternative_partition_2.png").is_file()
    assert manifest.is_file()
