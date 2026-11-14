from __future__ import annotations

import json

from PIL import Image, ImageDraw

from rasterregions import BBox, crop, detect_regions, tighten_bbox
from rasterregions.io import write_detection


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
