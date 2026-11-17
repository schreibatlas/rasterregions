"""Write overlays and diagnostics for opt-in spaced-region segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import detect_spaced_regions
from rasterregions.io import detection_payload

ROOT = Path(__file__).resolve().parent
BUILD = ROOT.parent / "build" / "examples" / "08_spaced_regions"


def _synthetic_page() -> Image.Image:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((15, 55, 49, 18), fill="black", width=3)
    draw.line((44, 55, 70, 18), fill="black", width=3)
    return image


def main() -> None:
    result = detect_spaced_regions(_synthetic_page(), mode="dark")
    BUILD.mkdir(parents=True, exist_ok=True)

    Image.fromarray(result.mask_result.mask.astype(np.uint8) * 255).save(
        BUILD / "input_mask.png"
    )

    row_overlay = result.image.copy()
    draw = ImageDraw.Draw(row_overlay)
    for row in result.rows:
        draw.rectangle(row.bbox.as_tuple(), outline=(30, 110, 220), width=1)
    row_overlay.save(BUILD / "row_overlay.png")

    component_overlay = result.image.copy()
    draw = ImageDraw.Draw(component_overlay)
    colors = ((230, 70, 70), (40, 160, 80), (40, 100, 220), (210, 130, 20))
    for index, component in enumerate(result.components):
        draw.point(
            list(zip(component.xs.tolist(), component.ys.tolist(), strict=True)),
            fill=colors[index % len(colors)],
        )
        draw.rectangle(component.bbox.as_tuple(), outline=colors[index % len(colors)])
    component_overlay.save(BUILD / "component_overlay.png")

    boundary_overlay = result.image.copy()
    draw = ImageDraw.Draw(boundary_overlay)
    for boundary in result.diagnostics.partition_boundaries:
        row = (
            next(
                (
                    diagnostic
                    for diagnostic in result.diagnostics.row_segmentation
                    if boundary.x in diagnostic.selected_boundaries
                ),
                None,
            )
            if boundary.selected
            else None
        )
        y0, y1 = row.y_range if row else (0, result.image.height)
        color = (230, 40, 40) if boundary.selected else (130, 130, 130)
        draw.line((boundary.x, y0, boundary.x, y1), fill=color, width=1)
    boundary_overlay.save(BUILD / "candidate_boundary_overlay.png")

    region_overlay = Image.new("RGB", result.image.size, "white")
    draw = ImageDraw.Draw(region_overlay)
    for index, region in enumerate(result.regions):
        color = colors[index % len(colors)]
        pixels = [
            (region.bbox.x0 + int(x), region.bbox.y0 + int(y))
            for y, x in zip(*np.nonzero(region.mask), strict=True)
        ]
        draw.point(pixels, fill=color)
        draw.rectangle(region.bbox.as_tuple(), outline=color, width=1)
    region_overlay.save(BUILD / "final_region_overlay.png")

    (BUILD / "diagnostics.json").write_text(
        json.dumps(detection_payload(result), indent=2) + "\n",
        encoding="utf-8",
    )
    print("status:", result.status)
    print("regions:", len(result.regions))
    print("artifacts:", BUILD)


if __name__ == "__main__":
    main()
