from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from .models import DetectionResult


def detection_payload(result: DetectionResult) -> dict:
    return {
        "source": result.source_path.name if result.source_path else None,
        "status": result.status,
        "expected_count": result.expected_count,
        "detected_count": len(result.regions),
        "mask": {
            "method": result.mask_result.method,
            "confidence": round(result.mask_result.confidence, 4),
            "warnings": list(result.mask_result.warnings),
        },
        "warnings": list(result.warnings),
        "regions": [
            {
                "id": region.region_id,
                "row": region.row_index,
                "column": region.column_index,
                "reading_order": region.reading_order,
                "bbox": list(region.bbox.as_tuple()),
                "confidence": round(region.confidence, 4),
                "component_indices": list(region.component_indices),
            }
            for region in result.regions
        ],
    }


def write_detection(result: DetectionResult, output_dir: str | Path) -> Path:
    """Write crops, mask, overlay, and JSON diagnostics."""
    output = Path(output_dir)
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    for region in result.regions:
        region.image.save(crops / f"{region.region_id}.png", format="PNG")

    from PIL import Image

    Image.fromarray(np.asarray(result.clean_mask, dtype=np.uint8) * 255).save(
        output / "mask.png", format="PNG"
    )
    overlay = result.image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for region in result.regions:
        x0, y0, x1, y1 = region.bbox.as_tuple()
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(230, 45, 35), width=2)
        draw.text((x0, max(0, y0 - 12)), region.region_id, fill=(25, 75, 220))
    overlay.save(output / "regions.png", format="PNG")

    manifest = output / "regions.json"
    manifest.write_text(
        json.dumps(detection_payload(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
