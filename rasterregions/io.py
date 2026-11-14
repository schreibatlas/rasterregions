from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import DetectionResult


def detection_payload(result: DetectionResult) -> dict:
    from . import __version__

    diagnostics = result.diagnostics
    decisions = diagnostics.component_decisions if diagnostics else result.component_decisions
    boundaries = diagnostics.partition_boundaries if diagnostics else ()
    width, height = diagnostics.image_size if diagnostics else result.image.size
    return {
        "rasterregions_version": __version__,
        "source": result.source_path.name if result.source_path else None,
        "status": result.status,
        "expected_count": result.expected_count,
        "detected_count": len(result.regions),
        "image": {"width": width, "height": height},
        "parameters": {
            "mode": diagnostics.mode if diagnostics else "auto",
            "padding": diagnostics.padding if diagnostics else None,
            "min_confidence": diagnostics.min_confidence if diagnostics else 0.3,
            "expected_count": result.expected_count,
            "expected_count_scope": result.expected_count_scope,
        },
        "mask": {
            "method": result.mask_result.method,
            "threshold": result.mask_result.threshold,
            "confidence": round(result.mask_result.confidence, 4),
            "foreground_fraction": round(float(result.mask_result.mask.mean()), 6),
            "warnings": list(result.mask_result.warnings),
        },
        "cleanup": [
            {
                "component_index": decision.component_index,
                "bbox": list(decision.bbox.as_tuple()),
                "area": decision.area,
                "action": decision.action,
                "reason": decision.reason,
            }
            for decision in decisions
        ],
        "rows": [
            {
                "row": index,
                "bbox": list(row.bbox.as_tuple()),
                "component_indices": list(row.component_indices),
            }
            for index, row in enumerate(result.rows)
        ],
        "partition_boundaries": [
            {
                "x": boundary.x,
                "projection": boundary.projection,
                "crossing_components": boundary.crossing_components,
                "confidence": round(boundary.confidence, 4),
            }
            for boundary in boundaries
        ],
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


def write_detection(
    result: DetectionResult,
    output_dir: str | Path,
    *,
    include_crops: bool = True,
    include_overlay: bool = True,
    include_mask: bool = True,
) -> Path:
    """Write selected image artifacts and the JSON diagnostics manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if include_crops:
        crops = output / "crops"
        crops.mkdir(parents=True, exist_ok=True)
        for region in result.regions:
            region.image.save(crops / f"{region.region_id}.png", format="PNG")

    if include_mask:
        Image.fromarray(np.asarray(result.clean_mask, dtype=np.uint8) * 255).save(
            output / "mask.png", format="PNG"
        )

    if include_overlay:
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
