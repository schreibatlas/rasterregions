from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import DetectionResult


def detection_payload(result: DetectionResult) -> dict:
    """Return the schema-versioned JSON-compatible representation of ``result``.

    Component indexes in the payload use the ``raw_mask`` namespace.
    """
    from . import __version__

    diagnostics = result.diagnostics
    decisions = diagnostics.component_decisions if diagnostics else result.component_decisions
    boundaries = diagnostics.partition_boundaries if diagnostics else ()
    width, height = diagnostics.image_size if diagnostics else result.image.size
    return {
        "rasterregions_version": __version__,
        "schema_version": 1,
        "component_index_namespace": "raw_mask",
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
            "roi": list(diagnostics.roi.as_tuple()) if diagnostics and diagnostics.roi else None,
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


def _atomic_save_image(image: Image.Image, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        image.save(temporary_path, format="PNG")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_text(text: str, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _remove_stale_crops(crops: Path, *, remove_directory: bool) -> None:
    if crops.is_dir():
        for path in crops.glob("region-*.png"):
            if path.is_file():
                path.unlink()
        if remove_directory:
            with contextlib.suppress(OSError):
                crops.rmdir()


def write_detection(
    result: DetectionResult,
    output_dir: str | Path,
    *,
    include_crops: bool = True,
    include_overlay: bool = True,
    include_mask: bool = True,
) -> Path:
    """Write selected image artifacts and publish ``regions.json`` last.

    Reruns remove stale managed crops and disabled mask/overlay artifacts while
    leaving unrelated files in the output directory untouched."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    crops = output / "crops"
    _remove_stale_crops(crops, remove_directory=not include_crops)
    if include_crops:
        crops.mkdir(parents=True, exist_ok=True)
        for region in result.regions:
            _atomic_save_image(region.image, crops / f"{region.region_id}.png")

    mask_path = output / "mask.png"
    if include_mask:
        mask = Image.fromarray(np.asarray(result.clean_mask, dtype=np.uint8) * 255)
        _atomic_save_image(mask, mask_path)
    else:
        mask_path.unlink(missing_ok=True)

    overlay_path = output / "regions.png"
    if include_overlay:
        overlay = result.image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for region in result.regions:
            x0, y0, x1, y1 = region.bbox.as_tuple()
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(230, 45, 35), width=2)
            draw.text((x0, max(0, y0 - 12)), region.region_id, fill=(25, 75, 220))
        _atomic_save_image(overlay, overlay_path)
    else:
        overlay_path.unlink(missing_ok=True)

    manifest = output / "regions.json"
    payload = json.dumps(detection_payload(result), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(payload, manifest)
    return manifest
