from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from ._cleanup import clean_page_mask
from ._grouping import _automatic_regions
from ._image import normalize_image
from ._partition import _expected_count_regions
from ._rows import _rows_from_components, detect_rows
from .mask import foreground_mask
from .models import (
    BBox,
    DetectionConfig,
    DetectionDiagnostics,
    DetectionResult,
    DetectionStatus,
    ExpectedCountScope,
    MaskMode,
    MaskResult,
    PartitionBoundary,
)

__all__ = ["detect_regions", "detect_rows"]


def _translate_roi_result(
    result: DetectionResult,
    full_image: Image.Image,
    source_path: Path | None,
    roi: BBox,
) -> DetectionResult:
    width, height = full_image.size
    roi_slice = (slice(roi.y0, roi.y1), slice(roi.x0, roi.x1))
    raw_mask = np.zeros((height, width), dtype=bool)
    clean_mask = np.zeros((height, width), dtype=bool)
    raw_mask[roi_slice] = result.mask_result.mask
    clean_mask[roi_slice] = result.clean_mask

    def offset_bbox(bbox: BBox) -> BBox:
        return BBox(
            bbox.x0 + roi.x0,
            bbox.y0 + roi.y0,
            bbox.x1 + roi.x0,
            bbox.y1 + roi.y0,
        )

    components = tuple(
        replace(
            component,
            bbox=offset_bbox(component.bbox),
            ys=component.ys + roi.y0,
            xs=component.xs + roi.x0,
        )
        for component in result.components
    )
    decisions = tuple(
        replace(decision, bbox=offset_bbox(decision.bbox))
        for decision in result.component_decisions
    )
    rows = tuple(replace(row, bbox=offset_bbox(row.bbox)) for row in result.rows)
    regions = tuple(replace(region, bbox=offset_bbox(region.bbox)) for region in result.regions)
    boundaries = tuple(
        replace(boundary, x=boundary.x + roi.x0)
        for boundary in (result.diagnostics.partition_boundaries if result.diagnostics else ())
    )
    mask_result = MaskResult(
        raw_mask,
        result.mask_result.confidence,
        result.mask_result.method,
        result.mask_result.warnings,
        threshold=result.mask_result.threshold,
    )
    diagnostics = result.diagnostics
    if diagnostics is not None:
        diagnostics = replace(
            diagnostics,
            image_size=(width, height),
            component_decisions=decisions,
            partition_boundaries=boundaries,
            foreground_fraction=float(raw_mask.mean()),
            roi=roi,
        )
    return DetectionResult(
        source_path=source_path,
        image=full_image,
        mask_result=mask_result,
        clean_mask=clean_mask,
        components=components,
        rows=rows,
        regions=regions,
        expected_count=result.expected_count,
        status=result.status,
        warnings=result.warnings,
        component_decisions=decisions,
        expected_count_scope=result.expected_count_scope,
        diagnostics=diagnostics,
    )


def detect_regions(
    source: str | Path | Image.Image | np.ndarray,
    *,
    expected_count: int | None = None,
    expected_count_scope: ExpectedCountScope = "row",
    mode: MaskMode = "auto",
    padding: int | None = None,
    min_confidence: float = 0.3,
    config: DetectionConfig | None = None,
    roi: BBox | None = None,
) -> DetectionResult:
    """Detect anonymous ordered foreground regions.

    With no ``expected_count``, rows and region groups are inferred. When a count
    is supplied, projection valleys and component-crossing costs partition a row
    or page. The default scope is ``row``; ``page`` constrains the total across
    rows. A caller should only provide a count when it has external evidence.

    ``min_confidence`` gates region or partition confidence, not mask confidence.
    Low-confidence automatic regions make the status ``ambiguous``. A missing or
    low-confidence expected-count partition makes the status ``failed`` and
    returns no regions.

    ``roi`` restricts analysis to a half-open bbox. Returned geometry and masks
    use full-image coordinates, and diagnostics record the ROI.
    """
    config = config or DetectionConfig()
    if expected_count_scope not in {"row", "page"}:
        raise ValueError("expected_count_scope must be row or page")
    if expected_count is not None and expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")

    if roi is not None:
        normalized = normalize_image(source)
        if roi.x1 > normalized.image.width or roi.y1 > normalized.image.height:
            raise ValueError("roi exceeds image bounds")
        cropped = normalized.image.crop(roi.as_tuple())
        result = detect_regions(
            cropped,
            expected_count=expected_count,
            expected_count_scope=expected_count_scope,
            mode=mode,
            padding=padding,
            min_confidence=min_confidence,
            config=config,
        )
        return _translate_roi_result(result, normalized.image, normalized.source_path, roi)
    normalized = normalize_image(source)
    image = normalized.image
    mask_result = foreground_mask(normalized, mode=mode, config=config.mask)
    cleaned = clean_page_mask(mask_result.mask, config.cleanup)
    rows = _rows_from_components(
        cleaned.components,
        cleaned.primary_area,
        cleaned.median_height,
        config.grouping,
    )
    if padding is None:
        padding = max(1, round(cleaned.median_height * 0.08)) if cleaned.median_height else 1
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    warnings = list(cleaned.warnings)
    status: DetectionStatus = "ok"
    partition_boundaries: tuple[PartitionBoundary, ...] = ()
    if expected_count is None:
        regions = _automatic_regions(
            image,
            cleaned.mask,
            cleaned.components,
            cleaned.primary_area,
            cleaned.median_height,
            rows,
            padding,
            config.grouping,
        )
        if not regions:
            status = "failed"
        elif any(region.confidence < min_confidence for region in regions):
            status = "ambiguous"
    else:
        selected = _expected_count_regions(
            image,
            cleaned.mask,
            cleaned.components,
            rows,
            cleaned.median_height,
            expected_count,
            padding,
            config.grouping,
            expected_count_scope,
            config.partition,
        )
        if selected is None:
            regions = []
            status = "failed"
            warnings.append(f"Unable to recover {expected_count} credible regions.")
        else:
            regions = list(selected.regions)
            partition_boundaries = selected.partition_boundaries
            if selected.confidence < min_confidence:
                regions = []
                status = "failed"
                warnings.append(
                    f"Unable to recover {expected_count} credible regions: "
                    f"best partition confidence is {selected.confidence:.3f}, "
                    f"below the required {min_confidence:.3f}."
                )

    diagnostics = DetectionDiagnostics(
        image_size=image.size,
        mask_threshold=mask_result.threshold,
        raw_component_count=len(cleaned.component_decisions),
        cleaned_component_count=len(cleaned.components),
        component_decisions=cleaned.component_decisions,
        partition_boundaries=partition_boundaries,
        mode=mode,
        padding=padding,
        min_confidence=min_confidence,
        expected_count=expected_count,
        expected_count_scope=expected_count_scope,
        mask_method=mask_result.method,
        mask_confidence=mask_result.confidence,
        foreground_fraction=float(mask_result.mask.mean()),
    )
    return DetectionResult(
        source_path=normalized.source_path,
        image=image,
        mask_result=mask_result,
        clean_mask=cleaned.mask,
        components=cleaned.components,
        rows=rows,
        regions=tuple(regions),
        expected_count=expected_count,
        status=status,
        warnings=tuple([*mask_result.warnings, *warnings]),
        component_decisions=cleaned.component_decisions,
        expected_count_scope=expected_count_scope,
        diagnostics=diagnostics,
    )
