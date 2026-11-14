from __future__ import annotations

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
    DetectionConfig,
    DetectionDiagnostics,
    DetectionResult,
    DetectionStatus,
    ExpectedCountScope,
    MaskMode,
    PartitionBoundary,
)

__all__ = ["detect_regions", "detect_rows"]


def detect_regions(
    source: str | Path | Image.Image | np.ndarray,
    *,
    expected_count: int | None = None,
    expected_count_scope: ExpectedCountScope = "row",
    mode: MaskMode = "auto",
    padding: int | None = None,
    min_confidence: float = 0.3,
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """Detect anonymous ordered foreground regions.

    With no ``expected_count``, all rows and region groups are inferred. When
    ``expected_count`` is supplied, projection valleys and component-crossing
    costs partition a row or page. The default expected-count scope is ``row``;
    use ``expected_count_scope="page"`` to constrain the total across all rows.

    ``min_confidence`` gates region/partition confidence only. Mask confidence
    remains a separate diagnostic and warning. A low-confidence automatic region
    makes status ambiguous, while a low-confidence expected-count partition fails.
    """
    config = config or DetectionConfig()
    if expected_count_scope not in {"row", "page"}:
        raise ValueError("expected_count_scope must be row or page")
    if expected_count is not None and expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")

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
