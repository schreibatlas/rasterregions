from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from ._cleanup import clean_page_mask
from ._grouping import (
    _assign_spacing_marks,
    _automatic_regions,
    _make_region,
    _spacing_body_components,
)
from ._image import normalize_image
from ._partition import _expected_count_regions, _row_partition_regions
from ._rows import (
    _rows_from_components,
    _spacing_rows_from_components,
    detect_rows,
)
from .mask import foreground_mask
from .models import (
    BBox,
    Component,
    DetectionConfig,
    DetectionDiagnostics,
    DetectionResult,
    DetectionStatus,
    ExpectedCountScope,
    MaskMode,
    MaskResult,
    PartitionBoundary,
    Region,
    RowSegmentationDiagnostics,
    SecondaryMarkAssignment,
)

__all__ = ["detect_regions", "detect_rows", "detect_spaced_regions"]


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


def detect_spaced_regions(
    source: str | Path | Image.Image | np.ndarray,
    *,
    mode: MaskMode = "auto",
    row_counts: Sequence[int | None] | None = None,
    min_confidence: float = 0.3,
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """Detect anonymous regions using row-scale spacing and projection evidence.

    ``row_counts`` optionally gives an exact positive region count for each detected
    row; ``None`` entries request automatic inference. Use ``DetectionConfig.spacing``
    to tune scale-relative padding, width priors, and ambiguity margins.
    """
    config = config or DetectionConfig()
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")
    normalized = normalize_image(source)
    image = normalized.image
    mask_result = foreground_mask(normalized, mode=mode, config=config.mask)
    cleaned = clean_page_mask(mask_result.mask, config.cleanup)
    layout = _spacing_rows_from_components(
        cleaned.components,
        config.grouping,
        config.spacing,
        cleaned.mask.shape,
    )
    rows = layout.rows
    if row_counts is None:
        requested: tuple[int | None, ...] = (None,) * len(rows)
    else:
        if isinstance(row_counts, (str, bytes)):
            raise ValueError(
                "row_counts must contain one positive integer or None per detected row"
            )
        requested = tuple(row_counts)
        if len(requested) != len(rows):
            raise ValueError("row_counts must contain one entry per detected row")
        if any(
            count is not None
            and (isinstance(count, bool) or not isinstance(count, int) or count < 1)
            for count in requested
        ):
            raise ValueError("row counts must be positive integers or None")

    body_components = _spacing_body_components(cleaned.components)
    body_ids = {component.index for component in body_components}
    component_by_index = {component.index: component for component in cleaned.components}
    assignment_by_index = {
        component.index: layout.component_row_assignments[index]
        for index, component in enumerate(cleaned.components)
    }
    ambiguous_row_ids = {
        component_index
        for indices in layout.ambiguous_component_indices
        for component_index in indices
    }
    unassigned_row_ids = set(layout.unassigned_component_indices)
    body_height = (
        float(np.median([component.height for component in body_components]))
        if body_components
        else 0.0
    )
    padding = max(0, round(body_height * config.spacing.padding_ratio))

    regions: list[Region] = []
    row_diagnostics: list[RowSegmentationDiagnostics] = []
    partition_boundaries: list[PartitionBoundary] = []
    spacing_warnings: list[str] = []
    failed_row = False
    low_confidence = False

    for row_index, (row, requested_count) in enumerate(zip(rows, requested, strict=True)):
        members = tuple(
            component
            for component in cleaned.components
            if component.index in row.component_indices
        )
        row_body_ids = layout.body_component_indices[row_index]
        row_bodies = [component_by_index[index] for index in row_body_ids]
        row_height = float(
            np.median([component.height for component in row_bodies or list(members)])
        )
        row_padding = max(0, round(row_height * config.spacing.padding_ratio))
        partition = _row_partition_regions(
            image,
            cleaned.mask,
            cleaned.components,
            row,
            row_index,
            cleaned.median_height,
            requested_count,
            row_padding,
            config.grouping,
            config.partition,
            config.spacing,
        )
        row_warnings: list[str] = []
        row_regions: list[Region] = []
        row_boundaries: tuple[PartitionBoundary, ...] = ()
        row_confidence = 0.0
        row_cost = 0.0
        margin = None
        width_deviation = None
        body_counts: tuple[int, ...] = ()
        if partition is None:
            target_text = (
                f"{requested_count} regions"
                if requested_count is not None
                else "a credible region partition"
            )
            row_warnings.append(f"Unable to recover {target_text} for row {row_index}.")
            spacing_warnings.extend(row_warnings)
            failed_row = True
        else:
            row_regions = list(partition.regions)
            row_boundaries = partition.partition_boundaries
            partition_boundaries.extend(row_boundaries)
            row_confidence = partition.confidence
            row_cost = partition.cost
            margin = partition.alternative_cost_margin
            width_deviation = partition.width_prior_deviation
            body_counts = partition.body_counts_per_region
            if (
                requested_count is None
                and margin is not None
                and margin < config.spacing.ambiguity_margin
            ):
                row_confidence = min(
                    row_confidence,
                    margin / max(1e-9, config.spacing.ambiguity_margin),
                )
                row_warnings.append("Several automatic region counts have similar partition costs.")
            if width_deviation is not None and width_deviation > 0.70:
                row_warnings.append(
                    "Region widths deviate strongly from the row width prior; a merge may remain."
                )
            if (
                requested_count is None
                and len(row_bodies) == 1
                and any(
                    boundary.valley_strength is not None
                    and boundary.valley_strength >= 0.80
                    and not boundary.selected
                    for boundary in row_boundaries
                )
            ):
                row_warnings.append(
                    "A strong internal valley remains inside one connected body; "
                    "the inferred count is ambiguous."
                )
            if any(count == 0 for count in body_counts):
                row_warnings.append("At least one region has no primary-body evidence.")
            if row_warnings:
                spacing_warnings.extend(row_warnings)
            if row_confidence < min_confidence or row_warnings:
                low_confidence = True

        if not row_regions:
            row_confidence = 0.0
        elif row_confidence < min_confidence:
            low_confidence = True
        center_source = row_bodies or list(members)
        row_diagnostics.append(
            RowSegmentationDiagnostics(
                row_index=row_index,
                bbox=row.bbox,
                component_indices=row.component_indices,
                y_range=(row.bbox.y0, row.bbox.y1),
                center_y=float(np.median([component.center_y for component in center_source])),
                baseline_y=float(max(component.bbox.y1 for component in center_source)),
                body_height=max(1.0, row_height),
                inferred_count=len(row_regions) or None,
                requested_count=requested_count,
                selected_boundaries=tuple(
                    boundary.x for boundary in row_boundaries if boundary.selected
                ),
                confidence=float(np.clip(row_confidence, 0.0, 1.0)),
                cost=max(0.0, row_cost),
                alternative_cost_margin=margin,
                width_prior_deviation=width_deviation,
                body_counts_per_region=body_counts,
                warnings=tuple(row_warnings),
                ambiguous_component_indices=layout.ambiguous_component_indices[row_index],
            )
        )
        for column_index, region in enumerate(row_regions):
            regions.append(
                replace(
                    region,
                    region_id=f"region-{len(regions):03d}",
                    row_index=row_index,
                    column_index=column_index,
                    reading_order=len(regions),
                )
            )

    def component_pixels_in_region(region: Region, component: Component) -> bool:
        ys = component.ys
        xs = component.xs
        in_box = (
            (ys >= region.bbox.y0)
            & (ys < region.bbox.y1)
            & (xs >= region.bbox.x0)
            & (xs < region.bbox.x1)
        )
        if not np.any(in_box):
            return False
        return bool(
            np.any(
                region.mask[
                    ys[in_box] - region.bbox.y0,
                    xs[in_box] - region.bbox.x0,
                ]
            )
        )

    body_groups = [
        tuple(
            component
            for component in body_components
            if component_pixels_in_region(region, component)
        )
        for region in regions
    ]
    row_assigned_marks = tuple(
        component
        for component in cleaned.components
        if component.index not in body_ids and assignment_by_index.get(component.index) is not None
    )
    assigned_decisions, assigned_marks = (
        _assign_spacing_marks(
            row_assigned_marks,
            body_groups,
            body_height,
            config.spacing,
        )
        if regions
        else ((), ())
    )
    decision_by_component = {decision.component_index: decision for decision in assigned_decisions}
    secondary_assignments = []
    for component in cleaned.components:
        if component.index in body_ids:
            continue
        if component.index in ambiguous_row_ids:
            decision = SecondaryMarkAssignment(component.index, None, None, None, None, "ambiguous")
        elif component.index in unassigned_row_ids:
            decision = SecondaryMarkAssignment(
                component.index, None, None, None, None, "unassigned"
            )
        else:
            decision = decision_by_component.get(
                component.index,
                SecondaryMarkAssignment(component.index, None, None, None, None, "unassigned"),
            )
        secondary_assignments.append(decision)
    secondary_assignments.sort(key=lambda item: item.component_index)

    region_masks: list[np.ndarray] = []
    for region in regions:
        full_mask = np.zeros(cleaned.mask.shape, dtype=bool)
        full_mask[region.bbox.y0 : region.bbox.y1, region.bbox.x0 : region.bbox.x1] = region.mask
        region_masks.append(full_mask)
    secondary_components = [
        component for component in cleaned.components if component.index not in body_ids
    ]
    for component in secondary_components:
        for full_mask in region_masks:
            full_mask[component.ys, component.xs] = False
    for local_region, marks in enumerate(assigned_marks):
        for component in marks:
            region_masks[local_region][component.ys, component.xs] = True

    page_bbox = BBox(0, 0, image.width, image.height)
    owned_mask = np.zeros(cleaned.mask.shape, dtype=bool)
    owned_regions: list[Region] = []
    for region, full_mask in zip(regions, region_masks, strict=True):
        ys, xs = np.nonzero(full_mask)
        if not len(xs):
            continue
        box = BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        component_indices = tuple(
            component.index
            for component in cleaned.components
            if np.any(full_mask[component.ys, component.xs])
        )
        owned_regions.append(
            _make_region(
                image,
                full_mask,
                box,
                region_id=region.region_id,
                row_index=region.row_index,
                column_index=region.column_index,
                reading_order=region.reading_order,
                confidence=region.confidence,
                component_indices=component_indices,
                padding=max(
                    0,
                    round(
                        row_diagnostics[region.row_index].body_height * config.spacing.padding_ratio
                    ),
                ),
                clip=page_bbox,
            )
        )
        owned_mask |= full_mask

    unassigned_mask = cleaned.mask & ~owned_mask
    unassigned_fraction = float(unassigned_mask.sum() / max(1, cleaned.mask.sum()))
    ambiguous_marks = {
        decision.component_index
        for decision in secondary_assignments
        if decision.status == "ambiguous"
    }
    row_diagnostics = [
        replace(
            diagnostic,
            ambiguous_component_indices=tuple(
                sorted(
                    set(diagnostic.ambiguous_component_indices)
                    | {
                        component_index
                        for component_index in ambiguous_marks
                        if assignment_by_index.get(component_index) == diagnostic.row_index
                    }
                )
            ),
        )
        for diagnostic in row_diagnostics
    ]
    if any(decision.status != "assigned" for decision in secondary_assignments):
        low_confidence = True
    if unassigned_fraction > 0.0:
        low_confidence = True
    if not owned_regions:
        status: DetectionStatus = "failed"
    elif failed_row or low_confidence:
        status = "ambiguous"
    else:
        status = "ok"

    margins = [
        diagnostic.alternative_cost_margin
        for diagnostic in row_diagnostics
        if diagnostic.alternative_cost_margin is not None
    ]
    deviations = [
        diagnostic.width_prior_deviation
        for diagnostic in row_diagnostics
        if diagnostic.width_prior_deviation is not None
    ]
    body_counts_per_region = tuple(
        count for diagnostic in row_diagnostics for count in diagnostic.body_counts_per_region
    )
    diagnostics = DetectionDiagnostics(
        image_size=image.size,
        mask_threshold=mask_result.threshold,
        raw_component_count=len(cleaned.component_decisions),
        cleaned_component_count=len(cleaned.components),
        component_decisions=cleaned.component_decisions,
        partition_boundaries=tuple(partition_boundaries),
        mode=mode,
        padding=padding,
        min_confidence=min_confidence,
        mask_method=mask_result.method,
        mask_confidence=mask_result.confidence,
        foreground_fraction=float(mask_result.mask.mean()),
        row_segmentation=tuple(row_diagnostics),
        secondary_mark_assignments=tuple(secondary_assignments),
        selected_region_count=len(owned_regions),
        row_count=len(rows),
        best_second_best_partition_margin=min(margins) if margins else None,
        width_prior_deviation=float(np.mean(deviations)) if deviations else None,
        body_counts_per_region=body_counts_per_region,
        unassigned_ink_fraction=unassigned_fraction,
    )
    return DetectionResult(
        source_path=normalized.source_path,
        image=image,
        mask_result=mask_result,
        clean_mask=cleaned.mask,
        components=cleaned.components,
        rows=rows,
        regions=tuple(owned_regions),
        status=status,
        warnings=tuple([*mask_result.warnings, *cleaned.warnings, *spacing_warnings]),
        component_decisions=cleaned.component_decisions,
        diagnostics=diagnostics,
        unassigned_mask=unassigned_mask,
    )
