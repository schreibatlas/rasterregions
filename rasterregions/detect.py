from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from ._cleanup import CleanedPage, clean_page_mask
from ._grouping import (
    _assign_spacing_marks,
    _automatic_regions,
    _BodyAnchor,
    _make_region,
    _merge_spacing_body_anchors,
    _spacing_attachment_score,
    _spacing_body_anchor_has_strong_valley,
    _spacing_body_anchors,
    _spacing_body_components,
)
from ._image import normalize_image
from ._partition import (
    RowPartition,
    _combine_page_partitions,
    _expected_count_regions,
    _row_partition_regions,
)
from ._rows import (
    SpacingRowLayout,
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
    PagePartitionHypothesis,
    PartitionBoundary,
    Region,
    RegionFrame,
    RegionHypothesis,
    RowBand,
    RowPartitionHypothesis,
    RowSegmentationDiagnostics,
    SecondaryMarkAssignment,
    SegmentationSearchLimitError,
    SpacedSegmentationHypotheses,
)

__all__ = [
    "detect_regions",
    "detect_rows",
    "detect_spaced_regions",
    "detect_spaced_region_hypotheses",
]


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


def _remove_unassigned_provisional_components(
    cleaned: CleanedPage,
    layout: SpacingRowLayout,
) -> CleanedPage:
    noise_indices = set(layout.unassigned_component_indices) & set(
        cleaned.provisional_component_indices
    )
    if not noise_indices:
        return cleaned

    clean_mask = cleaned.mask.copy()
    retained_components: list[Component] = []
    for component in cleaned.components:
        if component.index in noise_indices:
            clean_mask[component.ys, component.xs] = False
        else:
            retained_components.append(component)
    decisions = tuple(
        replace(
            decision,
            action="removed",
            reason="tiny component with no plausible body target",
            role="noise",
        )
        if decision.component_index in noise_indices
        else decision
        for decision in cleaned.component_decisions
    )
    warnings = (
        *cleaned.warnings,
        *(f"Removed component {index}: tiny isolated noise." for index in sorted(noise_indices)),
    )
    return replace(
        cleaned,
        mask=clean_mask,
        components=tuple(retained_components),
        warnings=warnings,
        component_decisions=decisions,
        provisional_component_indices=tuple(
            index for index in cleaned.provisional_component_indices if index not in noise_indices
        ),
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


def _combine_unconstrained_rows(
    options_by_row: Sequence[Sequence[RowPartition]],
    config: DetectionConfig,
) -> tuple[tuple[tuple[RowPartition, ...], ...], bool]:
    states: list[tuple[float, tuple[RowPartition, ...]]] = [(0.0, ())]
    visited = 0
    truncated = False
    for row_options in options_by_row:
        candidates: list[tuple[float, tuple[RowPartition, ...]]] = []
        for cost, selected in states:
            for option in row_options:
                if visited >= config.spacing.max_page_states:
                    truncated = True
                    break
                visited += 1
                candidates.append((cost + option.cost, (*selected, option)))
            if truncated:
                break
        candidates.sort(
            key=lambda item: (
                item[0],
                tuple((row.cut_positions, len(row.regions)) for row in item[1]),
            )
        )
        if len(candidates) > config.spacing.hypothesis_limit:
            truncated = True
        states = candidates[: config.spacing.hypothesis_limit]
        if not states:
            break
    return tuple(rows for _, rows in states), truncated


def _row_from_body_anchors(
    image: Image.Image,
    found: tuple[Component, ...],
    component_indices: Sequence[int],
    page_shape: tuple[int, int],
    row_index: int,
    padding: int,
    anchors: Sequence[_BodyAnchor] | None = None,
    edit_cost: float = 0.0,
) -> RowPartition | None:
    source_anchors = _spacing_body_anchors(found, component_indices)
    anchors = source_anchors if anchors is None else anchors
    if not anchors:
        return None
    regions = []
    for column_index, anchor in enumerate(anchors):
        mask = np.zeros(page_shape, dtype=bool)
        mask[
            anchor.bbox.y0 : anchor.bbox.y1,
            anchor.bbox.x0 : anchor.bbox.x1,
        ] = anchor.mask
        regions.append(
            _make_region(
                image,
                mask,
                anchor.bbox,
                region_id=f"region-{row_index:03d}-{column_index:03d}",
                row_index=row_index,
                column_index=column_index,
                reading_order=column_index,
                confidence=1.0,
                component_indices=anchor.component_indices,
                padding=padding,
            )
        )
    anchor_positions = {
        component_index: position
        for position, anchor in enumerate(source_anchors)
        for component_index in anchor.component_indices
    }
    merged_groups = tuple(
        tuple(sorted(anchor_positions[index] for index in anchor.component_indices))
        for anchor in anchors
        if len(anchor.component_indices) > 1
    )
    return RowPartition(
        regions=tuple(regions),
        confidence=1.0,
        cost=edit_cost,
        partition_boundaries=(),
        body_counts_per_region=tuple(len(anchor.component_indices) for anchor in anchors),
        anchor_count=len(source_anchors),
        segmentation_strategy="hybrid" if merged_groups else "anchors",
        attempted_counts=(len(regions),) if merged_groups else (),
        merged_anchor_groups=merged_groups,
    )


def _row_with_local_anchor_repairs(
    image: Image.Image,
    mask: np.ndarray,
    found: tuple[Component, ...],
    row: RowBand,
    row_index: int,
    component_indices: Sequence[int],
    median_height: float,
    padding: int,
    config: DetectionConfig,
    min_confidence: float,
    target_count: int | None = None,
) -> RowPartition | None:
    anchors = _spacing_body_anchors(found, component_indices)
    partition = _row_from_body_anchors(
        image, found, component_indices, mask.shape, row_index, padding, anchors
    )
    if partition is None:
        return None
    suspicious = [
        index
        for index, anchor in enumerate(anchors)
        if _spacing_body_anchor_has_strong_valley(anchor)
    ]
    split_counts: dict[int, int] = {}
    if target_count is None:
        split_counts = {index: 2 for index in suspicious}
    elif target_count > len(anchors):
        if not suspicious and len(anchors) == 1:
            suspicious = [0]
        if suspicious:
            split_index = max(suspicious, key=lambda index: (anchors[index].width, -index))
            split_counts[split_index] = target_count - len(anchors) + 1
    components = {component.index: component for component in found}
    regions: list[Region] = []
    boundaries: list[PartitionBoundary] = []
    body_counts: list[int] = []
    cost = 0.0
    confidence = 1.0
    truncated = False
    ambiguous_repair = False
    attempted_counts: list[int] = []
    truncated_counts: list[int] = []
    split_anchor_indices: list[int] = []
    for index, (anchor, original) in enumerate(zip(anchors, partition.regions, strict=True)):
        split_count = split_counts.get(index)
        if split_count is None:
            regions.append(original)
            body_counts.append(len(anchor.component_indices))
            continue
        attempted_counts.append(split_count)
        component = components[anchor.component_indices[0]]
        try:
            repaired = _row_partition_regions(
                image,
                mask,
                (component,),
                row,
                row_index,
                median_height,
                split_count,
                padding,
                config.grouping,
                config.partition,
                config.spacing,
            )
        except SegmentationSearchLimitError:
            if target_count is not None:
                return None
            repaired = None
            truncated_counts.append(split_count)
            truncated = True
        if repaired is None or len(repaired.regions) != split_count:
            if target_count is not None:
                return None
            regions.append(original)
            body_counts.append(len(anchor.component_indices))
            ambiguous_repair = True
            continue
        split_anchor_indices.append(index)
        regions.extend(repaired.regions)
        boundaries.extend(repaired.partition_boundaries)
        body_counts.extend(repaired.body_counts_per_region)
        cost += repaired.cost + config.spacing.complexity_penalty * (split_count - 1)
        confidence = min(confidence, repaired.confidence)
        truncated |= repaired.truncated
        if repaired.truncated:
            truncated_counts.append(split_count)
    if target_count is not None and len(regions) != target_count:
        return None
    regions = [
        replace(region, column_index=index, reading_order=index)
        for index, region in enumerate(regions)
    ]
    if ambiguous_repair:
        confidence = min(confidence, min_confidence * 0.5)
    return replace(
        partition,
        regions=tuple(regions),
        confidence=confidence,
        cost=cost,
        partition_boundaries=tuple(boundaries),
        body_counts_per_region=tuple(body_counts),
        truncated=truncated,
        anchor_count=len(anchors),
        segmentation_strategy="hybrid" if split_anchor_indices else "anchors",
        attempted_counts=tuple(attempted_counts),
        truncated_counts=tuple(truncated_counts),
        split_anchor_indices=tuple(split_anchor_indices),
    )


def detect_spaced_regions(
    source: str | Path | Image.Image | np.ndarray,
    *,
    mode: MaskMode = "auto",
    row_counts: Sequence[int | None] | None = None,
    expected_count: int | None = None,
    min_confidence: float = 0.3,
    config: DetectionConfig | None = None,
    _row_hypotheses_out: list[tuple[RowPartition, ...]] | None = None,
    _page_options_out: list[tuple[RowPartition, ...]] | None = None,
    _truncated_out: list[bool] | None = None,
) -> DetectionResult:
    """Detect anonymous regions with spacing evidence and optional page count."""
    config = config or DetectionConfig()
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")
    normalized = normalize_image(source)
    image = normalized.image
    mask_result = foreground_mask(normalized, mode=mode, config=config.mask)
    cleaned = clean_page_mask(mask_result.mask, config.cleanup)
    provisional_ids = set(cleaned.provisional_component_indices)
    primary_components = tuple(
        component for component in cleaned.components if component.index not in provisional_ids
    )
    layout = _spacing_rows_from_components(
        cleaned.components,
        config.grouping,
        config.spacing,
        cleaned.mask.shape,
        primary_components=primary_components,
    )
    row_assignment_by_index = {
        component.index: layout.component_row_assignments[index]
        for index, component in enumerate(cleaned.components)
    }
    cleaned = _remove_unassigned_provisional_components(cleaned, layout)
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

    component_by_index = {component.index: component for component in cleaned.components}
    body_ids = {
        component_index
        for row_component_indices in layout.body_component_indices
        for component_index in row_component_indices
    }
    body_components = [component_by_index[index] for index in sorted(body_ids)]
    assignment_by_index = row_assignment_by_index
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

    page_options: tuple[tuple[RowPartition, ...], ...] = ()
    selected_partitions: tuple[RowPartition, ...] | None = None
    page_search_truncated = False
    page_allocation_failure: str | None = None
    row_partition_search_limited = False
    attempted_counts_by_row: list[tuple[int, ...]] = [() for _ in rows]
    truncated_counts_by_row: list[tuple[int, ...]] = [() for _ in rows]

    def try_partition_row(
        row: RowBand,
        row_index: int,
        expected: int | None,
        row_padding: int,
        row_components: tuple[Component, ...] | None = None,
    ) -> tuple[RowPartition | None, bool]:
        try:
            partition = _row_partition_regions(
                image,
                cleaned.mask,
                tuple(
                    component
                    for component in cleaned.components
                    if component.index in layout.body_component_indices[row_index]
                ),
                row,
                row_index,
                cleaned.median_height,
                expected,
                row_padding,
                config.grouping,
                config.partition,
                config.spacing,
            )
        except SegmentationSearchLimitError:
            return None, True
        return partition, False

    if expected_count is not None:
        options_by_row: list[tuple[RowPartition, ...]] = []
        count_candidates_truncated = False
        for row_index, row in enumerate(rows):
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
            requested_count = requested[row_index]
            if requested_count is not None:
                candidate_counts = [requested_count]
            else:
                candidate_limit = config.spacing.max_count_candidates
                count_candidates_truncated |= expected_count > candidate_limit
                if expected_count <= candidate_limit:
                    candidate_counts = list(range(1, expected_count + 1))
                else:
                    target = expected_count / max(1, len(rows))
                    spaced = np.linspace(1, expected_count, candidate_limit).round().astype(int)
                    candidate_counts = sorted(
                        set(spaced.tolist())
                        | {max(1, int(np.floor(target))), max(1, int(np.ceil(target)))}
                    )
                    if len(candidate_counts) > candidate_limit:
                        candidate_counts = sorted(
                            candidate_counts,
                            key=lambda count: (abs(count - target), count),
                        )[:candidate_limit]
                        candidate_counts.sort()
            count_options: list[RowPartition] = []
            seen_options: set[tuple[int, ...]] = set()
            attempted_counts: list[int] = []
            truncated_counts: list[int] = []
            for count in candidate_counts:
                attempted_counts.append(count)
                partition, search_limited = try_partition_row(row, row_index, count, row_padding)
                row_partition_search_limited |= search_limited
                count_candidates_truncated |= search_limited
                if search_limited or (partition is not None and partition.truncated):
                    truncated_counts.append(count)
                if partition is None:
                    continue
                for option in (partition, *partition.alternatives):
                    if option.cut_positions not in seen_options:
                        seen_options.add(option.cut_positions)
                        count_options.append(option)
            attempted_counts_by_row[row_index] = tuple(attempted_counts)
            truncated_counts_by_row[row_index] = tuple(truncated_counts)
            count_options = [
                replace(
                    option,
                    anchor_count=len(row_body_ids),
                    attempted_counts=tuple(attempted_counts),
                    truncated_counts=tuple(truncated_counts),
                )
                for option in count_options
            ]
            options_by_row.append(tuple(count_options))
        page_options, page_search_truncated = _combine_page_partitions(
            options_by_row, expected_count, config.spacing
        )
        page_search_truncated |= count_candidates_truncated
        if _row_hypotheses_out is not None:
            _row_hypotheses_out.extend(options_by_row)
        if _page_options_out is not None:
            _page_options_out.extend(page_options)
        if page_options:
            selected_partitions = page_options[0]
            requested = tuple(len(partition.regions) for partition in selected_partitions)
        else:
            page_allocation_failure = (
                f"No feasible row allocation sums to expected_count={expected_count}."
            )
            if page_search_truncated:
                page_allocation_failure += " Search budgets limited the candidate allocations."
    elif _page_options_out is not None:
        _page_options_out.clear()
    regions: list[Region] = []
    row_diagnostics: list[RowSegmentationDiagnostics] = []
    partition_boundaries: list[PartitionBoundary] = []
    spacing_warnings = [page_allocation_failure] if page_allocation_failure else []
    failed_row = page_allocation_failure is not None
    low_confidence = False

    row_partition_options: list[tuple[RowPartition, ...]] = []
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
        row_search_limited = False
        if page_allocation_failure is not None:
            partition = None
            row_search_limited = bool(truncated_counts_by_row[row_index])
        elif selected_partitions is not None:
            partition = selected_partitions[row_index]
            row_search_limited = bool(truncated_counts_by_row[row_index])
        elif requested_count is None and expected_count is None:
            partition = _row_with_local_anchor_repairs(
                image,
                cleaned.mask,
                cleaned.components,
                row,
                row_index,
                row_body_ids,
                cleaned.median_height,
                row_padding,
                config,
                min_confidence,
            )
            if partition is None:
                partition, search_limited = try_partition_row(
                    row, row_index, requested_count, row_padding
                )
                row_partition_search_limited |= search_limited
                row_search_limited |= search_limited
        elif expected_count is None and requested_count is not None:
            anchors = _spacing_body_anchors(cleaned.components, row_body_ids)
            if requested_count == len(anchors):
                partition = _row_from_body_anchors(
                    image,
                    cleaned.components,
                    row_body_ids,
                    cleaned.mask.shape,
                    row_index,
                    row_padding,
                    anchors,
                )
            elif requested_count < len(anchors):
                merged = _merge_spacing_body_anchors(anchors, requested_count)
                if merged is not None:
                    partition = _row_from_body_anchors(
                        image,
                        cleaned.components,
                        row_body_ids,
                        cleaned.mask.shape,
                        row_index,
                        row_padding,
                        merged,
                        config.spacing.complexity_penalty * (len(anchors) - requested_count),
                    )
                else:
                    partition = None
            else:
                partition = _row_with_local_anchor_repairs(
                    image,
                    cleaned.mask,
                    cleaned.components,
                    row,
                    row_index,
                    row_body_ids,
                    cleaned.median_height,
                    row_padding,
                    config,
                    min_confidence,
                    requested_count,
                )
            if partition is None:
                partition, search_limited = try_partition_row(
                    row, row_index, requested_count, row_padding
                )
                row_partition_search_limited |= search_limited
                row_search_limited |= search_limited
        else:
            partition, search_limited = try_partition_row(
                row, row_index, requested_count, row_padding
            )
            row_partition_search_limited |= search_limited
            row_search_limited |= search_limited
        partition_options = (partition, *partition.alternatives) if partition is not None else ()
        row_partition_options.append(partition_options)
        if expected_count is None and _row_hypotheses_out is not None:
            _row_hypotheses_out.append(partition_options)
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
                anchor_count=(
                    partition.anchor_count
                    if partition is not None and partition.anchor_count is not None
                    else len(row_body_ids)
                ),
                segmentation_strategy=(
                    partition.segmentation_strategy
                    if partition is not None and partition.segmentation_strategy is not None
                    else "partition"
                    if requested_count is not None or expected_count is not None
                    else "anchors"
                ),
                attempted_counts=(
                    partition.attempted_counts
                    if partition is not None and partition.attempted_counts
                    else attempted_counts_by_row[row_index]
                    or ((requested_count,) if requested_count is not None else ())
                ),
                truncated_counts=(
                    partition.truncated_counts
                    if partition is not None and partition.truncated_counts
                    else truncated_counts_by_row[row_index]
                    or (
                        (requested_count,)
                        if row_search_limited and requested_count is not None
                        else ()
                    )
                ),
                split_anchor_indices=partition.split_anchor_indices if partition else (),
                merged_anchor_groups=partition.merged_anchor_groups if partition else (),
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

    if expected_count is None:
        page_options, page_search_truncated = (
            _combine_unconstrained_rows(row_partition_options, config) if rows else ((), False)
        )
        if _page_options_out is not None:
            _page_options_out.extend(page_options)

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
            region_scales=tuple(
                row_diagnostics[region.row_index].body_height for region in regions
            ),
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
            candidate_scores = [
                (score, region_index)
                for region_index, bodies in enumerate(body_groups)
                if (
                    score := _spacing_attachment_score(
                        component,
                        bodies,
                        row_diagnostics[regions[region_index].row_index].body_height,
                    )
                )
                is not None
                and score <= 2.5
            ]
            candidate_scores.sort()
            decision = SecondaryMarkAssignment(
                component.index,
                None,
                None,
                None,
                None,
                "ambiguous",
                candidate_region_indices=tuple(
                    region_index for _, region_index in candidate_scores[:2]
                ),
            )
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
    role_by_index = {component_index: "primary" for component_index in body_ids}
    role_by_index.update(
        {
            assignment.component_index: (
                "secondary"
                if assignment.status == "assigned"
                else "ambiguous"
                if assignment.status == "ambiguous"
                else "secondary-candidate"
            )
            for assignment in secondary_assignments
        }
    )
    cleaned = replace(
        cleaned,
        component_decisions=tuple(
            replace(
                decision,
                role=role_by_index.get(decision.component_index, decision.role),
            )
            for decision in cleaned.component_decisions
        ),
    )
    ambiguous_components_by_region = {
        region_index: tuple(
            sorted(
                assignment.component_index
                for assignment in secondary_assignments
                if assignment.status == "ambiguous"
                and region_index in assignment.candidate_region_indices
            )
        )
        for region_index in range(len(regions))
    }

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

    region_index_by_id = {region.region_id: index for index, region in enumerate(regions)}
    owned_regions = [
        replace(
            region,
            primary_component_indices=tuple(
                index for index in region.component_indices if role_by_index.get(index) == "primary"
            ),
            secondary_component_indices=tuple(
                index
                for index in region.component_indices
                if role_by_index.get(index) == "secondary"
            ),
            ambiguous_component_indices=ambiguous_components_by_region[
                region_index_by_id[region.region_id]
            ],
        )
        for region in owned_regions
    ]
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
    row_search_truncated = any(
        option.truncated for options in row_partition_options for option in options
    )
    search_truncated = page_search_truncated or row_search_truncated or row_partition_search_limited
    search_limit = (
        "row-partition-budget"
        if row_partition_search_limited or row_search_truncated
        else "page-allocation-budget"
        if page_search_truncated
        else None
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
        expected_count=expected_count,
        expected_count_scope="page" if expected_count is not None else "row",
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
        search_truncated=search_truncated,
        search_limit=search_limit,
    )
    result = DetectionResult(
        source_path=normalized.source_path,
        image=image,
        mask_result=mask_result,
        clean_mask=cleaned.mask,
        components=cleaned.components,
        rows=rows,
        regions=tuple(owned_regions),
        expected_count=expected_count,
        expected_count_scope="page" if expected_count is not None else "row",
        status=status,
        warnings=tuple([*mask_result.warnings, *cleaned.warnings, *spacing_warnings]),
        component_decisions=cleaned.component_decisions,
        diagnostics=diagnostics,
        unassigned_mask=unassigned_mask,
    )
    if _truncated_out is not None:
        _truncated_out.append(search_truncated)
    return result


def _region_frame(result: DetectionResult, row_index: int) -> RegionFrame:
    diagnostics = result.diagnostics
    row_diagnostic = (
        diagnostics.row_segmentation[row_index]
        if diagnostics is not None and row_index < len(diagnostics.row_segmentation)
        else None
    )
    row = result.rows[row_index]
    return RegionFrame(
        baseline_y=row_diagnostic.baseline_y if row_diagnostic else float(row.bbox.y1),
        body_height=row_diagnostic.body_height if row_diagnostic else float(row.bbox.height),
        row_center_y=row_diagnostic.center_y if row_diagnostic else row.bbox.center_y,
        row_index=row_index,
    )


def _row_boundary_positions(
    row: RowBand,
    regions: Sequence[Region],
    partition: RowPartition | None,
) -> tuple[int, ...]:
    if partition is not None and len(partition.cut_positions) == len(regions) + 1:
        return partition.cut_positions
    internal = tuple(
        sorted(
            boundary
            for boundary in (
                [item.x for item in partition.partition_boundaries if item.selected]
                if partition is not None
                else []
            )
            if row.bbox.x0 < boundary < row.bbox.x1
        )
    )
    if len(internal) != len(regions) - 1:
        internal = tuple(region.bbox.x1 for region in regions[:-1])
    return (row.bbox.x0, *internal, row.bbox.x1)


def _selected_page_hypothesis(
    result: DetectionResult,
    selected_partitions: Sequence[RowPartition] | None,
) -> PagePartitionHypothesis | None:
    if not result.regions:
        return None
    assert result.unassigned_mask is not None
    diagnostics = result.diagnostics
    rows: list[RowPartitionHypothesis] = []
    regions_by_row = [
        sorted(
            (region for region in result.regions if region.row_index == row_index),
            key=lambda region: region.reading_order,
        )
        for row_index in range(len(result.rows))
    ]
    assignments = diagnostics.secondary_mark_assignments if diagnostics else ()
    body_ids = {component.index for component in _spacing_body_components(result.components)}
    for row_index, row_regions in enumerate(regions_by_row):
        if not row_regions:
            continue
        row = result.rows[row_index]
        row_diagnostic = (
            diagnostics.row_segmentation[row_index]
            if diagnostics and row_index < len(diagnostics.row_segmentation)
            else None
        )
        part = (
            selected_partitions[row_index]
            if selected_partitions and row_index < len(selected_partitions)
            else None
        )
        frame = _region_frame(result, row_index)
        region_hypotheses = tuple(
            RegionHypothesis(
                bbox=region.bbox,
                mask=region.mask,
                component_indices=region.component_indices,
                primary_component_indices=region.primary_component_indices,
                secondary_component_indices=region.secondary_component_indices,
                ambiguous_component_indices=region.ambiguous_component_indices,
                frame=frame,
                cost=row_diagnostic.cost if row_diagnostic else 0.0,
                confidence=region.confidence,
                warnings=row_diagnostic.warnings if row_diagnostic else (),
                cut_crossing_cost=(
                    part.cut_crossing_cost / max(1, len(row_regions)) if part else 0.0
                ),
                body_count=(
                    row_diagnostic.body_counts_per_region[region_index]
                    if row_diagnostic and region_index < len(row_diagnostic.body_counts_per_region)
                    else sum(index in body_ids for index in region.component_indices)
                ),
            )
            for region_index, region in enumerate(row_regions)
        )
        row_assignments = tuple(
            assignment
            for assignment in assignments
            if assignment.component_index in row.component_indices
            or (
                row_diagnostic
                and assignment.component_index in row_diagnostic.ambiguous_component_indices
            )
        )
        boundaries = _row_boundary_positions(row, row_regions, part)
        if len(boundaries) != len(row_regions) + 1:
            boundaries = (
                row.bbox.x0,
                *tuple(region.bbox.x1 for region in row_regions[:-1]),
                row.bbox.x1,
            )
        rows.append(
            RowPartitionHypothesis(
                row_index=row_index,
                count=len(row_regions),
                boundaries=boundaries,
                regions=region_hypotheses,
                cost=row_diagnostic.cost if row_diagnostic else 0.0,
                unassigned_ink_fraction=(
                    result.unassigned_mask[row.bbox.y0 : row.bbox.y1].sum()
                    / max(1, result.clean_mask[row.bbox.y0 : row.bbox.y1].sum())
                ),
                secondary_mark_assignments=row_assignments,
                boundary_evidence=(
                    part.partition_boundaries
                    if part
                    else tuple(
                        boundary
                        for boundary in (diagnostics.partition_boundaries if diagnostics else ())
                        if boundary.x in boundaries[1:-1]
                    )
                ),
                cut_crossing_cost=part.cut_crossing_cost if part else 0.0,
            )
        )
    unassigned_fraction = diagnostics.unassigned_ink_fraction if diagnostics else 0.0
    return PagePartitionHypothesis(
        row_hypotheses=tuple(rows),
        count=len(result.regions),
        cost=sum(row.cost for row in rows) + unassigned_fraction,
        unassigned_ink_fraction=unassigned_fraction,
        unassigned_mask=result.unassigned_mask,
    )


def _row_hypothesis_from_partition(
    result: DetectionResult,
    row_index: int,
    partition: RowPartition,
    region_offset: int,
) -> RowPartitionHypothesis:
    frame = _region_frame(result, row_index)
    body_ids = {
        decision.component_index
        for decision in result.component_decisions
        if decision.role == "primary"
    }
    regions = tuple(
        RegionHypothesis(
            bbox=region.bbox,
            mask=region.mask,
            component_indices=region.component_indices,
            frame=frame,
            cost=partition.cost / max(1, len(partition.regions)),
            confidence=partition.confidence,
            warnings=(),
            cut_crossing_cost=(partition.cut_crossing_cost / max(1, len(partition.regions))),
            body_count=sum(index in body_ids for index in region.component_indices),
            primary_component_indices=tuple(
                index for index in region.component_indices if index in body_ids
            ),
            secondary_component_indices=tuple(
                index for index in region.component_indices if index not in body_ids
            ),
        )
        for region in partition.regions
    )
    row_assignments: list[SecondaryMarkAssignment] = []
    row = result.rows[row_index]
    for component in result.components:
        if component.index in body_ids or component.index not in row.component_indices:
            continue
        owners = [
            local_index
            for local_index, region in enumerate(partition.regions)
            if component.index in region.component_indices
        ]
        if len(owners) == 1:
            row_assignments.append(
                SecondaryMarkAssignment(
                    component.index,
                    region_offset + owners[0],
                    None,
                    None,
                    None,
                    "assigned",
                )
            )
        else:
            row_assignments.append(
                SecondaryMarkAssignment(component.index, None, None, None, None, "unassigned")
            )
    return RowPartitionHypothesis(
        row_index=row_index,
        count=len(regions),
        boundaries=_row_boundary_positions(result.rows[row_index], partition.regions, partition),
        regions=regions,
        cost=partition.cost,
        unassigned_ink_fraction=0.0,
        secondary_mark_assignments=tuple(row_assignments),
        boundary_evidence=partition.partition_boundaries,
        cut_crossing_cost=partition.cut_crossing_cost,
    )


def _page_hypotheses_for_partitions(
    result: DetectionResult,
    partitions: Sequence[RowPartition],
    config: DetectionConfig,
) -> tuple[tuple[PagePartitionHypothesis, ...], bool]:
    if not partitions:
        return (), False
    region_specs = [
        (row_index, row_partition, region)
        for row_index, row_partition in enumerate(partitions)
        for region in row_partition.regions
    ]
    if not region_specs:
        return (), False
    shape = result.clean_mask.shape
    body_ids = {
        component_index
        for _, _, region in region_specs
        for component_index in region.component_indices
    }
    body_components = [component for component in result.components if component.index in body_ids]
    secondary = [component for component in result.components if component.index not in body_ids]
    base_masks: list[np.ndarray] = []
    body_groups: list[tuple[Component, ...]] = []
    for _, _, region in region_specs:
        full = np.zeros(shape, dtype=bool)
        full[region.bbox.y0 : region.bbox.y1, region.bbox.x0 : region.bbox.x1] = region.mask
        base_masks.append(full)
        body_groups.append(
            tuple(
                component
                for component in result.components
                if component.index in body_ids and np.any(full[component.ys, component.xs])
            )
        )
    for component in secondary:
        for full in base_masks:
            full[component.ys, component.xs] = False

    row_for_component = {
        component_index: row_index
        for row_index, row in enumerate(result.rows)
        for component_index in row.component_indices
    }
    mark_options: list[
        tuple[
            Component,
            float | None,
            float | None,
            float | None,
            tuple[int, ...],
            tuple[tuple[int | None, float], ...],
        ]
    ] = []
    body_height = (
        float(np.median([component.height for component in body_components]))
        if body_components
        else 1.0
    )
    for component in sorted(secondary, key=lambda item: item.index):
        row_index = row_for_component.get(component.index)
        scored: list[tuple[float, int]] = []
        for region_index, (target_row, _, _) in enumerate(region_specs):
            if row_index is not None and target_row != row_index:
                continue
            row_scale = (
                result.diagnostics.row_segmentation[target_row].body_height
                if result.diagnostics and target_row < len(result.diagnostics.row_segmentation)
                else body_height
            )
            score = _spacing_attachment_score(component, body_groups[region_index], row_scale)
            if score is not None:
                scored.append((score, region_index))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_score = scored[0][0] if scored and scored[0][0] <= 2.5 else None
        second_score = scored[1][0] if best_score is not None and len(scored) > 1 else None
        margin = (
            (second_score - best_score) / max(1.0, abs(second_score))
            if best_score is not None and second_score is not None
            else None
        )
        candidate_region_indices = (
            tuple(target for _, target in scored[:2])
            if margin is not None and margin < config.spacing.mark_assignment_margin
            else ()
        )
        if best_score is None:
            choices: tuple[tuple[int | None, float], ...] = ((None, 0.0),)
        else:
            eligible = (
                scored[:2]
                if margin is not None and margin < config.spacing.mark_assignment_margin
                else scored[:1]
            )
            choices = tuple((target, score) for score, target in eligible) + ((None, 2.5),)
        mark_options.append(
            (
                component,
                best_score,
                second_score,
                margin,
                candidate_region_indices,
                choices,
            )
        )

    branches: list[tuple[float, tuple[int | None, ...]]] = [(0.0, ())]
    branches_truncated = False
    for _, _, _, _, _, choices in mark_options:
        expanded = [
            (cost + choice_cost, (*selected, target))
            for cost, selected in branches
            for target, choice_cost in choices
        ]
        expanded.sort(
            key=lambda item: (
                item[0],
                tuple(-1 if target is None else target for target in item[1]),
            )
        )
        if len(expanded) > config.spacing.max_secondary_mark_branches:
            branches_truncated = True
        branches = expanded[: config.spacing.max_secondary_mark_branches]

    pages: list[PagePartitionHypothesis] = []
    for branch_cost, targets in branches:
        masks = [mask.copy() for mask in base_masks]
        owned = np.zeros(shape, dtype=bool)
        for mask in masks:
            mask &= ~owned
            owned |= mask
        assignments: list[SecondaryMarkAssignment] = []
        for (
            component,
            best,
            second,
            margin,
            candidate_region_indices,
            _,
        ), target in zip(mark_options, targets, strict=True):
            if target is None:
                assignments.append(
                    SecondaryMarkAssignment(
                        component.index,
                        None,
                        best,
                        second,
                        margin,
                        "unassigned",
                        candidate_region_indices=candidate_region_indices,
                    )
                )
                continue
            new_pixels = ~owned[component.ys, component.xs]
            ys, xs = component.ys[new_pixels], component.xs[new_pixels]
            masks[target][ys, xs] = True
            owned[ys, xs] = True
            assignments.append(
                SecondaryMarkAssignment(
                    component.index,
                    target,
                    best,
                    second,
                    margin,
                    "assigned",
                    candidate_region_indices=candidate_region_indices,
                )
            )
        unassigned = result.clean_mask & ~owned
        row_region_hypotheses: dict[int, list[RegionHypothesis]] = {}
        for region_spec_index, ((row_index, row_partition, _), mask) in enumerate(
            zip(region_specs, masks, strict=True)
        ):
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            bbox = BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            local_mask = mask[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
            component_indices = tuple(
                component.index
                for component in result.components
                if np.any(mask[component.ys, component.xs])
            )
            hypothesis = RegionHypothesis(
                bbox=bbox,
                mask=local_mask,
                component_indices=component_indices,
                frame=_region_frame(result, row_index),
                cost=row_partition.cost / max(1, len(row_partition.regions)),
                confidence=row_partition.confidence,
                warnings=(),
                cut_crossing_cost=(
                    row_partition.cut_crossing_cost / max(1, len(row_partition.regions))
                ),
                body_count=sum(index in body_ids for index in component_indices),
                primary_component_indices=tuple(
                    index for index in component_indices if index in body_ids
                ),
                secondary_component_indices=tuple(
                    index for index in component_indices if index not in body_ids
                ),
                ambiguous_component_indices=tuple(
                    component.index
                    for component, _, _, _, candidates, _ in mark_options
                    if region_spec_index in candidates
                ),
            )
            row_region_hypotheses.setdefault(row_index, []).append(hypothesis)

        row_hypotheses: list[RowPartitionHypothesis] = []
        for row_index, partition in enumerate(partitions):
            current_regions = tuple(row_region_hypotheses.get(row_index, ()))
            if not current_regions:
                continue
            row = result.rows[row_index]
            boundaries = partition.cut_positions
            if len(boundaries) != len(current_regions) + 1:
                left = boundaries[0] if boundaries else row.bbox.x0
                right = boundaries[-1] if boundaries else row.bbox.x1
                internal = tuple(
                    (left_region.bbox.x1 + right_region.bbox.x0) // 2
                    for left_region, right_region in zip(
                        current_regions, current_regions[1:], strict=False
                    )
                )
                boundaries = (left, *internal, right)
            row_unassigned = unassigned[row.bbox.y0 : row.bbox.y1]
            row_foreground = result.clean_mask[row.bbox.y0 : row.bbox.y1]
            row_assignments = tuple(
                assignment
                for assignment in assignments
                if row_for_component.get(assignment.component_index) == row_index
            )
            row_hypotheses.append(
                RowPartitionHypothesis(
                    row_index=row_index,
                    count=len(current_regions),
                    boundaries=boundaries,
                    regions=current_regions,
                    cost=partition.cost,
                    unassigned_ink_fraction=float(
                        row_unassigned.sum() / max(1, row_foreground.sum())
                    ),
                    secondary_mark_assignments=row_assignments,
                    boundary_evidence=partition.partition_boundaries,
                    cut_crossing_cost=partition.cut_crossing_cost,
                )
            )
        if not row_hypotheses:
            continue
        unassigned_fraction = float(unassigned.sum() / max(1, result.clean_mask.sum()))
        pages.append(
            PagePartitionHypothesis(
                row_hypotheses=tuple(row_hypotheses),
                count=sum(row.count for row in row_hypotheses),
                cost=sum(row.cost for row in row_hypotheses) + branch_cost + unassigned_fraction,
                unassigned_ink_fraction=unassigned_fraction,
                unassigned_mask=unassigned,
            )
        )
    return tuple(pages), branches_truncated


def detect_spaced_region_hypotheses(
    source: str | Path | Image.Image | np.ndarray,
    *,
    mode: MaskMode = "auto",
    expected_count: int | None = None,
    row_counts: Sequence[int | None] | None = None,
    min_confidence: float = 0.3,
    config: DetectionConfig | None = None,
) -> SpacedSegmentationHypotheses:
    """Return selected spaced regions with bounded row and page alternatives."""
    selected_config = config or DetectionConfig()
    row_options: list[tuple[RowPartition, ...]] = []
    page_options: list[tuple[RowPartition, ...]] = []
    truncated_output: list[bool] = []
    selected = detect_spaced_regions(
        source,
        mode=mode,
        expected_count=expected_count,
        row_counts=row_counts,
        min_confidence=min_confidence,
        config=selected_config,
        _row_hypotheses_out=row_options,
        _page_options_out=page_options,
        _truncated_out=truncated_output,
    )
    selected_page = _selected_page_hypothesis(
        selected,
        page_options[0] if page_options else None,
    )
    page_hypotheses: list[PagePartitionHypothesis] = []
    if selected_page is not None:
        page_hypotheses.append(selected_page)
    truncated = bool(truncated_output and truncated_output[0])
    if page_options:
        selected_signature = (
            _page_hypothesis_signature(selected_page, selected.clean_mask)
            if selected_page
            else None
        )
        candidates: list[PagePartitionHypothesis] = []
        seen_signatures = {selected_signature} if selected_signature is not None else set()
        for partitions in page_options:
            variants, branch_truncated = _page_hypotheses_for_partitions(
                selected,
                partitions,
                selected_config,
            )
            truncated |= branch_truncated
            for page in variants:
                signature = _page_hypothesis_signature(page, selected.clean_mask)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                candidates.append(page)
        candidates.sort(
            key=lambda page: (
                page.cost,
                tuple((row.count, row.boundaries) for row in page.row_hypotheses),
                _page_hypothesis_signature(page, selected.clean_mask),
            )
        )
        remaining = selected_config.spacing.hypothesis_limit - len(page_hypotheses)
        if len(candidates) > remaining:
            truncated = True
        page_hypotheses.extend(candidates[:remaining])
    row_hypotheses: list[tuple[RowPartitionHypothesis, ...]] = []
    selected_rows = selected_page.row_hypotheses if selected_page else ()
    for row_index, options in enumerate(row_options):
        row_candidates: list[RowPartitionHypothesis] = []
        selected_row_by_index = {row.row_index: row for row in selected_rows}
        if row_index in selected_row_by_index:
            row_candidates.append(selected_row_by_index[row_index])
        region_offset = sum(len(row.regions) for row in selected_rows if row.row_index < row_index)
        for option in sorted(
            options,
            key=lambda item: (item.cost, len(item.regions), item.cut_positions),
        ):
            candidate = _row_hypothesis_from_partition(selected, row_index, option, region_offset)
            if not any(
                existing.boundaries == candidate.boundaries
                and tuple(region.bbox for region in existing.regions)
                == tuple(region.bbox for region in candidate.regions)
                for existing in row_candidates
            ):
                row_candidates.append(candidate)
            if len(row_candidates) >= selected_config.spacing.hypothesis_limit:
                if len(options) > len(row_candidates):
                    truncated = True
                break
        row_hypotheses.append(tuple(row_candidates))
    failure_reason = None
    if expected_count is not None and not page_hypotheses:
        failure_reason = next(
            (warning for warning in selected.warnings if "No feasible row allocation" in warning),
            "No feasible page partition was returned.",
        )
    return SpacedSegmentationHypotheses(
        selected=selected,
        row_hypotheses=tuple(row_hypotheses),
        page_hypotheses=tuple(page_hypotheses),
        truncated=truncated,
        failure_reason=failure_reason,
    )


def _page_ownership_signature(
    page: PagePartitionHypothesis | None,
    shape_mask: np.ndarray,
) -> tuple[bytes, ...] | None:
    if page is None:
        return None
    signatures = []
    for region in page.regions:
        full = np.zeros(shape_mask.shape, dtype=bool)
        full[region.bbox.y0 : region.bbox.y1, region.bbox.x0 : region.bbox.x1] = region.mask
        signatures.append(full.tobytes())
    return tuple(signatures)


def _page_hypothesis_signature(
    page: PagePartitionHypothesis,
    shape_mask: np.ndarray,
) -> tuple[
    tuple[bytes, ...],
    tuple[tuple[int, tuple[int, ...], tuple[tuple[int, int, str], ...]], ...],
]:
    ownership = _page_ownership_signature(page, shape_mask)
    assert ownership is not None
    rows = tuple(
        (
            row.row_index,
            row.boundaries,
            tuple(
                (
                    assignment.component_index,
                    (
                        -1
                        if assignment.target_region_index is None
                        else assignment.target_region_index
                    ),
                    assignment.status,
                )
                for assignment in row.secondary_mark_assignments
            ),
        )
        for row in page.row_hypotheses
    )
    return ownership, rows
