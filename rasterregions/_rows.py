from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._cleanup import clean_page_mask
from ._grouping import (
    _body_components,
    _satellite_score,
    _spacing_attachment_score,
    _spacing_body_components,
)
from .models import BBox, Component, DetectionConfig, GroupingConfig, RowBand, SpacingConfig


def _rows_from_components(
    found: tuple[Component, ...],
    primary_area: float,
    median_height: float,
    grouping: GroupingConfig,
) -> tuple[RowBand, ...]:
    bodies = _body_components(found, primary_area, median_height, grouping)
    if not bodies:
        return ()
    ordered = sorted(bodies, key=lambda component: component.center_y)
    rows: list[list[Component]] = []
    for component in ordered:
        if not rows or component.center_y - np.median(
            [member.center_y for member in rows[-1]]
        ) > max(1.0, median_height * grouping.row_center_gap_ratio):
            rows.append([component])
        else:
            rows[-1].append(component)
    return tuple(
        RowBand(
            bbox=BBox(
                min(component.bbox.x0 for component in row),
                min(component.bbox.y0 for component in row),
                max(component.bbox.x1 for component in row),
                max(component.bbox.y1 for component in row),
            ),
            component_indices=tuple(component.index for component in row),
        )
        for row in rows
    )


def detect_rows(
    mask: np.ndarray,
    *,
    config: DetectionConfig | None = None,
) -> tuple[RowBand, ...]:
    """Detect coherent foreground rows from component-center clustering."""
    config = config or DetectionConfig()
    ink = np.asarray(mask, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    cleaned = clean_page_mask(ink, config.cleanup)
    return _rows_from_components(
        cleaned.components,
        cleaned.primary_area,
        cleaned.median_height,
        config.grouping,
    )


def _row_y_bounds(
    row: RowBand,
    found: tuple[Component, ...],
    median_height: float,
    page_height: int,
    config: GroupingConfig,
) -> tuple[int, int]:
    body_indices = set(row.component_indices)
    bodies = [component for component in found if component.index in body_indices]
    satellites = [
        component
        for component in found
        if component.index not in body_indices
        and any(
            _satellite_score(component, body, median_height, config) is not None for body in bodies
        )
    ]
    top = min([row.bbox.y0, *(component.bbox.y0 for component in satellites)])
    bottom = max([row.bbox.y1, *(component.bbox.y1 for component in satellites)])
    return max(0, top), min(page_height, bottom)


@dataclass(frozen=True)
class SpacingRowLayout:
    """Spacing-mode row assignment, with ambiguous and unassigned components explicit."""

    rows: tuple[RowBand, ...]
    body_component_indices: tuple[tuple[int, ...], ...]
    ambiguous_component_indices: tuple[tuple[int, ...], ...]
    unassigned_component_indices: tuple[int, ...]
    component_row_assignments: tuple[int | None, ...]


def _spacing_rows_from_components(
    found: tuple[Component, ...],
    grouping: GroupingConfig,
    spacing: SpacingConfig,
    page_shape: tuple[int, int],
) -> SpacingRowLayout:
    """Cluster primary-body centers, then assign satellites once or retain ambiguity."""
    if not found:
        return SpacingRowLayout((), (), (), (), ())
    bodies = _spacing_body_components(found)
    ordered = sorted(bodies, key=lambda component: (component.center_y, component.index))
    body_scale = float(np.median([component.height for component in ordered]))
    rows_of_bodies: list[list[Component]] = []
    for component in ordered:
        if not rows_of_bodies:
            rows_of_bodies.append([component])
            continue
        centers = np.asarray([member.center_y for member in rows_of_bodies[-1]], dtype=float)
        center = float(np.median(centers))
        spread = float(np.median(np.abs(centers - center))) if len(centers) > 1 else 0.0
        gap_limit = max(
            1.0,
            body_scale * grouping.row_center_gap_ratio,
            spread * 3.0,
        )
        if component.center_y - center > gap_limit:
            rows_of_bodies.append([component])
        else:
            rows_of_bodies[-1].append(component)

    body_ids = {component.index for component in bodies}
    row_members = [list(row) for row in rows_of_bodies]
    ambiguous: list[list[int]] = [[] for _ in rows_of_bodies]
    unassigned: list[int] = []
    component_rows: dict[int, int | None] = {
        component.index: row_index
        for row_index, row in enumerate(rows_of_bodies)
        for component in row
    }
    for component in found:
        if component.index in body_ids:
            continue
        scores = [
            (score, row_index)
            for row_index, row in enumerate(rows_of_bodies)
            if (score := _spacing_attachment_score(component, row, body_scale)) is not None
        ]
        scores.sort()
        if not scores or scores[0][0] > 2.5:
            unassigned.append(component.index)
            component_rows[component.index] = None
            continue
        best_score, best_row = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else None
        margin = (
            (second_score - best_score) / max(1.0, abs(second_score))
            if second_score is not None
            else None
        )
        if margin is not None and margin < spacing.row_assignment_margin:
            ambiguous[best_row].append(component.index)
            component_rows[component.index] = None
            continue
        row_members[best_row].append(component)
        component_rows[component.index] = best_row

    height, _ = page_shape
    body_centers = [float(np.median([item.center_y for item in row])) for row in rows_of_bodies]
    result_rows: list[RowBand] = []
    body_indices: list[tuple[int, ...]] = []
    for row_index, (body_row, members) in enumerate(zip(rows_of_bodies, row_members, strict=True)):
        cell_top = (
            0
            if row_index == 0
            else int((body_centers[row_index - 1] + body_centers[row_index]) / 2.0)
        )
        cell_bottom = (
            height
            if row_index + 1 == len(rows_of_bodies)
            else int((body_centers[row_index] + body_centers[row_index + 1]) / 2.0)
        )
        x0 = min(component.bbox.x0 for component in members)
        x1 = max(component.bbox.x1 for component in members)
        y0 = max(cell_top, min(component.bbox.y0 for component in members))
        y1 = min(cell_bottom, max(component.bbox.y1 for component in members))
        if y1 <= y0:
            y0 = min(component.bbox.y0 for component in body_row)
            y1 = max(component.bbox.y1 for component in body_row)
        result_rows.append(
            RowBand(
                BBox(x0, y0, x1, y1),
                tuple(sorted(component.index for component in members)),
            )
        )
        body_indices.append(tuple(sorted(component.index for component in body_row)))

    return SpacingRowLayout(
        tuple(result_rows),
        tuple(body_indices),
        tuple(tuple(sorted(indices)) for indices in ambiguous),
        tuple(sorted(unassigned)),
        tuple(component_rows.get(component.index) for component in found),
    )
