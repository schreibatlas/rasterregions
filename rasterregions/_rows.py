from __future__ import annotations

import numpy as np

from ._cleanup import clean_page_mask
from ._grouping import _body_components, _satellite_score
from .models import BBox, Component, DetectionConfig, GroupingConfig, RowBand


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
