from __future__ import annotations

import numpy as np
from PIL import Image

from .models import BBox, Component, GroupingConfig, Region, RowBand
from .regions import crop, tighten_bbox


def _size_scale(found: tuple[Component, ...] | list[Component]) -> tuple[float, float]:
    if not found:
        return 0.0, 0.0
    areas = np.asarray([component.area for component in found], dtype=float)
    upper_half = np.sort(areas)[len(areas) // 2 :]
    primary_area = float(np.median(upper_half))
    body = [component for component in found if component.area >= primary_area * 0.2]
    median_height = float(np.median([component.height for component in body])) if body else 0.0
    return primary_area, median_height


def _body_components(
    found: tuple[Component, ...] | list[Component],
    primary_area: float,
    median_height: float,
    config: GroupingConfig | None = None,
) -> list[Component]:
    if not found:
        return []
    config = config or GroupingConfig()
    minimum_area = primary_area * config.body_min_area_ratio
    minimum_height = median_height * config.body_min_height_ratio
    bodies = [
        component
        for component in found
        if component.area >= minimum_area
        and (component.height >= minimum_height or component.width >= minimum_height)
    ]
    return bodies or [max(found, key=lambda component: component.area)]


def _group_nearby_bodies(
    bodies: list[Component], median_height: float, config: GroupingConfig
) -> list[list[Component]]:
    groups: list[list[Component]] = []
    for component in sorted(bodies, key=lambda item: item.bbox.x0):
        if not groups:
            groups.append([component])
            continue
        previous = groups[-1]
        previous_right = max(member.bbox.x1 for member in previous)
        previous_top = min(member.bbox.y0 for member in previous)
        previous_bottom = max(member.bbox.y1 for member in previous)
        gap = component.bbox.x0 - previous_right
        overlap = min(previous_bottom, component.bbox.y1) - max(previous_top, component.bbox.y0)
        if gap <= max(1.0, median_height * config.body_join_gap_ratio) and overlap > 0:
            previous.append(component)
        else:
            groups.append([component])
    return groups


def _satellite_score(
    satellite: Component,
    body: Component,
    median_height: float,
    config: GroupingConfig,
) -> float | None:
    if satellite.area >= body.area * config.satellite_max_area_ratio:
        return None
    horizontal_gap = max(
        body.bbox.x0 - satellite.bbox.x1,
        satellite.bbox.x0 - body.bbox.x1,
        0,
    )
    vertical_gap = max(
        body.bbox.y0 - satellite.bbox.y1,
        satellite.bbox.y0 - body.bbox.y1,
        0,
    )
    if (
        horizontal_gap > median_height * config.satellite_max_horizontal_gap_ratio
        or vertical_gap > median_height * config.satellite_max_vertical_gap_ratio
    ):
        return None
    overlaps_x = max(
        0,
        min(body.bbox.x1, satellite.bbox.x1) - max(body.bbox.x0, satellite.bbox.x0),
    )
    return (
        horizontal_gap / max(1.0, median_height)
        + vertical_gap / max(1.0, median_height)
        - 0.35 * overlaps_x / max(1.0, median_height)
    )


def _make_region(
    image: Image.Image,
    mask: np.ndarray,
    bbox: BBox,
    *,
    region_id: str,
    row_index: int,
    column_index: int,
    reading_order: int,
    confidence: float,
    component_indices: tuple[int, ...],
    padding: int,
    clip: BBox | None = None,
) -> Region:
    height, width = mask.shape
    clip = clip or BBox(0, 0, width, height)
    expanded = BBox(
        max(clip.x0, bbox.x0 - padding),
        max(clip.y0, bbox.y0 - padding),
        min(clip.x1, bbox.x1 + padding),
        min(clip.y1, bbox.y1 + padding),
    )
    tight = tighten_bbox(mask, expanded, pad=padding)
    return Region(
        region_id=region_id,
        bbox=tight,
        row_index=row_index,
        column_index=column_index,
        reading_order=reading_order,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        component_indices=component_indices,
        image=crop(image, tight),
        mask=mask[tight.y0 : tight.y1, tight.x0 : tight.x1],
    )


def _automatic_regions(
    image: Image.Image,
    mask: np.ndarray,
    found: tuple[Component, ...],
    primary_area: float,
    median_height: float,
    rows: tuple[RowBand, ...],
    padding: int,
    grouping: GroupingConfig,
) -> list[Region]:
    bodies = _body_components(found, primary_area, median_height, grouping)
    body_indices = {component.index for component in bodies}
    groups_by_row = [
        _group_nearby_bodies(
            [component for component in bodies if component.index in row.component_indices],
            median_height,
            grouping,
        )
        for row in rows
    ]
    attached_by_row: list[list[list[Component]]] = [
        [[] for _ in groups] for groups in groups_by_row
    ]
    satellites = [component for component in found if component.index not in body_indices]
    for satellite in satellites:
        options: list[tuple[float, int, int]] = []
        for row_index, groups in enumerate(groups_by_row):
            for group_index, group in enumerate(groups):
                scores = [
                    score
                    for body in group
                    if (score := _satellite_score(satellite, body, median_height, grouping))
                    is not None
                ]
                if scores:
                    options.append((min(scores), row_index, group_index))
        if options:
            _, row_index, group_index = min(options)
            attached_by_row[row_index][group_index].append(satellite)

    regions: list[Region] = []
    reading_order = 0
    for row_index, groups in enumerate(groups_by_row):
        for column_index, group in enumerate(groups):
            marks = attached_by_row[row_index][column_index]
            members = [*group, *marks]
            box = BBox(
                min(component.bbox.x0 for component in members),
                min(component.bbox.y0 for component in members),
                max(component.bbox.x1 for component in members),
                max(component.bbox.y1 for component in members),
            )
            next_group = groups[column_index + 1] if column_index + 1 < len(groups) else None
            previous_group = groups[column_index - 1] if column_index else None
            left_gap = (
                group[0].bbox.x0 - max(component.bbox.x1 for component in previous_group)
                if previous_group
                else median_height
            )
            right_gap = (
                min(component.bbox.x0 for component in next_group)
                - max(component.bbox.x1 for component in group)
                if next_group
                else median_height
            )
            separation = float(
                np.clip(min(left_gap, right_gap) / max(1.0, median_height * 0.5), 0.0, 1.0)
            )
            body_areas = np.asarray([component.area for component in group], dtype=float)
            median_area = float(np.median(body_areas))
            size_consistency = float(
                np.clip(
                    1.0 - np.median(np.abs(body_areas - median_area)) / max(1.0, median_area),
                    0.0,
                    1.0,
                )
            )
            confidence = 0.65 * separation + 0.35 * size_consistency
            regions.append(
                _make_region(
                    image,
                    mask,
                    box,
                    region_id=f"region-{reading_order:03d}",
                    row_index=row_index,
                    column_index=column_index,
                    reading_order=reading_order,
                    confidence=confidence,
                    component_indices=tuple(sorted(component.index for component in members)),
                    padding=padding,
                )
            )
            reading_order += 1
    return regions
