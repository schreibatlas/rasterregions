from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from ._grouping import _make_region, _size_scale
from ._rows import _row_y_bounds
from .models import (
    BBox,
    Component,
    ExpectedCountScope,
    GroupingConfig,
    PartitionBoundary,
    PartitionConfig,
    Region,
    RowBand,
)


@dataclass(frozen=True)
class PartitionResult:
    cut_positions: tuple[int, ...]
    confidence: float
    cost: float
    partition_boundaries: tuple[PartitionBoundary, ...]


@dataclass(frozen=True)
class RowPartition:
    regions: tuple[Region, ...]
    confidence: float
    cost: float
    partition_boundaries: tuple[PartitionBoundary, ...]


@dataclass(frozen=True)
class ExpectedCountResult:
    regions: tuple[Region, ...]
    confidence: float
    partition_boundaries: tuple[PartitionBoundary, ...]


def _candidate_cut_positions(projection: np.ndarray, config: PartitionConfig) -> list[int]:
    width = len(projection)
    if width < 2:
        return []
    positions = {0, width}
    zero = projection == 0
    edges = np.diff(np.r_[False, zero, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends, strict=True):
        if start > 0 and end < width:
            positions.add((int(start) + int(end) - 1) // 2)
    scale = (
        float(np.percentile(projection[projection > 0], config.candidate_valley_percentile))
        if np.any(projection)
        else 0.0
    )
    for x in range(1, width - 1):
        if (
            projection[x] <= projection[x - 1]
            and projection[x] <= projection[x + 1]
            and projection[x] <= scale
        ):
            positions.add(x)
    return sorted(positions)


def _partition_row(
    row_mask: np.ndarray,
    expected_count: int,
    found: tuple[Component, ...],
    *,
    x_offset: int,
    config: PartitionConfig,
) -> PartitionResult | None:
    projection = np.asarray(row_mask, dtype=np.uint8).sum(axis=0)
    occupied = np.flatnonzero(projection)
    if occupied.size == 0:
        return None
    left, right = int(occupied[0]), int(occupied[-1]) + 1
    projection = projection[left:right]
    cuts = _candidate_cut_positions(projection, config)
    if len(cuts) < expected_count + 1:
        return None

    nonzero_scale = max(
        1.0, float(np.percentile(projection[projection > 0], config.normalization_percentile))
    )
    crossing: dict[int, int] = {}
    for cut in cuts[1:-1]:
        absolute_x = x_offset + left + cut
        crossing[cut] = sum(c.bbox.x0 < absolute_x < c.bbox.x1 for c in found)
    boundary_cost = {
        cut: float(projection[cut] / nonzero_scale) * config.projection_cost
        + crossing.get(cut, 0) * config.crossing_cost
        for cut in cuts[1:-1]
    }
    prefix = np.r_[0.0, np.cumsum(projection, dtype=float)]
    component_scale, median_height = _size_scale(found)
    primary_threshold = component_scale * config.primary_area_ratio

    def segment_cost(start: int, end: int) -> float:
        width = end - start
        ink = prefix[end] - prefix[start]
        if width <= 0 or ink <= 0:
            return float("inf")
        absolute_start = x_offset + left + start
        absolute_end = x_offset + left + end
        body_count = sum(
            c.area >= primary_threshold and absolute_start <= c.center_x < absolute_end
            for c in found
        )
        component_cost = config.body_mismatch_cost * abs(body_count - 1)
        narrow_cost = (
            max(0.0, median_height * config.narrow_width_ratio - width)
            / max(1.0, median_height)
            * config.narrow_segment_cost
        )
        area_cost = (
            config.small_segment_cost
            if ink < max(1.0, float(prefix[-1]) * config.small_segment_fraction)
            else 0.0
        )
        return component_cost + narrow_cost + area_cost

    costs = np.full((expected_count + 1, len(cuts)), np.inf)
    previous = np.full((expected_count + 1, len(cuts)), -1, dtype=int)
    costs[0, 0] = 0.0
    for segments in range(1, expected_count + 1):
        for end_index in range(1, len(cuts)):
            end = cuts[end_index]
            for start_index in range(end_index):
                if not np.isfinite(costs[segments - 1, start_index]):
                    continue
                start = cuts[start_index]
                extra_boundary = boundary_cost[start] if start_index else 0.0
                score = costs[segments - 1, start_index] + segment_cost(start, end) + extra_boundary
                if score < costs[segments, end_index]:
                    costs[segments, end_index] = score
                    previous[segments, end_index] = start_index
    final_index = len(cuts) - 1
    if not np.isfinite(costs[expected_count, final_index]):
        return None

    boundary_indexes = [final_index]
    cursor = final_index
    for segments in range(expected_count, 0, -1):
        cursor = int(previous[segments, cursor])
        if cursor < 0:
            return None
        boundary_indexes.append(cursor)
    boundary_indexes.reverse()
    boundaries = [left + cuts[index] for index in boundary_indexes]
    boundary_confidences: list[float] = []
    partition_boundaries: list[PartitionBoundary] = []
    for boundary in boundaries[1:-1]:
        local_cut = boundary - left
        valley = float(projection[local_cut] / nonzero_scale)
        component_penalty = crossing.get(local_cut, 0)
        confidence = float(
            np.clip(
                1.0 - valley - config.boundary_crossing_confidence_penalty * component_penalty,
                0.0,
                1.0,
            )
        )
        boundary_confidences.append(confidence)
        partition_boundaries.append(
            PartitionBoundary(
                x=x_offset + boundary,
                projection=float(projection[local_cut]),
                crossing_components=component_penalty,
                confidence=confidence,
            )
        )
    segment_confidences: list[float] = []
    row_height = row_mask.shape[0]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        width = end - start
        local_start = start - left
        local_end = end - left
        ink = prefix[local_end] - prefix[local_start]
        density = ink / max(1, row_height * width)
        body_count = sum(
            c.area >= primary_threshold and x_offset + start <= c.center_x < x_offset + end
            for c in found
        )
        body_score = 1.0 if body_count else 0.0
        density_score = float(np.clip(1.0 - abs(density - config.density_target) / 0.55, 0.0, 1.0))
        width_score = float(np.clip(width / max(1.0, median_height * 0.5), 0.0, 1.0))
        segment_confidences.append(0.5 * body_score + 0.3 * density_score + 0.2 * width_score)

    boundary_confidence = min(boundary_confidences, default=1.0)
    segment_confidence = min(segment_confidences, default=0.0)
    confidence = min(boundary_confidence, segment_confidence)
    return PartitionResult(
        tuple(boundaries),
        confidence,
        float(costs[expected_count, final_index]),
        tuple(partition_boundaries),
    )


def _row_partition_regions(
    image: Image.Image,
    mask: np.ndarray,
    found: tuple[Component, ...],
    row: RowBand,
    row_index: int,
    median_height: float,
    expected_count: int,
    padding: int,
    grouping: GroupingConfig,
    partition_config: PartitionConfig,
) -> RowPartition | None:
    row_y0, row_y1 = _row_y_bounds(row, found, median_height, mask.shape[0], grouping)
    row_mask = mask[row_y0:row_y1, :]
    row_found = tuple(
        component
        for component in found
        if component.bbox.y0 < row_y1 and component.bbox.y1 > row_y0
    )
    partition = _partition_row(
        row_mask,
        expected_count,
        row_found,
        x_offset=0,
        config=partition_config,
    )
    if partition is None:
        return None
    boundaries = partition.cut_positions
    confidence = partition.confidence
    cost = partition.cost
    partition_boundaries = partition.partition_boundaries
    regions: list[Region] = []
    for column_index, (x0, x1) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        segment = mask[row_y0:row_y1, x0:x1]
        ys, xs = np.nonzero(segment)
        if not len(xs):
            return None
        box = BBox(x0, row_y0 + int(ys.min()), x1, row_y0 + int(ys.max()) + 1)
        member_indices = tuple(
            component.index
            for component in found
            if component.bbox.x0 < x1
            and component.bbox.x1 > x0
            and component.bbox.y0 < row_y1
            and component.bbox.y1 > row_y0
        )
        regions.append(
            _make_region(
                image,
                mask,
                box,
                region_id=f"region-{column_index:03d}",
                row_index=row_index,
                column_index=column_index,
                reading_order=column_index,
                confidence=confidence,
                component_indices=member_indices,
                padding=padding,
                clip=BBox(x0, row_y0, x1, row_y1),
            )
        )
    return RowPartition(
        tuple(regions),
        confidence,
        cost + (1.0 - confidence),
        partition_boundaries,
    )


def _expected_count_regions(
    image: Image.Image,
    mask: np.ndarray,
    found: tuple[Component, ...],
    rows: tuple[RowBand, ...],
    median_height: float,
    expected_count: int,
    padding: int,
    grouping: GroupingConfig,
    scope: ExpectedCountScope,
    partition_config: PartitionConfig,
) -> ExpectedCountResult | None:
    if scope == "row":
        candidates = [
            option
            for row_index, row in enumerate(rows)
            if (
                option := _row_partition_regions(
                    image,
                    mask,
                    found,
                    row,
                    row_index,
                    median_height,
                    expected_count,
                    padding,
                    grouping,
                    partition_config,
                )
            )
            is not None
        ]
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda item: (
                item.confidence,
                sum(int(region.mask.sum()) for region in item.regions),
            ),
        )
        return ExpectedCountResult(
            selected.regions,
            selected.confidence,
            selected.partition_boundaries,
        )

    options_by_row: list[dict[int, RowPartition]] = []
    for row_index, row in enumerate(rows):
        row_options: dict[int, RowPartition] = {}
        for count in range(1, expected_count + 1):
            option = _row_partition_regions(
                image,
                mask,
                found,
                row,
                row_index,
                median_height,
                count,
                padding,
                grouping,
                partition_config,
            )
            if option is not None:
                row_options[count] = option
        options_by_row.append(row_options)

    states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for row_index, options in enumerate(options_by_row):
        next_states: dict[int, tuple[float, list[tuple[int, int]]]] = {}
        for used, (current_cost, choices) in states.items():
            for count, option in options.items():
                total = used + count
                if total > expected_count:
                    continue
                candidate = (current_cost + option.cost, [*choices, (row_index, count)])
                if total not in next_states or candidate[0] < next_states[total][0]:
                    next_states[total] = candidate
        states = next_states
    if expected_count not in states:
        return None

    _, choices = states[expected_count]
    regions: list[Region] = []
    confidences: list[float] = []
    partition_boundaries: list[PartitionBoundary] = []
    for row_index, count in choices:
        row_partition = options_by_row[row_index][count]
        confidences.append(row_partition.confidence)
        partition_boundaries.extend(row_partition.partition_boundaries)
        for column_index, region in enumerate(row_partition.regions):
            reading_order = len(regions)
            regions.append(
                replace(
                    region,
                    region_id=f"region-{reading_order:03d}",
                    row_index=row_index,
                    column_index=column_index,
                    reading_order=reading_order,
                )
            )
    return ExpectedCountResult(
        tuple(regions),
        min(confidences, default=0.0),
        tuple(partition_boundaries),
    )
