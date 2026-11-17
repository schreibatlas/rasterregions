from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from PIL import Image

from ._grouping import _make_region, _size_scale, _spacing_body_components
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
    SpacingConfig,
)


@dataclass(frozen=True)
class PartitionResult:
    cut_positions: tuple[int, ...]
    confidence: float
    cost: float
    partition_boundaries: tuple[PartitionBoundary, ...]

    alternative_cost_margin: float | None = None
    width_prior_deviation: float | None = None
    body_counts_per_region: tuple[int, ...] = ()


@dataclass(frozen=True)
class RowPartition:
    regions: tuple[Region, ...]
    confidence: float
    cost: float
    partition_boundaries: tuple[PartitionBoundary, ...]
    alternative_cost_margin: float | None = None
    width_prior_deviation: float | None = None
    body_counts_per_region: tuple[int, ...] = ()


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


def _spacing_candidate_boundaries(
    projection: np.ndarray,
    found: tuple[Component, ...],
    *,
    x_offset: int,
    left: int,
    config: PartitionConfig,
    spacing: SpacingConfig,
    scale: float,
) -> tuple[list[int], tuple[PartitionBoundary, ...], dict[int, float]]:
    """Find scale-smoothed valleys and plausible component-center midpoints."""
    width = len(projection)
    if width < 2:
        return [0, width], (), {}
    smooth_width = min(width, max(1, int(round(max(1.0, scale) * 0.08))))
    if smooth_width > 1:
        kernel = np.ones(smooth_width, dtype=float) / smooth_width
        smoothed = np.convolve(projection.astype(float), kernel, mode="same")
    else:
        smoothed = projection.astype(float)
    occupied = smoothed[smoothed > 0]
    if occupied.size == 0:
        return [0, width], (), {}
    normalization = max(1.0, scale)
    valley_level = float(np.percentile(occupied, config.candidate_valley_percentile))
    candidate_kinds: dict[
        int, Literal["zero-run", "valley", "local-minimum", "component-midpoint"]
    ] = {}

    zero = projection == 0
    edges = np.diff(np.r_[False, zero, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends, strict=True):
        if start > 0 and end < width:
            candidate_kinds[(int(start) + int(end) - 1) // 2] = "zero-run"

    for x in range(1, width - 1):
        if (
            smoothed[x] <= smoothed[x - 1]
            and smoothed[x] <= smoothed[x + 1]
            and (smoothed[x] < smoothed[x - 1] or smoothed[x] < smoothed[x + 1])
            and smoothed[x] <= valley_level
        ):
            candidate_kinds.setdefault(x, "local-minimum")

    likely_bodies = _spacing_body_components(found)
    for left_body, right_body in zip(
        sorted(likely_bodies, key=lambda item: item.center_x)[:-1],
        sorted(likely_bodies, key=lambda item: item.center_x)[1:],
        strict=True,
    ):
        midpoint = int(round((left_body.center_x + right_body.center_x) / 2.0))
        local = midpoint - x_offset - left
        if 0 < local < width and smoothed[local] <= max(valley_level * 1.5, scale * 0.5):
            candidate_kinds.setdefault(local, "component-midpoint")

    candidate_positions = sorted(candidate_kinds)
    cap = min(
        256,
        max(32, int(width / max(1.0, scale * 0.08))),
        max(32, spacing.max_inferred_count * 12),
    )
    if len(candidate_positions) > cap:
        candidate_positions = sorted(
            candidate_positions,
            key=lambda x: (smoothed[x] / normalization, x),
        )[:cap]
        candidate_positions.sort()

    boundaries: list[PartitionBoundary] = []
    boundary_cost: dict[int, float] = {}
    for cut in candidate_positions:
        absolute_x = x_offset + left + cut
        crossing = tuple(
            component.index
            for component in found
            if component.bbox.x0 < absolute_x < component.bbox.x1
        )
        normalized_projection = float(smoothed[cut] / normalization)
        strength = float(np.clip(1.0 - normalized_projection, 0.0, 1.0))
        confidence = float(
            np.clip(
                strength
                - config.boundary_crossing_confidence_penalty
                * len(crossing)
                * min(1.0, normalized_projection),
                0.0,
                1.0,
            )
        )
        boundary_cost[cut] = normalized_projection * config.projection_cost + len(
            crossing
        ) * config.crossing_cost * min(1.0, normalized_projection)
        boundaries.append(
            PartitionBoundary(
                x=absolute_x,
                projection=float(projection[cut]),
                crossing_components=len(crossing),
                confidence=confidence,
                component_indices=crossing,
                selected=False,
                kind=candidate_kinds[cut],
                valley_strength=strength,
            )
        )
    return [0, *candidate_positions, width], tuple(boundaries), boundary_cost


def _partition_row(
    row_mask: np.ndarray,
    expected_count: int | None,
    found: tuple[Component, ...],
    *,
    x_offset: int,
    config: PartitionConfig,
    spacing_config: SpacingConfig | None = None,
) -> PartitionResult | None:
    projection = np.asarray(row_mask, dtype=np.uint8).sum(axis=0).astype(float)
    occupied = np.flatnonzero(projection)
    if occupied.size == 0:
        return None
    left, right = int(occupied[0]), int(occupied[-1]) + 1
    projection = projection[left:right]
    component_scale, median_height = _size_scale(found)
    spacing_bodies = _spacing_body_components(found) if spacing_config else []
    if spacing_config:
        width_source = spacing_bodies or list(found)
        row_width_prior = float(np.median([component.width for component in width_source]))
        if spacing_config.width_prior_ratio is not None:
            row_width_prior *= spacing_config.width_prior_ratio
        row_width_prior = max(1.0, row_width_prior)
    else:
        row_width_prior = max(1.0, median_height)

    if expected_count is None:
        if spacing_config is None:
            return None
        row_width = len(projection)
        smallest_width = row_width_prior * spacing_config.min_region_width_ratio
        largest_width = row_width_prior * spacing_config.max_region_width_ratio
        minimum_count = max(1, int(row_width / max(1.0, largest_width)))
        maximum_count = min(
            spacing_config.max_inferred_count,
            max(minimum_count, int(np.ceil(row_width / max(1.0, smallest_width)))),
        )
        alternatives: list[tuple[float, int, PartitionResult]] = []
        for count in range(minimum_count, maximum_count + 1):
            result = _partition_row(
                row_mask,
                count,
                found,
                x_offset=x_offset,
                config=config,
                spacing_config=spacing_config,
            )
            if result is not None:
                adjusted_cost = result.cost + spacing_config.complexity_penalty * (count - 1)
                alternatives.append((adjusted_cost, count, result))
        if not alternatives:
            return None
        alternatives.sort(key=lambda item: (item[0], item[1], item[2].cut_positions))
        best_cost, _, best = alternatives[0]
        margin = None
        if len(alternatives) > 1:
            second_cost = alternatives[1][0]
            margin = max(0.0, second_cost - best_cost) / max(1.0, abs(second_cost))
        return replace(best, cost=best_cost, alternative_cost_margin=margin)

    if spacing_config is None:
        cuts = _candidate_cut_positions(projection, config)
        candidate_boundaries: tuple[PartitionBoundary, ...] = ()
        candidate_costs: dict[int, float] = {}
        normalized_scale = max(
            1.0,
            float(np.percentile(projection[projection > 0], config.normalization_percentile)),
        )
        crossing: dict[int, int] = {}
        for cut in cuts[1:-1]:
            absolute_x = x_offset + left + cut
            crossing[cut] = sum(
                component.bbox.x0 < absolute_x < component.bbox.x1 for component in found
            )
        boundary_cost = {
            cut: float(projection[cut] / normalized_scale) * config.projection_cost
            + crossing.get(cut, 0) * config.crossing_cost
            for cut in cuts[1:-1]
        }
    else:
        cuts, candidate_boundaries, candidate_costs = _spacing_candidate_boundaries(
            projection,
            found,
            x_offset=x_offset,
            left=left,
            config=config,
            spacing=spacing_config,
            scale=median_height,
        )
        normalized_scale = max(1.0, median_height)
        crossing = {
            cut: sum(
                component.bbox.x0 < x_offset + left + cut < component.bbox.x1 for component in found
            )
            for cut in cuts[1:-1]
        }
        boundary_cost = candidate_costs
    if len(cuts) < expected_count + 1:
        return None

    prefix = np.r_[0.0, np.cumsum(projection, dtype=float)]
    primary_threshold = component_scale * config.primary_area_ratio
    body_indices = {component.index for component in spacing_bodies}
    minimum_width = (
        row_width_prior * spacing_config.min_region_width_ratio
        if spacing_config
        else median_height * config.narrow_width_ratio
    )
    maximum_width = (
        row_width_prior * spacing_config.max_region_width_ratio if spacing_config else float("inf")
    )

    def segment_body_count(start: int, end: int) -> int:
        absolute_start = x_offset + left + start
        absolute_end = x_offset + left + end
        if spacing_config and len(body_indices) == 1:
            return sum(
                component.index in body_indices
                and bool(np.any((component.xs >= absolute_start) & (component.xs < absolute_end)))
                for component in found
            )
        if spacing_config:
            return sum(
                component.index in body_indices
                and absolute_start <= component.center_x < absolute_end
                for component in found
            )
        return sum(
            component.area >= primary_threshold
            and absolute_start <= component.center_x < absolute_end
            for component in found
        )

    def segment_cost(start: int, end: int) -> float:
        width = end - start
        ink = prefix[end] - prefix[start]
        if width <= 0 or ink <= 0:
            return float("inf")
        body_count = segment_body_count(start, end)
        component_cost = (
            config.body_mismatch_cost * (2.0 if spacing_config else 1.0) * abs(body_count - 1)
        )
        narrow_cost = (
            max(0.0, minimum_width - width)
            / max(1.0, row_width_prior if spacing_config else median_height)
            * config.narrow_segment_cost
        )
        wide_cost = (
            max(0.0, width - maximum_width) / max(1.0, row_width_prior) * config.narrow_segment_cost
            if spacing_config
            else 0.0
        )
        area_cost = (
            config.small_segment_cost
            if ink < max(1.0, float(prefix[-1]) * config.small_segment_fraction)
            else 0.0
        )
        if spacing_config:
            width_deviation = abs(np.log(max(1.0, width) / row_width_prior))
            width_cost = 0.20 * float(width_deviation)
            density = ink / max(1.0, float(row_mask.shape[0] * width))
            density_cost = 0.20 * abs(density - config.density_target)
            return component_cost + narrow_cost + wide_cost + width_cost + density_cost + area_cost
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
                extra_boundary = boundary_cost.get(start, 0.0) if start_index else 0.0
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
    selected_x = {x_offset + boundary for boundary in boundaries[1:-1]}
    candidate_by_x = {boundary.x: boundary for boundary in candidate_boundaries}
    boundary_confidences: list[float] = []
    selected_boundaries: list[PartitionBoundary] = []
    for boundary in boundaries[1:-1]:
        local_cut = boundary - left
        candidate = candidate_by_x.get(x_offset + boundary) if spacing_config else None
        if candidate is not None:
            confidence = candidate.confidence
        else:
            normalized_valley = float(projection[local_cut] / normalized_scale)
            component_penalty = crossing.get(local_cut, 0)
            confidence = float(
                np.clip(
                    1.0
                    - normalized_valley
                    - config.boundary_crossing_confidence_penalty * component_penalty,
                    0.0,
                    1.0,
                )
            )
        boundary_confidences.append(confidence)
        if spacing_config:
            candidate = candidate_by_x.get(x_offset + boundary)
            if candidate is not None:
                selected_boundaries.append(replace(candidate, selected=True))
        else:
            selected_boundaries.append(
                PartitionBoundary(
                    x=x_offset + boundary,
                    projection=float(projection[local_cut]),
                    crossing_components=component_penalty,
                    confidence=confidence,
                )
            )

    segment_confidences: list[float] = []
    body_counts: list[int] = []
    width_deviations: list[float] = []
    row_height = row_mask.shape[0]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        width = end - start
        local_start = start - left
        local_end = end - left
        ink = prefix[local_end] - prefix[local_start]
        density = ink / max(1, row_height * width)
        body_count = segment_body_count(start - left, end - left)
        body_counts.append(int(body_count))
        body_score = 1.0 if body_count else 0.0
        density_score = float(np.clip(1.0 - abs(density - config.density_target) / 0.55, 0.0, 1.0))
        if spacing_config:
            deviation = abs(np.log(max(1.0, width) / row_width_prior))
            width_deviations.append(float(deviation))
            width_score = float(np.exp(-deviation))
            body_score = float(np.exp(-abs(body_count - 1)))
        else:
            width_score = float(np.clip(width / max(1.0, median_height * 0.5), 0.0, 1.0))
        segment_confidences.append(0.5 * body_score + 0.3 * density_score + 0.2 * width_score)

    if spacing_config:
        selected_boundaries = [
            replace(candidate, selected=candidate.x in selected_x)
            for candidate in candidate_boundaries
        ]
    boundary_confidence = min(boundary_confidences, default=1.0)
    segment_confidence = min(segment_confidences, default=0.0)
    confidence = min(boundary_confidence, segment_confidence)
    return PartitionResult(
        tuple(boundaries),
        confidence,
        float(costs[expected_count, final_index]),
        tuple(selected_boundaries),
        width_prior_deviation=(float(np.mean(width_deviations)) if width_deviations else None),
        body_counts_per_region=tuple(body_counts),
    )


def _row_partition_regions(
    image: Image.Image,
    mask: np.ndarray,
    found: tuple[Component, ...],
    row: RowBand,
    row_index: int,
    median_height: float,
    expected_count: int | None,
    padding: int,
    grouping: GroupingConfig,
    partition_config: PartitionConfig,
    spacing_config: SpacingConfig | None = None,
) -> RowPartition | None:
    if spacing_config:
        row_y0, row_y1 = row.bbox.y0, row.bbox.y1
        row_component_ids = set(row.component_indices)
        row_found = tuple(component for component in found if component.index in row_component_ids)
        region_mask = np.zeros_like(mask, dtype=bool)
        for component in row_found:
            region_mask[component.ys, component.xs] = True
    else:
        row_y0, row_y1 = _row_y_bounds(row, found, median_height, mask.shape[0], grouping)
        row_found = tuple(
            component
            for component in found
            if component.bbox.y0 < row_y1 and component.bbox.y1 > row_y0
        )
        region_mask = mask
    row_mask = region_mask[row_y0:row_y1, :]
    partition = _partition_row(
        row_mask,
        expected_count,
        row_found,
        x_offset=0,
        config=partition_config,
        spacing_config=spacing_config,
    )
    if partition is None:
        return None
    boundaries = partition.cut_positions
    confidence = partition.confidence
    cost = partition.cost
    partition_boundaries = partition.partition_boundaries
    regions: list[Region] = []
    for column_index, (x0, x1) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        segment = region_mask[row_y0:row_y1, x0:x1]
        ys, xs = np.nonzero(segment)
        if not len(xs):
            return None
        box = BBox(x0, row_y0 + int(ys.min()), x1, row_y0 + int(ys.max()) + 1)
        member_source = row_found if spacing_config else found
        member_indices = tuple(
            component.index
            for component in member_source
            if (
                x0 <= component.center_x < x1
                if spacing_config
                else component.bbox.x0 < x1 and component.bbox.x1 > x0
            )
            and component.bbox.y0 < row_y1
            and component.bbox.y1 > row_y0
        )
        regions.append(
            _make_region(
                image,
                region_mask,
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
        partition.alternative_cost_margin,
        partition.width_prior_deviation,
        partition.body_counts_per_region,
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
