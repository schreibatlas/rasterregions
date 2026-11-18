from __future__ import annotations

from bisect import insort
from collections.abc import Sequence
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
    SegmentationSearchLimitError,
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
    alternatives: tuple[PartitionResult, ...] = ()
    truncated: bool = False
    cut_crossing_cost: float = 0.0


@dataclass(frozen=True)
class RowPartition:
    regions: tuple[Region, ...]
    confidence: float
    cost: float
    partition_boundaries: tuple[PartitionBoundary, ...]
    alternative_cost_margin: float | None = None
    width_prior_deviation: float | None = None
    body_counts_per_region: tuple[int, ...] = ()

    alternatives: tuple[RowPartition, ...] = ()
    truncated: bool = False
    cut_positions: tuple[int, ...] = ()
    cut_crossing_cost: float = 0.0
    anchor_count: int | None = None
    segmentation_strategy: Literal["anchors", "hybrid", "partition"] | None = None
    attempted_counts: tuple[int, ...] = ()
    truncated_counts: tuple[int, ...] = ()
    split_anchor_indices: tuple[int, ...] = ()
    merged_anchor_groups: tuple[tuple[int, ...], ...] = ()


def _combine_page_partitions(
    options_by_row: Sequence[Sequence[RowPartition]],
    expected_count: int,
    spacing: SpacingConfig,
) -> tuple[tuple[tuple[RowPartition, ...], ...], bool]:
    """Return bounded, deterministic exact-total row allocations."""
    limit = spacing.hypothesis_limit
    states: dict[int, list[tuple[float, tuple[RowPartition, ...]]]] = {0: [(0.0, ())]}
    visited = 0
    truncated = False
    for row_options in options_by_row:
        next_states: dict[int, list[tuple[float, tuple[RowPartition, ...]]]] = {}
        for used_count in sorted(states):
            for current_cost, selected_rows in states[used_count]:
                for option in sorted(
                    row_options,
                    key=lambda row: (row.cut_positions[-1] if row.cut_positions else 0, row.cost),
                ):
                    total = used_count + len(option.regions)
                    if total > expected_count:
                        continue
                    if visited >= spacing.max_page_states:
                        truncated = True
                        break
                    visited += 1
                    bucket = next_states.setdefault(total, [])
                    bucket.append((current_cost + option.cost, (*selected_rows, option)))
                if truncated:
                    break
            if truncated:
                break
        for total, candidates in next_states.items():
            candidates.sort(
                key=lambda item: (
                    item[0],
                    tuple((row.cut_positions, len(row.regions)) for row in item[1]),
                )
            )
            if len(candidates) > limit:
                truncated = True
                next_states[total] = candidates[:limit]
        states = next_states
        if truncated and not states:
            break
    candidates = states.get(expected_count, [])
    if len(candidates) > limit:
        truncated = True
    return tuple(rows for _, rows in candidates[:limit]), truncated


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
    local_minima = (
        np.flatnonzero(
            (projection[1:-1] <= projection[:-2])
            & (projection[1:-1] <= projection[2:])
            & (projection[1:-1] <= scale)
        )
        + 1
    )
    if local_minima.size:
        plateau_starts = np.flatnonzero(np.diff(local_minima) > 1) + 1
        for plateau in np.split(local_minima, plateau_starts):
            positions.add((int(plateau[0]) + int(plateau[-1])) // 2)
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
) -> tuple[list[int], tuple[PartitionBoundary, ...], dict[int, float], dict[int, float], bool]:
    """Find bounded scale-smoothed valleys and plausible component-center midpoints."""
    width = len(projection)
    if width < 2:
        return [0, width], (), {}, {}, False
    smooth_width = min(width, max(1, int(round(max(1.0, scale) * 0.08))))
    if smooth_width > 1:
        kernel = np.ones(smooth_width, dtype=float) / smooth_width
        smoothed = np.convolve(projection.astype(float), kernel, mode="same")
    else:
        smoothed = projection.astype(float)
    occupied = smoothed[smoothed > 0]
    if occupied.size == 0:
        return [0, width], (), {}, {}, False
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
    ordered_bodies = sorted(likely_bodies, key=lambda item: item.center_x)
    for left_body, right_body in zip(ordered_bodies[:-1], ordered_bodies[1:], strict=True):
        midpoint = int(round((left_body.center_x + right_body.center_x) / 2.0))
        local = midpoint - x_offset - left
        if 0 < local < width and smoothed[local] <= max(valley_level * 1.5, scale * 0.5):
            candidate_kinds.setdefault(local, "component-midpoint")

    candidate_positions = sorted(candidate_kinds)
    cap = spacing.max_candidate_boundaries
    candidates_truncated = len(candidate_positions) > cap
    if candidates_truncated:
        candidate_positions = sorted(
            candidate_positions,
            key=lambda x: (smoothed[x] / normalization, x),
        )[:cap]
        candidate_positions.sort()

    boundaries: list[PartitionBoundary] = []
    boundary_cost: dict[int, float] = {}
    boundary_crossing_cost: dict[int, float] = {}
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
        crossing_cost = len(crossing) * config.crossing_cost * min(1.0, normalized_projection)
        boundary_crossing_cost[cut] = crossing_cost
        boundary_cost[cut] = normalized_projection * config.projection_cost + crossing_cost
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
                crossing_cost=crossing_cost,
            )
        )
    return (
        [0, *candidate_positions, width],
        tuple(boundaries),
        boundary_cost,
        boundary_crossing_cost,
        candidates_truncated,
    )


def _partition_row(
    row_mask: np.ndarray,
    expected_count: int | None,
    found: tuple[Component, ...],
    *,
    x_offset: int,
    config: PartitionConfig,
    spacing_config: SpacingConfig | None = None,
    _forced_cut_positions: tuple[int, ...] | None = None,
    _collect_alternatives: bool = True,
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
        candidate_counts = list(range(minimum_count, maximum_count + 1))
        count_truncated = len(candidate_counts) > spacing_config.max_count_candidates
        if count_truncated:
            target_count = row_width / max(1.0, row_width_prior)
            candidate_counts = sorted(
                candidate_counts,
                key=lambda count: (abs(count - target_count), count),
            )[: spacing_config.max_count_candidates]
            candidate_counts.sort()
        count_alternatives: list[tuple[float, int, PartitionResult]] = []
        for count in candidate_counts:
            result = _partition_row(
                row_mask,
                count,
                found,
                x_offset=x_offset,
                config=config,
                spacing_config=spacing_config,
                _collect_alternatives=_collect_alternatives,
            )
            if result is not None:
                result_options = (result, *result.alternatives)
                for option in result_options:
                    adjusted_cost = option.cost + spacing_config.complexity_penalty * abs(
                        count - len(spacing_bodies)
                    )
                    count_alternatives.append((adjusted_cost, count, option))
        if not count_alternatives:
            if count_truncated:
                raise SegmentationSearchLimitError(
                    "count-candidate budget exhausted before a feasible partition"
                )
            return None
        count_alternatives.sort(key=lambda item: (item[0], item[1], item[2].cut_positions))
        selected_options = count_alternatives[: spacing_config.hypothesis_limit]
        best_cost, _, best = selected_options[0]
        count_costs: dict[int, float] = {}
        for candidate_cost, count, _ in count_alternatives:
            count_costs[count] = min(count_costs.get(count, float("inf")), candidate_cost)
        ordered_count_costs = sorted(count_costs.values())
        margin = None
        if len(ordered_count_costs) > 1:
            second_count_cost = ordered_count_costs[1]
            margin = max(0.0, second_count_cost - best_cost) / max(1.0, abs(second_count_cost))
        rest = tuple(
            replace(
                option,
                cost=cost,
                alternatives=(),
                truncated=option.truncated or count_truncated,
            )
            for cost, _, option in selected_options[1:]
        )
        return replace(
            best,
            cost=best_cost,
            alternative_cost_margin=margin,
            alternatives=rest,
            truncated=best.truncated or count_truncated,
        )

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
        boundary_crossing_cost = {
            cut: crossing.get(cut, 0) * config.crossing_cost for cut in cuts[1:-1]
        }
        candidates_truncated = False
    else:
        (
            cuts,
            candidate_boundaries,
            candidate_costs,
            boundary_crossing_cost,
            candidates_truncated,
        ) = _spacing_candidate_boundaries(
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
    if _forced_cut_positions is not None:
        forced = set(_forced_cut_positions)
        cuts = [cut for cut in cuts if cut in forced]
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

    top_k = spacing_config.hypothesis_limit if spacing_config else 1
    max_states = spacing_config.max_partition_states if spacing_config else 100_000
    final_index = len(cuts) - 1
    valid_starts_by_end: dict[int, tuple[int, ...]] = {}
    segment_costs: dict[tuple[int, int], float] = {}
    for end_index in range(1, len(cuts)):
        end = cuts[end_index]
        valid_starts = []
        for start_index in range(end_index):
            start = cuts[start_index]
            width = end - start
            if spacing_config and not minimum_width <= width <= maximum_width:
                continue
            valid_starts.append(start_index)
            segment_costs[start_index, end_index] = segment_cost(start, end)
        valid_starts_by_end[end_index] = tuple(valid_starts)

    paths_by_state: dict[tuple[int, int], list[tuple[float, tuple[int, ...]]]] = {
        (0, 0): [(0.0, (0,))]
    }
    visited_states = 0
    state_budget_exhausted = False
    path_budget_exhausted = False
    for segments in range(1, expected_count + 1):
        remaining_segments = expected_count - segments
        for end_index in range(segments, final_index - remaining_segments + 1):
            candidates: list[tuple[float, tuple[int, ...]]] = []
            for start_index in valid_starts_by_end[end_index]:
                prefixes = paths_by_state.get((segments - 1, start_index), ())
                if not prefixes:
                    continue
                start = cuts[start_index]
                extra_boundary = boundary_cost.get(start, 0.0) if start_index else 0.0
                segment_score = segment_costs[start_index, end_index] + extra_boundary
                if not np.isfinite(segment_score):
                    continue
                for prefix_cost, prefix_path in prefixes:
                    if visited_states >= max_states:
                        state_budget_exhausted = True
                        break
                    visited_states += 1
                    insort(candidates, (prefix_cost + segment_score, (*prefix_path, end_index)))
                    if len(candidates) > top_k:
                        candidates.pop()
                        if spacing_config is not None:
                            path_budget_exhausted = True
                if state_budget_exhausted:
                    break
            if candidates:
                paths_by_state[(segments, end_index)] = candidates
            if state_budget_exhausted:
                break
        if state_budget_exhausted:
            break
    final_paths = paths_by_state.get((expected_count, final_index), [])
    search_truncated = candidates_truncated or state_budget_exhausted or path_budget_exhausted
    if not final_paths:
        if candidates_truncated or state_budget_exhausted:
            raise SegmentationSearchLimitError(
                "partition search budget exhausted before a feasible result"
            )
        return None
    if len(final_paths) > top_k:
        search_truncated = True
        final_paths = final_paths[:top_k]
    selected_path_cost, selected_path = final_paths[0]
    boundary_indexes = list(selected_path)
    boundaries = [left + cuts[index] for index in boundary_indexes]
    selected_x = {x_offset + boundary for boundary in boundaries[1:-1]}
    candidate_by_x = {boundary.x: boundary for boundary in candidate_boundaries}
    boundary_confidences: list[float] = []
    selected_boundaries: list[PartitionBoundary] = []
    for boundary in boundaries[1:-1]:
        local_cut = boundary - left
        component_penalty = crossing.get(local_cut, 0)
        boundary_candidate = candidate_by_x.get(x_offset + boundary) if spacing_config else None
        if boundary_candidate is not None:
            confidence = boundary_candidate.confidence
        else:
            normalized_valley = float(projection[local_cut] / normalized_scale)
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
            if boundary_candidate is not None:
                selected_boundaries.append(replace(boundary_candidate, selected=True))
        else:
            selected_boundaries.append(
                PartitionBoundary(
                    x=x_offset + boundary,
                    projection=float(projection[local_cut]),
                    crossing_components=component_penalty,
                    confidence=confidence,
                    crossing_cost=component_penalty * config.crossing_cost,
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
    margin = None
    if len(final_paths) > 1:
        second_cost = final_paths[1][0]
        margin = max(0.0, second_cost - selected_path_cost) / max(1.0, abs(second_cost))
    cut_crossing_cost = sum(
        boundary_crossing_cost.get(cuts[index], 0.0) for index in boundary_indexes[1:-1]
    )
    result = PartitionResult(
        tuple(boundaries),
        confidence,
        selected_path_cost,
        tuple(selected_boundaries),
        alternative_cost_margin=margin,
        width_prior_deviation=(float(np.mean(width_deviations)) if width_deviations else None),
        body_counts_per_region=tuple(body_counts),
        cut_crossing_cost=cut_crossing_cost,
    )
    if _collect_alternatives and len(final_paths) > 1:
        partition_alternatives: list[PartitionResult] = []
        for _, alternative_path in final_paths[1:]:
            alternative = _partition_row(
                row_mask,
                expected_count,
                found,
                x_offset=x_offset,
                config=config,
                spacing_config=spacing_config,
                _forced_cut_positions=tuple(cuts[index] for index in alternative_path),
                _collect_alternatives=False,
            )
            if alternative is not None:
                partition_alternatives.append(alternative)
        result = replace(result, alternatives=tuple(partition_alternatives))
    return replace(result, truncated=search_truncated)


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
    _partition_override: PartitionResult | None = None,
    _collect_alternatives: bool = True,
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
    partition = _partition_override
    if partition is None:
        partition = _partition_row(
            row_mask,
            expected_count,
            row_found,
            x_offset=0,
            config=partition_config,
            spacing_config=spacing_config,
            _collect_alternatives=_collect_alternatives,
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
    anchor_components = tuple(
        sorted(
            _spacing_body_components(row_found),
            key=lambda component: (component.center_x, component.index),
        )
    )
    anchor_positions = {component.index: index for index, component in enumerate(anchor_components)}
    split_anchor_indices = tuple(
        index
        for index, component in enumerate(anchor_components)
        if any(
            boundary.selected and component.bbox.x0 < boundary.x < component.bbox.x1
            for boundary in partition_boundaries
        )
    )
    merged_anchor_groups = tuple(
        anchor_group
        for region in regions
        if len(
            anchor_group := tuple(
                anchor_positions[index]
                for index in region.component_indices
                if index in anchor_positions
            )
        )
        > 1
    )
    attempted_count = expected_count if expected_count is not None else len(regions)
    result = RowPartition(
        tuple(regions),
        confidence,
        cost + (1.0 - confidence),
        partition_boundaries,
        partition.alternative_cost_margin,
        partition.width_prior_deviation,
        partition.body_counts_per_region,
        truncated=partition.truncated,
        cut_positions=partition.cut_positions,
        cut_crossing_cost=partition.cut_crossing_cost,
        anchor_count=len(anchor_components),
        segmentation_strategy="partition",
        attempted_counts=(attempted_count,),
        truncated_counts=(attempted_count,) if partition.truncated else (),
        split_anchor_indices=split_anchor_indices,
        merged_anchor_groups=merged_anchor_groups,
    )
    if _collect_alternatives and partition.alternatives:
        alternatives: list[RowPartition] = []
        for alternative_partition in partition.alternatives:
            alternative = _row_partition_regions(
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
                spacing_config,
                _partition_override=alternative_partition,
                _collect_alternatives=False,
            )
            if alternative is not None:
                alternatives.append(alternative)
        result = replace(result, alternatives=tuple(alternatives))
    return result


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
