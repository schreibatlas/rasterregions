from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from PIL import Image

from .models import (
    AnchorSplitEvidence,
    BBox,
    Component,
    GroupingConfig,
    Region,
    RowBand,
    SecondaryMarkAssignment,
    SpacingConfig,
)
from .regions import crop, tighten_bbox


class _MarkAssignmentDetails(TypedDict, total=False):
    area_ratio: float
    horizontal_overlap_ratio: float
    horizontal_center_distance: float
    vertical_gap_ratio: float
    bbox_expansion_ratio: float


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


def _spacing_body_components(
    found: tuple[Component, ...] | list[Component],
) -> list[Component]:
    """Classify primary bodies using robust row-scale area and extent evidence."""
    if not found:
        return []
    ordered = sorted(found, key=lambda component: component.area, reverse=True)
    upper = ordered[: max(1, (len(ordered) + 1) // 2)]
    area_scale = float(np.median([component.area for component in upper]))
    extent_scale = float(np.median([max(component.width, component.height) for component in upper]))
    bodies = [
        component
        for component in found
        if (
            component.area >= area_scale * 0.18
            and max(component.width, component.height) >= extent_scale * 0.30
        )
        or (
            component.area >= area_scale * 0.04
            and max(component.width, component.height) >= extent_scale * 0.65
        )
        or (component.area >= area_scale * 0.08 and component.width >= extent_scale * 0.75)
    ]
    return bodies or [ordered[0]]


@dataclass(frozen=True)
class _BodyAnchor:
    component_indices: tuple[int, ...]
    mask: np.ndarray
    bbox: BBox
    center_x: float
    width: int
    height: int
    area: int


def _spacing_body_anchors(
    found: tuple[Component, ...], component_indices: Sequence[int]
) -> tuple[_BodyAnchor, ...]:
    components = {component.index: component for component in found}
    anchors = []
    for index in component_indices:
        component = components[index]
        bbox = component.bbox
        mask = np.zeros((bbox.height, bbox.width), dtype=bool)
        mask[component.ys - bbox.y0, component.xs - bbox.x0] = True
        anchors.append(
            _BodyAnchor(
                component_indices=(component.index,),
                mask=mask,
                bbox=bbox,
                center_x=component.center_x,
                width=component.width,
                height=component.height,
                area=component.area,
            )
        )
    return tuple(
        sorted(anchors, key=lambda anchor: (anchor.center_x, anchor.bbox.x0, anchor.bbox.y0))
    )


def _merge_body_anchors(left: _BodyAnchor, right: _BodyAnchor) -> _BodyAnchor:
    bbox = BBox(
        min(left.bbox.x0, right.bbox.x0),
        min(left.bbox.y0, right.bbox.y0),
        max(left.bbox.x1, right.bbox.x1),
        max(left.bbox.y1, right.bbox.y1),
    )
    mask = np.zeros((bbox.height, bbox.width), dtype=bool)
    for anchor in (left, right):
        mask[
            anchor.bbox.y0 - bbox.y0 : anchor.bbox.y1 - bbox.y0,
            anchor.bbox.x0 - bbox.x0 : anchor.bbox.x1 - bbox.x0,
        ] |= anchor.mask
    return _BodyAnchor(
        component_indices=tuple(sorted((*left.component_indices, *right.component_indices))),
        mask=mask,
        bbox=bbox,
        center_x=bbox.center_x,
        width=bbox.width,
        height=bbox.height,
        area=left.area + right.area,
    )


def _merge_spacing_body_anchors(
    anchors: Sequence[_BodyAnchor], target_count: int
) -> tuple[_BodyAnchor, ...] | None:
    merged = list(anchors)
    if target_count < 1 or target_count >= len(merged):
        return None
    scale_count = max(1, (len(merged) + 1) // 2)
    median_width = float(np.median(sorted(anchor.width for anchor in merged)[-scale_count:]))
    median_height = float(np.median(sorted(anchor.height for anchor in merged)[-scale_count:]))
    median_area = float(np.median(sorted(anchor.area for anchor in merged)[-scale_count:]))
    while len(merged) > target_count:
        candidates: list[tuple[float, int]] = []
        for index, (left, right) in enumerate(zip(merged, merged[1:], strict=False)):
            gap = right.bbox.x0 - left.bbox.x1
            vertical_overlap = min(left.bbox.y1, right.bbox.y1) - max(left.bbox.y0, right.bbox.y0)
            overlap_ratio = vertical_overlap / max(1, min(left.height, right.height))
            small_areas = left.area <= 0.65 * median_area and right.area <= 0.65 * median_area
            small_widths = left.width <= 0.65 * median_width and right.width <= 0.65 * median_width
            combined_width = max(left.bbox.x1, right.bbox.x1) - min(left.bbox.x0, right.bbox.x0)
            if (
                gap <= max(1.0, median_height * 0.08)
                and overlap_ratio >= 0.75
                and (small_areas or small_widths)
                and combined_width <= median_width * 1.5
            ):
                score = abs(gap) / max(1.0, median_height) + abs(
                    np.log(max(1.0, combined_width) / max(1.0, median_width))
                )
                candidates.append((float(score), index))
        if not candidates:
            return None
        _, index = min(candidates)
        merged[index : index + 2] = [_merge_body_anchors(merged[index], merged[index + 1])]
    return tuple(merged)


def _spacing_body_anchor_has_strong_valley(anchor: _BodyAnchor) -> bool:
    projection = anchor.mask.sum(axis=0)
    if projection.size < 4:
        return False
    start = projection.size // 4
    stop = projection.size - start
    for index in range(start, stop):
        valley = int(projection[index])
        left_peak = int(projection[:index].max())
        right_peak = int(projection[index + 1 :].max())
        if valley * 4 < min(left_peak, right_peak) and min(left_peak, right_peak) >= 3:
            return True
    return False


def _spacing_body_anchor_split_evidence(
    anchor: _BodyAnchor,
    anchor_index: int,
    median_width: float | None,
    config: SpacingConfig,
) -> AnchorSplitEvidence | None:
    projection = anchor.mask.sum(axis=0).astype(int)
    if projection.size < 4:
        return None
    start = max(1, projection.size // 5)
    stop = min(projection.size - 1, (projection.size * 4) // 5)
    minima = [
        x
        for x in range(start, stop)
        if projection[x] <= projection[x - 1] and projection[x] <= projection[x + 1]
    ]
    if not minima:
        return None
    cut = min(
        minima,
        key=lambda x: (
            projection[x] / max(1, min(int(projection[:x].max()), int(projection[x + 1 :].max()))),
            x,
        ),
    )
    left_peak = int(projection[:cut].max())
    right_peak = int(projection[cut + 1 :].max())
    neck_reference = max(1, min(left_peak, right_peak))
    valley_ratio = float(projection[cut] / neck_reference)
    low_limit = max(1, int(np.floor(neck_reference * config.auto_split_max_neck_width_ratio)))
    neck_start = cut
    neck_stop = cut
    while neck_start > 0 and projection[neck_start - 1] <= low_limit:
        neck_start -= 1
    while neck_stop + 1 < projection.size and projection[neck_stop + 1] <= low_limit:
        neck_stop += 1
    neck_width = neck_stop - neck_start + 1
    neck_ratio = max(valley_ratio, neck_width / max(1, anchor.width))
    left = anchor.mask[:, :cut]
    right = anchor.mask[:, cut:]
    total_area = max(1, anchor.area)
    left_area_fraction = float(left.sum() / total_area)
    right_area_fraction = float(right.sum() / total_area)

    def child_height(mask: np.ndarray) -> float:
        ys = np.flatnonzero(mask.any(axis=1))
        return (int(ys[-1]) - int(ys[0]) + 1) / max(1, anchor.height) if len(ys) else 0.0

    left_height_ratio = child_height(left)
    right_height_ratio = child_height(right)
    width_ratio = (
        anchor.width / median_width if median_width is not None and median_width > 0 else None
    )
    narrow_neck = neck_ratio <= config.auto_split_max_neck_width_ratio
    substantial_children = (
        min(left_area_fraction, right_area_fraction) >= config.auto_split_min_child_area_fraction
    )
    tall_children = (
        min(left_height_ratio, right_height_ratio) >= config.auto_split_min_child_height_ratio
    )
    width_anomaly = width_ratio is None or width_ratio >= config.auto_split_min_width_ratio
    reasons = []
    reasons.append("narrow-neck" if narrow_neck else "neck-too-wide")
    reasons.append("substantial-children" if substantial_children else "child-area-too-small")
    reasons.append("child-heights-supported" if tall_children else "child-height-too-small")
    if width_ratio is None:
        reasons.append("width-anomaly-not-applicable")
    else:
        reasons.append("wide-anchor" if width_anomaly else "anchor-width-not-anomalous")
    signals = (
        float(np.clip(1.0 - neck_ratio, 0.0, 1.0)),
        float(np.clip(min(left_area_fraction, right_area_fraction) / 0.25, 0.0, 1.0)),
        float(np.clip(min(left_height_ratio, right_height_ratio) / 0.70, 0.0, 1.0)),
        float(np.clip(width_ratio / config.auto_split_min_width_ratio, 0.0, 1.0))
        if width_ratio is not None
        else 1.0,
    )
    score = float(np.mean(signals))
    accepted = narrow_neck and substantial_children and tall_children and width_anomaly
    return AnchorSplitEvidence(
        anchor_index=anchor_index,
        cut_x=anchor.bbox.x0 + cut,
        cut_range=(anchor.bbox.x0 + neck_start, anchor.bbox.x0 + neck_stop + 1),
        valley_strength=float(np.clip(1.0 - valley_ratio, 0.0, 1.0)),
        neck_width=neck_width,
        neck_width_ratio=float(np.clip(neck_ratio, 0.0, 1.0)),
        left_area_fraction=left_area_fraction,
        right_area_fraction=right_area_fraction,
        left_height_ratio=left_height_ratio,
        right_height_ratio=right_height_ratio,
        anchor_width_ratio=width_ratio,
        score=score,
        accepted=accepted,
        reasons=tuple(reasons),
    )


def _split_spacing_body_anchor(
    anchor: _BodyAnchor, cut_x: int
) -> tuple[_BodyAnchor, _BodyAnchor] | None:
    local_cut = cut_x - anchor.bbox.x0
    if not 0 < local_cut < anchor.width:
        return None
    children = []
    for x0, x1 in ((0, local_cut), (local_cut, anchor.width)):
        mask = anchor.mask[:, x0:x1]
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        child_mask = np.array(mask[top:bottom, left:right], copy=True)
        bbox = BBox(
            anchor.bbox.x0 + x0 + left,
            anchor.bbox.y0 + top,
            anchor.bbox.x0 + x0 + right,
            anchor.bbox.y0 + bottom,
        )
        children.append(
            _BodyAnchor(
                component_indices=anchor.component_indices,
                mask=child_mask,
                bbox=bbox,
                center_x=bbox.center_x,
                width=bbox.width,
                height=bbox.height,
                area=int(child_mask.sum()),
            )
        )
    return children[0], children[1]


def _spacing_attachment_metrics(
    component: Component,
    targets: Sequence[Component],
    scale: float,
    config: SpacingConfig,
) -> tuple[float, float, float, float, float, float, bool] | None:
    """Return score, geometry, and eligibility for the best target."""
    if not targets:
        return None
    scale = max(1.0, scale)
    options = []
    for target in targets:
        horizontal_gap = max(
            target.bbox.x0 - component.bbox.x1,
            component.bbox.x0 - target.bbox.x1,
            0,
        )
        vertical_gap = max(
            target.bbox.y0 - component.bbox.y1,
            component.bbox.y0 - target.bbox.y1,
            0,
        )
        area_ratio = component.area / max(1.0, target.area)
        expanded_width = max(target.bbox.x1, component.bbox.x1) - min(
            target.bbox.x0, component.bbox.x0
        )
        expanded_height = max(target.bbox.y1, component.bbox.y1) - min(
            target.bbox.y0, component.bbox.y0
        )
        expansion = expanded_width * expanded_height / (target.width * target.height)
        overlap_x = max(
            0,
            min(target.bbox.x1, component.bbox.x1) - max(target.bbox.x0, component.bbox.x0),
        )
        overlap_y = max(
            0,
            min(target.bbox.y1, component.bbox.y1) - max(target.bbox.y0, component.bbox.y0),
        )
        overlap_ratio = overlap_x / max(1, min(target.width, component.width))
        center_distance = abs(component.center_x - target.center_x) / scale
        vertical_gap_ratio = vertical_gap / scale
        score = (
            horizontal_gap / scale
            + 0.7 * vertical_gap_ratio
            - 0.30 * overlap_x / scale
            - 0.20 * overlap_y / scale
            + 0.15 * max(0.0, area_ratio - 0.45)
            + config.secondary_bbox_expansion_weight * (expansion - 1.0)
        )
        eligible = (
            config.secondary_mark_min_area_ratio
            <= area_ratio
            <= config.secondary_mark_max_area_ratio
            and horizontal_gap <= config.secondary_mark_max_horizontal_gap_ratio * scale
            and vertical_gap <= config.secondary_mark_max_vertical_gap_ratio * scale
            and expansion <= config.secondary_mark_max_bbox_expansion_ratio
        )
        options.append(
            (
                max(0.0, score),
                area_ratio,
                overlap_ratio,
                center_distance,
                vertical_gap_ratio,
                expansion,
                eligible,
            )
        )
    eligible_options = [option for option in options if option[-1]]
    return min(eligible_options or options)


def _spacing_attachment_score(
    component: Component,
    targets: Sequence[Component],
    scale: float,
    config: SpacingConfig | None = None,
) -> float | None:
    metrics = _spacing_attachment_metrics(component, targets, scale, config or SpacingConfig())
    return metrics[0] if metrics is not None and metrics[-1] else None


def _is_suspicious_lateral_speck(mark: Component, targets: Sequence[Component]) -> bool:
    return any(
        mark.area <= 4
        and target.bbox.y0 + target.height * 0.25
        < mark.center_y
        < target.bbox.y1 - target.height * 0.25
        and (mark.bbox.x1 <= target.bbox.x0 or mark.bbox.x0 >= target.bbox.x1)
        for target in targets
    )


def _assign_spacing_marks(
    marks: tuple[Component, ...] | list[Component],
    bodies_by_region: Sequence[Sequence[Component]],
    median_height: float,
    config: SpacingConfig,
    *,
    region_indices: tuple[int, ...] | None = None,
    region_scales: tuple[float, ...] | None = None,
) -> tuple[tuple[SecondaryMarkAssignment, ...], tuple[tuple[Component, ...], ...]]:
    """Assign each plausible secondary component to at most one region."""
    indices = region_indices or tuple(range(len(bodies_by_region)))
    assigned: list[list[Component]] = [[] for _ in bodies_by_region]
    decisions: list[SecondaryMarkAssignment] = []
    for mark in sorted(marks, key=lambda component: component.index):
        if any(_is_suspicious_lateral_speck(mark, bodies) for bodies in bodies_by_region):
            decisions.append(
                SecondaryMarkAssignment(
                    mark.index,
                    None,
                    None,
                    None,
                    None,
                    "unassigned",
                    reason="lateral mark is not aligned with a body",
                )
            )
            continue
        options: list[tuple[tuple[float, float, float, float, float, float, bool], int, int]] = []
        rejected: list[tuple[tuple[float, float, float, float, float, float, bool], int, int]] = []
        for local_index, (region_index, bodies) in enumerate(
            zip(indices, bodies_by_region, strict=True)
        ):
            scale = region_scales[local_index] if region_scales is not None else median_height
            metrics = _spacing_attachment_metrics(mark, bodies, scale, config)
            if metrics is not None:
                entry = (metrics, region_index, local_index)
                (options if metrics[-1] else rejected).append(entry)
        options.sort(key=lambda item: (item[0], item[1]))
        rejected.sort(key=lambda item: (item[0], item[1]))
        if not options or options[0][0][0] > 2.5:
            evidence = options[0][0] if options else rejected[0][0] if rejected else None
            details = _mark_assignment_details(evidence)
            decisions.append(
                SecondaryMarkAssignment(
                    mark.index,
                    None,
                    evidence[0] if evidence else None,
                    None,
                    None,
                    "unassigned",
                    reason="no nearby body passed mark geometry and bbox-expansion limits",
                    **details,
                )
            )
            continue
        best_metrics, target, local_target = options[0]
        best_score = best_metrics[0]
        second_score = options[1][0][0] if len(options) > 1 else None
        margin = (
            (second_score - best_score) / max(1.0, abs(second_score))
            if second_score is not None
            else None
        )
        details = _mark_assignment_details(best_metrics)
        if margin is not None and margin < config.mark_assignment_margin:
            decisions.append(
                SecondaryMarkAssignment(
                    mark.index,
                    None,
                    best_score,
                    second_score,
                    margin,
                    "ambiguous",
                    candidate_region_indices=tuple(item[1] for item in options[:2]),
                    reason="best-target score margin is below the configured threshold",
                    **details,
                )
            )
            continue
        assigned[local_target].append(mark)
        decisions.append(
            SecondaryMarkAssignment(
                mark.index,
                target,
                best_score,
                second_score,
                margin,
                "assigned",
                reason="plausible mark with clear best target and bounded bbox expansion",
                **details,
            )
        )
    return tuple(decisions), tuple(tuple(group) for group in assigned)


def _mark_assignment_details(
    metrics: tuple[float, float, float, float, float, float, bool] | None,
) -> _MarkAssignmentDetails:
    if metrics is None:
        return {}
    return {
        "area_ratio": metrics[1],
        "horizontal_overlap_ratio": metrics[2],
        "horizontal_center_distance": metrics[3],
        "vertical_gap_ratio": metrics[4],
        "bbox_expansion_ratio": metrics[5],
    }


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
