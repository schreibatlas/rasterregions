from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .components import components as label_components
from .mask import foreground_mask
from .models import BBox, Component, DetectionResult, MaskMode, Region, RowBand
from .regions import crop, tighten_bbox


def _load_image(source: str | Path | Image.Image | np.ndarray) -> tuple[Image.Image, Path | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        with Image.open(path) as image:
            return image.convert("RGB"), path
    if isinstance(source, Image.Image):
        return source.convert("RGB"), None
    array = np.asarray(source)
    if array.ndim == 2:
        return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").convert("RGB"), None
    if array.ndim == 3 and array.shape[2] in (3, 4):
        if array.dtype != np.uint8:
            finite = np.asarray(array, dtype=float)
            if not np.isfinite(finite).all():
                raise ValueError("image pixels must be finite")
            if finite.max(initial=0.0) <= 1.0:
                finite *= 255.0
            array = np.clip(np.rint(finite), 0, 255).astype(np.uint8)
        mode = "RGBA" if array.shape[2] == 4 else "RGB"
        return Image.fromarray(array, mode=mode).convert("RGB"), None
    raise ValueError("source must be a path, PIL image, or grayscale/RGB/RGBA array")


def _size_scale(found: tuple[Component, ...] | list[Component]) -> tuple[float, float]:
    if not found:
        return 0.0, 0.0
    areas = np.asarray([component.area for component in found], dtype=float)
    upper_half = np.sort(areas)[len(areas) // 2 :]
    primary_area = float(np.median(upper_half))
    body = [component for component in found if component.area >= primary_area * 0.2]
    median_height = float(np.median([component.height for component in body])) if body else 0.0
    return primary_area, median_height


def _is_page_artifact(component: Component, width: int, height: int) -> bool:
    box_area = component.width * component.height
    touches_border = (
        component.bbox.x0 == 0
        or component.bbox.y0 == 0
        or component.bbox.x1 == width
        or component.bbox.y1 == height
    )
    if touches_border and (box_area / (width * height) > 0.05 or component.width > width * 0.8):
        return True
    if component.width >= width * 0.2 and component.width / max(1, component.height) >= 12:
        return True
    if component.height >= height * 0.2 and component.height / max(1, component.width) >= 12:
        return True
    return box_area / (width * height) > 0.2


def _clean_page_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[Component, ...], float, float, list[str]]:
    height, width = mask.shape
    found = label_components(mask)
    if not found:
        return mask.copy(), (), 0.0, 0.0, ["No foreground components were detected."]
    non_artifacts = [c for c in found if not _is_page_artifact(c, width, height)]
    primary_area, _ = _size_scale(non_artifacts)
    cleaned = np.array(mask, dtype=bool, copy=True)
    warnings: list[str] = []
    for component in found:
        reason = None
        if _is_page_artifact(component, width, height):
            reason = "page border, rule, or oversized artifact"
        elif primary_area and component.area < primary_area * 0.008:
            reason = "tiny isolated noise"
        if reason:
            cleaned[component.ys, component.xs] = False
            warnings.append(f"Removed component {component.index}: {reason}.")
    kept = label_components(cleaned)
    primary_area, median_height = _size_scale(kept)
    return cleaned, kept, primary_area, median_height, warnings


def _body_components(
    found: tuple[Component, ...] | list[Component], primary_area: float, median_height: float
) -> list[Component]:
    if not found:
        return []
    minimum_area = primary_area * 0.18
    minimum_height = median_height * 0.25
    bodies = [
        component
        for component in found
        if component.area >= minimum_area
        and (component.height >= minimum_height or component.width >= minimum_height)
    ]
    return bodies or [max(found, key=lambda component: component.area)]


def detect_rows(mask: np.ndarray) -> tuple[RowBand, ...]:
    """Detect coherent foreground rows from component-center clustering."""
    ink = np.asarray(mask, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    _, found, primary_area, median_height, _ = _clean_page_mask(ink)
    bodies = _body_components(found, primary_area, median_height)
    if not bodies:
        return ()
    ordered = sorted(bodies, key=lambda component: component.center_y)
    rows: list[list[Component]] = []
    for component in ordered:
        if not rows or component.center_y - np.median(
            [member.center_y for member in rows[-1]]
        ) > max(1.0, median_height * 0.72):
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


def _row_y_bounds(
    row: RowBand,
    found: tuple[Component, ...],
    median_height: float,
    page_height: int,
) -> tuple[int, int]:
    body_indices = set(row.component_indices)
    bodies = [component for component in found if component.index in body_indices]
    satellites = [
        component
        for component in found
        if component.index not in body_indices
        and any(_satellite_score(component, body, median_height) is not None for body in bodies)
    ]
    top = min([row.bbox.y0, *(component.bbox.y0 for component in satellites)])
    bottom = max([row.bbox.y1, *(component.bbox.y1 for component in satellites)])
    return max(0, top), min(page_height, bottom)


def _group_nearby_bodies(bodies: list[Component], median_height: float) -> list[list[Component]]:
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
        if gap <= max(1.0, median_height * 0.12) and overlap > 0:
            previous.append(component)
        else:
            groups.append([component])
    return groups


def _satellite_score(satellite: Component, body: Component, median_height: float) -> float | None:
    if satellite.area >= body.area * 0.6:
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
    if horizontal_gap > median_height * 0.4 or vertical_gap > median_height * 1.25:
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


def _candidate_cut_positions(projection: np.ndarray) -> list[int]:
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
    scale = float(np.percentile(projection[projection > 0], 35)) if np.any(projection) else 0.0
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
) -> tuple[list[int], float] | None:
    projection = np.asarray(row_mask, dtype=np.uint8).sum(axis=0)
    occupied = np.flatnonzero(projection)
    if occupied.size == 0:
        return None
    left, right = int(occupied[0]), int(occupied[-1]) + 1
    projection = projection[left:right]
    cuts = _candidate_cut_positions(projection)
    if len(cuts) < expected_count + 1:
        return None

    nonzero_scale = max(1.0, float(np.percentile(projection[projection > 0], 80)))
    crossing: dict[int, int] = {}
    for cut in cuts[1:-1]:
        absolute_x = x_offset + left + cut
        crossing[cut] = sum(c.bbox.x0 < absolute_x < c.bbox.x1 for c in found)
    boundary_cost = {
        cut: float(projection[cut] / nonzero_scale) * 3.0 + crossing.get(cut, 0) * 5.0
        for cut in cuts[1:-1]
    }
    prefix = np.r_[0.0, np.cumsum(projection, dtype=float)]
    component_scale, median_height = _size_scale(found)
    primary_threshold = component_scale * 0.18

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
        component_cost = 0.8 * abs(body_count - 1)
        narrow_cost = max(0.0, median_height * 0.15 - width) / max(1.0, median_height) * 4.0
        area_cost = 2.0 if ink < max(1.0, float(prefix[-1]) * 0.006) else 0.0
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
    confidences: list[float] = []
    for boundary in boundaries[1:-1]:
        local_cut = boundary - left
        valley = float(projection[local_cut] / nonzero_scale)
        component_penalty = crossing.get(local_cut, 0)
        confidences.append(float(np.clip(1.0 - valley - 0.55 * component_penalty, 0.0, 1.0)))
    return boundaries, min(confidences, default=0.0)


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
) -> list[Region]:
    bodies = _body_components(found, primary_area, median_height)
    body_indices = {component.index for component in bodies}
    regions: list[Region] = []
    reading_order = 0

    for row_index, row in enumerate(rows):
        row_bodies = [c for c in bodies if c.index in row.component_indices]
        if not row_bodies:
            continue
        groups = _group_nearby_bodies(row_bodies, median_height)
        satellites = [c for c in found if c.index not in body_indices]
        attached: list[list[Component]] = [[] for _ in groups]
        for satellite in satellites:
            options: list[tuple[float, int]] = []
            for group_index, group in enumerate(groups):
                scores = [
                    score
                    for body in group
                    if (score := _satellite_score(satellite, body, median_height)) is not None
                ]
                if scores:
                    options.append((min(scores), group_index))
            if options:
                _, group_index = min(options)
                attached[group_index].append(satellite)

        for column_index, (group, marks) in enumerate(zip(groups, attached, strict=True)):
            members = [*group, *marks]
            box = BBox(
                min(c.bbox.x0 for c in members),
                min(c.bbox.y0 for c in members),
                max(c.bbox.x1 for c in members),
                max(c.bbox.y1 for c in members),
            )
            next_group = groups[column_index + 1] if column_index + 1 < len(groups) else None
            previous_group = groups[column_index - 1] if column_index else None
            left_gap = (
                group[0].bbox.x0 - max(c.bbox.x1 for c in previous_group)
                if previous_group
                else median_height
            )
            right_gap = (
                min(c.bbox.x0 for c in next_group) - max(c.bbox.x1 for c in group)
                if next_group
                else median_height
            )
            confidence = 0.92 if min(left_gap, right_gap) >= median_height * 0.15 else 0.58
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
                    component_indices=tuple(sorted(c.index for c in members)),
                    padding=padding,
                )
            )
            reading_order += 1
    return regions


def _expected_count_regions(
    image: Image.Image,
    mask: np.ndarray,
    found: tuple[Component, ...],
    rows: tuple[RowBand, ...],
    median_height: float,
    expected_count: int,
    padding: int,
) -> tuple[list[Region], float] | None:
    candidates: list[tuple[list[Region], float]] = []
    for row_index, row in enumerate(rows):
        row_y0, row_y1 = _row_y_bounds(row, found, median_height, mask.shape[0])
        row_mask = mask[row_y0:row_y1, :]
        partition = _partition_row(row_mask, expected_count, found, x_offset=0)
        if partition is None:
            continue
        boundaries, confidence = partition
        regions: list[Region] = []
        for column_index, (x0, x1) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            segment = mask[row_y0:row_y1, x0:x1]
            ys, xs = np.nonzero(segment)
            if not len(xs):
                regions = []
                break
            box = BBox(x0, row_y0 + int(ys.min()), x1, row_y0 + int(ys.max()) + 1)
            member_indices = tuple(
                c.index
                for c in found
                if c.bbox.x0 < x1 and c.bbox.x1 > x0 and c.bbox.y0 < row_y1 and c.bbox.y1 > row_y0
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
        if len(regions) == expected_count:
            candidates.append((regions, confidence))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[1], sum(int(r.mask.sum()) for r in item[0])))


def detect_regions(
    source: str | Path | Image.Image | np.ndarray,
    *,
    expected_count: int | None = None,
    mode: MaskMode = "auto",
    padding: int | None = None,
    min_confidence: float = 0.3,
) -> DetectionResult:
    """Detect anonymous ordered foreground regions.

    With no ``expected_count``, all rows and region groups are inferred. When
    ``expected_count`` is supplied, the best row is partitioned into exactly
    that many regions using projection valleys and component-crossing costs.
    """
    if expected_count is not None and expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between zero and one")

    image, source_path = _load_image(source)
    mask_result = foreground_mask(image, mode=mode)
    clean_mask, found, primary_area, median_height, warnings = _clean_page_mask(mask_result.mask)
    rows = detect_rows(clean_mask)
    if padding is None:
        padding = max(1, round(median_height * 0.08)) if median_height else 1
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    status: str = "ok"
    if expected_count is None:
        regions = _automatic_regions(
            image, clean_mask, found, primary_area, median_height, rows, padding
        )
        if not regions:
            status = "failed"
        elif any(region.confidence < min_confidence for region in regions):
            status = "ambiguous"
    else:
        selected = _expected_count_regions(
            image, clean_mask, found, rows, median_height, expected_count, padding
        )
        if selected is None:
            regions = []
            status = "failed"
            warnings.append(f"Unable to recover {expected_count} credible regions.")
        else:
            regions, confidence = selected
            if confidence < min_confidence:
                regions = []
                status = "failed"
                warnings.append(
                    f"Unable to recover {expected_count} credible regions: "
                    f"best partition confidence is {confidence:.3f}, "
                    f"below the required {min_confidence:.3f}."
                )

    return DetectionResult(
        source_path=source_path,
        image=image,
        mask_result=mask_result,
        clean_mask=clean_mask,
        components=found,
        rows=rows,
        regions=tuple(regions),
        expected_count=expected_count,
        status=status,  # type: ignore[arg-type]
        warnings=tuple([*mask_result.warnings, *warnings]),
    )
