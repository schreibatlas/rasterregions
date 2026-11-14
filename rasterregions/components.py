from __future__ import annotations

import numpy as np

from .models import BBox, Component, IsolationCandidate, IsolationResult


def components(mask: np.ndarray) -> tuple[Component, ...]:
    """Label 8-connected foreground components.

    The implementation is dependency-light and intentionally avoids requiring
    SciPy or scikit-image.
    """
    ink = np.asarray(mask, dtype=bool)
    if ink.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    visited = np.zeros(ink.shape, dtype=bool)
    height, width = ink.shape
    result: list[Component] = []

    for start_y, start_x in zip(*np.nonzero(ink), strict=True):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
        ys: list[int] = []
        xs: list[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if ink[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        ya = np.asarray(ys, dtype=int)
        xa = np.asarray(xs, dtype=int)
        result.append(
            Component(
                index=len(result),
                bbox=BBox(int(xa.min()), int(ya.min()), int(xa.max()) + 1, int(ya.max()) + 1),
                area=int(ya.size),
                ys=ya,
                xs=xa,
            )
        )
    return tuple(result)


def isolate_component(
    mask: np.ndarray,
    *,
    center: tuple[float, float],
    competing_centers: tuple[tuple[float, float], ...] = (),
) -> IsolationResult:
    """Select the component nearest ``center`` and explain rejected candidates."""
    found = components(mask)
    if not found:
        return IsolationResult(center=center, mask=np.asarray(mask, dtype=bool).copy())

    def min_distance_squared(component: Component, point: tuple[float, float]) -> float:
        return float(((component.ys - point[1]) ** 2 + (component.xs - point[0]) ** 2).min())

    kept = min(found, key=lambda item: min_distance_squared(item, center))
    selected = np.zeros(np.asarray(mask).shape, dtype=bool)
    records: list[IsolationCandidate] = []
    for component in found:
        is_kept = component.index == kept.index
        reason = "nearest-to-center"
        if is_kept:
            selected[component.ys, component.xs] = True
        elif competing_centers and min(
            min_distance_squared(component, neighbor) for neighbor in competing_centers
        ) < min_distance_squared(component, center):
            reason = "nearest-to-competing-center"
        else:
            reason = "not-nearest-component"
        records.append(
            IsolationCandidate(
                bbox=component.bbox,
                area=component.area,
                kept=is_kept,
                reason=reason,
            )
        )
    return IsolationResult(center=center, mask=selected, candidates=tuple(records))
