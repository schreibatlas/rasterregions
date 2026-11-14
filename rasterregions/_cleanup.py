from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grouping import _size_scale
from .components import components as label_components
from .models import CleanupConfig, Component, ComponentDecision


@dataclass(frozen=True)
class CleanedPage:
    mask: np.ndarray
    components: tuple[Component, ...]
    primary_area: float
    median_height: float
    warnings: tuple[str, ...]
    component_decisions: tuple[ComponentDecision, ...]


def _is_page_artifact(
    component: Component,
    width: int,
    height: int,
    config: CleanupConfig | None = None,
) -> bool:
    config = config or CleanupConfig()
    page_area = max(1, width * height)
    box_area = component.width * component.height
    box_fraction = box_area / page_area
    area_fraction = component.area / page_area
    fill_ratio = component.area / max(1, box_area)
    touches_border = (
        component.bbox.x0 == 0
        or component.bbox.y0 == 0
        or component.bbox.x1 == width
        or component.bbox.y1 == height
    )
    horizontal_rule = (
        component.width / max(1, width) >= config.rule_min_span_ratio
        and component.width / max(1, component.height) >= config.rule_aspect_ratio
    )
    vertical_rule = (
        component.height / max(1, height) >= config.rule_min_span_ratio
        and component.height / max(1, component.width) >= config.rule_aspect_ratio
    )
    if horizontal_rule or vertical_rule:
        return True
    if touches_border and (
        component.width >= width * config.border_span_ratio
        or component.height >= height * config.border_span_ratio
    ):
        return True
    if (
        touches_border
        and box_fraction > config.border_bbox_ratio
        and fill_ratio < config.border_fill_ratio
    ):
        return True
    return area_fraction > config.max_component_area_ratio


def clean_page_mask(mask: np.ndarray, config: CleanupConfig | None = None) -> CleanedPage:
    config = config or CleanupConfig()
    height, width = mask.shape
    found = label_components(mask)
    if not found:
        return CleanedPage(
            mask.copy(),
            (),
            0.0,
            0.0,
            ("No foreground components were detected.",),
            (),
        )
    artifacts = {
        component.index: _is_page_artifact(component, width, height, config) for component in found
    }
    non_artifacts = [component for component in found if not artifacts[component.index]]
    primary_area, _ = _size_scale(non_artifacts)
    cleaned = np.array(mask, dtype=bool, copy=True)
    warnings: list[str] = []
    decisions: list[ComponentDecision] = []
    kept_components: list[Component] = []
    for component in found:
        reason = None
        if artifacts[component.index]:
            reason = "page border, rule, or oversized artifact"
        elif primary_area and component.area < primary_area * config.tiny_area_ratio:
            reason = "tiny isolated noise"
        if reason:
            cleaned[component.ys, component.xs] = False
            warnings.append(f"Removed component {component.index}: {reason}.")
        else:
            kept_components.append(component)
        decisions.append(
            ComponentDecision(
                component.index,
                component.bbox,
                component.area,
                "removed" if reason else "kept",
                reason,
            )
        )
    kept = tuple(kept_components)
    primary_area, median_height = _size_scale(kept)
    return CleanedPage(
        cleaned,
        kept,
        primary_area,
        median_height,
        tuple(warnings),
        tuple(decisions),
    )
