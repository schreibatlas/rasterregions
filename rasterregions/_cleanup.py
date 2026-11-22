from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grouping import _size_scale
from .components import components as label_components
from .models import (
    CleanupConfig,
    Component,
    ComponentDecision,
    _component_scale_features,
)


@dataclass(frozen=True)
class CleanedPage:
    """Retained mask after hard artifact removal and before contextual disposition."""

    mask: np.ndarray
    components: tuple[Component, ...]
    primary_area: float
    median_height: float
    warnings: tuple[str, ...]
    component_decisions: tuple[ComponentDecision, ...]

    provisional_component_indices: tuple[int, ...] = ()


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
    scale_candidates = [component for component in non_artifacts if component.area > 5]
    micro_ids: set[int] = set()
    body_area = 0.0
    body_height = body_width = 0.0
    bodies: list[Component] = []
    scale_features = {}
    if scale_candidates:
        ordered = sorted(scale_candidates, key=lambda component: component.area)
        upper = ordered[len(ordered) // 2 :]
        body_area = float(np.median([component.area for component in upper]))
        bodies = [component for component in ordered if component.area >= body_area * 0.2]
        body_area = float(np.median([component.area for component in bodies]))
        body_height = float(np.median([component.height for component in bodies]))
        body_width = float(np.median([component.width for component in bodies]))
        scale_features = {
            component.index: _component_scale_features(
                component, body_area, body_height, body_width
            )
            for component in non_artifacts
        }
        area_limit = (
            config.hard_micro_noise_max_area_ratio
            if config.hard_micro_noise_max_area_ratio is not None
            else config.micro_noise_max_area_ratio
        )
        height_limit = (
            config.hard_micro_noise_max_height_ratio
            if config.hard_micro_noise_max_height_ratio is not None
            else config.micro_noise_max_height_ratio
        )
        width_limit = (
            config.hard_micro_noise_max_width_ratio
            if config.hard_micro_noise_max_width_ratio is not None
            else config.micro_noise_max_width_ratio
        )
        micro_ids = {
            component.index
            for component in non_artifacts
            if scale_features[component.index].area_ratio < area_limit
            and scale_features[component.index].height_ratio < height_limit
            and scale_features[component.index].width_ratio < width_limit
        }
    retained = [component for component in non_artifacts if component.index not in micro_ids]
    provisional_scale, _ = _size_scale(retained)
    provisional = tuple(
        component
        for component in retained
        if component.area
        < max(
            provisional_scale * config.tiny_area_ratio,
            body_area * config.secondary_mark_max_area_ratio,
        )
        and component.height < body_height * 0.65
        and component.width < body_width * 0.65
    )
    provisional_ids = {component.index for component in provisional}
    primary_candidates = [
        component for component in retained if component.index not in provisional_ids
    ]
    primary_area, median_height = _size_scale(primary_candidates or retained)

    cleaned = np.array(mask, dtype=bool, copy=True)
    warnings: list[str] = []
    decisions: list[ComponentDecision] = []
    kept_components: list[Component] = []
    for component in found:
        features = scale_features.get(component.index)
        reason = None
        if artifacts[component.index]:
            reason = "page border, rule, or oversized artifact"
            cleaned[component.ys, component.xs] = False
            warnings.append(f"Removed component {component.index}: {reason}.")
        elif component.index in micro_ids:
            reason = "below hard handwriting-scale threshold"
            cleaned[component.ys, component.xs] = False
            warnings.append(f"Removed component {component.index}: {reason}.")
        elif component.index in provisional_ids:
            reason = "plausible detached handwriting mark retained for contextual assignment"
            kept_components.append(component)
        else:
            kept_components.append(component)
        decisions.append(
            ComponentDecision(
                component.index,
                component.bbox,
                component.area,
                "removed" if artifacts[component.index] or component.index in micro_ids else "kept",
                reason,
                role=(
                    "artifact"
                    if artifacts[component.index]
                    else "micro-noise"
                    if component.index in micro_ids
                    else "secondary-candidate"
                    if component.index in provisional_ids
                    else "primary"
                ),
                body_area_ratio=features.area_ratio if features else None,
                body_height_ratio=features.height_ratio if features else None,
                body_width_ratio=features.width_ratio if features else None,
                aspect_ratio=features.aspect_ratio if features else None,
            )
        )

    return CleanedPage(
        cleaned,
        tuple(kept_components),
        primary_area,
        median_height,
        tuple(warnings),
        tuple(decisions),
        tuple(component.index for component in provisional if component.index not in micro_ids),
    )
