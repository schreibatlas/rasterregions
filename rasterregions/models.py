from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

MaskMethod = Literal["grayscale", "dark", "color", "hybrid", "adaptive"]
MaskMode = Literal["auto", "grayscale", "dark", "color", "hybrid", "adaptive"]
ExpectedCountScope = Literal["row", "page"]
DetectionStatus = Literal["ok", "ambiguous", "failed"]
ComponentRole = Literal[
    "primary",
    "secondary-candidate",
    "secondary",
    "noise",
    "artifact",
    "ambiguous",
    "micro-noise",
]


IsolationReason = Literal[
    "nearest-to-center",
    "nearest-to-competing-center",
    "not-nearest-component",
]


@dataclass(frozen=True, order=True)
class BBox:
    """Half-open pixel rectangle ``[x0, x1) × [y0, y1)``."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if min(self.x0, self.y0) < 0:
            raise ValueError("bbox coordinates must be nonnegative")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bbox must have positive width and height")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


@dataclass(frozen=True)
class MaskResult:
    """A read-only two-dimensional foreground mask and its extraction quality."""

    mask: np.ndarray
    confidence: float
    method: MaskMethod
    warnings: tuple[str, ...] = ()

    threshold: float | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("mask confidence must be between zero and one")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "confidence", float(self.confidence))


@dataclass(frozen=True)
class ForegroundEvidence:
    """A boolean foreground mask and its signed continuous decision field.

    The field is sampled at pixel centers. Nonnegative values denote foreground,
    including the zero-valued classification boundary.
    """

    mask: np.ndarray
    decision_field: np.ndarray = field(repr=False, compare=False)
    confidence: float
    method: MaskMethod
    warnings: tuple[str, ...] = ()
    threshold: float | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        decision_field = np.asarray(self.decision_field, dtype=np.float32)
        if mask.ndim != 2 or decision_field.ndim != 2:
            raise ValueError("mask and decision_field must be 2D arrays")
        if mask.shape != decision_field.shape:
            raise ValueError("mask and decision_field must have the same shape")
        if not np.isfinite(decision_field).all():
            raise ValueError("decision_field values must be finite")
        if not np.array_equal(mask, decision_field >= 0.0):
            raise ValueError("mask must equal decision_field >= 0")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("foreground evidence confidence must be between zero and one")

        mask = np.array(mask, dtype=bool, copy=True)
        decision_field = np.array(decision_field, dtype=np.float32, copy=True)
        mask.setflags(write=False)
        decision_field.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "decision_field", decision_field)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _pyramid_shape(shape: tuple[int, int], field_name: str) -> tuple[int, int]:
    try:
        dimensions = tuple(shape)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a (height, width) pair") from exc
    if len(dimensions) != 2 or any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or dimension <= 0
        for dimension in dimensions
    ):
        raise ValueError(f"{field_name} must contain positive integer dimensions")
    return int(dimensions[0]), int(dimensions[1])


@dataclass(frozen=True)
class ForegroundPyramidLevel:
    """Fractional foreground occupancy at one pyramid resolution.

    ``coverage`` is fitting evidence. The thresholded ``mask`` is only a
    visualization/convenience representation, not a topology-preserving
    substitute for the source mask.
    """

    index: int
    source_shape: tuple[int, int]
    shape: tuple[int, int]
    scale_x: float
    scale_y: float
    coverage: np.ndarray
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("pyramid level index must be a nonnegative integer")
        source_shape = _pyramid_shape(self.source_shape, "source_shape")
        shape = _pyramid_shape(self.shape, "shape")
        if isinstance(self.scale_x, bool) or not isfinite(self.scale_x):
            raise ValueError("scale_x must be finite")
        if isinstance(self.scale_y, bool) or not isfinite(self.scale_y):
            raise ValueError("scale_y must be finite")
        expected_scale_x = shape[1] / source_shape[1]
        expected_scale_y = shape[0] / source_shape[0]
        if not np.isclose(self.scale_x, expected_scale_x) or not np.isclose(
            self.scale_y, expected_scale_y
        ):
            raise ValueError("pyramid level scales must agree with source and level shapes")
        if isinstance(self.mask_threshold, bool) or not isfinite(self.mask_threshold):
            raise ValueError("mask_threshold must be finite and between zero and one")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be finite and between zero and one")

        coverage = np.asarray(self.coverage, dtype=np.float32)
        if coverage.ndim != 2 or coverage.shape != shape:
            raise ValueError("coverage dimensions must match the pyramid level shape")
        if not np.isfinite(coverage).all() or np.any((coverage < 0.0) | (coverage > 1.0)):
            raise ValueError("coverage values must be finite and between zero and one")
        coverage = np.array(coverage, dtype=np.float32, copy=True)
        coverage.setflags(write=False)
        object.__setattr__(self, "source_shape", source_shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "scale_x", float(self.scale_x))
        object.__setattr__(self, "scale_y", float(self.scale_y))
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "mask_threshold", float(self.mask_threshold))

    @property
    def mask(self) -> np.ndarray:
        """Return a read-only thresholded view for visualization and convenience."""
        mask = self.coverage >= self.mask_threshold
        mask.setflags(write=False)
        return mask

    @property
    def ink_mass(self) -> float:
        """Return the sum of foreground occupancy in level pixels."""
        return float(self.coverage.sum())

    @property
    def normalized_ink_mass(self) -> float:
        """Return occupancy mass in approximate source-pixel units."""
        return self.ink_mass / (self.scale_x * self.scale_y)


@dataclass(frozen=True)
class ForegroundPyramid:
    """Coarse-to-fine fitting levels backed by authoritative source evidence."""

    evidence: ForegroundEvidence
    levels: tuple[ForegroundPyramidLevel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ForegroundEvidence):
            raise ValueError("evidence must be a ForegroundEvidence instance")
        levels = tuple(self.levels)
        if not levels or any(not isinstance(level, ForegroundPyramidLevel) for level in levels):
            raise ValueError("levels must contain at least one ForegroundPyramidLevel")
        source_shape = _pyramid_shape(self.evidence.mask.shape, "evidence mask shape")
        if any(level.source_shape != source_shape for level in levels):
            raise ValueError("all pyramid levels must use the evidence source shape")
        if any(level.index != index for index, level in enumerate(levels)):
            raise ValueError("pyramid level indexes must be consecutive from zero")
        for previous, current in zip(levels, levels[1:], strict=False):
            if (
                current.shape[0] < previous.shape[0]
                or current.shape[1] < previous.shape[1]
                or current.scale_x * current.scale_y <= previous.scale_x * previous.scale_y
            ):
                raise ValueError("pyramid levels must be ordered coarse to fine")
        final = levels[-1]
        expected_coverage = self.evidence.mask.astype(np.float32)
        if final.shape != source_shape or not np.array_equal(final.coverage, expected_coverage):
            raise ValueError("the final pyramid level must exactly preserve the source mask")
        if any(level.mask_threshold != levels[0].mask_threshold for level in levels):
            raise ValueError("all pyramid levels must use the same mask threshold")
        object.__setattr__(self, "levels", levels)


@dataclass(frozen=True)
class Component:
    """An 8-connected foreground component with row-major index and pixel coordinates."""

    index: int
    bbox: BBox
    area: int
    ys: np.ndarray = field(repr=False, compare=False)
    xs: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.index < 0 or self.area <= 0:
            raise ValueError("component index must be nonnegative and area must be positive")
        ys = np.asarray(self.ys)
        xs = np.asarray(self.xs)
        if (
            ys.ndim != 1
            or xs.ndim != 1
            or ys.size != self.area
            or xs.size != self.area
            or ys.dtype.kind not in "iu"
            or xs.dtype.kind not in "iu"
        ):
            raise ValueError("component coordinate lengths must match area")
        ys = np.array(ys, dtype=np.intp, copy=True)
        xs = np.array(xs, dtype=np.intp, copy=True)
        if (
            np.any(ys < self.bbox.y0)
            or np.any(ys >= self.bbox.y1)
            or np.any(xs < self.bbox.x0)
            or np.any(xs >= self.bbox.x1)
        ):
            raise ValueError("component coordinates must lie inside bbox")
        ys.setflags(write=False)
        xs.setflags(write=False)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "xs", xs)

    @property
    def width(self) -> int:
        return self.bbox.width

    @property
    def height(self) -> int:
        return self.bbox.height

    @property
    def center_x(self) -> float:
        return self.bbox.center_x

    @property
    def center_y(self) -> float:
        return self.bbox.center_y


@dataclass(frozen=True)
class ComponentDecision:
    """A raw-mask component's cleanup disposition and explanation."""

    component_index: int
    bbox: BBox
    area: int
    action: Literal["kept", "removed"]
    reason: str | None = None
    role: ComponentRole = "primary"
    body_area_ratio: float | None = None


@dataclass(frozen=True)
class PartitionBoundary:
    """A candidate partition cut with projection, crossings, and heuristic confidence."""

    x: int
    projection: float
    crossing_components: int
    confidence: float
    valley_strength: float | None = None
    component_indices: tuple[int, ...] = ()
    crossing_cost: float = 0.0
    selected: bool = True
    kind: Literal["zero-run", "valley", "local-minimum", "component-midpoint"] = "valley"

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))

        if not isfinite(self.crossing_cost) or self.crossing_cost < 0.0:
            raise ValueError("boundary crossing_cost must be finite and nonnegative")


@dataclass(frozen=True)
class AnchorSplitEvidence:
    """Measured evidence for splitting one primary body anchor."""

    anchor_index: int
    cut_x: int
    cut_range: tuple[int, int]
    valley_strength: float
    neck_width: int
    neck_width_ratio: float
    left_area_fraction: float
    right_area_fraction: float
    left_height_ratio: float
    right_height_ratio: float
    anchor_width_ratio: float | None
    score: float
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.anchor_index < 0 or self.cut_x < 0 or self.neck_width < 1:
            raise ValueError("split evidence indexes and neck width must be positive")
        if len(self.cut_range) != 2 or not self.cut_range[0] <= self.cut_x < self.cut_range[1]:
            raise ValueError("cut_x must lie inside the positive cut_range")
        values = (
            self.valley_strength,
            self.neck_width_ratio,
            self.left_area_fraction,
            self.right_area_fraction,
            self.left_height_ratio,
            self.right_height_ratio,
            self.score,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("split evidence values must be finite")
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("split evidence ratios and score must be between zero and one")
        if self.anchor_width_ratio is not None and (
            not isfinite(self.anchor_width_ratio) or self.anchor_width_ratio <= 0.0
        ):
            raise ValueError("anchor_width_ratio must be finite and positive")
        object.__setattr__(self, "cut_range", tuple(self.cut_range))
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True)
class CleanupConfig:
    """Artifact thresholds and maximum micro-noise ratios to median body dimensions.

    Micro-noise is removed only when all three ratios are below their limits.
    """

    tiny_area_ratio: float = 0.008
    rule_aspect_ratio: float = 12.0
    rule_min_span_ratio: float = 0.20
    border_span_ratio: float = 0.80
    border_bbox_ratio: float = 0.20
    border_fill_ratio: float = 0.15
    max_component_area_ratio: float = 0.50

    micro_noise_max_area_ratio: float = 0.02
    micro_noise_max_height_ratio: float = 0.08
    micro_noise_max_width_ratio: float = 0.08
    secondary_mark_max_area_ratio: float = 0.20
    micro_noise_mark_max_vertical_gap_ratio: float = 0.50
    micro_noise_mark_max_horizontal_gap_ratio: float = 0.40

    def __post_init__(self) -> None:
        values = (
            self.tiny_area_ratio,
            self.rule_aspect_ratio,
            self.rule_min_span_ratio,
            self.border_span_ratio,
            self.border_bbox_ratio,
            self.border_fill_ratio,
            self.max_component_area_ratio,
            self.micro_noise_max_area_ratio,
            self.micro_noise_max_height_ratio,
            self.micro_noise_max_width_ratio,
            self.secondary_mark_max_area_ratio,
            self.micro_noise_mark_max_vertical_gap_ratio,
            self.micro_noise_mark_max_horizontal_gap_ratio,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("cleanup configuration values must be finite")
        fractions = (
            self.tiny_area_ratio,
            self.rule_min_span_ratio,
            self.border_span_ratio,
            self.border_bbox_ratio,
            self.border_fill_ratio,
            self.max_component_area_ratio,
            self.micro_noise_max_area_ratio,
            self.micro_noise_max_height_ratio,
            self.micro_noise_max_width_ratio,
            self.secondary_mark_max_area_ratio,
            self.micro_noise_mark_max_vertical_gap_ratio,
            self.micro_noise_mark_max_horizontal_gap_ratio,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("cleanup fractions must be between zero and one")
        if self.rule_aspect_ratio <= 0.0:
            raise ValueError("rule_aspect_ratio must be positive")


@dataclass(frozen=True)
class GroupingConfig:
    """Ratios controlling body grouping, row clustering, and satellite attachment."""

    body_min_area_ratio: float = 0.18
    body_min_height_ratio: float = 0.25
    row_center_gap_ratio: float = 0.72
    body_join_gap_ratio: float = 0.12
    satellite_max_area_ratio: float = 0.60
    satellite_max_horizontal_gap_ratio: float = 0.40
    satellite_max_vertical_gap_ratio: float = 1.25

    def __post_init__(self) -> None:
        values = (
            self.body_min_area_ratio,
            self.body_min_height_ratio,
            self.row_center_gap_ratio,
            self.body_join_gap_ratio,
            self.satellite_max_area_ratio,
            self.satellite_max_horizontal_gap_ratio,
            self.satellite_max_vertical_gap_ratio,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("grouping configuration values must be finite")
        fractions = (
            self.body_min_area_ratio,
            self.body_min_height_ratio,
            self.satellite_max_area_ratio,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("grouping area and height ratios must be between zero and one")
        distances = (
            self.row_center_gap_ratio,
            self.body_join_gap_ratio,
            self.satellite_max_horizontal_gap_ratio,
            self.satellite_max_vertical_gap_ratio,
        )
        if any(value < 0.0 for value in distances):
            raise ValueError("grouping distance ratios must be nonnegative")


@dataclass(frozen=True)
class MaskConfig:
    """Thresholding and mask-quality parameters for foreground extraction."""

    saturation_threshold: float = 0.20
    color_max_value: float = 250.0
    paper_percentile: float = 95.0
    dark_delta: float = 32.0
    neutral_dark_delta: float = 80.0
    adaptive_window_size: int = 31
    adaptive_offset: float = 10.0
    minimum_foreground_fraction: float = 0.00005
    neutral_support_fraction: float = 0.0005
    target_foreground_fraction: float = 0.08
    maximum_foreground_fraction: float = 0.50
    long_structure_occupancy: float = 0.75
    contrast_confidence_scale: float = 128.0
    color_confidence_saturation_scale: float = 0.5
    threshold_max: float = 240.0
    dark_threshold_min: float = 12.0
    adaptive_global_ceiling_max: float = 250.0
    adaptive_global_delta: float = 8.0

    def __post_init__(self) -> None:
        values = (
            self.saturation_threshold,
            self.color_max_value,
            self.paper_percentile,
            self.dark_delta,
            self.neutral_dark_delta,
            self.adaptive_offset,
            self.minimum_foreground_fraction,
            self.neutral_support_fraction,
            self.target_foreground_fraction,
            self.maximum_foreground_fraction,
            self.long_structure_occupancy,
            self.contrast_confidence_scale,
            self.color_confidence_saturation_scale,
            self.threshold_max,
            self.dark_threshold_min,
            self.adaptive_global_ceiling_max,
            self.adaptive_global_delta,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("mask configuration values must be finite")
        if (
            isinstance(self.adaptive_window_size, bool)
            or not isinstance(self.adaptive_window_size, int)
            or self.adaptive_window_size < 3
            or self.adaptive_window_size % 2 == 0
        ):
            raise ValueError("adaptive_window_size must be an odd integer of at least 3")
        unit_values = (
            self.saturation_threshold,
            self.minimum_foreground_fraction,
            self.neutral_support_fraction,
            self.target_foreground_fraction,
            self.maximum_foreground_fraction,
            self.long_structure_occupancy,
        )
        if any(not 0.0 <= value <= 1.0 for value in unit_values):
            raise ValueError("mask fractions and saturation must be between zero and one")
        if not 0.0 <= self.paper_percentile <= 100.0:
            raise ValueError("paper_percentile must be between zero and 100")
        if not 0.0 <= self.color_max_value <= 255.0:
            raise ValueError("color_max_value must be between zero and 255")
        if any(
            value < 0.0
            for value in (
                self.dark_delta,
                self.neutral_dark_delta,
                self.adaptive_offset,
                self.adaptive_global_delta,
            )
        ):
            raise ValueError("mask threshold deltas must be nonnegative")
        if not 0.0 <= self.dark_threshold_min <= self.threshold_max <= 255.0:
            raise ValueError("dark threshold bounds must be between zero and 255")
        if not 0.0 <= self.adaptive_global_ceiling_max <= 255.0:
            raise ValueError("adaptive_global_ceiling_max must be between zero and 255")
        if self.neutral_support_fraction == 0.0:
            raise ValueError("neutral_support_fraction must be positive")
        if self.contrast_confidence_scale <= 0.0 or self.color_confidence_saturation_scale <= 0.0:
            raise ValueError("mask confidence scales must be positive")


@dataclass(frozen=True)
class PartitionConfig:
    """Costs and thresholds used by expected-count partition search."""

    crossing_cost: float = 5.0
    projection_cost: float = 3.0
    primary_area_ratio: float = 0.18
    narrow_width_ratio: float = 0.15
    small_segment_fraction: float = 0.006
    boundary_crossing_confidence_penalty: float = 0.55
    density_target: float = 0.08
    candidate_valley_percentile: float = 35.0
    normalization_percentile: float = 80.0
    body_mismatch_cost: float = 0.8
    narrow_segment_cost: float = 4.0
    small_segment_cost: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.crossing_cost,
            self.projection_cost,
            self.primary_area_ratio,
            self.narrow_width_ratio,
            self.small_segment_fraction,
            self.boundary_crossing_confidence_penalty,
            self.density_target,
            self.candidate_valley_percentile,
            self.normalization_percentile,
            self.body_mismatch_cost,
            self.narrow_segment_cost,
            self.small_segment_cost,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("partition configuration values must be finite")
        fractions = (
            self.primary_area_ratio,
            self.narrow_width_ratio,
            self.small_segment_fraction,
            self.boundary_crossing_confidence_penalty,
            self.density_target,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("partition ratios and confidence penalty must be between zero and one")
        if not 0.0 <= self.candidate_valley_percentile <= 100.0:
            raise ValueError("candidate_valley_percentile must be between zero and 100")
        if not 0.0 <= self.normalization_percentile <= 100.0:
            raise ValueError("normalization_percentile must be between zero and 100")
        costs = (
            self.crossing_cost,
            self.projection_cost,
            self.body_mismatch_cost,
            self.narrow_segment_cost,
            self.small_segment_cost,
        )
        if any(value < 0.0 for value in costs):
            raise ValueError("partition costs must be nonnegative")


@dataclass(frozen=True)
class SpacingConfig:
    """Scale-relative controls for spacing-based region segmentation."""

    padding_ratio: float = 0.08
    width_prior_ratio: float | None = None
    min_region_width_ratio: float = 0.15
    max_region_width_ratio: float = 3.5
    max_inferred_count: int = 24
    complexity_penalty: float = 0.35
    auto_split_min_width_ratio: float = 1.35
    auto_split_max_neck_width_ratio: float = 0.20
    auto_split_min_child_area_fraction: float = 0.16
    auto_split_min_child_height_ratio: float = 0.45
    auto_split_min_improvement: float = 0.15
    ambiguity_margin: float = 0.10
    mark_assignment_margin: float = 0.15
    row_assignment_margin: float = 0.15
    secondary_mark_min_area_ratio: float = 0.001
    secondary_mark_max_area_ratio: float = 0.60
    secondary_mark_max_vertical_gap_ratio: float = 0.65
    secondary_mark_max_horizontal_gap_ratio: float = 0.75
    secondary_mark_max_bbox_expansion_ratio: float = 3.0
    secondary_bbox_expansion_weight: float = 0.4

    hypothesis_limit: int = 5
    max_candidate_boundaries: int = 256
    max_count_candidates: int = 24
    max_partition_states: int = 100_000
    max_page_states: int = 50_000
    max_secondary_mark_branches: int = 32

    def __post_init__(self) -> None:
        values = (
            self.padding_ratio,
            self.min_region_width_ratio,
            self.max_region_width_ratio,
            self.complexity_penalty,
            self.ambiguity_margin,
            self.mark_assignment_margin,
            self.row_assignment_margin,
            self.auto_split_min_width_ratio,
            self.auto_split_max_neck_width_ratio,
            self.auto_split_min_child_area_fraction,
            self.auto_split_min_child_height_ratio,
            self.auto_split_min_improvement,
            self.secondary_mark_min_area_ratio,
            self.secondary_mark_max_area_ratio,
            self.secondary_mark_max_vertical_gap_ratio,
            self.secondary_mark_max_horizontal_gap_ratio,
            self.secondary_mark_max_bbox_expansion_ratio,
            self.secondary_bbox_expansion_weight,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("spacing configuration values must be finite")
        if self.width_prior_ratio is not None and (
            not isfinite(self.width_prior_ratio) or self.width_prior_ratio <= 0.0
        ):
            raise ValueError("width_prior_ratio must be finite and positive when supplied")
        if self.padding_ratio < 0.0:
            raise ValueError("padding_ratio must be nonnegative")
        if (
            self.min_region_width_ratio <= 0.0
            or self.max_region_width_ratio < self.min_region_width_ratio
        ):
            raise ValueError("region width ratios must be positive and ordered")
        if (
            isinstance(self.max_inferred_count, bool)
            or not isinstance(self.max_inferred_count, int)
            or self.max_inferred_count < 1
        ):
            raise ValueError("max_inferred_count must be a positive integer")
        limits = (
            self.hypothesis_limit,
            self.max_candidate_boundaries,
            self.max_count_candidates,
            self.max_partition_states,
            self.max_page_states,
            self.max_secondary_mark_branches,
        )
        if any(
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 for limit in limits
        ):
            raise ValueError("spacing search limits must be positive integers")
        if self.complexity_penalty < 0.0:
            raise ValueError("complexity_penalty must be nonnegative")
        margins = (self.ambiguity_margin, self.mark_assignment_margin, self.row_assignment_margin)
        if any(not 0.0 <= margin <= 1.0 for margin in margins):
            raise ValueError("spacing confidence margins must be between zero and one")
        if self.auto_split_min_width_ratio < 1.0:
            raise ValueError("auto_split_min_width_ratio must be at least one")
        auto_split_fractions = (
            self.auto_split_max_neck_width_ratio,
            self.auto_split_min_child_area_fraction,
            self.auto_split_min_child_height_ratio,
            self.auto_split_min_improvement,
        )
        if any(not 0.0 <= value <= 1.0 for value in auto_split_fractions):
            raise ValueError("automatic split ratios and improvement must be between zero and one")
        if not (
            0.0 <= self.secondary_mark_min_area_ratio <= self.secondary_mark_max_area_ratio <= 1.0
        ):
            raise ValueError("secondary mark area ratios must be ordered and in [0, 1]")
        if (
            min(
                self.secondary_mark_max_vertical_gap_ratio,
                self.secondary_mark_max_horizontal_gap_ratio,
                self.secondary_bbox_expansion_weight,
            )
            < 0.0
            or self.secondary_mark_max_bbox_expansion_ratio < 1.0
        ):
            raise ValueError("secondary mark distance and expansion limits must be nonnegative")


@dataclass(frozen=True)
class DetectionConfig:
    """Combined immutable configuration for detection stages."""

    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    spacing: SpacingConfig = field(default_factory=SpacingConfig)


@dataclass(frozen=True)
class DetectionDiagnostics:
    """Detection parameters and intermediate decisions for diagnostics."""

    image_size: tuple[int, int]
    mask_threshold: float | None
    raw_component_count: int
    cleaned_component_count: int
    component_decisions: tuple[ComponentDecision, ...]
    partition_boundaries: tuple[PartitionBoundary, ...] = ()
    mode: MaskMode = "auto"
    padding: int = 0
    min_confidence: float = 0.3
    expected_count: int | None = None
    expected_count_scope: ExpectedCountScope = "row"
    mask_method: MaskMethod = "grayscale"
    mask_confidence: float = 0.0
    foreground_fraction: float = 0.0
    roi: BBox | None = None

    cleanup_config: CleanupConfig | None = None
    spacing_config: SpacingConfig | None = None
    row_segmentation: tuple[RowSegmentationDiagnostics, ...] = ()
    secondary_mark_assignments: tuple[SecondaryMarkAssignment, ...] = ()
    selected_region_count: int | None = None
    row_count: int | None = None
    best_second_best_partition_margin: float | None = None
    width_prior_deviation: float | None = None
    body_counts_per_region: tuple[int, ...] = ()
    unassigned_ink_fraction: float = 0.0
    search_truncated: bool = False
    search_limit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_decisions", tuple(self.component_decisions))
        object.__setattr__(self, "partition_boundaries", tuple(self.partition_boundaries))
        object.__setattr__(self, "row_segmentation", tuple(self.row_segmentation))
        object.__setattr__(
            self, "secondary_mark_assignments", tuple(self.secondary_mark_assignments)
        )
        object.__setattr__(self, "body_counts_per_region", tuple(self.body_counts_per_region))
        if self.cleanup_config is not None and not isinstance(self.cleanup_config, CleanupConfig):
            raise ValueError("cleanup_config must be CleanupConfig")
        if self.spacing_config is not None and not isinstance(self.spacing_config, SpacingConfig):
            raise ValueError("spacing_config must be SpacingConfig")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.body_counts_per_region
        ):
            raise ValueError("body counts must be nonnegative integers")
        if self.width_prior_deviation is not None and (
            not isfinite(self.width_prior_deviation) or self.width_prior_deviation < 0.0
        ):
            raise ValueError("width_prior_deviation must be finite and nonnegative")
        if (
            not isfinite(self.unassigned_ink_fraction)
            or not 0.0 <= self.unassigned_ink_fraction <= 1.0
        ):
            raise ValueError("unassigned_ink_fraction must be between zero and one")
        for count in (self.selected_region_count, self.row_count):
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError("diagnostic counts must be nonnegative integers")
        if self.best_second_best_partition_margin is not None and (
            not isfinite(self.best_second_best_partition_margin)
            or self.best_second_best_partition_margin < 0.0
        ):
            raise ValueError("partition margin must be finite and nonnegative")


@dataclass(frozen=True)
class RowBand:
    """A row envelope and the raw-mask component indexes assigned to it."""

    bbox: BBox
    component_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))


@dataclass(frozen=True)
class RowSegmentationDiagnostics:
    """Stable row-level ownership, partition, and confidence evidence."""

    row_index: int
    bbox: BBox
    component_indices: tuple[int, ...]
    y_range: tuple[int, int]
    center_y: float
    baseline_y: float
    body_height: float
    inferred_count: int | None
    requested_count: int | None
    selected_boundaries: tuple[int, ...]
    confidence: float
    cost: float
    alternative_cost_margin: float | None
    width_prior_deviation: float | None = None
    body_counts_per_region: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    ambiguous_component_indices: tuple[int, ...] = ()
    anchor_count: int | None = None
    segmentation_strategy: Literal["anchors", "hybrid", "partition"] | None = None
    attempted_counts: tuple[int, ...] = ()
    truncated_counts: tuple[int, ...] = ()
    split_anchor_indices: tuple[int, ...] = ()
    merged_anchor_groups: tuple[tuple[int, ...], ...] = ()
    baseline_cost: float | None = None
    repair_count: int = 0
    split_evidence: tuple[AnchorSplitEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))
        object.__setattr__(self, "selected_boundaries", tuple(self.selected_boundaries))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self, "ambiguous_component_indices", tuple(self.ambiguous_component_indices)
        )
        object.__setattr__(self, "body_counts_per_region", tuple(self.body_counts_per_region))
        object.__setattr__(self, "attempted_counts", tuple(self.attempted_counts))
        object.__setattr__(self, "truncated_counts", tuple(self.truncated_counts))
        object.__setattr__(self, "split_anchor_indices", tuple(self.split_anchor_indices))
        object.__setattr__(self, "split_evidence", tuple(self.split_evidence))
        merged_groups = tuple(tuple(group) for group in self.merged_anchor_groups)
        object.__setattr__(self, "merged_anchor_groups", merged_groups)
        if self.anchor_count is not None and (
            isinstance(self.anchor_count, bool)
            or not isinstance(self.anchor_count, int)
            or self.anchor_count < 0
        ):
            raise ValueError("anchor_count must be a nonnegative integer")
        if self.segmentation_strategy not in {None, "anchors", "hybrid", "partition"}:
            raise ValueError("unsupported segmentation strategy")
        for counts in (self.attempted_counts, self.truncated_counts):
            if any(
                isinstance(count, bool) or not isinstance(count, int) or count < 1
                for count in counts
            ):
                raise ValueError("attempted counts must be positive integers")
        if self.repair_count < 0:
            raise ValueError("repair_count must be nonnegative")
        if self.baseline_cost is not None and (
            not isfinite(self.baseline_cost) or self.baseline_cost < 0.0
        ):
            raise ValueError("baseline_cost must be finite and nonnegative")
        if any(not isinstance(item, AnchorSplitEvidence) for item in self.split_evidence):
            raise ValueError("split_evidence must contain AnchorSplitEvidence values")
        if any(index < 0 for index in self.split_anchor_indices):
            raise ValueError("split anchor indices must be nonnegative")
        if any(len(group) < 2 or any(index < 0 for index in group) for group in merged_groups):
            raise ValueError("merged anchor groups must contain multiple nonnegative indices")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.body_counts_per_region
        ):
            raise ValueError("body counts must be nonnegative integers")
        if self.width_prior_deviation is not None and (
            not isfinite(self.width_prior_deviation) or self.width_prior_deviation < 0.0
        ):
            raise ValueError("width_prior_deviation must be finite and nonnegative")
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise ValueError("row_index must be a nonnegative integer")
        if self.y_range[0] < 0 or self.y_range[1] <= self.y_range[0]:
            raise ValueError("y_range must be a positive half-open interval")
        if not self.bbox.y0 <= self.y_range[0] < self.y_range[1] <= self.bbox.y1:
            raise ValueError("y_range must lie within the row bbox")
        finite_values = (
            self.center_y,
            self.baseline_y,
            self.body_height,
            self.confidence,
            self.cost,
        )
        if not all(isfinite(value) for value in finite_values):
            raise ValueError("row diagnostic values must be finite")
        if self.body_height <= 0.0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "body_height must be positive and confidence must be between zero and one"
            )
        if self.cost < 0.0:
            raise ValueError("row partition cost must be nonnegative")
        if self.alternative_cost_margin is not None and (
            not isfinite(self.alternative_cost_margin) or self.alternative_cost_margin < 0.0
        ):
            raise ValueError("alternative_cost_margin must be finite and nonnegative")
        for count in (self.inferred_count, self.requested_count):
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 1
            ):
                raise ValueError("row counts must be positive integers")


@dataclass(frozen=True)
class SecondaryMarkAssignment:
    """A globally considered secondary component and its assignment margin."""

    component_index: int
    target_region_index: int | None
    best_score: float | None
    second_best_score: float | None
    margin: float | None
    status: Literal["assigned", "ambiguous", "unassigned"]
    candidate_region_indices: tuple[int, ...] = ()
    area_ratio: float | None = None
    horizontal_overlap_ratio: float | None = None
    horizontal_center_distance: float | None = None
    vertical_gap_ratio: float | None = None
    bbox_expansion_ratio: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.component_index, bool)
            or not isinstance(self.component_index, int)
            or self.component_index < 0
        ):
            raise ValueError("component_index must be a nonnegative integer")
        if self.target_region_index is not None and (
            isinstance(self.target_region_index, bool)
            or not isinstance(self.target_region_index, int)
            or self.target_region_index < 0
        ):
            raise ValueError("target_region_index must be a nonnegative integer")
        scores = (self.best_score, self.second_best_score, self.margin)
        if any(value is not None and not isfinite(value) for value in scores):
            raise ValueError("mark assignment scores must be finite")
        metrics = (
            self.area_ratio,
            self.horizontal_overlap_ratio,
            self.horizontal_center_distance,
            self.vertical_gap_ratio,
            self.bbox_expansion_ratio,
        )
        if any(value is not None and not isfinite(value) for value in metrics):
            raise ValueError("mark assignment geometry must be finite")
        if any(
            value is not None and not 0.0 <= value <= 1.0
            for value in (
                self.area_ratio,
                self.horizontal_overlap_ratio,
                self.vertical_gap_ratio,
            )
        ):
            raise ValueError("mark assignment ratios must be between zero and one")
        if self.horizontal_center_distance is not None and self.horizontal_center_distance < 0.0:
            raise ValueError("horizontal center distance must be nonnegative")
        if self.bbox_expansion_ratio is not None and self.bbox_expansion_ratio < 1.0:
            raise ValueError("bbox expansion ratio must be at least one")
        if self.margin is not None and self.margin < 0.0:
            raise ValueError("mark assignment margin must be nonnegative")
        if self.status == "assigned" and self.target_region_index is None:
            raise ValueError("assigned marks must name a target region")
        if self.status != "assigned" and self.target_region_index is not None:
            raise ValueError("unassigned or ambiguous marks cannot name a target region")
        candidate_indices = tuple(self.candidate_region_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in candidate_indices
        ):
            raise ValueError("candidate region indexes must be nonnegative integers")
        object.__setattr__(self, "candidate_region_indices", candidate_indices)


@dataclass(frozen=True)
class Region:
    """An anonymous box, crop, and mask with reading-order metadata."""

    region_id: str
    bbox: BBox
    row_index: int
    column_index: int
    reading_order: int
    confidence: float
    component_indices: tuple[int, ...]

    image: Image.Image = field(repr=False, compare=False)
    mask: np.ndarray = field(repr=False, compare=False)
    primary_component_indices: tuple[int, ...] = ()
    secondary_component_indices: tuple[int, ...] = ()
    ambiguous_component_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))
        object.__setattr__(self, "primary_component_indices", tuple(self.primary_component_indices))
        object.__setattr__(
            self, "secondary_component_indices", tuple(self.secondary_component_indices)
        )
        object.__setattr__(
            self, "ambiguous_component_indices", tuple(self.ambiguous_component_indices)
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("region confidence must be between zero and one")
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != (self.bbox.height, self.bbox.width):
            raise ValueError("region mask dimensions must match bbox")
        if self.image.size != (self.bbox.width, self.bbox.height):
            raise ValueError("region image dimensions must match bbox")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class DetectionResult:
    """Detection output, diagnostics, immutable masks, and independent image snapshot."""

    source_path: Path | None
    image: Image.Image = field(repr=False, compare=False)
    mask_result: MaskResult
    clean_mask: np.ndarray = field(repr=False, compare=False)
    components: tuple[Component, ...]
    rows: tuple[RowBand, ...]
    regions: tuple[Region, ...]
    expected_count: int | None = None
    status: DetectionStatus = "ok"
    warnings: tuple[str, ...] = ()
    component_decisions: tuple[ComponentDecision, ...] = ()
    expected_count_scope: ExpectedCountScope = "row"
    diagnostics: DetectionDiagnostics | None = None
    unassigned_mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "regions", tuple(self.regions))
        clean = np.asarray(self.clean_mask, dtype=bool)
        if clean.shape != self.mask_result.mask.shape:
            raise ValueError("clean mask must match raw mask shape")
        clean = np.array(clean, copy=True)
        clean.setflags(write=False)
        unassigned = (
            np.zeros_like(clean)
            if self.unassigned_mask is None
            else np.asarray(self.unassigned_mask, dtype=bool)
        )
        provided_unassigned_mask = self.unassigned_mask is not None
        if unassigned.shape != clean.shape:
            raise ValueError("unassigned mask must match clean mask shape")
        unassigned = np.array(unassigned, copy=True)
        unassigned.setflags(write=False)
        object.__setattr__(self, "unassigned_mask", unassigned)
        object.__setattr__(self, "clean_mask", clean)
        if (
            provided_unassigned_mask
            and self.diagnostics is not None
            and self.diagnostics.row_count is not None
        ):
            ownership = np.zeros_like(clean)
            for region in self.regions:
                bbox = region.bbox
                if bbox.x1 > clean.shape[1] or bbox.y1 > clean.shape[0]:
                    raise ValueError("region ownership must stay inside cleaned foreground bounds")
                local = ownership[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
                if np.any(local & region.mask):
                    raise ValueError("region ownership masks must be pairwise disjoint")
                local |= region.mask
            if np.any(ownership & unassigned) or not np.array_equal(ownership | unassigned, clean):
                raise ValueError("region and unassigned masks must conserve cleaned foreground")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "component_decisions", tuple(self.component_decisions))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

        if self.diagnostics is not None:
            diagnostics = self.diagnostics
            consistent = (
                diagnostics.image_size == self.image.size
                and diagnostics.mask_threshold == self.mask_result.threshold
                and diagnostics.raw_component_count == len(self.component_decisions)
                and diagnostics.cleaned_component_count == len(self.components)
                and diagnostics.component_decisions == self.component_decisions
                and diagnostics.expected_count == self.expected_count
                and diagnostics.expected_count_scope == self.expected_count_scope
                and diagnostics.mask_method == self.mask_result.method
                and diagnostics.mask_confidence == self.mask_result.confidence
                and diagnostics.foreground_fraction == float(self.mask_result.mask.mean())
            )
            if not consistent:
                raise ValueError("diagnostics must agree with canonical result and mask fields")


@dataclass(frozen=True)
class RegionFrame:
    """Generic row geometry for downstream region normalization."""

    baseline_y: float
    body_height: float
    row_center_y: float
    row_index: int

    def __post_init__(self) -> None:
        values = (self.baseline_y, self.body_height, self.row_center_y)
        if not all(isfinite(value) for value in values) or self.body_height <= 0.0:
            raise ValueError("region frame values must be finite and body_height positive")
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise ValueError("row_index must be a nonnegative integer")


@dataclass(frozen=True)
class RegionHypothesis:
    """One anonymous foreground region within a segmentation hypothesis."""

    bbox: BBox
    mask: np.ndarray = field(repr=False, compare=False)
    component_indices: tuple[int, ...]
    frame: RegionFrame
    cost: float
    confidence: float
    warnings: tuple[str, ...]
    cut_crossing_cost: float = 0.0

    body_count: int = 0
    primary_component_indices: tuple[int, ...] = ()
    secondary_component_indices: tuple[int, ...] = ()
    ambiguous_component_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != (self.bbox.height, self.bbox.width):
            raise ValueError("region hypothesis mask dimensions must match bbox")
        if not all(
            isfinite(value) for value in (self.cost, self.confidence, self.cut_crossing_cost)
        ):
            raise ValueError("region hypothesis scores must be finite")
        if (
            isinstance(self.body_count, bool)
            or not isinstance(self.body_count, int)
            or self.body_count < 0
        ):
            raise ValueError("body_count must be a nonnegative integer")
        if self.cost < 0.0 or self.cut_crossing_cost < 0.0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("region hypothesis costs must be nonnegative and confidence in [0, 1]")
        snapshot = np.array(mask, copy=True)
        snapshot.setflags(write=False)
        object.__setattr__(self, "mask", snapshot)
        object.__setattr__(self, "component_indices", tuple(self.component_indices))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "primary_component_indices", tuple(self.primary_component_indices))
        object.__setattr__(
            self, "secondary_component_indices", tuple(self.secondary_component_indices)
        )
        object.__setattr__(
            self, "ambiguous_component_indices", tuple(self.ambiguous_component_indices)
        )


@dataclass(frozen=True)
class RowPartitionHypothesis:
    """A bounded row partition candidate and its ownership evidence."""

    row_index: int
    count: int
    boundaries: tuple[int, ...]
    regions: tuple[RegionHypothesis, ...]
    cost: float
    unassigned_ink_fraction: float
    secondary_mark_assignments: tuple[SecondaryMarkAssignment, ...]

    boundary_evidence: tuple[PartitionBoundary, ...] = ()
    cut_crossing_cost: float = 0.0
    baseline_cost: float | None = None
    split_evidence: tuple[AnchorSplitEvidence, ...] = ()
    split_anchor_indices: tuple[int, ...] = ()
    repair_count: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise ValueError("row_index must be a nonnegative integer")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("count must be a positive integer")
        boundaries = tuple(self.boundaries)
        regions = tuple(self.regions)
        if (
            len(regions) != self.count
            or len(boundaries) != self.count + 1
            or any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False))
        ):
            raise ValueError("row hypothesis count and boundaries must match its regions")
        if not isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("row hypothesis cost must be finite and nonnegative")
        if not isfinite(self.cut_crossing_cost) or self.cut_crossing_cost < 0.0:
            raise ValueError("cut_crossing_cost must be finite and nonnegative")
        if (
            not isfinite(self.unassigned_ink_fraction)
            or not 0.0 <= self.unassigned_ink_fraction <= 1.0
        ):
            raise ValueError("unassigned_ink_fraction must be between zero and one")
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self, "secondary_mark_assignments", tuple(self.secondary_mark_assignments)
        )
        object.__setattr__(self, "boundary_evidence", tuple(self.boundary_evidence))
        object.__setattr__(self, "split_evidence", tuple(self.split_evidence))
        if self.baseline_cost is not None and (
            not isfinite(self.baseline_cost) or self.baseline_cost < 0.0
        ):
            raise ValueError("baseline_cost must be finite and nonnegative")
        if any(not isinstance(item, AnchorSplitEvidence) for item in self.split_evidence):
            raise ValueError("split_evidence must contain AnchorSplitEvidence values")
        object.__setattr__(self, "split_anchor_indices", tuple(self.split_anchor_indices))
        if self.repair_count < 0 or any(index < 0 for index in self.split_anchor_indices):
            raise ValueError("split anchors and repair count must be nonnegative")


@dataclass(frozen=True)
class PagePartitionHypothesis:
    """One page allocation assembled from row partition hypotheses."""

    row_hypotheses: tuple[RowPartitionHypothesis, ...]
    count: int
    cost: float
    unassigned_ink_fraction: float
    unassigned_mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = tuple(self.row_hypotheses)
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("page hypothesis count must be a positive integer")
        if sum(row.count for row in rows) != self.count:
            raise ValueError("page hypothesis row counts must sum to count")
        if not isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("page hypothesis cost must be finite and nonnegative")
        if (
            not isfinite(self.unassigned_ink_fraction)
            or not 0.0 <= self.unassigned_ink_fraction <= 1.0
        ):
            raise ValueError("unassigned_ink_fraction must be between zero and one")
        object.__setattr__(self, "row_hypotheses", rows)
        if self.unassigned_mask is not None:
            unassigned = np.asarray(self.unassigned_mask, dtype=bool)
            if unassigned.ndim != 2:
                raise ValueError("page unassigned_mask must be two-dimensional")
            unassigned = np.array(unassigned, copy=True)
            unassigned.setflags(write=False)
            object.__setattr__(self, "unassigned_mask", unassigned)

    @property
    def regions(self) -> tuple[RegionHypothesis, ...]:
        return tuple(region for row in self.row_hypotheses for region in row.regions)


@dataclass(frozen=True)
class SpacedSegmentationHypotheses:
    """Selected segmentation and bounded row/page alternatives."""

    selected: DetectionResult
    row_hypotheses: tuple[tuple[RowPartitionHypothesis, ...], ...]
    page_hypotheses: tuple[PagePartitionHypothesis, ...]
    truncated: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        rows = tuple(tuple(hypotheses) for hypotheses in self.row_hypotheses)
        pages = tuple(self.page_hypotheses)
        if self.failure_reason is not None and not self.failure_reason:
            raise ValueError("failure_reason must be nonempty when supplied")
        if pages:
            selected_regions = self.selected.regions
            hypothesis_regions = pages[0].regions
            if len(selected_regions) != len(hypothesis_regions) or any(
                region.bbox != hypothesis.bbox or not np.array_equal(region.mask, hypothesis.mask)
                for region, hypothesis in zip(selected_regions, hypothesis_regions, strict=True)
            ):
                raise ValueError("selected result must match page hypothesis zero")
            for page in pages:
                if page.unassigned_mask is None:
                    raise ValueError("page hypotheses must include an unassigned_mask")
                unassigned = page.unassigned_mask
                if unassigned.shape != self.selected.clean_mask.shape:
                    raise ValueError("page unassigned_mask must match selected foreground shape")
                ownership = np.zeros_like(self.selected.clean_mask)
                for region in page.regions:
                    bbox = region.bbox
                    if bbox.x1 > ownership.shape[1] or bbox.y1 > ownership.shape[0]:
                        raise ValueError("page region hypothesis exceeds foreground bounds")
                    local = ownership[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
                    if np.any(local & region.mask):
                        raise ValueError("page region ownership masks must be pairwise disjoint")
                    local |= region.mask
                if np.any(ownership & unassigned) or not np.array_equal(
                    ownership | unassigned, self.selected.clean_mask
                ):
                    raise ValueError("page hypotheses must conserve cleaned foreground ownership")
        object.__setattr__(self, "row_hypotheses", rows)
        object.__setattr__(self, "page_hypotheses", pages)


class SegmentationSearchLimitError(RuntimeError):
    """Raised when a configured segmentation search budget prevents a safe result."""


@dataclass(frozen=True)
class IsolationCandidate:
    """A component's disposition and reason in an isolation proposal."""

    bbox: BBox
    area: int
    kept: bool
    reason: IsolationReason


@dataclass(frozen=True)
class IsolationResult:
    """A selected component mask and decisions for every candidate component."""

    center: tuple[float, float]
    mask: np.ndarray = field(repr=False, compare=False)
    candidates: tuple[IsolationCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("isolation mask must be 2D")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
