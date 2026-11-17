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
        object.__setattr__(self, "warnings", tuple(self.warnings))


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


@dataclass(frozen=True)
class PartitionBoundary:
    """A candidate partition cut with projection, crossings, and heuristic confidence."""

    x: int
    projection: float
    crossing_components: int
    confidence: float
    valley_strength: float | None = None
    component_indices: tuple[int, ...] = ()
    selected: bool = True
    kind: Literal["zero-run", "valley", "local-minimum", "component-midpoint"] = "valley"

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))


@dataclass(frozen=True)
class CleanupConfig:
    """Thresholds controlling border, rule, and tiny-component cleanup."""

    tiny_area_ratio: float = 0.008
    rule_aspect_ratio: float = 12.0
    rule_min_span_ratio: float = 0.20
    border_span_ratio: float = 0.80
    border_bbox_ratio: float = 0.20
    border_fill_ratio: float = 0.15
    max_component_area_ratio: float = 0.50

    def __post_init__(self) -> None:
        values = (
            self.tiny_area_ratio,
            self.rule_aspect_ratio,
            self.rule_min_span_ratio,
            self.border_span_ratio,
            self.border_bbox_ratio,
            self.border_fill_ratio,
            self.max_component_area_ratio,
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
    ambiguity_margin: float = 0.10
    mark_assignment_margin: float = 0.15
    row_assignment_margin: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.padding_ratio,
            self.min_region_width_ratio,
            self.max_region_width_ratio,
            self.complexity_penalty,
            self.ambiguity_margin,
            self.mark_assignment_margin,
            self.row_assignment_margin,
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
        if self.complexity_penalty < 0.0:
            raise ValueError("complexity_penalty must be nonnegative")
        margins = (self.ambiguity_margin, self.mark_assignment_margin, self.row_assignment_margin)
        if any(not 0.0 <= margin <= 1.0 for margin in margins):
            raise ValueError("spacing confidence margins must be between zero and one")


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

    row_segmentation: tuple[RowSegmentationDiagnostics, ...] = ()
    secondary_mark_assignments: tuple[SecondaryMarkAssignment, ...] = ()
    selected_region_count: int | None = None
    row_count: int | None = None
    best_second_best_partition_margin: float | None = None
    width_prior_deviation: float | None = None
    body_counts_per_region: tuple[int, ...] = ()
    unassigned_ink_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_decisions", tuple(self.component_decisions))
        object.__setattr__(self, "partition_boundaries", tuple(self.partition_boundaries))
        object.__setattr__(self, "row_segmentation", tuple(self.row_segmentation))
        object.__setattr__(
            self, "secondary_mark_assignments", tuple(self.secondary_mark_assignments)
        )
        object.__setattr__(self, "body_counts_per_region", tuple(self.body_counts_per_region))
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))
        object.__setattr__(self, "selected_boundaries", tuple(self.selected_boundaries))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self, "ambiguous_component_indices", tuple(self.ambiguous_component_indices)
        )
        object.__setattr__(self, "body_counts_per_region", tuple(self.body_counts_per_region))
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
        if self.margin is not None and self.margin < 0.0:
            raise ValueError("mark assignment margin must be nonnegative")
        if self.status == "assigned" and self.target_region_index is None:
            raise ValueError("assigned marks must name a target region")
        if self.status != "assigned" and self.target_region_index is not None:
            raise ValueError("unassigned or ambiguous marks cannot name a target region")


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))
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
        if unassigned.shape != clean.shape:
            raise ValueError("unassigned mask must match clean mask shape")
        unassigned = np.array(unassigned, copy=True)
        unassigned.setflags(write=False)
        object.__setattr__(self, "unassigned_mask", unassigned)
        object.__setattr__(self, "clean_mask", clean)
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
