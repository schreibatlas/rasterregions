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
    component_index: int
    bbox: BBox
    area: int
    action: Literal["kept", "removed"]
    reason: str | None = None


@dataclass(frozen=True)
class PartitionBoundary:
    x: int
    projection: float
    crossing_components: int
    confidence: float


@dataclass(frozen=True)
class CleanupConfig:
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
class DetectionConfig:
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)


@dataclass(frozen=True)
class DetectionDiagnostics:
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_decisions", tuple(self.component_decisions))
        object.__setattr__(self, "partition_boundaries", tuple(self.partition_boundaries))


@dataclass(frozen=True)
class RowBand:
    bbox: BBox
    component_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_indices", tuple(self.component_indices))


@dataclass(frozen=True)
class Region:
    """An anonymous region with an independent mutable PIL image snapshot."""

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
    """Detection result whose PIL image is an independent mutable snapshot."""

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "regions", tuple(self.regions))
        clean = np.asarray(self.clean_mask, dtype=bool)
        if clean.shape != self.mask_result.mask.shape:
            raise ValueError("clean mask must match raw mask shape")
        clean = np.array(clean, copy=True)
        clean.setflags(write=False)
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
    bbox: BBox
    area: int
    kept: bool
    reason: str


@dataclass(frozen=True)
class IsolationResult:
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
