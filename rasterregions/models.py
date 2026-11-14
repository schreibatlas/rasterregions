from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class GroupingConfig:
    body_min_area_ratio: float = 0.18
    body_min_height_ratio: float = 0.25
    row_center_gap_ratio: float = 0.72
    body_join_gap_ratio: float = 0.12
    satellite_max_area_ratio: float = 0.60
    satellite_max_horizontal_gap_ratio: float = 0.40
    satellite_max_vertical_gap_ratio: float = 1.25


@dataclass(frozen=True)
class DetectionConfig:
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)


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


@dataclass(frozen=True)
class Region:
    """An anonymous detected region in reading order."""

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
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("isolation mask must be 2D")
        mask = np.array(mask, copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
