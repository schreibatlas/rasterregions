"""Raster-region detection primitives.

The package deliberately knows nothing about OCR, handwriting, glyph labels, or
application-specific semantics. It detects foreground pixels and anonymous
ordered regions only.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .components import components, isolate_component
from .detect import (
    detect_regions,
    detect_rows,
    detect_spaced_region_hypotheses,
    detect_spaced_regions,
)
from .mask import foreground_evidence, foreground_mask
from .models import (
    BBox,
    CleanupConfig,
    Component,
    ComponentDecision,
    ComponentRole,
    DetectionConfig,
    DetectionDiagnostics,
    DetectionResult,
    DetectionStatus,
    ExpectedCountScope,
    ForegroundEvidence,
    ForegroundPyramid,
    ForegroundPyramidLevel,
    GroupingConfig,
    IsolationCandidate,
    IsolationReason,
    IsolationResult,
    MaskConfig,
    MaskMethod,
    MaskMode,
    MaskResult,
    PagePartitionHypothesis,
    PartitionBoundary,
    PartitionConfig,
    Region,
    RegionFrame,
    RegionHypothesis,
    RowBand,
    RowPartitionHypothesis,
    RowSegmentationDiagnostics,
    SecondaryMarkAssignment,
    SegmentationSearchLimitError,
    SpacedSegmentationHypotheses,
    SpacingConfig,
)
from .pyramid import foreground_pyramid
from .regions import crop, tighten_bbox

try:
    from ._version import __version__
except ImportError:
    try:
        __version__ = version("rasterregions")
    except PackageNotFoundError:  # source checkout before installation
        __version__ = "0+unknown"

__all__ = [
    "BBox",
    "CleanupConfig",
    "Component",
    "ComponentDecision",
    "ComponentRole",
    "DetectionConfig",
    "DetectionDiagnostics",
    "DetectionResult",
    "DetectionStatus",
    "ForegroundEvidence",
    "ForegroundPyramid",
    "ForegroundPyramidLevel",
    "ExpectedCountScope",
    "GroupingConfig",
    "IsolationReason",
    "MaskConfig",
    "IsolationCandidate",
    "IsolationResult",
    "MaskMethod",
    "MaskMode",
    "MaskResult",
    "PagePartitionHypothesis",
    "PartitionBoundary",
    "PartitionConfig",
    "Region",
    "RegionFrame",
    "RegionHypothesis",
    "RowBand",
    "RowPartitionHypothesis",
    "RowSegmentationDiagnostics",
    "SecondaryMarkAssignment",
    "SegmentationSearchLimitError",
    "SpacedSegmentationHypotheses",
    "SpacingConfig",
    "__version__",
    "components",
    "crop",
    "detect_regions",
    "detect_spaced_regions",
    "detect_spaced_region_hypotheses",
    "detect_rows",
    "foreground_pyramid",
    "foreground_evidence",
    "foreground_mask",
    "isolate_component",
    "tighten_bbox",
]
