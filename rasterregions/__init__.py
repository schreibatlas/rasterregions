"""Raster-region detection primitives.

The package deliberately knows nothing about OCR, handwriting, glyph labels, or
application-specific semantics. It detects foreground pixels and anonymous
ordered regions only.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .components import components, isolate_component
from .detect import detect_regions, detect_rows
from .mask import foreground_mask
from .models import (
    BBox,
    CleanupConfig,
    Component,
    ComponentDecision,
    DetectionConfig,
    DetectionDiagnostics,
    DetectionResult,
    DetectionStatus,
    ExpectedCountScope,
    GroupingConfig,
    IsolationCandidate,
    IsolationReason,
    IsolationResult,
    MaskConfig,
    MaskMethod,
    MaskMode,
    MaskResult,
    PartitionBoundary,
    PartitionConfig,
    Region,
    RowBand,
)
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
    "DetectionConfig",
    "DetectionDiagnostics",
    "DetectionResult",
    "DetectionStatus",
    "ExpectedCountScope",
    "GroupingConfig",
    "IsolationReason",
    "MaskConfig",
    "IsolationCandidate",
    "IsolationResult",
    "MaskMethod",
    "MaskMode",
    "MaskResult",
    "PartitionBoundary",
    "PartitionConfig",
    "Region",
    "RowBand",
    "__version__",
    "components",
    "crop",
    "detect_regions",
    "detect_rows",
    "foreground_mask",
    "isolate_component",
    "tighten_bbox",
]
