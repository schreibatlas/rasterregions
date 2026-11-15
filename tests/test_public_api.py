from __future__ import annotations

import rasterregions


def test_public_api_and_version_exist() -> None:
    assert callable(rasterregions.detect_regions)
    assert callable(rasterregions.foreground_mask)
    assert isinstance(rasterregions.__version__, str)
    assert rasterregions.DetectionConfig().cleanup == rasterregions.CleanupConfig()
    assert rasterregions.DetectionDiagnostics
    assert rasterregions.ComponentDecision
    assert rasterregions.PartitionBoundary
    assert rasterregions.ExpectedCountScope
    assert rasterregions.IsolationReason
