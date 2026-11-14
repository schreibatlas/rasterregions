import numpy as np
import pytest
from PIL import Image

from rasterregions.models import (
    BBox,
    Component,
    ComponentDecision,
    DetectionDiagnostics,
    DetectionResult,
    IsolationCandidate,
    IsolationResult,
    MaskResult,
    Region,
    RowBand,
)


def _component() -> Component:
    return Component(
        index=2,
        bbox=BBox(1, 1, 3, 3),
        area=4,
        ys=np.array([1, 1, 2, 2]),
        xs=np.array([1, 2, 1, 2]),
    )


def _detection(diagnostics: DetectionDiagnostics | None = None) -> DetectionResult:
    mask = np.zeros((10, 10), dtype=bool)
    component = _component()
    decision = ComponentDecision(2, component.bbox, component.area, "kept")
    region = Region(
        "region-000",
        component.bbox,
        0,
        0,
        0,
        0.8,
        [2],
        Image.new("RGB", (2, 2)),
        np.ones((2, 2), dtype=bool),
    )
    return DetectionResult(
        source_path=None,
        image=Image.new("RGB", (10, 10)),
        mask_result=MaskResult(mask, 0.8, "dark"),
        clean_mask=mask,
        components=[component],
        rows=[RowBand(BBox(1, 1, 3, 3), [2])],
        regions=[region],
        warnings=["warning"],
        component_decisions=[decision],
        diagnostics=diagnostics,
    )


def test_component_coordinate_arrays_are_copied_and_read_only() -> None:
    ys = np.array([1, 1, 2, 2])
    xs = np.array([1, 2, 1, 2])
    component = Component(2, BBox(1, 1, 3, 3), 4, ys, xs)

    assert not component.ys.flags.writeable
    assert not component.xs.flags.writeable
    ys[0] = 9
    assert component.ys[0] == 1
    with pytest.raises(ValueError):
        component.ys[0] = 9


def test_component_validates_coordinate_area_consistency() -> None:
    with pytest.raises(ValueError, match="coordinate"):
        Component(0, BBox(1, 1, 3, 3), 3, np.array([1, 1]), np.array([1, 2]))


def test_frozen_models_normalize_sequence_fields() -> None:
    decision = ComponentDecision(2, BBox(1, 1, 3, 3), 4, "kept")
    diagnostics = DetectionDiagnostics(
        image_size=(10, 10),
        mask_threshold=None,
        raw_component_count=1,
        cleaned_component_count=1,
        component_decisions=[decision],
        mask_method="dark",
        mask_confidence=0.8,
        foreground_fraction=0.0,
    )
    result = _detection(diagnostics)
    isolation = IsolationResult(
        center=(1.0, 1.0),
        mask=np.zeros((10, 10), dtype=bool),
        candidates=[IsolationCandidate(BBox(1, 1, 3, 3), 4, True, "nearest")],
    )

    assert isinstance(result.components, tuple)
    assert isinstance(result.rows, tuple)
    assert isinstance(result.regions, tuple)
    assert isinstance(result.warnings, tuple)
    assert isinstance(result.component_decisions, tuple)
    assert isinstance(result.rows[0].component_indices, tuple)
    assert isinstance(result.regions[0].component_indices, tuple)
    assert isinstance(diagnostics.component_decisions, tuple)
    assert isinstance(isolation.candidates, tuple)


def test_detection_result_rejects_diagnostic_fact_drift() -> None:
    decision = ComponentDecision(2, BBox(1, 1, 3, 3), 4, "kept")
    diagnostics = DetectionDiagnostics(
        image_size=(10, 10),
        mask_threshold=None,
        raw_component_count=1,
        cleaned_component_count=1,
        component_decisions=[decision],
        expected_count=1,
        mask_method="dark",
        mask_confidence=0.8,
        foreground_fraction=0.0,
    )

    with pytest.raises(ValueError, match="diagnostics"):
        _detection(diagnostics)
