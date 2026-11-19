import numpy as np
import pytest
from PIL import Image

from rasterregions.models import (
    BBox,
    Component,
    ComponentDecision,
    DetectionDiagnostics,
    DetectionResult,
    ForegroundEvidence,
    IsolationCandidate,
    IsolationResult,
    MaskResult,
    Region,
    RowBand,
    RowSegmentationDiagnostics,
    SecondaryMarkAssignment,
    SpacingConfig,
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
        candidates=[IsolationCandidate(BBox(1, 1, 3, 3), 4, True, "nearest-to-center")],
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


def test_spacing_diagnostic_models_validate_and_keep_immutable_sequences() -> None:
    config = SpacingConfig(width_prior_ratio=1.25, max_inferred_count=12)
    row = RowSegmentationDiagnostics(
        row_index=0,
        bbox=BBox(2, 3, 22, 30),
        component_indices=[1, 2],
        y_range=(3, 30),
        center_y=15.0,
        baseline_y=28.0,
        body_height=20.0,
        inferred_count=2,
        requested_count=2,
        selected_boundaries=[12],
        confidence=0.8,
        cost=0.2,
        anchor_count=2,
        segmentation_strategy="hybrid",
        attempted_counts=[3],
        truncated_counts=[4],
        split_anchor_indices=[1],
        merged_anchor_groups=[[0, 1]],
        alternative_cost_margin=0.3,
        warnings=["one valley is weak"],
    )
    assignment = SecondaryMarkAssignment(3, None, 0.2, 0.18, 0.02, "ambiguous")

    assert config.width_prior_ratio == 1.25
    assert row.component_indices == (1, 2)
    assert row.selected_boundaries == (12,)
    assert row.warnings == ("one valley is weak",)
    assert row.anchor_count == 2
    assert row.segmentation_strategy == "hybrid"
    assert row.attempted_counts == (3,)
    assert row.truncated_counts == (4,)
    assert row.split_anchor_indices == (1,)
    assert row.merged_anchor_groups == ((0, 1),)
    assert assignment.status == "ambiguous"
    with pytest.raises(ValueError, match="positive integer"):
        SpacingConfig(max_inferred_count=True)
    with pytest.raises(ValueError, match="positive integers"):
        RowSegmentationDiagnostics(
            row_index=0,
            bbox=BBox(2, 3, 22, 30),
            component_indices=(),
            y_range=(3, 30),
            center_y=15.0,
            baseline_y=28.0,
            body_height=20.0,
            inferred_count=0,
            requested_count=None,
            selected_boundaries=(),
            confidence=0.8,
            cost=0.2,
            alternative_cost_margin=None,
        )


def test_detection_result_unassigned_mask_is_a_read_only_snapshot() -> None:
    from dataclasses import replace

    unassigned = np.zeros((10, 10), dtype=bool)
    unassigned[7, 8] = True
    result = replace(_detection(), unassigned_mask=unassigned)

    assert result.unassigned_mask is not None
    assert result.unassigned_mask[7, 8]
    assert not result.unassigned_mask.flags.writeable
    unassigned[7, 8] = False
    assert result.unassigned_mask[7, 8]


def test_foreground_evidence_owns_immutable_consistent_snapshots() -> None:
    mask = np.array([[True, False], [False, True]])
    decision = np.array([[0.2, -0.3], [-0.1, 0.4]], dtype=np.float64)
    evidence = ForegroundEvidence(mask, decision, 0.75, "dark")

    assert evidence.mask.dtype == np.bool_
    assert evidence.decision_field.dtype == np.float32
    assert np.array_equal(evidence.mask, evidence.decision_field >= 0.0)
    assert not evidence.mask.flags.writeable
    assert not evidence.decision_field.flags.writeable
    mask[0, 0] = False
    decision[0, 0] = -1.0
    assert evidence.mask[0, 0]
    assert evidence.decision_field[0, 0] == np.float32(0.2)
    with pytest.raises(ValueError):
        evidence.mask[0, 0] = False
    with pytest.raises(ValueError):
        evidence.decision_field[0, 0] = -1.0


def test_foreground_evidence_rejects_invalid_shapes_values_and_signs() -> None:
    mask = np.array([[True, False]])
    decision = np.array([[1.0, -1.0]])

    with pytest.raises(ValueError, match="same shape"):
        ForegroundEvidence(mask, np.zeros((2, 1)), 0.5, "dark")
    with pytest.raises(ValueError, match="finite"):
        ForegroundEvidence(mask, np.array([[np.inf, -1.0]]), 0.5, "dark")
    with pytest.raises(ValueError, match="decision_field >= 0"):
        ForegroundEvidence(mask, np.array([[-1.0, -1.0]]), 0.5, "dark")
    with pytest.raises(ValueError, match="confidence"):
        ForegroundEvidence(mask, decision, 1.1, "dark")


def test_foreground_evidence_zero_is_foreground() -> None:
    evidence = ForegroundEvidence(
        np.array([[True, False]]), np.array([[0.0, -0.5]]), 1.0, "grayscale"
    )

    assert evidence.mask[0, 0]
    assert evidence.decision_field[0, 0] == 0.0
