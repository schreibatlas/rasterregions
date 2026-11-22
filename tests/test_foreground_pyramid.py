from __future__ import annotations

import numpy as np
import pytest

from rasterregions import (
    ForegroundEvidence,
    ForegroundPyramid,
    ForegroundPyramidLevel,
    components,
    foreground_pyramid,
)

# BOX area resampling should preserve source-equivalent mass within this relative tolerance.
MASS_RELATIVE_TOLERANCE = 0.05


def _evidence(mask: np.ndarray) -> ForegroundEvidence:
    return ForegroundEvidence(
        mask,
        np.where(mask, 1.0, -1.0).astype(np.float32),
        1.0,
        "dark",
    )


def _synthetic_mask(name: str) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    if name == "rectangle":
        mask[16:48, 20:44] = True
    elif name == "diagonal":
        mask[np.arange(5, 59), np.arange(5, 59)] = True
    elif name == "dot":
        mask[7, 9] = True
    elif name == "separated":
        mask[12:20, 12:20] = True
        mask[40:48, 40:48] = True
    elif name == "ring":
        mask[16:48, 16:48] = True
        mask[22:42, 22:42] = False
    else:
        raise AssertionError(f"unknown synthetic mask: {name}")
    return mask


@pytest.mark.parametrize("name", ["rectangle", "diagonal", "dot", "separated", "ring"])
def test_foreground_pyramid_preserves_occupancy_mass_and_source(name: str) -> None:
    evidence = _evidence(_synthetic_mask(name))
    pyramid = foreground_pyramid(evidence, scales=(0.125, 0.25, 0.5))

    assert pyramid.evidence is evidence
    assert [level.shape for level in pyramid.levels] == [(8, 8), (16, 16), (32, 32), (64, 64)]
    for level in pyramid.levels[:-1]:
        assert np.isfinite(level.coverage).all()
        assert np.all((level.coverage >= 0.0) & (level.coverage <= 1.0))
        assert np.any(level.coverage > 0.0)
        assert level.normalized_ink_mass == pytest.approx(
            float(evidence.mask.sum()), rel=MASS_RELATIVE_TOLERANCE
        )

    finest = pyramid.levels[-1]
    assert finest.shape == evidence.mask.shape
    assert finest.coverage.dtype == np.float32
    assert np.array_equal(finest.coverage, evidence.mask.astype(np.float32))
    assert np.array_equal(finest.mask, evidence.mask)
    assert len(components(pyramid.evidence.mask)) == len(components(evidence.mask))


def test_detached_dot_survives_as_fractional_occupancy_below_mask_threshold() -> None:
    evidence = _evidence(_synthetic_mask("dot"))
    pyramid = foreground_pyramid(evidence, scales=(0.125, 1.0))

    coarse = pyramid.levels[0]
    assert 0.0 < float(coarse.coverage.sum()) < 0.5
    assert not coarse.mask.any()
    assert len(components(evidence.mask)) == 1
    assert np.array_equal(pyramid.levels[-1].mask, evidence.mask)


def test_pyramid_scales_are_normalized_and_duplicate_dimensions_removed() -> None:
    evidence = _evidence(np.array([[True, False]], dtype=bool))
    pyramid = foreground_pyramid(evidence, scales=(0.2, 0.4, 0.6))

    assert [level.shape for level in pyramid.levels] == [(1, 1), (1, 2)]
    assert [level.index for level in pyramid.levels] == [0, 1]
    assert pyramid.levels[-1].scale_x == 1.0
    assert pyramid.levels[-1].scale_y == 1.0
    assert np.array_equal(pyramid.levels[-1].coverage, evidence.mask.astype(np.float32))


def test_pyramid_levels_are_coarse_to_fine_and_always_include_source_resolution() -> None:
    evidence = _evidence(np.eye(17, 23, dtype=bool))
    pyramid = foreground_pyramid(evidence, scales=(0.25, 0.5))

    assert [level.shape for level in pyramid.levels] == [(4, 6), (8, 12), (17, 23)]
    assert all(
        left.scale_x * left.scale_y < right.scale_x * right.scale_y
        for left, right in zip(pyramid.levels, pyramid.levels[1:], strict=False)
    )
    assert pyramid.levels[-1].source_shape == evidence.mask.shape
    assert pyramid.levels[-1].shape == evidence.mask.shape


def test_pyramid_coverage_is_an_immutable_snapshot() -> None:
    coverage = np.array([[0.25, 1.0]], dtype=np.float64)
    level = ForegroundPyramidLevel(
        index=0,
        source_shape=(1, 2),
        shape=(1, 2),
        scale_x=1.0,
        scale_y=1.0,
        coverage=coverage,
    )

    assert level.coverage.dtype == np.float32
    assert not level.coverage.flags.writeable
    assert not level.mask.flags.writeable
    coverage[0, 0] = 0.0
    assert level.coverage[0, 0] == np.float32(0.25)
    with pytest.raises(ValueError):
        level.coverage[0, 0] = 0.0


def test_pyramid_rejects_empty_or_invalid_scales() -> None:
    evidence = _evidence(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="not be empty"):
        foreground_pyramid(evidence, scales=())


@pytest.mark.parametrize(
    "scales",
    [
        (0.0,),
        (-0.1,),
        (1.1,),
        (np.nan,),
        (np.inf,),
        (0.5, 0.4),
        (0.5, 0.5),
    ],
)
def test_pyramid_rejects_invalid_scales(scales: tuple[float, ...]) -> None:
    evidence = _evidence(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError):
        foreground_pyramid(evidence, scales=scales)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_pyramid_rejects_invalid_mask_threshold(threshold: float) -> None:
    evidence = _evidence(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="mask_threshold"):
        foreground_pyramid(evidence, mask_threshold=threshold)


@pytest.mark.parametrize(
    ("coverage", "message"),
    [
        (np.array([[np.nan, 0.0]], dtype=np.float32), "finite"),
        (np.array([[0.0, 1.1]], dtype=np.float32), "between zero and one"),
    ],
)
def test_pyramid_level_rejects_invalid_coverage(coverage: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ForegroundPyramidLevel(
            index=0,
            source_shape=(1, 2),
            shape=(1, 2),
            scale_x=1.0,
            scale_y=1.0,
            coverage=coverage,
        )


def test_foreground_pyramid_container_requires_exact_finest_level() -> None:
    evidence = _evidence(np.array([[True, False]], dtype=bool))
    coarse = ForegroundPyramidLevel(
        index=0,
        source_shape=(1, 2),
        shape=(1, 1),
        scale_x=0.5,
        scale_y=1.0,
        coverage=np.array([[0.5]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="final pyramid level"):
        ForegroundPyramid(evidence, (coarse,))


def test_repeated_pyramid_construction_is_deterministic() -> None:
    evidence = _evidence(_synthetic_mask("ring"))
    first = foreground_pyramid(evidence, scales=(0.17, 0.42, 0.8))
    second = foreground_pyramid(evidence, scales=(0.17, 0.42, 0.8))

    assert [level.shape for level in first.levels] == [level.shape for level in second.levels]
    for left, right in zip(first.levels, second.levels, strict=True):
        assert left.coverage.dtype == right.coverage.dtype == np.float32
        assert np.array_equal(left.coverage, right.coverage)
