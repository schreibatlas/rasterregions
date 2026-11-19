from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

import rasterregions.mask as mask_module
from rasterregions import foreground_evidence, foreground_mask


def _mode_image(mode: str) -> np.ndarray:
    if mode == "dark":
        image = np.full((48, 48), 180, dtype=np.uint8)
        image[12:36, 22:26] = 20
        return image
    if mode == "adaptive":
        image = np.tile(np.linspace(220, 255, 64, dtype=np.uint8), (48, 1))
        image[12:36, 30:34] = 30
        return image

    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    if mode in {"grayscale", "hybrid", "auto"}:
        image[10:38, 12:16] = 0
    if mode in {"color", "hybrid", "auto"}:
        image[10:38, 30:34] = (20, 70, 220)
    return image


@pytest.mark.parametrize("mode", ["grayscale", "dark", "adaptive", "color", "hybrid", "auto"])
def test_evidence_field_matches_mask_api_and_metadata(mode: str) -> None:
    image = _mode_image(mode)
    options = {"adaptive_window_size": 15} if mode == "adaptive" else {}

    mask_result = foreground_mask(image, mode=mode, **options)
    evidence = foreground_evidence(image, mode=mode, **options)

    assert np.array_equal(evidence.mask, mask_result.mask)
    assert evidence.method == mask_result.method
    assert evidence.threshold == mask_result.threshold
    assert evidence.confidence == mask_result.confidence
    assert evidence.warnings == mask_result.warnings
    assert evidence.decision_field.shape == evidence.mask.shape
    assert evidence.decision_field.dtype == np.float32
    assert np.isfinite(evidence.decision_field).all()
    assert not evidence.mask.flags.writeable
    assert not evidence.decision_field.flags.writeable
    assert np.array_equal(evidence.mask, evidence.decision_field >= 0.0)
    assert not hasattr(mask_result, "decision_field")


def test_grayscale_evidence_uses_unquantized_luminance_and_zero_is_foreground() -> None:
    image = np.full((40, 40), 255, dtype=np.uint8)
    image[10:30, 18:22] = 0

    evidence = foreground_evidence(image, mode="grayscale")

    assert evidence.threshold == 0.0
    assert evidence.decision_field[10, 18] == 0.0
    assert evidence.mask[10, 18]


def test_color_field_respects_saturation_and_exclusive_value_constraints() -> None:
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    image[0, 0] = (0, 0, 220)
    image[0, 1] = (0, 0, 250)
    image[0, 2] = (160, 200, 160)

    evidence = foreground_evidence(image, mode="color")

    assert evidence.mask[0, 0]
    assert evidence.decision_field[0, 0] > 0.0
    assert not evidence.mask[0, 1]
    assert evidence.decision_field[0, 1] < 0.0
    assert evidence.mask[0, 2]
    assert evidence.decision_field[0, 2] >= 0.0


def test_mask_only_api_does_not_construct_a_decision_field(monkeypatch) -> None:
    image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(image).rectangle((12, 12, 19, 19), fill="black")

    def fail_if_called(*args, **kwargs):
        pytest.fail("foreground_mask must not construct a decision field")

    monkeypatch.setattr(mask_module, "_decision_field", fail_if_called)

    result = foreground_mask(image, mode="dark")

    assert result.mask.any()


def _bilinear_sample(field: np.ndarray, x: float, y: float) -> float:
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = x - x0
    dy = y - y0
    return float(
        field[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + field[y0, x0 + 1] * dx * (1.0 - dy)
        + field[y0 + 1, x0] * (1.0 - dx) * dy
        + field[y0 + 1, x0 + 1] * dx * dy
    )


def _crossing_distance(values: np.ndarray, distances: np.ndarray) -> float:
    crossings = np.flatnonzero((values[:-1] >= 0.0) & (values[1:] < 0.0))
    assert crossings.size
    index = int(crossings[0])
    fraction = values[index] / (values[index] - values[index + 1])
    return float(distances[index] + fraction * (distances[index + 1] - distances[index]))


def test_antialiased_grayscale_edges_localize_fractional_threshold_crossings(monkeypatch) -> None:
    monkeypatch.setattr(mask_module, "_otsu_threshold_u8", lambda _: 127)
    height = width = 96
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    cases = [(0.0, 8.25, 48.0), (0.0, 8.75, 48.0)]
    for angle in (17.0, 43.0, 71.0):
        radians = np.deg2rad(angle)
        normal_x = float(np.cos(radians))
        normal_y = float(np.sin(radians))
        cases.extend(
            (angle, 48.25 + offset * normal_x, 48.25 + offset * normal_y)
            for offset in np.arange(0.0, 1.0, 0.1)
        )

    distances = np.linspace(-0.75, 0.75, 121)
    expected = (127.0 - 127.5) / 255.0
    field_errors = []
    boolean_errors = []
    for angle, center_x, center_y in cases:
        radians = np.deg2rad(angle)
        normal_x = float(np.cos(radians))
        normal_y = float(np.sin(radians))
        signed_distance = (xx - center_x) * normal_x + (yy - center_y) * normal_y
        image = np.clip(127.5 + 255.0 * signed_distance, 0.0, 255.0).astype(np.float32)
        evidence = foreground_evidence(image, mode="grayscale")
        sample_x = center_x + distances * normal_x
        sample_y = center_y + distances * normal_y
        continuous_values = np.array(
            [
                _bilinear_sample(evidence.decision_field, x, y)
                for x, y in zip(sample_x, sample_y, strict=True)
            ]
        )
        boolean_field = np.where(evidence.mask, 1.0, -1.0)
        boolean_values = np.array(
            [_bilinear_sample(boolean_field, x, y) for x, y in zip(sample_x, sample_y, strict=True)]
        )
        field_errors.append(abs(_crossing_distance(continuous_values, distances) - expected))
        boolean_errors.append(abs(_crossing_distance(boolean_values, distances) - expected))

    assert np.median(field_errors) <= 0.10
    assert np.percentile(field_errors, 95) <= 0.20
    assert np.median(field_errors) < np.median(boolean_errors) * 0.5


def test_adaptive_field_tracks_smooth_local_thresholds() -> None:
    width = 100
    background = np.tile(np.linspace(180.0, 240.0, width, dtype=np.float32), (64, 1))
    background[20:44, 18:22] = 150.0
    background[20:44, 76:80] = 150.0

    evidence = foreground_evidence(
        background, mode="adaptive", adaptive_window_size=31, adaptive_offset=10.0
    )

    assert evidence.mask[30, 19]
    assert evidence.mask[30, 77]
    assert not evidence.mask[5, 10]
    assert evidence.decision_field[30, 77] > evidence.decision_field[30, 19]
    assert np.any(evidence.decision_field != np.rint(evidence.decision_field))
    assert np.array_equal(evidence.mask, evidence.decision_field >= 0.0)


def test_hybrid_field_unions_colored_and_neutral_dark_evidence() -> None:
    image = np.full((48, 64, 3), 255, dtype=np.uint8)
    image[12:36, 12:16] = 100
    image[12:36, 40:44] = (20, 70, 220)

    evidence = foreground_evidence(image, mode="hybrid")

    assert evidence.mask[20, 13]
    assert evidence.mask[20, 41]
    assert not evidence.mask[0, 0]
    assert evidence.decision_field[20, 13] >= 0.0
    assert evidence.decision_field[20, 41] >= 0.0
    assert evidence.decision_field[0, 0] < 0.0
