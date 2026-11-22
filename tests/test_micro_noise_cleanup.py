import numpy as np
import pytest

from rasterregions import CleanupConfig
from rasterregions._cleanup import clean_page_mask


def test_micro_noise_removed_and_dot_and_narrow_fragment_retained() -> None:
    mask = np.zeros((220, 440), dtype=bool)
    for x in (30, 130, 230):
        mask[90:150, x : x + 60] = True
    mask[70:76, 55:61] = True
    mask[60:70, 320] = True
    mask[20, 20] = True
    mask[20, 30:32] = True
    mask[20, 40:43] = True
    mask[20:22, 50:52] = True
    mask[21, 52] = True

    cleaned = clean_page_mask(mask)
    removed = [d for d in cleaned.component_decisions if d.role == "micro-noise"]
    assert sorted(d.area for d in removed) == [1, 2, 3, 5]
    assert all(d.action == "removed" and d.body_area_ratio is not None for d in removed)
    assert cleaned.mask[70:76, 55:61].all()
    dot = next(d for d in cleaned.component_decisions if d.area == 36)
    assert dot.role == "secondary-candidate"
    assert dot.component_index in cleaned.provisional_component_indices
    assert cleaned.mask[60:70, 320].all()
    assert not cleaned.mask[20, 20]


def test_body_scale_ignores_page_rule_and_many_pixel_specks() -> None:
    mask = np.zeros((250, 500), dtype=bool)
    mask[80:140, 40:100] = True
    mask[80:140, 170:230] = True
    mask[80:140, 300:360] = True
    mask[220, :] = True
    for x in range(10, 490, 5):
        mask[20, x] = True
    mask[60:66, 65:71] = True

    cleaned = clean_page_mask(mask)
    assert sum(d.role == "micro-noise" for d in cleaned.component_decisions) == 96
    assert any(d.role == "artifact" for d in cleaned.component_decisions)
    dot = next(d for d in cleaned.component_decisions if d.area == 36)
    assert dot.action == "kept"
    assert dot.role == "secondary-candidate"
    assert dot.body_area_ratio == pytest.approx(0.01)


def test_plausible_dot_is_assigned_and_remote_mark_does_not_expand_region() -> None:
    from PIL import Image

    from rasterregions import detect_spaced_regions

    mask = np.full((200, 300), 255, dtype=np.uint8)
    mask[90:150, 40:100] = 0
    mask[70:76, 65:71] = 0
    mask[20:26, 200:206] = 0
    result = detect_spaced_regions(Image.fromarray(mask), mode="dark")
    dot = next(d for d in result.component_decisions if d.bbox.x0 == 65)
    remote = next(d for d in result.component_decisions if d.bbox.x0 == 200)
    assert dot.role == "secondary"
    assert sum(dot.component_index in r.component_indices for r in result.regions) == 1
    assert remote.action == "kept"
    assert all(remote.component_index not in r.component_indices for r in result.regions)
    assert result.regions[0].bbox.x1 < 200
    from rasterregions.io import detection_payload

    assignments = result.diagnostics.secondary_mark_assignments
    dot_assignment = next(
        item for item in assignments if item.component_index == dot.component_index
    )
    remote_assignment = next(
        item for item in assignments if item.component_index == remote.component_index
    )
    assert dot_assignment.status == "assigned"
    assert dot_assignment.area_ratio is not None
    assert dot_assignment.vertical_gap_ratio is not None
    assert dot_assignment.bbox_expansion_ratio is not None
    assert remote_assignment.status == "unassigned"
    assert remote_assignment.reason is not None
    serialized = detection_payload(result)["spacing"]["secondary_mark_assignments"]
    serialized_dot = next(
        item for item in serialized if item["component_index"] == dot.component_index
    )
    assert serialized_dot["bbox_expansion_ratio"] is not None


@pytest.mark.parametrize(
    "name",
    [
        "micro_noise_max_area_ratio",
        "micro_noise_max_height_ratio",
        "micro_noise_max_width_ratio",
        "micro_noise_mark_max_vertical_gap_ratio",
        "micro_noise_mark_max_horizontal_gap_ratio",
    ],
)
def test_micro_noise_ratios_are_validated(name: str) -> None:
    with pytest.raises(ValueError):
        CleanupConfig(**{name: 1.1})
