from __future__ import annotations

import numpy as np

from rasterregions import components, isolate_component


def test_components_are_8_connected_and_boxed() -> None:
    mask = np.zeros((12, 20), dtype=bool)
    mask[2:5, 3:6] = True
    mask[7:10, 13:18] = True

    found = components(mask)

    assert len(found) == 2
    assert found[0].bbox.as_tuple() == (3, 2, 6, 5)
    assert found[0].area == 9
    assert found[1].bbox.as_tuple() == (13, 7, 18, 10)


def test_isolate_component_selects_nearest_anchor() -> None:
    mask = np.zeros((20, 40), dtype=bool)
    mask[2:8, 2:10] = True
    mask[2:8, 25:35] = True

    result = isolate_component(mask, center=(28.0, 5.0))

    assert result.mask[5, 28]
    assert not result.mask[5, 5]
    assert [item.kept for item in result.candidates] == [False, True]
