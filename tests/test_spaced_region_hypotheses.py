from __future__ import annotations

import numpy as np

from rasterregions import SpacingConfig
from rasterregions._partition import _partition_row
from rasterregions.components import components
from rasterregions.models import PartitionConfig


def test_row_partitioner_returns_deterministic_bounded_top_k() -> None:
    mask = np.zeros((40, 100), dtype=bool)
    mask[5:35, 10:30] = True
    mask[5:35, 70:90] = True
    found = tuple(components(mask))
    config = SpacingConfig(hypothesis_limit=3)

    first = _partition_row(
        mask,
        2,
        found,
        x_offset=0,
        config=PartitionConfig(),
        spacing_config=config,
    )
    second = _partition_row(
        mask,
        2,
        found,
        x_offset=0,
        config=PartitionConfig(),
        spacing_config=config,
    )

    assert first is not None and second is not None
    assert first.cut_positions == second.cut_positions
    assert [item.cut_positions for item in first.alternatives] == [
        item.cut_positions for item in second.alternatives
    ]
    hypotheses = (first, *first.alternatives)
    assert len(hypotheses) == config.hypothesis_limit
    assert [item.cost for item in hypotheses] == sorted(item.cost for item in hypotheses)
    assert first.truncated
    assert all(len(item.cut_positions) == 3 for item in hypotheses)


def test_auto_count_penalty_is_symmetric_around_body_anchor_count() -> None:
    mask = np.zeros((40, 100), dtype=bool)
    mask[5:35, 10:30] = True
    mask[5:35, 70:90] = True
    found = tuple(components(mask))
    penalty = 8.0

    result = _partition_row(
        mask,
        None,
        found,
        x_offset=0,
        config=PartitionConfig(),
        spacing_config=SpacingConfig(complexity_penalty=penalty, max_region_width_ratio=5.0),
    )

    assert result is not None
    options = (result, *result.alternatives)
    one_region = next(item for item in options if len(item.cut_positions) == 2)
    assert len(result.cut_positions) == 3
    assert one_region.cost >= penalty
    assert result.cost < one_region.cost
