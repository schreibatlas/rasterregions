from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from rasterregions import DetectionConfig, SpacingConfig, detect_spaced_region_hypotheses


def test_equally_plausible_dot_owners_are_preserved_with_unassigned_branch() -> None:
    image = Image.new("RGB", (130, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 25, 38, 55), fill="black")
    draw.rectangle((75, 25, 98, 55), fill="black")
    draw.ellipse((52, 12, 55, 15), fill="black")
    config = DetectionConfig(
        spacing=SpacingConfig(mark_assignment_margin=0.99, hypothesis_limit=12)
    )

    hypotheses = detect_spaced_region_hypotheses(
        image,
        mode="dark",
        row_counts=[2],
        config=config,
    )

    assignments = [
        assignment
        for page in hypotheses.page_hypotheses
        for row in page.row_hypotheses
        for assignment in row.secondary_mark_assignments
    ]
    assert hypotheses.selected.diagnostics.secondary_mark_assignments[0].status == "ambiguous"
    assert {assignment.target_region_index for assignment in assignments} >= {None, 0, 1}
    for page in hypotheses.page_hypotheses:
        ownership = np.zeros(hypotheses.selected.clean_mask.shape, dtype=bool)
        for region in page.regions:
            local = ownership[region.bbox.y0 : region.bbox.y1, region.bbox.x0 : region.bbox.x1]
            assert not np.any(local & region.mask)
            local |= region.mask
        assert np.array_equal(ownership | page.unassigned_mask, hypotheses.selected.clean_mask)
