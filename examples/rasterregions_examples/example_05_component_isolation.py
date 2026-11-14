"""Label connected components and isolate the component nearest an anchor."""

from pathlib import Path

import numpy as np
from PIL import Image

from rasterregions import components, foreground_mask, isolate_component

ROOT = Path(__file__).resolve().parent
BUILD = ROOT.parents[1] / "build" / "rasterregions_examples"


def main() -> None:
    source = ROOT / "assets" / "05_component_isolation.png"
    output = BUILD / "05_component_isolation"
    output.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        mask_result = foreground_mask(image, mode="dark")
    found = components(mask_result.mask)
    assert len(found) == 4
    print(
        "components:",
        [(component.index, component.bbox.as_tuple(), component.area) for component in found],
    )

    isolated = isolate_component(
        mask_result.mask,
        center=(355.0, 95.0),
        competing_centers=((70.0, 90.0), (210.0, 100.0)),
    )
    assert sum(candidate.kept for candidate in isolated.candidates) == 1
    Image.fromarray(np.asarray(isolated.mask, dtype=np.uint8) * 255).save(
        output / "isolated_mask.png"
    )
    print(
        "candidates:",
        [
            (candidate.bbox.as_tuple(), candidate.kept, candidate.reason)
            for candidate in isolated.candidates
        ],
    )


if __name__ == "__main__":
    main()
