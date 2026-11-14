"""Label connected components and isolate the component nearest an anchor."""

from pathlib import Path

import numpy as np
from PIL import Image

from rasterregions import components, foreground_mask, isolate_component

ROOT = Path(__file__).resolve().parent
source = ROOT / "assets" / "05_component_isolation.png"
out = ROOT / "outputs" / "05_component_isolation"
out.mkdir(parents=True, exist_ok=True)

image = Image.open(source)
mask_result = foreground_mask(image, mode="dark")
found = components(mask_result.mask)
assert len(found) == 4
print("components:", [(c.index, c.bbox.as_tuple(), c.area) for c in found])

isolated = isolate_component(
    mask_result.mask,
    center=(355.0, 95.0),
    competing_centers=((70.0, 90.0), (210.0, 100.0)),
)
assert sum(candidate.kept for candidate in isolated.candidates) == 1
Image.fromarray(np.asarray(isolated.mask, dtype=np.uint8) * 255).save(out / "isolated_mask.png")
print("candidates:", [(c.bbox.as_tuple(), c.kept, c.reason) for c in isolated.candidates])
