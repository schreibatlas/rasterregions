"""Detect saturated colored marks while ignoring neutral gray worksheet guides."""

from pathlib import Path

from PIL import Image

from rasterregions import detect_regions, foreground_mask
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
source = ROOT / "assets" / "03_color_on_grid.png"
out = ROOT / "outputs" / "03_color_mask"
out.mkdir(parents=True, exist_ok=True)

image = Image.open(source)
mask = foreground_mask(image, mode="auto")
print("selected mask method:", mask.method, "confidence:", round(mask.confidence, 3))
assert mask.method == "color"

result = detect_regions(source, mode="auto")
assert result.status == "ok"
assert len(result.regions) == 3
assert not result.clean_mask[40, 100]
write_detection(result, out)
print("detected regions:", len(result.regions))
