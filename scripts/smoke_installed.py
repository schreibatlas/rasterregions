from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageDraw

import rasterregions
from rasterregions import detect_regions, foreground_mask

assert rasterregions.__version__ == version("rasterregions")
assert Path(rasterregions.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())

image = Image.new("RGB", (120, 60), "white")
draw = ImageDraw.Draw(image)
draw.rectangle((15, 10, 35, 40), fill="black")
draw.rectangle((75, 10, 95, 40), fill="black")
mask = foreground_mask(image, mode="dark")
result = detect_regions(image, expected_count=2)
assert mask.mask[20, 20]
assert result.status == "ok" and len(result.regions) == 2
print(f"rasterregions {rasterregions.__version__}: {rasterregions.__file__}")
