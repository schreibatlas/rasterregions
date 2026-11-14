"""Force dark-ink segmentation for non-white / shaded paper."""

from pathlib import Path

from rasterregions import detect_regions
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
source = ROOT / "assets" / "04_dark_ink_gray_paper.png"
out = ROOT / "outputs" / "04_dark_paper"

result = detect_regions(source, mode="dark")
assert result.status == "ok"
assert len(result.regions) == 3
write_detection(result, out)
print("status:", result.status)
print("mask confidence:", round(result.mask_result.confidence, 3))
print("regions:", [r.bbox.as_tuple() for r in result.regions])
