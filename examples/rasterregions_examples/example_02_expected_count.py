"""Partition one row into an exact expected number of anonymous regions."""

from pathlib import Path

from rasterregions import detect_regions
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
source = ROOT / "assets" / "02_expected_count_three.png"
out = ROOT / "outputs" / "02_expected_count"

result = detect_regions(source, expected_count=3)
manifest = write_detection(result, out)

assert result.status == "ok"
assert len(result.regions) == 3
print("bboxes:", [r.bbox.as_tuple() for r in result.regions])
print("confidences:", [round(r.confidence, 3) for r in result.regions])
print("manifest:", manifest)
