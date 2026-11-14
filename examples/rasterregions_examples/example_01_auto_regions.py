"""Automatic multi-row detection with artifact/noise cleanup and reading order."""

from pathlib import Path

from rasterregions import detect_regions
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
source = ROOT / "assets" / "01_multirow_sparse.png"
out = ROOT / "outputs" / "01_auto_regions"

result = detect_regions(source)
assert result.status == "ok"
assert len(result.regions) == 8
assert [region.row_index for region in result.regions] == [0] * 4 + [1] * 4
manifest = write_detection(result, out)

print(f"status={result.status} regions={len(result.regions)} mask={result.mask_result.method}")
for region in result.regions:
    print(
        region.region_id,
        "row",
        region.row_index,
        "col",
        region.column_index,
        "bbox",
        region.bbox.as_tuple(),
        "confidence",
        f"{region.confidence:.2f}",
    )
print("warnings:", *result.warnings, sep="\n- ")
print("manifest:", manifest)
