"""Partition one row into an exact expected number of anonymous regions."""

from pathlib import Path

from rasterregions import detect_regions
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
BUILD = ROOT.parent / "build" / "examples"


def main() -> None:
    source = ROOT / "assets" / "02_expected_count_three.png"
    result = detect_regions(source, expected_count=3)
    manifest = write_detection(result, BUILD / "02_expected_count")

    assert result.status == "ok"
    assert len(result.regions) == 3
    print("bboxes:", [region.bbox.as_tuple() for region in result.regions])
    print("confidences:", [round(region.confidence, 3) for region in result.regions])
    print("manifest:", manifest)


if __name__ == "__main__":
    main()
