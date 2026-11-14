"""Force dark-ink segmentation for non-white or shaded paper."""

from pathlib import Path

from rasterregions import detect_regions
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
BUILD = ROOT.parents[1] / "build" / "rasterregions_examples"


def main() -> None:
    source = ROOT / "assets" / "04_dark_ink_gray_paper.png"
    result = detect_regions(source, mode="dark")
    assert result.status == "ok"
    assert len(result.regions) == 3
    write_detection(result, BUILD / "04_dark_paper")
    print("status:", result.status)
    print("mask confidence:", round(result.mask_result.confidence, 3))
    print("regions:", [region.bbox.as_tuple() for region in result.regions])


if __name__ == "__main__":
    main()
