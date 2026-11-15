"""Detect saturated colored marks while ignoring neutral gray worksheet guides."""

from pathlib import Path

from PIL import Image

from rasterregions import detect_regions, foreground_mask
from rasterregions.io import write_detection

ROOT = Path(__file__).resolve().parent
BUILD = ROOT.parent / "build" / "examples"


def main() -> None:
    source = ROOT / "assets" / "03_color_on_grid.png"
    with Image.open(source) as image:
        mask = foreground_mask(image, mode="auto")
    print("selected mask method:", mask.method, "confidence:", round(mask.confidence, 3))
    assert mask.mask[100, 57]
    assert not mask.mask[10, 100]

    result = detect_regions(source, mode="auto")
    assert result.status == "ok"
    assert len(result.regions) == 3
    assert not result.clean_mask[40, 100]
    write_detection(result, BUILD / "03_color_mask")
    print("detected regions:", len(result.regions))


if __name__ == "__main__":
    main()
