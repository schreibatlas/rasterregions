"""Write bounded spaced-segmentation hypotheses and visual diagnostics."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from rasterregions import detect_spaced_region_hypotheses
from rasterregions.io import write_spaced_hypotheses

ROOT = Path(__file__).resolve().parent
BUILD = ROOT.parent / "build" / "examples" / "08_spaced_regions"


def _synthetic_page() -> Image.Image:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((15, 55, 49, 18), fill="black", width=3)
    draw.line((44, 55, 70, 18), fill="black", width=3)
    return image


def main() -> None:
    hypotheses = detect_spaced_region_hypotheses(_synthetic_page(), mode="dark")
    BUILD.mkdir(parents=True, exist_ok=True)
    write_spaced_hypotheses(hypotheses, BUILD)
    print("status:", hypotheses.selected.status)
    print("selected regions:", len(hypotheses.selected.regions))
    print("page hypotheses:", len(hypotheses.page_hypotheses))
    print("search truncated:", hypotheses.truncated)
    print("artifacts:", BUILD)


if __name__ == "__main__":
    main()
