"""Generate a synthetic page and detect its anonymous regions."""

from pathlib import Path

from PIL import Image, ImageDraw

from rasterregions import detect_regions
from rasterregions.io import write_detection


def main() -> None:
    output = Path("build/synthetic-regions")
    output.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (300, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 38, 60, 76), outline="black", width=5)
    draw.line((130, 30, 130, 80), fill="black", width=5)
    draw.line((220, 40, 250, 75), fill="black", width=5)
    source = output / "input.png"
    image.save(source)

    result = detect_regions(source, expected_count=3)
    write_detection(result, output / "detected")
    print(result.status, [region.bbox.as_tuple() for region in result.regions])


if __name__ == "__main__":
    main()
