from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def draw_glyph(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str, *, fill="black", width=5):
    if kind == "e":
        draw.line(
            [
                (x, y + 18),
                (x + 24, y + 18),
                (x + 27, y + 12),
                (x + 23, y + 5),
                (x + 13, y + 2),
                (x + 5, y + 6),
                (x + 2, y + 15),
                (x + 5, y + 26),
                (x + 14, y + 31),
                (x + 24, y + 29),
            ],
            fill=fill,
            width=width,
        )
    elif kind == "l":
        draw.line((x + 10, y, x + 10, y + 32), fill=fill, width=width)
    elif kind == "i":
        draw.line((x + 10, y + 11, x + 10, y + 32), fill=fill, width=width)
        draw.ellipse((x + 7, y, x + 13, y + 6), fill=fill)
    elif kind == "n":
        draw.line(
            [
                (x, y + 32),
                (x, y + 7),
                (x + 8, y + 3),
                (x + 18, y + 5),
                (x + 24, y + 12),
                (x + 24, y + 32),
            ],
            fill=fill,
            width=width,
        )
    elif kind == "box":
        draw.rounded_rectangle((x, y, x + 28, y + 28), radius=5, outline=fill, width=width)
    elif kind == "cross":
        draw.line((x + 14, y, x + 14, y + 30), fill=fill, width=width)
        draw.line((x, y + 15, x + 28, y + 15), fill=fill, width=width)
    else:
        raise ValueError(kind)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS.glob("*.png"):
        asset.unlink()

    image = Image.new("RGB", (520, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 519, 259), outline=(20, 20, 20), width=2)
    for x, kind in ((35, "e"), (150, "i"), (275, "n"), (405, "l")):
        draw_glyph(draw, x, 40, kind)
    for x, kind in ((45, "n"), (175, "box"), (300, "i"), (420, "cross")):
        draw_glyph(draw, x, 155, kind)
    draw.line((15, 125, 505, 125), fill=(190, 190, 190), width=1)
    draw.point((495, 18), fill="black")
    image.save(ASSETS / "01_multirow_sparse.png")

    image = Image.new("RGB", (460, 150), "white")
    draw = ImageDraw.Draw(image)
    draw_glyph(draw, 35, 55, "e")
    draw_glyph(draw, 190, 45, "i")
    draw_glyph(draw, 350, 55, "n")
    draw.line((10, 125, 450, 125), fill=(175, 175, 175), width=1)
    draw.point((440, 12), fill="black")
    image.save(ASSETS / "02_expected_count_three.png")

    image = Image.new("RGB", (480, 220), "white")
    draw = ImageDraw.Draw(image)
    for x in range(40, 480, 40):
        draw.line((x, 0, x, 219), fill=(215, 215, 215), width=1)
    for y in range(40, 220, 40):
        draw.line((0, y, 479, y), fill=(215, 215, 215), width=1)
    for x, kind, color in (
        (55, "box", (20, 90, 220)),
        (205, "cross", (210, 45, 70)),
        (355, "n", (20, 155, 85)),
    ):
        draw_glyph(draw, x, 85, kind, fill=color, width=6)
    image.save(ASSETS / "03_color_on_grid.png")

    width, height = 500, 210
    image = Image.new("L", (width, height), 190)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = max(
                0,
                min(255, 185 + int(16 * x / (width - 1)) + int(6 * y / (height - 1))),
            )
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, kind in ((45, "e"), (195, "i"), (350, "n")):
        draw_glyph(draw, x, 80, kind, fill=(25, 25, 25), width=6)
    image.save(ASSETS / "04_dark_ink_gray_paper.png")

    image = Image.new("RGB", (420, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((35, 55, 105, 125), fill=(20, 20, 20))
    draw.rectangle((175, 65, 245, 135), fill=(20, 20, 20))
    draw.polygon([(320, 135), (355, 55), (390, 135)], fill=(20, 20, 20))
    draw.ellipse((205, 25, 215, 35), fill=(20, 20, 20))
    image.save(ASSETS / "05_component_isolation.png")

    image = Image.new("RGB", (420, 150), "white")
    draw = ImageDraw.Draw(image)
    for x in (35, 165, 295):
        draw.rounded_rectangle((x, 48, x + 72, 108), radius=10, outline="black", width=7)
    draw.line((107, 78, 165, 78), fill="black", width=2)
    draw.line((237, 78, 295, 78), fill="black", width=2)
    image.save(ASSETS / "06_touching_objects.png")

    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 75, 68, 110), fill="black")
    draw.rectangle((190, 75, 203, 110), fill=(20, 70, 220))
    draw.rectangle((325, 75, 338, 110), fill="black")
    draw.line((5, 155, 394, 155), fill=(180, 180, 180), width=1)
    image.save(ASSETS / "07_mixed_black_color.png")

    image = Image.new("RGB", (120, 80), "white")
    ImageDraw.Draw(image).rectangle((25, 20, 94, 59), fill="black")
    image.save(ASSETS / "08_single_region.png")

    print(f"Wrote assets to {ASSETS}")


if __name__ == "__main__":
    main()
