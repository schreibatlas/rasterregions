from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


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


# 1. Sparse multi-row page with border, rule and tiny noise.
img = Image.new("RGB", (520, 260), "white")
d = ImageDraw.Draw(img)
d.rectangle((0, 0, 519, 259), outline=(20, 20, 20), width=2)
for x, k in ((35, "e"), (150, "i"), (275, "n"), (405, "l")):
    draw_glyph(d, x, 40, k)
for x, k in ((45, "n"), (175, "box"), (300, "i"), (420, "cross")):
    draw_glyph(d, x, 155, k)
d.line((15, 125, 505, 125), fill=(190, 190, 190), width=1)
d.point((495, 18), fill="black")
img.save(ASSETS / "01_multirow_sparse.png")

# 2. One row with three glyph-like groups and detached marks; good expected_count case.
img = Image.new("RGB", (460, 150), "white")
d = ImageDraw.Draw(img)
draw_glyph(d, 35, 55, "e")
draw_glyph(d, 190, 45, "i")
draw_glyph(d, 350, 55, "n")
d.line((10, 125, 450, 125), fill=(175, 175, 175), width=1)
d.point((440, 12), fill="black")
img.save(ASSETS / "02_expected_count_three.png")

# 3. Colored foreground on a worksheet-style gray grid.
img = Image.new("RGB", (480, 220), "white")
d = ImageDraw.Draw(img)
for x in range(40, 480, 40):
    d.line((x, 0, x, 219), fill=(215, 215, 215), width=1)
for y in range(40, 220, 40):
    d.line((0, y, 479, y), fill=(215, 215, 215), width=1)
for x, k, c in (
    (55, "box", (20, 90, 220)),
    (205, "cross", (210, 45, 70)),
    (355, "n", (20, 155, 85)),
):
    draw_glyph(d, x, 85, k, fill=c, width=6)
img.save(ASSETS / "03_color_on_grid.png")

# 4. Uneven gray paper with dark ink.
w, h = 500, 210
img = Image.new("L", (w, h), 190)
p = img.load()
for y in range(h):
    for x in range(w):
        # deterministic low-frequency paper shading, about 170..210
        p[x, y] = max(0, min(255, 185 + int(16 * x / (w - 1)) + int(6 * y / (h - 1))))
img = img.convert("RGB")
d = ImageDraw.Draw(img)
for x, k in ((45, "e"), (195, "i"), (350, "n")):
    draw_glyph(d, x, 80, k, fill=(25, 25, 25), width=6)
img.save(ASSETS / "04_dark_ink_gray_paper.png")

# 5. Component isolation: several disconnected blobs around anchors.
img = Image.new("RGB", (420, 220), "white")
d = ImageDraw.Draw(img)
d.ellipse((35, 55, 105, 125), fill=(20, 20, 20))
d.rectangle((175, 65, 245, 135), fill=(20, 20, 20))
d.polygon([(320, 135), (355, 55), (390, 135)], fill=(20, 20, 20))
d.ellipse((205, 25, 215, 35), fill=(20, 20, 20))
img.save(ASSETS / "05_component_isolation.png")

# 6. A deliberately difficult case: touching bodies in one row.
img = Image.new("RGB", (420, 150), "white")
d = ImageDraw.Draw(img)
# three block-like symbols joined by thin bridges
for x in (35, 165, 295):
    d.rounded_rectangle((x, 48, x + 72, 108), radius=10, outline="black", width=7)
d.line((107, 78, 165, 78), fill="black", width=2)
d.line((237, 78, 295, 78), fill="black", width=2)
img.save(ASSETS / "06_touching_objects.png")

print(f"Wrote assets to {ASSETS}")
