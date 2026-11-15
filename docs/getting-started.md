# Installation and quickstart

## Install

`rasterregions` requires Python 3.10 or newer, NumPy 1.24 or newer, and Pillow 10 or newer.

```bash
python -m pip install rasterregions
```

For a source checkout, install the package and development tools with:

```bash
python -m pip install -e ".[dev]"
```

## Detect regions

```python
from rasterregions import detect_regions
from rasterregions.io import write_detection

result = detect_regions("page.png")
for region in result.regions:
    print(region.region_id, region.bbox.as_tuple(), region.confidence)
    region.image.save(f"{region.region_id}.png")

manifest = write_detection(result, "build/page")
print(result.status, manifest)
```

Regions are anonymous crops. `rasterregions` does not recognize text, glyphs, or other application-specific classes. See [Concepts](guide.md) for input and coordinate rules, then [Detection](detection.md) for expected counts, regions of interest, and status handling.

## Mask only

```python
from rasterregions import foreground_mask

mask = foreground_mask("page.png", mode="auto")
print(mask.method, mask.confidence, mask.warnings)
```

The mask is boolean, with `True` for foreground. `mask.mask` is read-only. See [Masking](masking.md) for the available modes and configuration.

## Command line

```bash
rasterregions detect page.png --output build/page
```

The command writes `regions.json`, `mask.png`, `regions.png`, and region crops by default. See the [CLI reference](cli.md) for options and exit status.
