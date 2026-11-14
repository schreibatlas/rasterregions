# rasterregions

`rasterregions` is a small, dependency-light Python library for turning raster
images into **anonymous, ordered crop regions**. It deliberately does not do
OCR and has no handwriting- or glyph-specific concepts.

It was extracted from the generic image-segmentation work in InkGrammar.

## MVP scope

- foreground-mask extraction for grayscale, dark-on-paper, colored, hybrid, and adaptive modes
- 8-connected component detection
- removal of obvious page borders, long rules, and tiny isolated noise
- row detection
- label-free region detection in reading order
- detached-mark/satellite attachment (for example dots above a body)
- optional `expected_count=N` partitioning using projection valleys and
  component-crossing costs
- bbox tightening and cropping
- nearest-component isolation around an anchor point
- JSON, mask, crop, and overlay diagnostics
- CLI and Python API

The public API returns geometry and confidence only. It never assigns semantic
labels.

## Install

```bash
python -m pip install -e .
```

Development install:

```bash
python -m pip install -e ".[dev]"
```

## Python API

Fully automatic detection:

```python
from rasterregions import detect_regions

result = detect_regions("page.png")
for region in result.regions:
    print(region.region_id, region.bbox, region.confidence)
    region.image.save(f"{region.region_id}.png")
```

Ask for exactly three regions:

```python
result = detect_regions("written.png", expected_count=3)
assert len(result.regions) == 3
```


`expected_count` is row-scoped by default. It partitions the best qualifying row
into the requested number of regions. To constrain the total over all detected
rows, use page scope:

```python
result = detect_regions(
    "worksheet.png", expected_count=8, expected_count_scope="page"
)
```

Confidence values are deterministic heuristic scores in `[0, 1]`, based on
separation and segment plausibility. They are not calibrated probabilities.


Cleanup and grouping heuristics can be tuned through a configuration object:

```python
from rasterregions import CleanupConfig, DetectionConfig, detect_regions

config = DetectionConfig(cleanup=CleanupConfig(rule_min_span_ratio=0.3))
result = detect_regions("page.png", config=config)
```

Mask-only operation:

```python
from rasterregions import foreground_mask

mask = foreground_mask(image, mode="auto")
print(mask.method, mask.confidence)
```


`auto` deterministically scores grayscale, dark-relative, color, and hybrid
candidates. Explicit `color` mode remains color-only. The opt-in `adaptive` mode
uses a local mean threshold for uneven illumination. `foreground_mask` also
accepts `adaptive_window_size` and `adaptive_offset`.

Component isolation:

```python
from rasterregions import isolate_component

proposal = isolate_component(mask, center=(120, 80))
```

## CLI

```bash
rasterregions detect page.png --output build/regions
```

For an expected number of regions:

```bash
rasterregions detect written.png --expected-count 3 --output build/written
```


For page-wide counts and machine-readable stdout:

```bash
rasterregions detect worksheet.png --expected-count 8 \
  --expected-count-scope page --json --no-crops --no-overlay --output build/page
```

Use `--no-crops`, `--no-overlay`, or `--no-mask` to omit individual image
artifacts. The JSON manifest is still written to the output directory.

The output directory contains:

```text
regions.json
mask.png
regions.png
crops/
  region-000.png
  region-001.png
  ...
```


`regions.json` adds the package version, image dimensions, effective parameters,
mask threshold and foreground fraction, cleanup decisions, row metadata, and
partition boundaries. Existing manifest keys remain available.

## Package layout

There is intentionally **no `src/` layer**:

```text
rasterregions/
  __init__.py
  __main__.py
  cli.py
  components.py
  detect.py
  io.py
  mask.py
  models.py
  regions.py
examples/
tests/
pyproject.toml
```

## Dynamic versioning

Versions come from Git tags through `setuptools-scm`; no version constant is
maintained by hand. The build configuration uses:

```toml
[project]
dynamic = ["version"]

[tool.setuptools_scm]
write_to = "rasterregions/_version.py"
fallback_version = "0.1.0.dev0"
```

A tagged checkout such as `v0.1.0` therefore builds as `0.1.0`. A source ZIP
without `.git` metadata uses the explicit development fallback.

## Design boundary

`rasterregions` owns:

```text
pixels -> foreground mask -> components -> rows -> anonymous regions/crops
```

Applications own:

```text
region -> OCR / glyph identity / handwriting semantics / domain labels
```

This boundary keeps the library reusable for document scans, forms,
worksheets, sprite extraction, diagrams, signatures, historical documents,
and handwriting pipelines.

## Current limitations

This is an MVP, not a general document-layout engine. Automatic grouping is
best for sparse pages containing separated foreground objects. Touching
objects, dense prose, rotated layouts, nested regions, and arbitrary document
layout require stronger algorithms. `expected_count` is intentionally offered
as a useful constrained mode for those cases where the caller knows how many
objects should be present.
