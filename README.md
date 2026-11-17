# rasterregions

`rasterregions` extracts anonymous, ordered crop regions from raster images. It provides foreground masks, connected components, rows, crops, and diagnostics. It does not perform OCR, assign glyph identities, or interpret document semantics.

## Install

```bash
python -m pip install rasterregions
```

Requires Python 3.10 or newer, NumPy 1.24 or newer, and Pillow 10 or newer. For a source checkout, use `python -m pip install -e ".[dev]"` to install development tools too.

## Quick start

```python
from rasterregions import detect_regions
from rasterregions.io import write_detection

result = detect_regions("page.png")
for region in result.regions:
    print(region.region_id, region.bbox.as_tuple(), region.confidence)

manifest = write_detection(result, "build/page")
print(result.status, manifest)
```

Set `expected_count` when external evidence provides a count. The default count scope partitions one qualifying row; `expected_count_scope="page"` constrains a count across rows. Pass a `BBox` as `roi` to constrain analysis while keeping full-image result coordinates.


For sparse objects separated by spacing evidence, opt into `detect_spaced_regions`. It supports per-row exact count hints or deterministic count inference, reports ambiguity, and keeps exact foreground ownership in each region mask. This does not change `detect_regions` defaults; see [Spaced-region segmentation](docs/spaced-segmentation.md).

```python
from rasterregions import detect_spaced_regions

result = detect_spaced_regions("page.png", row_counts=[None, 4])
print(result.status, len(result.regions), result.unassigned_mask.sum())
```

## Capabilities

- deterministic grayscale, dark-relative, color, hybrid, and adaptive foreground masks;
- 8-connected components, row grouping, anonymous ordered regions, and crops;
- optional expected-count partitioning using projection valleys and component-crossing costs;
- opt-in spacing-based row segmentation with per-row count hints, exact pixel ownership, and ambiguity diagnostics;
- nearest-component isolation proposals with explicit reason values;
- JSON, mask, crop, and overlay diagnostics through Python and the CLI.

Confidence values are heuristic scores, not calibrated probabilities. Automatic grouping is intended for separated foreground objects rather than dense prose or general document layout.

## Documentation

- [Installation and quickstart](docs/getting-started.md)
- [Concepts and input contracts](docs/guide.md)
- [Foreground masking](docs/masking.md)
- [Region detection](docs/detection.md)
- [Spaced-region segmentation](docs/spaced-segmentation.md)
- [Components and isolation](docs/components.md)
- [Diagnostics and output files](docs/diagnostics.md)
- [Command-line interface](docs/cli.md)
- [Python API reference](docs/api.md)
- [Compatibility and schema policy](docs/compatibility.md)
- [Changelog](CHANGELOG.md)
- [Manual release procedure](docs/releasing.md)
- [Visual examples](docs/examples.md)

## Examples

Run the complete deterministic example bundle from the repository root:

```bash
python examples/run_all.py
```

Inputs are regenerated from source. Diagnostics are written under `build/examples/`. See [the examples guide](docs/examples.md) for input/output figures and commands to regenerate or check them.
