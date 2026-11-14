# User guide

## Installation

Install the runtime package with pip:

```bash
python -m pip install rasterregions
```

The runtime dependencies are NumPy and Pillow. For development, install the optional tools with `python -m pip install -e ".[dev]"`.

## Supported input and coordinate contract

`detect_regions` and `foreground_mask` accept a path, a Pillow image, or a NumPy array. Arrays may be two-dimensional grayscale or have three RGB or four RGBA channels. Values must be finite numeric values. Floating-point arrays in `[0, 1]` are scaled to `[0, 255]`; conversion clips values to that range. RGBA inputs are composited over white.

Paths and Pillow inputs are transposed using their EXIF orientation before detection. Coordinates, masks, overlays, and crops therefore refer to the orientation-corrected image. NumPy arrays have no EXIF metadata and retain their supplied orientation. Bounding boxes use half-open coordinates `(x0, y0, x1, y1)`, where the right and bottom edges are excluded.

## Detection

Automatic detection returns anonymous regions in row-major reading order:

```python
from rasterregions import detect_regions

result = detect_regions("page.jpg")
for region in result.regions:
    print(region.reading_order, region.bbox, region.confidence)
```

Rows and columns are zero-based. Regions within a row are ordered from left to right. The library does not infer right-to-left ordering or semantic labels.

An expected count requests projection-based partitioning:

```python
result = detect_regions("worksheet.png", expected_count=8)
```

The default `expected_count_scope="row"` partitions the best qualifying row. Use `expected_count_scope="page"` to constrain the total across detected rows. A caller should only provide a count when it has external evidence for that count.

## Masking and configuration

`foreground_mask` supports `auto`, `grayscale`, `dark`, `color`, `hybrid`, and `adaptive` modes. Auto mode scores candidate masks. Explicit color mode remains color-only. Adaptive mode uses a local mean threshold for uneven illumination.

`DetectionConfig` groups `CleanupConfig`, `GroupingConfig`, `MaskConfig`, and `PartitionConfig`. These frozen configuration objects validate finite values and domain constraints. Adaptive settings can be passed through detection without adding command-line options:

```python
from rasterregions import DetectionConfig, MaskConfig, detect_regions

config = DetectionConfig(mask=MaskConfig(adaptive_window_size=21, adaptive_offset=8.0))
result = detect_regions("page.png", mode="adaptive", config=config)
```

`foreground_mask` also accepts the `adaptive_window_size` and `adaptive_offset` keyword arguments directly.

## Status and confidence

Statuses are `ok`, `ambiguous`, and `failed`:

- Automatic detection is `ambiguous` if any returned region has confidence below `min_confidence`. The regions remain available for review.
- Expected-count detection is `failed` if no credible partition is found or the best partition confidence is below `min_confidence`. In that case, the result has no regions.
- Mask confidence is a separate diagnostic. It contributes warnings, but does not gate detection status.

Mask, region, and partition confidence values are deterministic heuristic scores in `[0, 1]`. They are not calibrated probabilities and should not be compared as though they share one evidence scale.

## Component identity and diagnostics

Cleanup decisions, rows, regions, and the JSON payload refer to component indexes in the original foreground mask. The payload declares `component_index_namespace: "raw_mask"`; removed components can therefore leave gaps in the indexes of retained components.

`DetectionResult.diagnostics` provides image dimensions, mask and cleanup information, parameters, and partition boundaries. `detection_payload(result)` returns a JSON-compatible representation with `schema_version` set to `1`.

## Writing output artifacts

`write_detection` writes the JSON manifest and optional crops, mask, and overlay:

```python
from rasterregions.io import write_detection

manifest_path = write_detection(result, "build/regions")
```

The function returns the `regions.json` path. Reruns remove stale managed crop files and disabled mask or overlay outputs while preserving unrelated files. Image artifacts are replaced atomically and the JSON manifest is published last as the completion marker.

The CLI exposes detection and artifact selection:

```bash
rasterregions detect page.png --expected-count 3 --output build/page
rasterregions detect page.png --json --no-crops --no-overlay --output build/page
```

Use `rasterregions detect --help` for the complete option list.
